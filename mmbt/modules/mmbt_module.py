import torch
import torch.nn as nn
import pytorch_lightning as pl
import mmbt.modules.trans.multi_biosig_transformer as bist
import numpy as np
from mmbt.modules import heads, objectives, mmbt_utils
from timm.models.layers import trunc_normal_


class MMBTransformerSS(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        # == Begin: 1. Build Models ==
        self.token_type_embeddings = nn.Embedding(2, config['hidden_size'])
        self.token_type_embeddings.apply(objectives.init_weights)
        self.global_token = nn.Parameter(torch.zeros(1, 1, config['hidden_size']))

        if self.hparams.config['load_path'] == "":
            self.transformer = getattr(bist, self.hparams.config["bist"])(
                pretrained=True, config=self.hparams.config
            )
        else:
            self.transformer = getattr(bist, self.hparams.config["bist"])(
                pretrained=False, config=self.hparams.config
            )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ppg_pooler = heads.Pooler(config["hidden_size"])
        self.ecg_pooler = heads.Pooler(config["hidden_size"])
        self.ppg_pooler.apply(objectives.init_weights)
        self.ecg_pooler.apply(objectives.init_weights)
        trunc_normal_(self.global_token, std=0.02)
        # == Optional: Load Pretraining Weights ==
        if (
            self.hparams.config["load_path"] != ""
            and self.hparams.config["finetune_first"]
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            # for param in self.transformer.blocks.parameters():
            #     param.requires_grad = False
            # for param in self.token_type_embeddings.parameters():
            #     param.requires_grad = False
            # print("use pre-finetune model")
        # == End  : 1. Build Models ==

        # == Begin: 2. Build Pre-Training Heads ==
        if config["loss_names"]["msm"] > 0:
            self.msm_head = heads.MSMHead(config)
            self.msm_head.apply(objectives.init_weights)
        # == End  : 2. Build Pre-Training Heads ==

        # == Begin 3. Build Heads For Downstream Tasks ==
        if config["loss_names"]["bpe"] > 0:
            self.bpe_head = heads.BPEHead(config["hidden_size"])
            self.bpe_head.apply(objectives.init_weights)

        if config["loss_names"]["afd"] > 0:
            self.afd_head = heads.AFDHead(config["hidden_size"])
            self.afd_head.apply(objectives.init_weights)

        if config["loss_names"]["ssc"] > 0:
            self.ssc_head = heads.SSCHead(config["hidden_size"])
            self.ssc_head.apply(objectives.init_weights)

        mmbt_utils.set_metrics(self)
        self.current_tasks = list()
        # == End:  3. Build Heads For Downstream Tasks ==


        # == Begin: 4. Load Models For Testing ==
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)

        # == End: 4. Load Models For Testing ==

    def patchify(self, sig, patch_size):
        """
        sig: [B, C, L], C=1 for ECG, C=3 for PPG
        x: [B, N, pacth_size*C]
        """
        assert len(sig.shape) == 3
        B, C, L = sig.shape
        num_patches = L // patch_size
        sig = sig[:, :, :num_patches*patch_size]
        x = sig.reshape(B, C, num_patches, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, num_patches, patch_size*C)
        return x

    def infer(self, batch, mask_both=False, is_train=None):
        ppg = batch["ppg_norm"]
        ecg = batch["ecg_norm"]
        B, _, _ = ppg.shape


        do_msm = "_msm" if mask_both else ""

        ppg_ids_keep, ecg_ids_keep = batch[f"ppg_input_patch_idx{do_msm}"], batch[f"ecg_input_patch_idx{do_msm}"]
        ppg_ids_keep = torch.cat(
            (torch.zeros([B, 1], dtype=ppg_ids_keep.dtype, device=ppg_ids_keep.device), ppg_ids_keep + 1), dim=1)
        ecg_ids_keep = torch.cat(
            (torch.zeros([B, 1], dtype=ecg_ids_keep.dtype, device=ecg_ids_keep.device), ecg_ids_keep + 1), dim=1)
        ecg_ids_restore = batch["ecg_ids_restore"] if do_msm else None
        ppg_ids_restore = batch["ppg_ids_restore"] if do_msm else None

        # patchify -> embedding -> concat cls token -> add position embedding
        ppg_embeds, ecg_embeds = self.transformer.multi_signal_embed(ppg,
                                                                     ecg)  # output: (B, L+1, H), L: sequence length, 1: cls token

        # save the unmasked embeddings for msm pretraining or all the embeddings for the downstream tasks
        ppg_embeds = torch.gather(ppg_embeds, dim=1,
                                  index=ppg_ids_keep.unsqueeze(-1).expand(-1, -1, ppg_embeds.shape[-1]))
        ecg_embeds = torch.gather(ecg_embeds, dim=1,
                                  index=ecg_ids_keep.unsqueeze(-1).expand(-1, -1, ecg_embeds.shape[-1]))

        ppg_embeds, ecg_embeds = (
            ppg_embeds + self.token_type_embeddings(
                torch.zeros_like(ppg_embeds[:, :, 0].to(torch.int64)
                                 )),
            ecg_embeds + self.token_type_embeddings(
                torch.full_like(ecg_embeds[:, :, 0].to(torch.int64), 1))
        )
        co_embeds = torch.cat([ppg_embeds, ecg_embeds], dim=1)
        x = co_embeds
        # x = ecg_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x)

        x = self.transformer.norm(x)
        ppg_feats, ecg_feats = (
            x[:, :ppg_embeds.shape[1]],
            x[:, ppg_embeds.shape[1]:],
        )
        # ppg_feats = x[:, :ppg_embeds.shape[1]]
        ppg_cls_feat = self.ppg_pooler(ppg_feats)
        ecg_cls_feat = self.ecg_pooler(ecg_feats)

        labels = batch['label'] if any("label" in k for k in batch.keys()) else None

        ret = {
            "ppg_feats": ppg_feats,
            "ecg_feats": ecg_feats,
            "ppg_cls_feat": ppg_cls_feat,
            "ecg_cls_feat": ecg_cls_feat,
            "ecg_ids_res": ecg_ids_restore,
            "ppg_ids_res": ppg_ids_restore,
            "label": labels,
        }
        return ret


    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        if "msm" in self.current_tasks:
            ret.update(objectives.compute_msm(self, batch))

        if "cep" in self.current_tasks:
            ret.update(objectives.compute_cep(self, batch))

        if "bpe" in self.current_tasks:
            ret.update(objectives.compute_bpe(self, batch))

        if "afd" in self.current_tasks:
            ret.update(objectives.compute_afd(self, batch))

        if "ssc" in self.current_tasks:
            ret.update(objectives.compute_ssc(self, batch))


        return ret


    def training_step(self, batch, batch_idx):
        """forward propagation, loss calculation, back propagation for single batch"""
        mmbt_utils.set_task(self)
        output = self(batch)
        # total_loss = sum([v for k, v in output.items() if "loss" in k])
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        return total_loss

    def training_epoch_end(self, outputs):
        mmbt_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        mmbt_utils.set_task(self)
        output = self(batch)

    def validation_epoch_end(self, outs):
        mmbt_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        mmbt_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["msm"] > 0:
            ret.update(objectives.msm_test_step(self, batch, output))

        if self.hparams.config["loss_names"]["bpe"] > 0:
            ret.update(objectives.bpe_test_step(self, batch, output))

        if self.hparams.config["loss_names"]["afd"] > 0:
            ret.update(objectives.afd_test_step(self, batch, output))

        if self.hparams.config["loss_names"]["ssc"] > 0:
            ret.update(objectives.ssc_test_step(self, batch, output))


        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["msm"] > 0:
            objectives.msm_test_wrapup(outs, model_name)

        if self.hparams.config["loss_names"]["bpe"] > 0:
            objectives.bpe_test_wrapup(outs, model_name)

        if self.hparams.config["loss_names"]["afd"] > 0:
            objectives.afd_test_wrapup(outs, model_name)

        if self.hparams.config["loss_names"]["ssc"] > 0:
            objectives.ssc_test_wrapup(outs, model_name)

        mmbt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return mmbt_utils.set_schedule(self)