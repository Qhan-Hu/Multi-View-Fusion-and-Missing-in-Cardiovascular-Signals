import torch
import torch.nn as nn
import pytorch_lightning as pl
import mmbt.modules.trans.multi_biosig_transformer_prompts as bist
from mmbt.modules import heads, objectives, mmbt_utils


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

        self.ppg_pooler = heads.Pooler(config["hidden_size"])
        self.ecg_pooler = heads.Pooler(config["hidden_size"])
        self.ppg_pooler.apply(objectives.init_weights)
        self.ecg_pooler.apply(objectives.init_weights)
        # == Optional: Load Pretraining Weights ==
        if (
            self.hparams.config["load_path"] != ""
            and self.hparams.config["finetune_first"]
            and not self.hparams.config["test_only"]
        ):
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.token_type_embeddings.parameters():
                param.requires_grad = False
            print("use pre-finetune model")
        # == End  : 1. Build Models ==

        # == Begin 2. Build Heads For Downstream Tasks ==
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
        # == End:  2. Build Heads For Downstream Tasks ==

        # == Begin: 3. Build Missing-Aware Prompt ==
        self.prompt_type = self.hparams.config["prompt_type"]
        prompt_length = self.hparams.config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams.config["hidden_size"]
        self.learnt_p = self.hparams.config["learnt_p"]
        self.prompt_layers = self.hparams.config["prompt_layers"]
        self.multi_layer_prompt = self.hparams.config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1
        
        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            complete_prompt[:,prompt_length//2:prompt_length//2+1,:].fill_(1)
        self.complete_prompt = nn.Parameter(complete_prompt)

        missing_ecg_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_ecg_prompt[:,2:3,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_ecg_prompt[:,prompt_length//2+2:prompt_length//2+3,:].fill_(1)
        self.missing_ecg_prompt = nn.Parameter(missing_ecg_prompt)

        missing_ppg_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_ppg_prompt[:,1:2,:].fill_(1)            
        if self.learnt_p and self.prompt_type == 'attention':
            missing_ppg_prompt[:,prompt_length//2+1:prompt_length//2+2,:].fill_(1)
        self.missing_ppg_prompt = nn.Parameter(missing_ppg_prompt)
        
        if not self.learnt_p:
            self.complete_prompt.requires_grad=False
            self.missing_ecg_prompt.requires_grad=False           
            self.missing_ppg_prompt.requires_grad=False

        # print(self.complete_prompt)
        # print(self.missing_ppg_prompt)
        # print(self.missing_ecg_prompt)
        # == End: 3. Build Missing-Aware Prompt ==



        # ===================== load downstream (test_only) ======================
        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.load_state_dict(state_dict, strict=False)




    def infer(self, batch, mask_both=False, is_train=None):
        ppg = batch["ppg_norm"]
        ecg = batch["ecg_norm"]

        if self.hparams.config["pos_embed"] == "CS":
            self.transformer.ppg_pos_embed.pos_embed = self.transformer.ppg_pos_embed.pos_embed.to(self.device)
            self.transformer.ecg_pos_embed.pos_embed = self.transformer.ecg_pos_embed.pos_embed.to(self.device)

        ppg_embeds, ecg_embeds = self.transformer.multi_signal_embed(ppg, ecg)

        ppg_embeds, ecg_embeds = (
            ppg_embeds + self.token_type_embeddings(
                torch.zeros_like(ppg_embeds[:, :, 0].to(torch.int64)
                                 )),
            ecg_embeds + self.token_type_embeddings(
                torch.full_like(ppg_embeds[:, :, 0].to(torch.int64), 1))
        )

        # instance wise missing aware prompts
        prompts = None
        for idx in range(len(ppg)):
            if batch["missing_type"][idx] == 0:
                prompt = self.complete_prompt
            elif batch["missing_type"][idx] == 1:
                prompt = self.missing_ecg_prompt
            elif batch["missing_type"][idx] == 2:
                prompt = self.missing_ppg_prompt
            #
            # if prompt.size(0) != 1:
            prompt = prompt.unsqueeze(0)

            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)

            
        co_embeds = torch.cat([ppg_embeds, ecg_embeds], dim=1)
        x = co_embeds.detach()

        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x,
                                   prompts=prompts[:,self.prompt_layers.index(i)],
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x)

        x = self.transformer.norm(x)

        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]
        elif self.prompt_type == 'attention':
            total_prompt_len = 0

        ppg_feats, ecg_feats = (
            x[:, total_prompt_len: total_prompt_len+ppg_embeds.shape[1]],
            x[:, total_prompt_len+ppg_embeds.shape[1]:],
        )
        ppg_cls_feat = self.ppg_pooler(ppg_feats)
        ecg_cls_feat = self.ecg_pooler(ecg_feats)

        ret = {
            "ppg_cls_feat": ppg_cls_feat,
            "ecg_cls_feat": ecg_cls_feat,
            "raw_cls_feats": x[:, 0],
            "label": batch['label'],
        }
        return ret


    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

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
        total_loss = sum([v for k, v in output.items() if "loss" in k])
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

        if self.hparams.config["loss_names"]["bpe"] > 0:
            ret.update(objectives.bpe_test_step(self, batch, output))

        if self.hparams.config["loss_names"]["afd"] > 0:
            ret.update(objectives.afd_test_step(self, batch, output))

        if self.hparams.config["loss_names"]["ssc"] > 0:
            ret.update(objectives.ssc_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["bpe"] > 0:
            objectives.bpe_test_wrapup(outs, model_name)

        if self.hparams.config["loss_names"]["afd"] > 0:
            objectives.afd_test_wrapup(outs, model_name)

        if self.hparams.config["loss_names"]["ssc"] > 0:
            objectives.ssc_test_wrapup(outs, model_name)

        mmbt_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        return mmbt_utils.set_schedule(self)