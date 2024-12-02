import matplotlib.pyplot as plt
import torch
import json
import os
import glob
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from einops import rearrange
import math



def compute_msm(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    infer = pl_module.infer(batch, mask_both=True)

    biosig_logits = pl_module.msm_head(infer)
    ppgs_targets = pl_module.patchify(batch["ppg_norm"], pl_module.hparams.config["ppg_patch_size"])
    ecgs_targets = pl_module.patchify(batch["ecg_norm"], pl_module.hparams.config["ecg_patch_size"])

    ppgs_loss_, ecgs_loss_ = F.l1_loss(biosig_logits["ppg_preds"], ppgs_targets, reduction="none"), F.l1_loss(biosig_logits["ecg_preds"], ecgs_targets, reduction="none")
    # ppgs_loss_, ecgs_loss_ = (biosig_logits["ppg_preds"] - ppgs_targets) ** 2, (biosig_logits["ecg_preds"] - ecgs_targets) ** 2
    ppgs_loss_, ecgs_loss_ = ppgs_loss_.mean(dim=-1), ecgs_loss_.mean(dim=-1)
    # binary mask: 0 is keep, 1 is remove
    ppgs_loss_, ecgs_loss_ = (ppgs_loss_*batch["ppg_mask_msm"]).sum()/ batch["ppg_mask_msm"].sum(), (ecgs_loss_*batch["ecg_mask_msm"]).sum()/ batch["ecg_mask_msm"].sum()

    total_loss_ = ppgs_loss_ + ecgs_loss_

    ret = {
        "msm_loss": total_loss_,
        "ppgs_pred": biosig_logits["ppg_preds"],
        "ppgs_target": ppgs_targets,
        "ecgs_pred": biosig_logits["ecg_preds"],
        "ecgs_target": ecgs_targets,
        "ppg_mask_msm": batch["ppg_mask_msm"],
        "ecg_mask_msm": batch["ecg_mask_msm"],
    }

    ppgs_loss = getattr(pl_module, f"{phase}_msm_ppg_loss")(ppgs_loss_)
    ecgs_loss = getattr(pl_module, f"{phase}_msm_ecg_loss")(ecgs_loss_)
    total_loss = getattr(pl_module, f"{phase}_msm_total_loss")(total_loss_)

    pl_module.log(f"msm/{phase}/ppg_loss", ppgs_loss)
    pl_module.log(f"msm/{phase}/ecg_loss", ecgs_loss)
    pl_module.log(f"msm/{phase}/total_loss", total_loss)

    return ret


def compute_cep(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    infer = pl_module.infer(batch, mask_both=False)
    ppg_feats = infer["ppg_cls_feat"]
    ecg_feats = infer["ecg_cls_feat"]
    ppg_feats = ppg_feats / ppg_feats.norm(dim=-1, keepdim=True)
    ecg_feats = ecg_feats / ecg_feats.norm(dim=-1, keepdim=True)

    logit_scale = pl_module.logit_scale.exp()
    logits_per_ppg = logit_scale * ppg_feats @ ecg_feats.t()
    logits_per_ecg = logits_per_ppg.t()
    labels = torch.arange(logits_per_ppg.size(0), device=logits_per_ppg.device, dtype=torch.long)
    total_loss = (F.cross_entropy(logits_per_ppg, labels) + F.cross_entropy(logits_per_ecg, labels)) / 2

    ret = {
        "cep_loss": total_loss,
        "cep_ppg_logits": logits_per_ppg,
        "cep_ecg_logits": logits_per_ecg,
    }

    loss = getattr(pl_module, f"{phase}_cep_loss")(ret["cep_loss"])
    acc = getattr(pl_module, f"{phase}_cep_accuracy")(ret["cep_ppg_logits"], labels)

    pl_module.log(f"cep/{phase}/loss", loss)
    pl_module.log(f"cep/{phase}/acc", acc)

    return ret


def compute_bpe(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    infer = pl_module.infer(batch, mask_both=False)

    bpcls_logits = pl_module.bpe_head(infer["ppg_cls_feat"])
    bpcls_labels = batch["label"]
    bpcls_labels_norm = (bpcls_labels - torch.tensor([61.87, 121.42]).to(pl_module.device)) / torch.Tensor([13.01, 22.10]).to(pl_module.device)
    # bpcls_labels_norm = torch.tensor(bpcls_labels_norm).to(pl_module.device)
    bp_loss = F.smooth_l1_loss(bpcls_logits, bpcls_labels_norm)

    ret = {
        "bpe_loss": bp_loss,
        "bpe_logits": bpcls_logits,
        "bpe_labels_raw": bpcls_labels,
        "bpe_labels_norm": bpcls_labels_norm,
    }

    # Scalar: the loss value of each train/val step is added to the attribute 'scalar',
    # the attribute 'scalar' counts the train/val step number.
    loss = getattr(pl_module, f"{phase}_bpe_loss")(ret["bpe_loss"])
    mse = getattr(pl_module, f"{phase}_bpe_mse")(
        ret["bpe_logits"], ret["bpe_labels_norm"]
    )

    pl_module.log(f"bpe/{phase}/loss", loss)
    pl_module.log(f"bpe/{phase}/mse", mse)
    # print(loss)

    return ret


def compute_afd(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    infer = pl_module.infer(batch, mask_both=False)

    afd_logits = pl_module.afd_head(infer["ppg_cls_feat"])
    afd_labels = batch["label"].squeeze(1).long()

    total_loss = F.cross_entropy(afd_logits, afd_labels)

    ret = {
        "afd_loss": total_loss,
        "afd_logits": afd_logits,
        "afd_labels": afd_labels,
    }

    loss = getattr(pl_module, f"{phase}_afd_loss")(ret["afd_loss"])
    acc = getattr(pl_module, f"{phase}_afd_accuracy")(afd_logits, afd_labels)

    pl_module.log(f"afd/{phase}/loss", loss)
    pl_module.log(f"afd/{phase}/acc", acc)

    return ret


def compute_ssc(pl_module, batch):
    phase = "train" if pl_module.training else "val"
    infer = pl_module.infer(batch, mask_both=False)

    ssc_logits = pl_module.ssc_head(infer["ppg_cls_feat"])
    ssc_labels = batch["label"].squeeze(1).long()

    total_loss = F.cross_entropy(ssc_logits, ssc_labels)

    ret = {
        "ssc_loss": total_loss,
        "ssc_logits": ssc_logits,
        "ssc_labels": ssc_labels,
    }

    loss = getattr(pl_module, f"{phase}_ssc_loss")(ret["ssc_loss"])
    acc = getattr(pl_module, f"{phase}_ssc_accuracy")(ssc_logits, ssc_labels)

    pl_module.log(f"ssc/{phase}/loss", loss)
    pl_module.log(f"ssc/{phase}/acc", acc, prog_bar=True)

    return ret


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def bpe_test_step(pl_module, batch, output):
    bpe_logits = output["bpe_logits"]
    bpe_labels_raw = output["bpe_labels_raw"]
    bpe_logits = bpe_logits * torch.Tensor([13.01, 22.10]).to(pl_module.device) + torch.tensor([61.87, 121.42]).to(pl_module.device)
    return {"labels": bpe_labels_raw, "preds": bpe_logits}


def msm_test_step(pl_module, batch, output):
    return{
        "ppgs_pred": output["ppgs_pred"],
        "ecgs_pred": output["ecgs_pred"],
        "ppgs_target": output["ppgs_target"],
        "ecgs_target": output["ecgs_target"],
        "ppg_mask_msm": output["ppg_mask_msm"],
        "ecg_mask_msm": output["ecg_mask_msm"],
    }


def afd_test_step(pl_module, batch, output):
    afd_logits = output["afd_logits"]
    afd_labels = output["afd_labels"]
    return {"labels": afd_labels, "preds": afd_logits}


def ssc_test_step(pl_module, batch, output):
    ssc_logits = output["ssc_logits"]
    ssc_labels = output["ssc_labels"]
    return {"labels": ssc_labels, "preds": ssc_logits}


def afd_test_wrapup(outs, model_name):
    afd_labels, afd_preds = list(), list()
    for out in outs:
        afd_labels += out["labels"].reshape(out["labels"].shape[0], -1)
        afd_preds += out["preds"].argmax(dim=-1).reshape(out["preds"].shape[0], -1)

    afd_labels = torch.cat(afd_labels, dim=0)
    afd_preds = torch.cat(afd_preds, dim=0)

    total_num = afd_labels.shape[0]
    correct_num = (afd_labels == afd_preds).sum().item()
    tp = sum((afd_labels == 1) & (afd_preds == 1)).item()
    fp = sum((afd_labels == 0) & (afd_preds == 1)).item()
    fn = sum((afd_labels == 1) & (afd_preds == 0)).item()
    AF_num = (afd_labels == 1).sum().item()
    NSR_num = (afd_labels == 0).sum().item()

    se = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    acc = correct_num / total_num
    print('-' * 32)
    print('numbers of AF sample:{}'.format(AF_num), 'numbers of NSR sample:{}'.format(NSR_num))
    print('Accuracy of Atrial Fibrillation Detection:{:.4f}'.format(acc))
    print('Sensitivity of Atrial Fibrillation Detection:{:.4f}'.format(se))
    print('F1 score of Atrial Fibrillation Detection:{:.4f}'.format(f1))
    print('-' * 32)


def ssc_test_wrapup(outs, model_name):
    ssc_labels, ssc_preds = list(), list()
    for out in outs:
        ssc_labels += out["labels"].reshape(out["labels"].shape[0], -1)
        ssc_preds += F.softmax(out["preds"], dim=-1).argmax(dim=-1).reshape(out["preds"].shape[0], -1)

    ssc_labels = torch.cat(ssc_labels, dim=0)
    ssc_preds = torch.cat(ssc_preds, dim=0)

    total_num = ssc_labels.shape[0]
    correct_num = (ssc_labels == ssc_preds).sum().item()
    wake_num = (ssc_labels == 0).sum().item()
    light_num = (ssc_labels == 1).sum().item()
    deep_num = (ssc_labels == 2).sum().item()
    rem_num = (ssc_labels == 3).sum().item()
    acc = correct_num / total_num
    f1 = f1_score(ssc_labels.cpu().numpy(), ssc_preds.cpu().numpy(), average="macro")

    print('-' * 32)
    print('numbers of wake sample:{}'.format(wake_num))
    print('numbers of light sample:{}'.format(light_num))
    print('numbers of deep sample:{}'.format(deep_num))
    print('numbers of rem sample:{}'.format(rem_num))
    print('Accuracy of Sleep Stage Classification:{:.4f}'.format(acc))
    print('F1-marco of Sleep Stage Classification:{:.4f}'.format(f1))
    print('-' * 32)

    cm = confusion_matrix(np.array(ssc_labels.cpu()), np.array(ssc_preds.cpu()))
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=["Wake", "Light", "Deep", "REM"],
                yticklabels=["Wake", "Light", "Deep", "REM"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"Sleep Staging CM.png")
    plt.close()

def msm_test_wrapup(outs, model_name):
    # rank = torch.distributed.get_rank()
    ppgs_target, ppgs_pred, ecgs_target, ecgs_pred, ppg_mask_msm, ecg_mask_msm = list(), list(), list(), list(), list(), list()
    for out in outs:
        ppgs_target += out["ppgs_target"].reshape(out["ppgs_target"].shape[0], out["ppgs_target"].shape[1], 3, -1)
        ecgs_target += out["ecgs_target"].reshape(out["ecgs_target"].shape[0], out["ecgs_target"].shape[1], -1)
        ppgs_pred += out["ppgs_pred"].reshape(out["ppgs_pred"].shape[0], out["ppgs_pred"].shape[1], 3, -1)
        ecgs_pred += out["ecgs_pred"].reshape(out["ecgs_pred"].shape[0], out["ecgs_pred"].shape[1], -1)
        ppg_mask_msm += out["ppg_mask_msm"]
        ecg_mask_msm += out["ecg_mask_msm"]

    tensor_rets = {
        "ppgs_target": torch.stack(ppgs_target, dim=0),
        "ecgs_target": torch.stack(ecgs_target, dim=0),
        "ppgs_pred": torch.stack(ppgs_pred, dim=0),
        "ecgs_pred": torch.stack(ecgs_pred, dim=0),
        "ppg_mask_msm": torch.stack(ppg_mask_msm, dim=0),
        "ecg_mask_msm": torch.stack(ecg_mask_msm, dim=0),
    }
    ppg_masked_ids = torch.stack(
        [torch.nonzero(tensor_rets["ppg_mask_msm"][i, :]).squeeze(1) for i in range(tensor_rets["ppg_mask_msm"].size(0))],
        dim=0
    )
    ecg_masked_ids = torch.stack(
        [torch.nonzero(tensor_rets["ecg_mask_msm"][i, :]).squeeze(1) for i in range(tensor_rets["ecg_mask_msm"].size(0))],
        dim=0
    )
    ppgs_seq_target = tensor_rets["ppgs_target"].permute(0, 2, 1, 3).reshape(tensor_rets["ppgs_target"].shape[0], tensor_rets["ppgs_target"].shape[2], -1).cpu()
    ecgs_seq_target = tensor_rets["ecgs_target"].reshape(tensor_rets["ecgs_target"].shape[0], -1).cpu()
    for i in range(ppgs_seq_target.shape[0]):
        # tensor_rets["ppgs_target"][i, ppg_masked_ids[i]] = torch.zeros_like(tensor_rets["ppgs_target"][i, ppg_masked_ids[i]])
        # tensor_rets["ecgs_target"][i, ecg_masked_ids[i]] = torch.zeros_like(tensor_rets["ecgs_target"][i, ecg_masked_ids[i]])
        tensor_rets["ppgs_target"][i, ppg_masked_ids[i]] = math.nan
        tensor_rets["ecgs_target"][i, ecg_masked_ids[i]] = math.nan
    ppgs_seq_target_masked = tensor_rets["ppgs_target"].permute(0, 2, 1, 3).reshape(tensor_rets["ppgs_target"].shape[0], tensor_rets["ppgs_target"].shape[2], -1).cpu().numpy()
    ecgs_seq_target_masked = tensor_rets["ecgs_target"].reshape(tensor_rets["ecgs_target"].shape[0], -1).cpu().numpy()

    for i in range(ppgs_seq_target.shape[0]):
        tensor_rets["ppgs_target"][i, ppg_masked_ids[i]] = tensor_rets["ppgs_pred"][i, ppg_masked_ids[i]]
        tensor_rets["ecgs_target"][i, ecg_masked_ids[i]] = tensor_rets["ecgs_pred"][i, ecg_masked_ids[i]]
    ppgs_seq_pred = tensor_rets["ppgs_target"].permute(0, 2, 1, 3).reshape(tensor_rets["ppgs_target"].shape[0], tensor_rets["ppgs_target"].shape[2], -1).cpu().numpy()
    ecgs_seq_pred = tensor_rets["ecgs_target"].reshape(tensor_rets["ecgs_target"].shape[0], -1).cpu().numpy()
    for i in range(10):
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, sharex=True)

        ax1.plot(ppgs_seq_target[i, 0, :])
        # ax1.set_xlabel("Raw PPG")

        ax2.plot(ppgs_seq_target_masked[i, 0, :])
        ax3.plot(ppgs_seq_pred[i, 0, :])
        # ax2.set_xlabel("Masked PPG")
        # ax3.set_xlabel("Reconstructed PPG")

        ax4.plot(ecgs_seq_target[i, :])
        # ax4.set_xlabel("Raw ECG")

        ax5.plot(ecgs_seq_target_masked[i, :])
        ax6.plot(ecgs_seq_pred[i, :])
        # ax5.set_xlabel("Masked ECG")
        # ax6.set_xlabel("Reconstructed ECG")

        # plt.rcParams['font.sans-serif'] = ['Times New Roman']
        # plt.rcParams['font.size'] = 12
        # plt.tight_layout()
        plt.savefig(f"test_{i}.png")
        plt.close()
    # ppgs_masked_target = torch.gather(tensor_rets["ppgs_target"],dim=1,index=ppg_masked_ids.unsqueeze(-1).expand(-1, -1, tensor_rets["ppg_mask_msm"].shape[-1]))



def bpe_test_wrapup(outs, model_name):
    # rank = torch.distributed.get_rank()
    labels, preds = list(), list()
    for out in outs:
        labels += out["labels"]
        preds += out["preds"]

    tensor_rets = {
        "bp_true_values": torch.stack(labels),
        "bp_pred_values": torch.stack(preds),
    }
    serialized_rets = {key: tensor.tolist() for key, tensor in tensor_rets.items()}
    with open(f"bpe_submit_{model_name}.json", "w") as fp:
        json.dump(serialized_rets, fp, indent=4)

    err = tensor_rets["bp_true_values"] - tensor_rets["bp_pred_values"]
    err_me = torch.mean(err, dim=0)
    err_mae = torch.mean(abs(err), dim=0)
    err_std = torch.std(err, dim=0)
    err_rmse_total = torch.sqrt(torch.mean(torch.pow(err, 2)))

    print('-' * 32)
    print('ME(DBP):{:.2f}'.format(err_me[0]), 'ME(SBP):{:.2f}'.format(err_me[1]))
    print('MAE(DBP):{:.2f}'.format(err_mae[0]), 'MAE(SBP):{:.2f}'.format(err_mae[1]))
    print('STD(DBP):{:.2f}'.format(err_std[0]), 'STD(SBP):{:.2f}'.format(err_std[1]))
    print('Total_RMSE_Loss:{:.2f}'.format(err_rmse_total))
    print('-' * 32)
