import torch
import random

from transformers.optimization import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from mmbt.gadgets.my_metrics import Scalar, Accuracy
from pytorch_lightning.metrics import MeanSquaredError, MeanAbsoluteError


def set_metrics(pl_module):
    for split in ["train", "val"]:
        for k, v in pl_module.hparams.config["loss_names"].items():
            if v == 0:
                continue

            elif k == "msm":
                setattr(pl_module, f"{split}_{k}_ppg_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_ecg_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_total_loss", Scalar())

            elif k == "cep":
                setattr(pl_module, f"{split}_cep_loss", Scalar())
                setattr(pl_module, f"{split}_cep_accuracy", Accuracy())

            elif k == "bpe":
                setattr(pl_module, f"{split}_{k}_mse", MeanSquaredError())
                setattr(pl_module, f"{split}_{k}_mae", MeanAbsoluteError())
                setattr(pl_module, f"{split}_{k}_loss", Scalar())

            elif k == "afd":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())

            elif k == "ssc":
                setattr(pl_module, f"{split}_{k}_loss", Scalar())
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())


def test_ablation(pl_module, loss_name, res):
    test_ratio = pl_module.hparams.config['test_ratio']
    exp_name = pl_module.hparams.config["test_exp_name"]
    test_type = pl_module.hparams.config["test_type"]       
    records = f'missing ratio: {test_ratio}, ' + res
    record_file = f'./records/{loss_name}/{loss_name}_{exp_name}_on_missing_{test_type}'
    with open(record_file, 'a+') as f:
        f.write(records+'\n')


def epoch_wrapup(pl_module):
    phase = "train" if pl_module.training else "val"
    the_metric = 0

    for loss_name, v in pl_module.hparams.config["loss_names"].items():
        if v < 1:
            continue
        value = 0

        if loss_name == "msm":
            value = getattr(pl_module, f"{phase}_{loss_name}_total_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/total_loss_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_total_loss").reset()

            value_ppg = getattr(pl_module, f"{phase}_{loss_name}_ppg_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/ppg_loss_epoch", value_ppg)
            getattr(pl_module, f"{phase}_{loss_name}_ppg_loss").reset()

            value_ecg = getattr(pl_module, f"{phase}_{loss_name}_ecg_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/ecg_loss_epoch", value_ecg)
            getattr(pl_module, f"{phase}_{loss_name}_ecg_loss").reset()

            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'mse: {:.4f}'.format(value)
                test_ablation(pl_module, loss_name, res)


        if loss_name == "cep":
            value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            acc = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", acc, prog_bar=True)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

            
        if loss_name == "bpe":
            value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()

            value2 = getattr(pl_module, f"{phase}_{loss_name}_mse").compute()
            pl_module.log(f"{loss_name}/{phase}/mse_epoch", value2)
            getattr(pl_module, f"{phase}_{loss_name}_mse").reset()
            
            if pl_module.hparams.config["test_exp_name"] is not None:
                res = 'mse: {:.4f}'.format(value)
                test_ablation(pl_module, loss_name, res)

        if loss_name == "afd":
            value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            acc = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", acc)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

        if loss_name == "ssc":
            value = getattr(pl_module, f"{phase}_{loss_name}_loss").compute()
            pl_module.log(f"{loss_name}/{phase}/loss_epoch", value)
            getattr(pl_module, f"{phase}_{loss_name}_loss").reset()
            acc = getattr(pl_module, f"{phase}_{loss_name}_accuracy").compute()
            pl_module.log(f"{loss_name}/{phase}/accuracy_epoch", acc)
            getattr(pl_module, f"{phase}_{loss_name}_accuracy").reset()

        the_metric += value

    pl_module.log(f"{phase}/the_metric", the_metric)


def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v > 0
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]
    
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["msm_head", "bpe_head", "afd_head", "ssc_head"]
    prompt_name = "prompt"
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]


    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },            
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )


def set_cnn_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]
    end_lr = pl_module.hparams.config["end_lr"]

    if optim_type == "adamw":
        optimizer = AdamW(
            pl_module.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.98), weight_decay=wd
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(pl_module.parameters(), lr=lr, weight_decay=wd)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(pl_module.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    if pl_module.trainer.max_steps is None:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )