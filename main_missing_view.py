import os
import copy
import pytorch_lightning as pl

from mmbt.config import ex
from mmbt.modules.mmbt_missing_aware_prompt_module import MMBTransformerSS
from mmbt.datamodules.multitask_datamodule import MTDataModule


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Data Modules
    dm = MTDataModule(_config)

    # Module
    model = MMBTransformerSS(_config)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    # wb_logger = pl.loggers.WandbLogger(project="MVF", name=run_name)
    # loggers = [tb_logger, wb_logger]
    loggers = tb_logger

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="min",
        save_last=True,
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val/the_metric",
        patience=10,
        verbose=True,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, early_stopping_callback, lr_callback]
    # callbacks = [early_stopping_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (
        _config["num_gpus"]
        if isinstance(_config["num_gpus"], list)
        else len(_config["num_gpus"])
    )
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config["max_epoch"] if max_steps is None else 100

    # Trainer
    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=1,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)