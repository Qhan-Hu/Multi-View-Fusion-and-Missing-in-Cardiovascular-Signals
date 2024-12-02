import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from mmbt.gadgets.my_dataCollator import DataCollatorforMaskedSignalModeling


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        self.draw_false_ppg = _config["draw_false_ppg"]
        self.draw_false_ecg = _config["draw_false_ecg"]
        self.dataset_ratio = _config["dataset_ratio"]  # ratio of dataset to use(only for train dataset)

        # Transformations
        self.train_transform_key = (
            "default"
            if _config["train_transform_key"] is None
            else _config["train_transform_key"]
        )
        self.val_transform_key = (
            "default"
            if _config["val_transform_key"] is None
            else _config["val_transform_key"]
        )

        # construct missing modality info
        self.missing_info = {
            'ratio': _config["missing_ratio"],
            'type': _config["missing_type"],
            'missing_table_root': _config["missing_table_root"]
        }
        # for bash execution
        if _config["test_ratio"] is not None:
            self.missing_info['ratio']['val'] = _config["test_ratio"]
            self.missing_info['ratio']['test'] = _config["test_ratio"]
        if _config["test_type"] is not None:
            self.missing_info['type']['val'] = _config["test_type"]
            self.missing_info['type']['test'] = _config["test_type"]

        assert _config["msm_prob"] > 0 and _config["msm_prob"] < 1, 'masking ratio must be kept between 0 and 1'
        self.msm_collator = DataCollatorforMaskedSignalModeling(
            ecg_patch_size=_config["ecg_patch_size"],
            ppg_patch_size=_config["ppg_patch_size"],
            msm_prob=_config["msm_prob"],
        )
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_key,
            split="train",
            draw_false_ppg=self.draw_false_ppg,
            draw_false_ecg=self.draw_false_ecg,
            missing_info=self.missing_info,
            dataset_ratio=self.dataset_ratio,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_key,
            split="val",
            draw_false_ppg=self.draw_false_ppg,
            draw_false_ecg=self.draw_false_ecg,
            missing_info=self.missing_info,
        )


    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_key,
            split="test",
            draw_false_ppg=self.draw_false_ppg,
            draw_false_ecg=self.draw_false_ecg,
            missing_info=self.missing_info,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()
            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
