# from .base_dataset import BaseDataset
import torch
import random, os
import argparse

import copy
from mmbt.config import ex
from mmbt.datasets.base_dataset import BaseDataset



class PretrainPulseDBDataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["PulseDB_train"]
        elif split == "val":
            names = ["PulseDB_val"]
        elif split == "test":
            names = ["PulseDB_test"]
        else:
            raise ValueError

        super().__init__(
            *args,
            **kwargs,
            names=names,
            ecg_column_name="ECG"
        )




    def __getitem__(self, index):

        index = self.index_mapper[index]
        suite = self.get_suite(index)

        ppg_tensor_norm = suite["ppg_multi_chan"]  # [3, L]: ppg, vpg, apg
        del suite["ppg_multi_chan"]
        del suite["ppg_index"]

        suite.update(
            {
                "ppg_norm": ppg_tensor_norm,
            }
        )

        return suite


@ex.automain
def run(_config):
    _config = copy.deepcopy(_config)
    return _config


if __name__ == "__main__":
    cfg = run()
    missing_info = {
        'ratio': cfg["missing_ratio"],
        'type': cfg["missing_type"],
        'missing_table_root': cfg["missing_table_root"]
    }
    hatememe_dataset = PulseDBDataset(data_dir=cfg['data_root'],
                                        split='val', missing_info=missing_info)
