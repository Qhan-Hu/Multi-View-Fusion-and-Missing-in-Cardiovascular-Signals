# from .base_dataset import BaseDataset
import torch
import random, os
import argparse

import copy
from mmbt.config import ex
from mmbt.datasets.base_dataset import BaseDataset



class PulseDBDataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["PulseDB_train"]
        elif split == "val":
            names = ["PulseDB_val"]
        elif split == "test":
            names = ["PulseDB_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            ecg_column_name="ECG"
        )

        # missing modality control
        missing_ratio = missing_info['ratio'][split]
        mratio = str(missing_ratio).replace('.', '')
        missing_type = missing_info['type'][split]
        missing_table_root = missing_info['missing_table_root']
        missing_table_name = f'PulseDB_{split}_missing_{missing_type}_{mratio}.pt'
        missing_table_path = os.path.join(missing_table_root, missing_table_name)

        # use ppg data to formulate missing table
        total_num = len(self.table['PPG'])

        # mratio_train/val/test = 0(complete modalities), create the missing table full of 0 element without saving
        # the missing table. Only the missing table of mratio > 0 could be saved. Thus, when mratio_train/val/test =
        # 0, the path of corresponding missing table would never exist. Only the command 'missing_table =
        # torch.zeros(total_num)' would be implemented in this if-else structure.
        if os.path.exists(missing_table_path):
            missing_table = torch.load(missing_table_path)
            if len(missing_table) != total_num:
                print('missing table mismatched!')
                exit()
        else:
            missing_table = torch.zeros(total_num)

            if missing_ratio > 0:
                missing_index = random.sample(range(total_num), int(total_num * missing_ratio))

                if missing_type == 'ecg':
                    missing_table[missing_index] = 1
                elif missing_type == 'ppg':
                    missing_table[missing_index] = 2
                elif missing_type == 'both':

                    missing_table[missing_index] = 1
                    missing_index_ppg = random.sample(missing_index, int(len(missing_index) / 2))
                    missing_table[missing_index_ppg] = 2

                torch.save(missing_table, missing_table_path)

        self.missing_table = missing_table


    def __getitem__(self, index):

        index = self.index_mapper[index]
        suite = self.get_suite(index)

        ppg_tensor_norm = suite["ppg_multi_chan"]  # [3, L]: ppg, vpg, apg
        del suite["ppg_multi_chan"]
        del suite["ppg_index"]
        # missing ppg, dummy ppg is all-zero ppg
        if self.missing_table[index] == 2:
            ppg_tensor_norm = torch.zeros(ppg_tensor_norm.size())

        ecg_sets = self.get_ecg(index)
        ecg_tensor_norm = ecg_sets["ecg_norm"]  # tensor: [1, L]

        # missing ecg, dummy ecg is all-zero ecg
        if self.missing_table[index] == 1:
            ecg_tensor_norm = torch.zeros(ecg_tensor_norm.size())

        # element in list is tensor: [2,]
        labels_tensor = torch.tensor(
            [self.table["DBP"][index].as_py(),
             self.table["SBP"][index].as_py()]
        )
        abp_tensor = torch.tensor(self.table["ABP"][index].as_py()).unsqueeze(0) # tensor: [1, L]

        suite.update(
            {
                "ppg_norm": ppg_tensor_norm,
                "ecg_norm": ecg_tensor_norm,
                "label": labels_tensor,
                "abp_raw": abp_tensor,
                "missing_type": torch.Tensor([self.missing_table[index].item()]),
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
