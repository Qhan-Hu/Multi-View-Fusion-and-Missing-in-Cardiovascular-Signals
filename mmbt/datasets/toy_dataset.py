import torch
import random, os
from mmbt.datasets.base_dataset import BaseDataset



class MESADataset(BaseDataset):
    def __init__(self, *args, split="", missing_info={}, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["MESA_train"]
        elif split == "val":
            names = ["MESA_val"]
        elif split == "test":
            names = ["MESA_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            ecg_column_name="ECG"
        )

        # missing modality control
        # missing_ratio = missing_info['ratio'][split]
        # mratio = str(missing_ratio).replace('.', '')
        # missing_type = missing_info['type'][split]
        # missing_table_root = missing_info['missing_table_root']
        # missing_table_name = f'PulseDB_{split}_missing_{missing_type}_{mratio}.pt'
        # missing_table_path = os.path.join(missing_table_root, missing_table_name)

        # use ppg data to formulate missing table
        total_num = len(self.table['PPG'])

        # mratio_train/val/test = 0(complete modalities), create the missing table full of 0 element without saving
        # the missing table. Only the missing table of mratio > 0 could be saved. Thus, when mratio_train/val/test =
        # 0, the path of corresponding missing table would never exist. Only the command 'missing_table =
        # torch.zeros(total_num)' would be implemented in this if-else structure.
        # if os.path.exists(missing_table_path):
        #     missing_table = torch.load(missing_table_path)
        #     if len(missing_table) != total_num:
        #         print('missing table mismatched!')
        #         exit()
        # else:
        #     missing_table = torch.zeros(total_num)
        #
        #     if missing_ratio > 0:
        #         missing_index = random.sample(range(total_num), int(total_num * missing_ratio))
        #
        #         if missing_type == 'ecg':
        #             missing_table[missing_index] = 1
        #         elif missing_type == 'ppg':
        #             missing_table[missing_index] = 2
        #         elif missing_type == 'both':
        #
        #             missing_table[missing_index] = 1
        #             missing_index_ppg = random.sample(missing_index, int(len(missing_index) / 2))
        #             missing_table[missing_index_ppg] = 2
        #
        #         torch.save(missing_table, missing_table_path)

        # self.missing_table = missing_table


    def __getitem__(self, index):

        index = self.index_mapper[index]


        labels_tensor = torch.tensor(
            self.table["SS"][index].as_py()
        ).unsqueeze(0) # tensor: [1, L], cognitive load

        suite = dict()
        suite.update(
            {
                "ppg_norm": self.get_ppg(index)["ppg_multi_chan"],
                "ecg_norm": self.get_ecg(index)["ecg_norm"],
                "label": labels_tensor,
            }
        )

        return suite


