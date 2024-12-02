import random
import torch
import pyarrow as pa
import numpy as np
import os
from mmbt.config import ex
from mmbt.gadgets.my_transform import key_to_transform
from scipy import signal
from collections import OrderedDict
import copy


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_dir: str,
            transform_key: str,
            names: list,
            ecg_column_name: str = "",
            draw_false_ppg=0,
            draw_false_ecg=0,
            missing_ratio={},
            missing_type={},
            dataset_ratio=1.0,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of ppgs
        ecg_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__()

        self.ecg_column_name = ecg_column_name
        self.draw_false_ppg = draw_false_ppg
        self.draw_false_ecg = draw_false_ecg
        self.data_dir = data_dir
        self.transforms = key_to_transform(transform_key)
        self.dataset_ratio = dataset_ratio


        tables = [
            pa.ipc.RecordBatchFileReader(
                pa.memory_map(f"{data_dir}/{name}.arrow", "r")
            ).read_all()
            for name in names
            if os.path.isfile(f"{data_dir}/{name}.arrow")
        ]



        self.table_names = list()
        for i, name in enumerate(names):
            self.table_names += [name] * len(tables[i])

        self.table = pa.concat_tables(tables, promote=True)

        if ecg_column_name != "":
            self.ecg_column_name = ecg_column_name
            self.all_ecgs = self.table[ecg_column_name].to_pandas().tolist()
        else:
            self.all_ecgs = list()


        self.index_mapper = OrderedDict()
        for i in range(len(self.table)):
            self.index_mapper[i] = i
        if dataset_ratio != 1.0:
            index_mapper_keys = list(self.index_mapper.keys())
            random.shuffle(index_mapper_keys)
            new_index_num = int(len(index_mapper_keys) * dataset_ratio)
            self.index_mapper = {i: self.index_mapper[index_mapper_keys[i]] for i in range(new_index_num)}


    def __len__(self):
        return len(self.index_mapper)

    def filter_ecg_signal(self, ecg_raw, fs=125):
        highpass_cutoff = 0.5
        b_highpass, a_highpass = signal.butter(4, highpass_cutoff, btype='high', fs=fs)
        highpass_filtered_signal = signal.filtfilt(b_highpass, a_highpass, ecg_raw)

        notch_freq = 50
        Q = 30
        b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs=fs)
        final_filtered_signal = signal.filtfilt(b_notch, a_notch, highpass_filtered_signal)
        final_filtered_signal_copy = final_filtered_signal.copy()  # to avoid numpy array with negative stride error
        del final_filtered_signal

        return final_filtered_signal_copy

    def filter_ppg_signal(self, ppg_raw, fs=125):
        passband_freq = [0.5, 8]
        order = 4
        ripple = 0.5
        stopband_attenuation = 60
        b, a = signal.iirfilter(order, passband_freq, btype='band', analog=False, ftype='cheby2', fs=fs, rp=ripple, rs=stopband_attenuation)
        filtered_ppg = signal.filtfilt(b, a, ppg_raw)

        return filtered_ppg

    def get_raw_ppg(self, index, ppg_key="PPG"):
        return np.array(self.table[ppg_key][index].as_py())

    def get_ppg(self, index, ppg_key="PPG"):
        ppg = self.get_raw_ppg(index, ppg_key=ppg_key)
        # ppg = self.filter_ppg_signal(self.get_raw_ppg(index, ppg_key=ppg_key))
        if np.std(ppg) != 0:
            ppg = (ppg - np.mean(ppg))/np.std(ppg)
        # ppg_augs = [torch.tensor(tr.augment(ppg)) for tr in self.transforms]

        ppg_diffs = self.get_ppg_diff(ppg)
        ppg_multi_chan = np.stack([ppg, ppg_diffs["vpg_norm"], ppg_diffs["apg_norm"]], axis=0)
        ppg_multi_chan = torch.tensor(self.transforms(ppg_multi_chan), dtype=torch.float32)
        return {
            "ppg_multi_chan": ppg_multi_chan,
            "ppg_index": index,
        }

    def get_ppg_diff(self, ppg_norm):
        ppg = ppg_norm

        vpg = np.diff(ppg)
        apg = np.diff(vpg)
        vpg = np.hstack((vpg, vpg[-1]))
        vpg = np.convolve(vpg, np.ones((round(3),)) / round(3), mode='same')
        if np.std(vpg) != 0:
            vpg_norm = vpg - np.mean(vpg) / np.std(vpg)
        else:
            vpg_norm = vpg

        apg = np.hstack((apg, [apg[-1], apg[-1]]))
        apg = np.convolve(apg, np.ones((round(3),)) / round(3), mode='same')
        if np.std(apg) != 0:
            apg_norm = apg - np.mean(apg) / np.std(apg)
        else:
            apg_norm = apg
        return {
            "vpg_norm": vpg_norm,
            "apg_norm": apg_norm,
        }

    def get_false_ppg(self, rep, ppg_key="PPG"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        ppg = self.get_raw_ppg(random_index, ppg_key=ppg_key)
        ppg = (ppg - np.mean(ppg)) / np.std(ppg)
        ppg_diffs = self.get_ppg_diff(ppg)
        ppg_multi_chan = np.stack([ppg, ppg_diffs["vpg_norm"], ppg_diffs["apg_norm"]], axis=0)
        ppg_multi_chan = torch.tensor(self.transforms(ppg_multi_chan), dtype=torch.float32)
        return {f"false_ppg_{rep}": ppg_multi_chan}

    def get_ecg(self, index):
        ecg = self.filter_ecg_signal(self.all_ecgs[index])
        if np.std(ecg) != 0:
            ecg = (ecg - np.mean(ecg))/np.std(ecg)
        ecg = self.transforms(ecg[None, :])
        ecg_tensor = torch.tensor(
            ecg, dtype=torch.float32
        )    # [1, L]
        return {
            "ecg_norm": ecg_tensor,
        }


    def get_false_ecg(self, rep):
        random_index = random.randint(0, len(self.index_mapper) - 1)

        index = self.index_mapper[random_index]
        ecg = self.all_ecgs[index]
        ecg = (ecg - np.mean(ecg)) / np.std(ecg)
        ecg = torch.tensor(self.transforms(ecg), dtype=torch.float32).unsqueeze(0)
        return {f"false_ecg_{rep}": ecg}

    def get_suite(self, index):
        result = None
        while result is None:
            try:
                ret = dict()
                ret.update(self.get_ppg(index))
                ret.update(self.get_ecg(index))

                for i in range(self.draw_false_ppg):
                    ret.update(self.get_false_ppg(i))
                for i in range(self.draw_false_ecg):
                    ret.update(self.get_false_ecg(i))
                result = True
            except Exception as e:
                print(f"Error while read file idx {index} in {self.table_names[0]} -> {e}")
                index = random.randint(0, len(self.index_mapper) - 1)

        return ret

    def collate(self, batch, msm_collator):
        """
        ppg -> ppg, vpg, apg.
        ppg, ecg: list type -> tensor type
        """
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        dict_batch = {
            k: torch.stack(dict_batch[k], dim=0) if not isinstance(dict_batch[k], torch.Tensor) else dict_batch[k] for k
            in keys}
        msm_info = msm_collator(dict_batch)
        num_patches = msm_info["num_patches"]
        dict_batch["ecg_input_patch_idx"] = torch.arange(num_patches).unsqueeze(0).expand(batch_size, -1)
        dict_batch["ecg_input_patch_idx_msm"] = msm_info["ecg_ids_keep"]  # mem indicates masked ecg modelling
        dict_batch["ecg_mask_msm"] = msm_info["ecg_mask"]
        dict_batch["ecg_ids_restore"] = msm_info["ecg_ids_restore"]
        dict_batch["ppg_input_patch_idx"] = torch.arange(num_patches).unsqueeze(0).expand(batch_size, -1)
        dict_batch["ppg_input_patch_idx_msm"] = msm_info["ppg_ids_keep"]  # mem indicates masked ppg modelling
        dict_batch["ppg_mask_msm"] = msm_info["ppg_mask"]
        dict_batch["ppg_ids_restore"] = msm_info["ppg_ids_restore"]

        return dict_batch


@ex.automain
def run(_config):
    _config = copy.deepcopy(_config)
    return _config


if __name__ == "__main__":
    cfg = run()
    missing_info = {
        'ratio': cfg["missing_ratio"],
        'type': cfg["missing_type"],
        'both_ratio': cfg["both_ratio"],
        'missing_table_root': cfg["missing_table_root"],
        'simulate_missing': cfg["simulate_missing"]
    }
    hatememe_dataset = BaseDataset(data_dir=cfg['data_root'], transform_keys= cfg['train_transform_keys'], ppg_size = cfg['ppg_size'])
