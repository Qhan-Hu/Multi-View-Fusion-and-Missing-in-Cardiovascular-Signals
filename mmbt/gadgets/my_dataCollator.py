import random
import torch
from typing import List, Dict
import itertools
from torch.distributions.dirichlet import Dirichlet
from einops import rearrange


class DataCollatorforMaskedSignalModeling:
    def __init__(self, ecg_patch_size=50, ppg_patch_size=50, msm_prob: float = 0.7, alpha: float=1, eps: float=1e-5):
        self.msm_prob = msm_prob
        self.ecg_patch_size = ecg_patch_size
        self.ppg_patch_size = ppg_patch_size
        self.alpha = alpha
        self.eps = eps


    def __call__(self, dict_batch:Dict[str, torch.Tensor])->Dict[str, torch.Tensor]:

        # ppg_keys, ecg_keys = [k for k in list(dict_batch.keys()) if "ppg_norm" in k], [k for k in list(dict_batch.keys()) if "ecg_norm" in k]
        # ppgs, ecgs = dict_batch[ppg_keys[0]], dict_batch[ecg_keys[0]]
        # B = ppgs.size(0)
        # num_patches_all = 2 * ppgs.size(-1) // self.ppg_patch_size
        # num_visible_patches = int(num_patches_all * (1 - self.msm_prob))
        #
        # alphas = [self.alpha] * 2
        # valid_modality_choices = torch.Tensor([list(i) for i in itertools.product([0, 1], repeat=2)][1:])
        # rand_per_sample_choice = torch.randint(0, len(valid_modality_choices), (B,))
        # alphas_tensor = torch.index_select(valid_modality_choices, 0, rand_per_sample_choice)
        # alphas_tensor = alphas_tensor * torch.tensor(alphas) + self.eps
        #
        # modality_sampling_dist = Dirichlet(alphas_tensor).sample()
        # visible_patches_per_modality = (modality_sampling_dist * num_visible_patches).round().long()

        ppg_keys, ecg_keys = [k for k in list(dict_batch.keys()) if "ppg_norm" in k], [k for k in list(dict_batch.keys()) if
                                                                                       "ecg_norm" in k]
        ppgs, ecgs = dict_batch[ppg_keys[0]], dict_batch[ecg_keys[0]]
        B = ppgs.size(0)
        num_patches = ppgs.size(-1) // self.ppg_patch_size

        num_keep = int(num_patches * (1 - self.msm_prob))

        ecg_ids_shuffle, ppg_ids_shuffle = torch.rand(B, num_patches).argsort(dim=-1), torch.rand(B, num_patches).argsort(dim=-1)
        ecg_ids_restore, ppg_ids_restore = torch.argsort(ecg_ids_shuffle, dim=-1), torch.argsort(ppg_ids_shuffle, dim=-1)
        ecg_ids_keep, ppg_ids_keep = ecg_ids_shuffle[:, :num_keep], ppg_ids_shuffle[:, :num_keep]
        # generate the binary mask: 0 is keep, 1 is remove
        ppg_mask, ecg_mask = torch.ones([B, num_patches]), torch.ones([B, num_patches])
        ppg_mask[:, :num_keep], ecg_mask[:, :num_keep] = 0, 0
        ppg_mask, ecg_mask = torch.gather(ppg_mask, dim=-1, index=ppg_ids_restore), torch.gather(ecg_mask, dim=-1,
                                                                                             index=ecg_ids_restore)

        return {
            "ecg_ids_keep": ecg_ids_keep,
            "ppg_ids_keep": ppg_ids_keep,
            "ecg_mask": ecg_mask,
            "ppg_mask": ppg_mask,
            "ecg_ids_restore": ecg_ids_restore,
            "ppg_ids_restore": ppg_ids_restore,
            "num_patches": num_patches,
                }
