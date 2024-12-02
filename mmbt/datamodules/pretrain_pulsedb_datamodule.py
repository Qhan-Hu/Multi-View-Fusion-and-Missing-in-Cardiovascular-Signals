from mmbt.datasets.pretrain_pulsedb_dataset import PretrainPulseDBDataset
from .datamodule_base import BaseDataModule


class PretrainPulseDBDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PretrainPulseDBDataset

    @property
    def dataset_name(self):
        return "PretrainPulseDB"

    def setup(self, stage):
        super().setup(stage)
