from mmbt.datasets.pulsedb_dataset import PulseDBDataset
from .datamodule_base import BaseDataModule


class PulseDBDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PulseDBDataset

    @property
    def dataset_name(self):
        return "PulseDB"

    def setup(self, stage):
        super().setup(stage)
