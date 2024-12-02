from mmbt.datasets.mesa_dataset import MESADataset
from .datamodule_base import BaseDataModule


class MESADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MESADataset

    @property
    def dataset_name(self):
        return "MESA"

    def setup(self, stage):
        super().setup(stage)
