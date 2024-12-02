from mmbt.datasets.adabase_dataset import ADABaseDataset
from .datamodule_base import BaseDataModule


class ADABaseDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ADABaseDataset

    @property
    def dataset_name(self):
        return "ADABase"

    def setup(self, stage):
        super().setup(stage)
