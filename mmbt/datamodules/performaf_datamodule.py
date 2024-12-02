from mmbt.datasets.performaf_dataset import PERFormAFDataset
from .datamodule_base import BaseDataModule


class PERFormAFDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PERFormAFDataset

    @property
    def dataset_name(self):
        return "PERFormAF"

    def setup(self, stage):
        super().setup(stage)
