from .pulsedb_datamodule import PulseDBDataModule
from .performaf_datamodule import PERFormAFDataModule
from .adabase_datamodule import ADABaseDataModule
from .mesa_datamodule import MESADataModule
from .pretrain_pulsedb_datamodule import PretrainPulseDBDataModule



_datamodules = {
    "pretrain_pulsedb": PretrainPulseDBDataModule,
    "pulsedb": PulseDBDataModule,
    "performaf": PERFormAFDataModule,
    "adabase": ADABaseDataModule,
    "mesa": MESADataModule,
}