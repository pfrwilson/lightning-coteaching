
from omegaconf import DictConfig

def get_datamodule(config: DictConfig):

    from .datamodule import DataModule

    return DataModule(config)
