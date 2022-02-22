
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


def get_logger(config):

    if config.type == 'wandb':
        return WandbLogger(
            name=config.name,
            save_dir=config.save_dir,
            project=config.project,
        )

    elif config.type == 'tensorboard':
        return TensorBoardLogger(
            save_dir=config.save_dir,
            name=config.name,
        )

    else:
        raise ValueError(f'no logger for type {config.type}')

