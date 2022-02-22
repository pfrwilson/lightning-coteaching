
import hydra
from omegaconf import DictConfig


@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):

    from src.data import DataModule
    dm = DataModule(config.data)

    module = None
    assert config.mode in ['coteaching', 'vanilla']
    if config.mode == 'coteaching':

        from src.coteaching import CoteachingModule
        module = CoteachingModule(
            config.model,
            config.loss,
            config.opt_config,
        )

    elif config.mode == 'vanilla':

        from src.coteaching import VanillaModule
        module = VanillaModule(
            config.model,
            config.loss,
            config.opt_config
        )

    from src.logging import get_logger
    logger = get_logger(config.logger)

    from pytorch_lightning import Trainer
    trainer = Trainer(**config.trainer, logger=logger)

    trainer.fit(module, dm)


if __name__ == '__main__':
    main()

