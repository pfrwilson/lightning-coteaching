
import hydra
from omegaconf import DictConfig


@hydra.main(config_path='config', config_name='config')
def main(config: DictConfig):

    train_dataset = M