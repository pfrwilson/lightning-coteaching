
from omegaconf import DictConfig


def build_model(config: DictConfig):
    from .classifier import CNN

    return CNN(**config)