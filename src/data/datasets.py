
from omegaconf import DictConfig

from torchvision.datasets import MNIST, CIFAR10, CIFAR100


def get_clean_dataset(split: str, config: DictConfig):

    if config.name == 'mnist':

        if split == 'train': 
            return MNIST(config.root, train=True, 
                         download=config.download)

        elif split in ['val', 'test']:
            return MNIST(config.root, train=False, 
                         download=config.download)
