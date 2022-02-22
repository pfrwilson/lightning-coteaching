import torchvision.transforms as T
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from .noisy_mnist import symmetric_flip_noisy_mnist
from .noisy_mnist import MNIST
from typing import Optional
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader
import torch


@dataclass
class DataConfig:
    use_augmentations: bool
    root: str
    download: bool
    noise_rate: float
    seed: int
    batch_size: int


cs = ConfigStore.instance()
cs.store('data', DataConfig)


class DataModule(LightningDataModule):

    def __init__(self, config: DataConfig):

        super().__init__()

        self.config = config
        self.transform = T.Compose([
            T.RandomResizedCrop(28, scale=(0.8, 1.0)) if self.config.use_augmentations
            else T.Lambda(lambda im: im),
            T.ToTensor()
        ])
        self.target_transform = T.Lambda(
            lambda num: torch.tensor(num).long()
        )

        self.train_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None) -> None:

        if self.config.noise_rate == 0:
            self.train_ds = MNIST(
                self.config.root,
                train=True,
                transform=self.transform,
                target_transform=self.target_transform,
                download=self.config.download,
            )

        else:
            self.train_ds = symmetric_flip_noisy_mnist(
                self.config.root,
                transform=self.transform,
                target_transform=self.target_transform,
                download=self.config.download,
                noise_rate=self.config.noise_rate,
                seed=self.config.seed
            )

        self.test_ds = MNIST(
            self.config.root,
            train=False,
            transform=self.transform,
            target_transform=self.target_transform,
            download=self.config.download,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.config.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.config.batch_size)
