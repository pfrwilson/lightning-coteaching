
import torchvision.transforms as T
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

from typing import Optional
from dataclasses import dataclass
from torch.utils.data import DataLoader, Subset
import torch

from .datasets import get_clean_dataset
from .noisy_dataset import get_noisy_dataset
from .preprocessing import get_preprocessing

class DataModule(LightningDataModule):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.config = config

        self.preprocessing = get_preprocessing(
            config.preprocessing
        )

        self.target_transform = T.Lambda(
            lambda num: torch.tensor(num).long()
        )

        self.clean_datasets = {}

    def setup(self, stage: Optional[str] = None) -> None:

        for split in ['train', 'val', 'test']:
            clean_dataset = get_clean_dataset(split, self.config.dataset.clean_dataset)
            clean_dataset.transform = self.preprocessing
            clean_dataset.target_transform = self.target_transform
            self.clean_datasets[split] = clean_dataset

        self.train_dataset = get_noisy_dataset(
            self.clean_datasets['train'],
            self.config.dataset.noisy_dataset.noise_type, 
            self.config.dataset.noisy_dataset.noise_rate,
            self.config.dataset.noisy_dataset.seed,  
            self.config.dataset.noisy_dataset.num_classes
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.clean_datasets['val'], batch_size=self.config.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.clean_datasets['test'], batch_size=self.config.batch_size)