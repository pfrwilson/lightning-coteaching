from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np


class LabelNoiseDataset(Dataset):

    def __init__(self, clean_dataset: Dataset, noise_transition_matrix: np.ndarray, seed=0):

        super().__init__()

        self.clean_dataset = clean_dataset
        self.noise_transition_matrix = noise_transition_matrix
        self.__labels = np.zeros(len(self.clean_dataset))
        self.seed = seed

        self.target_transform = getattr(self.clean_dataset, 'target_transform', None)
        self.transform = getattr(self.clean_dataset, 'transform', None)
        clean_dataset.target_transform = None
        clean_dataset.transform = None

        rng = np.random.RandomState(seed=seed)
        for idx, (image, label) in enumerate(self.clean_dataset):
            transition_probs = self.noise_transition_matrix[label]
            new_label = np.argmax(
                rng.multinomial(1, transition_probs)
            )
            self.__labels[idx] = new_label

    def __getitem__(self, idx):
        item, _ = self.clean_dataset[idx]
        noisy_label = self.__labels[idx]

        item = self.transform(item) if self.transform else item
        noisy_label = self.target_transform(noisy_label) if self.target_transform else noisy_label

        return item, noisy_label

    def __len__(self):
        return len(self.clean_dataset)


def symmetric_flip_noisy_mnist(
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        noise_rate=.5,
        seed=0
):
    ds = MNIST(root, train, transform, target_transform, download)

    noise_transition_matrix = np.eye(10) * (1 - noise_rate)
    noise_transition_matrix = np.where(noise_transition_matrix != 0,
                                       noise_transition_matrix, noise_rate / 9)

    return LabelNoiseDataset(
        ds, noise_transition_matrix, seed=seed
    )
