from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np


def symmetric_noise_transition_matrix(noise_rate, n_classes):

    matrix = np.eye(n_classes) * (1 - noise_rate)
    matrix = np.where(matrix != 0, matrix, noise_rate / (n_classes - 1))

    return matrix

def get_noisy_dataset(clean_dataset, noise_type, noise_rate, seed, n_classes):

    if noise_type == 'symmetric':
        return LabelNoiseDataset(
            clean_dataset, 
            noise_transition_matrix=symmetric_noise_transition_matrix(
                noise_rate, n_classes
            ),
            seed=seed
        )

    else: 
        raise NotImplementedError(f'noise type {noise_type} not supported. ')

class LabelNoiseDataset(Dataset):

    def __init__(self, clean_dataset: Dataset, noise_transition_matrix: np.ndarray, seed=0, ):

        super().__init__()

        self.clean_dataset = clean_dataset
        self.noise_transition_matrix = noise_transition_matrix
        self.__labels = np.zeros(len(self.clean_dataset))
        self.__noise_or_not = np.zeros(len(self.clean_dataset)).astype(bool)
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
            
            if new_label != label:
                self.__noise_or_not[idx] = True

    def __getitem__(self, idx):
        item, _ = self.clean_dataset[idx]
        noisy_label = self.__labels[idx]
        noise_or_not = self.__noise_or_not[idx]

        item = self.transform(item) if self.transform else item
        noisy_label = self.target_transform(noisy_label) if self.target_transform else noisy_label
        noise_or_not = self.target_transform(noise_or_not) if self.target_transform else noise_or_not

        return item, noisy_label, noise_or_not

    def __len__(self):
        return len(self.clean_dataset)




