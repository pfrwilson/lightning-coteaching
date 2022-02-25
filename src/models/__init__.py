
from torch import nn 

from dataclasses import dataclass

from omegaconf import DictConfig

    
def build_model(config: DictConfig):
    
    if config.type == 'logistic_regression':
        
        return nn.Sequential(
            nn.Flatten(), 
            nn.Linear(28 * 28, 10)
        )
    
    elif config.type == 'paper_original_cnn':

        from .orig_paper_model import CNN
        return CNN(input_channel=1, n_outputs=10)

    elif config.type == 'our_cnn':
        
        from .classifier import CNN
        return CNN()
    
