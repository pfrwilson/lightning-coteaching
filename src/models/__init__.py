
from torch import nn 
from .classifier import CNN

from dataclasses import dataclass

from omegaconf import DictConfig

    
def build_model(config: DictConfig):
    
    if config.type == 'logistic_regression':
        
        return nn.Sequential(
            nn.Flatten(), 
            nn.Linear(28 * 28, 10)
        )
        
    elif config.type == 'cnn':
        
        return CNN()
    
