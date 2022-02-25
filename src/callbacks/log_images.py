
import pytorch_lightning as pl
from pytorch_lightning import Callback
from typing import *
from pytorch_lightning.loggers import WandbLogger
import torch

class LogTrainingImageSamples(Callback):

    def __init__(self, batch_indices_to_log=[0]):
        super().__init__()
        self.batch_indices_to_log = batch_indices_to_log

    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, unused: Optional[int] = 0) -> None:
        
        if pl_module.current_epoch == 0 and batch_idx in self.batch_indices_to_log \
            and type(pl_module.logger) is WandbLogger:

            x, y, noise_indices = batch
        
            image_batch = x
            images = [img.cpu().numpy() for img in image_batch]
            labels = [int(label.cpu().item()) for label in y]
            noise_or_not = [flag.item() for flag in noise_indices]

            pl_module.logger.log_image(
                key='training examples', 
                images = images, 
                caption = [f'label = {labels[idx]}, noise = {noise_or_not[idx]}' \
                    for idx in range(len(images))]
            )

class LogTestImageSamples(Callback):

    def on_predict_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
                               batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        if batch_idx == 0 and type(pl_module.logger) is WandbLogger:
            
            x, y = batch

            logits = pl_module(x)
            y_hat = torch.argmax(logits, dim=-1, keepdim=True)

            images = [ image.cpu().numpy() for image in x ]
            labels = [ label.item() for label in y_hat ]

            pl_module.logger.log_image(
                k='test_predictions',
                images=images, 
                captions=[f'pred: {label}' for label in labels]
            )

