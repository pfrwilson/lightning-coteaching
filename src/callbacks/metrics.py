

import pytorch_lightning as pl
from typing import *
from pytorch_lightning.callbacks import Callback
from torchmetrics import Metric

class MetricCallback(Callback):

    def __init__(self, name: str, type: Union['train', 'val'], metric: Metric):
        super().__init__()

    def on_init_end(self, trainer: "pl.Trainer") -> None:
        return super().on_init_end(trainer)

 