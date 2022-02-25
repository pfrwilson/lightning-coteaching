
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy

from .models import build_model


def get_loss(loss_config):

    if loss_config.name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(
            reduction='none'
        )


class CoteachingModule(LightningModule):
    
    def __init__(self, model_config: DictConfig,
                 loss_config: DictConfig,
                 opt_config: DictConfig):

        super().__init__()

        self.model1 = build_model(model_config)
        self.model2 = build_model(model_config)

        self.loss_fn = get_loss(loss_config)

        self.opt_config = opt_config

        self.train_accuracy = Accuracy(num_classes=10)
        self.val_accuracy = Accuracy(num_classes=10)
        self.test_accuracy = Accuracy(num_classes=10)

        self.save_hyperparameters()

    def forward(self, x):
        return self.model1(x)

    def configure_optimizers(self):

        opt1 = torch.optim.Adam(self.model1.parameters(), lr=self.opt_config.lr)
        opt2 = torch.optim.Adam(self.model2.parameters(), lr=self.opt_config.lr)

        return [opt1, opt2]

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        x = batch[0]
        y = batch[1]
        noise_indices = batch[2]

        logits_1 = self.model1(x)
        logits_2 = self.model2(x)

        self.train_accuracy(logits_1, y)

        losses = self.coteaching_loss(logits_1, logits_2, y)

        self.log('train/accuracy', self.train_accuracy, on_step=False, on_epoch=True)
        self.log_dict({'train/loss_1': losses[0], 'train/loss_2': losses[1]})

        return losses[optimizer_idx]

    def validation_step(self, batch, batch_idx):

        x = batch[0]
        y = batch[1]

        logits_1 = self.model1(x)

        self.val_accuracy(logits_1, y)

        self.log('val/accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('forget_rate', self.get_forget_rate(), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):

        x, y = batch

        logits_1 = self.model1(x)

        self.test_accuracy(logits_1, y)

        self.log('test/accuracy', self.test_accuracy, on_step=False, on_epoch=True)
                         
    def coteaching_loss(self, logits_1, logits_2, targets):

        batch_size = logits_1.shape[0]

        num_to_keep = int((1 - self.get_forget_rate()) * batch_size)

        loss_1 = self.loss_fn(logits_1, targets).detach()
        loss_2 = self.loss_fn(logits_2, targets).detach()

        ind_1_sorted = torch.argsort(loss_1)
        ind_2_sorted = torch.argsort(loss_2)
        ind_1_clean = ind_1_sorted[:num_to_keep]
        ind_2_clean = ind_2_sorted[:num_to_keep]

        loss_1_clean = torch.mean(self.loss_fn(
            logits_1[ind_1_clean],
            targets[ind_1_clean]
        ))

        loss_2_clean = torch.mean(self.loss_fn(
            logits_2[ind_2_clean],
            targets[ind_2_clean]
        ))

        return loss_1_clean, loss_2_clean, ind_1_clean, ind_2_clean

    def get_forget_rate(self):

        t = self.current_epoch
        t_k = self.opt_config.final_forget_rate_epoch

        ratio = min(t / t_k, 1)
        return self.opt_config.final_forget_rate * ratio
