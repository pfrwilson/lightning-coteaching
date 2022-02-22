
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
from torchmetrics import Accuracy

from .models import build_model


def get_loss(loss_config):

    if loss_config.name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(
            reduction='none'
        )


class CoteachingMNIST(LightningModule):
    
    def __init__(self, model_config: DictConfig,
                 loss_config: DictConfig,
                 final_forget_rate: float,
                 final_forget_rate_epoch: int):

        super().__init__()

        self.model1 = build_model(model_config)
        self.model2 = build_model(model_config)

        self.loss_fn = get_loss(loss_config)

        self.forget_rate = final_forget_rate
        self.final_forget_rate_epoch = final_forget_rate_epoch

        self.automatic_optimization = False

        self.accuracy = Accuracy(num_classes=10)

        self.save_hyperparameters()

    def configure_optimizers(self):

        opt1 = torch.optim.Adam(self.model1.parameters())
        opt2 = torch.optim.Adam(self.model2.parameters())

        return opt1, opt2

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        
        logits_1 = self.model1(x)
        logits_2 = self.model2(x)

        loss_1, loss_2 = self.coteaching_loss(logits_1, logits_2, y)

        opt1, opt2 = self.optimizers()

        opt1.zero_grad()
        self.manual_backward(loss_1)
        opt2.step()

        opt2.zero_grad()
        self.manual_backward(loss_2)
        opt2.step()

        self.log_dict({'loss_1': loss_1, 'loss_2': loss_2})

    def validation_step(self, batch, batch_idx):

        x, y = batch

        logits_1 = self.model1(x)

        self.accuracy(logits_1, y)

        self.log('val/accuracy', self.accuracy, on_step=False, on_epoch=True)

    def coteaching_loss(self, logits_1, logits_2, targets):

        batch_size = logits_1.size[0]

        num_to_keep = int((1 - self.get_forget_rate()) * batch_size)

        loss_1 = self.loss_fn(logits_1, targets)
        loss_2 = self.loss_fn(logits_2, targets)

        loss_1_clean_indices = torch.sort(loss_2).indices[:num_to_keep]
        loss_2_clean_indices = torch.sort(loss_1).indices[:num_to_keep]

        loss_1_clean = torch.mean(self.loss_fn(
            logits_1[loss_1_clean_indices],
            targets[loss_1_clean_indices]
        ))

        loss_2_clean = torch.mean(self.loss_fn(
            logits_2[loss_2_clean_indices],
            targets[loss_2_clean_indices]
        ))

        return loss_1_clean, loss_2_clean

    def get_forget_rate(self):

        t = self.current_epoch
        t_k = self.final_forget_rate_epoch

        ratio = min(t / t_k, 1)
        return self.final_forget_rate * ratio

