
from omegaconf import DictConfig


def get_callbacks(config: DictConfig):

    callbacks = []

    if config.get('early_stopping') is not None:
        from pytorch_lightning.callbacks import EarlyStopping
        callbacks.append(EarlyStopping(**config.early_stopping))

    if config.log_training_sample_images:
        from .log_images import LogTrainingImageSamples
        callbacks.append(LogTrainingImageSamples())

    if config.log_test_pred_samples:
        from .log_images import LogTestImageSamples
        callbacks.append(LogTestImageSamples())

    return callbacks