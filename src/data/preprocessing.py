

import torchvision.transforms as T

def get_preprocessing(config):

    return T.Compose([
            T.RandomResizedCrop(config.target_shape, scale=config.scale) if config.use_augmentations
            else T.Lambda(lambda im: im),
            T.ToTensor()
        ])

    