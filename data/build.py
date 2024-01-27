import os
from torch.utils import data
from .train_datamodule import MuckSeg_FCMAE_DataModule
from .predict_dataloader import MuckSeg_Dataset_OriginalImage


def build_datamodule(config):
    return MuckSeg_FCMAE_DataModule(
        data_path=os.path.join(config.ENVIRONMENT.DATA_BASE_PATH, config.DATA.DATA_PATH),
        config=config,
        **config.DATA.DATAMODULE
    )


def build_inference_dataloader(config):
    return data.DataLoader(
        dataset=MuckSeg_Dataset_OriginalImage(
            data_path=os.path.join(config.ENVIRONMENT.DATA_BASE_PATH, config.PREDICT.DATA_PATH),
            image_mean=config.PREDICT.IMAGE_MEAN,
            image_std=config.PREDICT.IMAGE_STD
        ),
        shuffle=False,
        **config.PREDICT.DATAMODULE
    )