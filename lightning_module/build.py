from lib.pytorch_framework.optimizers import build_optimizer
from lib.pytorch_framework.lr_schedulers import build_scheduler
from models import build_model
from utils.visualization import build_visualizer
from .module import MuckSeg_FCMAE_Lightning_Module


def build_lightning_module(config):

    model, model_hparams = build_model(config.MODEL)
    optimizer, opt_hparams = build_optimizer(model, config.TRAIN.OPTIMIZER)
    scheduler_config, sche_params = build_scheduler(optimizer, config.TRAIN.LR_SCHEDULER)
    test_visualizer, featuremap_visualizer = build_visualizer(config.VISUALIZATION)
    hparams = model_hparams.copy()
    hparams.update(opt_hparams)
    hparams.update(sche_params)

    return MuckSeg_FCMAE_Lightning_Module(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler_config,
        config=config,
        test_visualizer=test_visualizer,
        featuremap_visualizer=featuremap_visualizer,
        featuremap_mask_ratio=config.VISUALIZATION.FEATUREMAP_MASK_RATIO,
        decoder_type=config.MODEL.DECODER_TYPE,
    ), hparams