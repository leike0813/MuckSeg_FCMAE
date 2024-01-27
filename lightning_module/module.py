# coding: utf-8
import os
from pathlib import Path
import shutil
import pytorch_lightning as L
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models.blocks.functional import upsample_mask
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from utils.prediction_cutter import PredictionCutter
from lib.lightning_framework.callbacks import PredictionWriter


class MuckSeg_FCMAE_Lightning_Module(L.LightningModule):
    def __init__(self, model, optimizer, scheduler, config, test_visualizer, featuremap_visualizer=None,
                 featuremap_mask_ratio=0.):
        super().__init__()
        self.validation_step_outputs = []
        self.predict_step_outputs = []
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.test_visualizer = test_visualizer
        self.featuremap_visualizer = featuremap_visualizer
        self.featuremap_mask_ratio = featuremap_mask_ratio
        self.current_validation_milestone = 0
        self.predict_step_outputs = []

    def forward(self, x):
        St = self.model.target_downsample_factor
        Pm = self.model._PATCH_SIZE_MODULUS
        x_orig = x
        if St != 1:
            x_orig = F.interpolate(x_orig, scale_factor=1 / St, mode='bilinear')
        x, mask = self.model.forward_features(x)  # B, C, H, W -> B, C, H/St, W/St
        mask = upsample_mask(mask, Pm // St).unsqueeze(1)  # B, H/P, W/P -> B, 1, H, W

        return x, x_orig, mask

    def forward_pred_and_loss(self, x):
        pic, lap, hog = x
        St = self.model.target_downsample_factor
        Pm = self.model._PATCH_SIZE_MODULUS
        pic_orig = pic
        if St != 1:
            pic_orig = F.interpolate(pic_orig, scale_factor=1 / St, mode='bilinear')
            lap = F.interpolate(lap, scale_factor=1 / St, mode='bilinear')
        target_pic = pic_orig
        target_pic = self.model.patchify(target_pic)
        target_lap = self.model.patchify(lap)
        target_hog = self.model.patchify(hog)
        x_pic, x_lap, x_hog, mask = self.model.forward_features(pic)
        pred = x_pic.detach()
        x_pic = self.model.patchify(x_pic)
        x_lap = self.model.patchify(x_lap)
        x_hog = self.model.patchify(x_hog)
        loss = self.model.loss_func((x_pic, x_lap, x_hog), (target_pic, target_lap, target_hog), mask)
        mask = upsample_mask(mask, Pm // St).unsqueeze(1)

        return pred, pic_orig, mask, loss

    def configure_optimizers(self):
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler
        }

    def get_run_basefld(self):
        if self.trainer.checkpoint_callback:
            _run_basefld = Path(self.trainer.checkpoint_callback.dirpath).parent
            if self.mlflow_logger_available:
                return _run_basefld if _run_basefld.name == self.logger.run_id else None
            else:
                return _run_basefld if _run_basefld == Path(self.logger.log_dir) else None
        else:
            return (Path(self.trainer.default_root_dir) / self.logger.experiment_id) / self.logger.run_id \
                if self.mlflow_logger_available else Path(self.trainer.logger.log_dir)

    def setup(self, stage):
        if stage in ['fit', 'test']:
            self.mlflow_logger_available = isinstance(self.logger, MLFlowLogger)
            self.run_basefld = self.get_run_basefld()
            if self.run_basefld:
                self.result_img_fld = self.run_basefld / 'result_img'
                self.featuremap_fld = self.run_basefld / 'feature_map'
                self.inference_example_fld = self.run_basefld / 'inference_example'

                self.test_visualizer.basefld = self.result_img_fld
                if self.featuremap_visualizer is not None:
                    self.featuremap_visualizer.basefld = self.featuremap_fld

    def teardown(self, stage):
        if stage == 'test':
            if self.run_basefld and (self.run_basefld / 'checkpoints').exists():
                shutil.rmtree(self.run_basefld / 'checkpoints')
            if hasattr(self, 'result_img_fld') and self.result_img_fld.exists():
                shutil.rmtree(self.result_img_fld)
        if stage == 'predict':
            if hasattr(self, 'inference_example_fld') and self.inference_example_fld.exists():
                shutil.rmtree(self.inference_example_fld)
            if hasattr(self, 'featuremap_fld') and self.featuremap_fld.exists():
                shutil.rmtree(self.featuremap_fld)

    def on_fit_start(self):
        config_filename = '{type}_{spec}_Recent.yaml'.format(type=self.config.MODEL.TYPE, spec=self.config.MODEL.SPEC_NAME)
        config_out_file = os.path.join(
            os.path.join(self.config.ENVIRONMENT.PROJECT_PATH, self.config.CONFIG_OUTPUT_PATH),
            config_filename)
        with open(config_out_file, "w") as f:
            f.write(self.config.dump_visible())
        if self.mlflow_logger_available:
            self.logger.experiment.log_artifact(self.logger._run_id, config_out_file, f'config')

        if self.config.FULL_DUMP:
            config_filename_full = '{type}_{spec}_Full.yaml'.format(type=self.config.MODEL.TYPE, spec=self.config.MODEL.SPEC_NAME)
            config_full_out_file = os.path.join(
                os.path.join(self.config.ENVIRONMENT.PROJECT_PATH, self.config.CONFIG_OUTPUT_PATH),
                config_filename_full)
            with open(config_full_out_file, "w") as f:
                f.write(self.config.dump())
            if self.mlflow_logger_available:
                self.logger.experiment.log_artifact(self.logger._run_id, config_full_out_file, f'config')

        if self.mlflow_logger_available:
            for path in self.config.MODEL.FILE_PATHS:
                self.logger.experiment.log_artifact(self.logger._run_id, path, f'model')

    def on_train_epoch_start(self):
        self.current_validation_milestone = 0

    def training_step(self, batch, batch_idx):
        return self.model(batch)

    def validation_step(self, batch, batch_idx):
        if batch_idx > 0:
            return self.model(batch)

        pred, orig_img, mask, loss = self.forward_pred_and_loss(batch)
        img_path = self.test_visualizer(orig_img, pred, batch_idx, mask=mask, name_suffix='epoch{ep}_{ms}'.format(
            ep=self.current_epoch, ms=self.current_validation_milestone
        ))
        if self.mlflow_logger_available:
            for i in range(len(img_path)):
                self.logger.experiment.log_artifact(
                    self.logger._run_id,
                    img_path[i],
                    f"validation_img"
                )

        return loss

    def on_validation_epoch_end(self) -> None:
        self.current_validation_milestone += 1

    def test_step(self, batch, batch_idx):
        pred, orig_img, mask, loss = self.forward_pred_and_loss(batch)
        img_path = self.test_visualizer(orig_img, pred, batch_idx, mask=mask)
        if isinstance(self.logger, MLFlowLogger):
            for i in range(len(img_path)):
                self.logger.experiment.log_artifact(
                    self.logger._run_id,
                    img_path[i],
                    f"result_img"
                )

        return loss

    def on_predict_start(self):
        for callback in self.trainer.callbacks:
            if isinstance(callback, PredictionWriter):
                if hasattr(self, 'inference_example_fld'): # Use this as the criterion for distinguish if the trainer is running at the training mode
                    callback.output_dir = self.inference_example_fld
                else:
                    if self.featuremap_visualizer is not None:
                        self.featuremap_visualizer.basefld = callback.output_dir / 'feature_map'

        self.prediction_cutter = PredictionCutter(self, self.config)
        self.predict_mode, self.patches_shape = self.prediction_cutter.find_predict_mode()

    def predict_step(self, batch, batch_idx):
        garbage_collection_cuda()

        orig_imgs, img_names = batch

        if batch_idx < self.config.VISUALIZATION.NUM_FEATUREMAP_SAMPLES:
            self.predict_step_outputs.append(
                TF.center_crop(orig_imgs[0].unsqueeze(0), (self.config.MODEL.IMAGE_SIZE, self.config.MODEL.IMAGE_SIZE))
            )

        orig_imgs_roi = self.prediction_cutter.cut_roi(orig_imgs)

        if self.predict_mode == self.prediction_cutter.PredictMode.FULL_SIZE:
            pred, orig_imgs, mask = self(orig_imgs_roi)
        elif self.predict_mode == self.prediction_cutter.PredictMode.THREE_FOLD:
            orig_imgs = self.prediction_cutter.cut_fold(orig_imgs_roi)
            pred = []
            mask = []
            for i in range(3):
                _ = self(orig_imgs[i])
                pred.append(TF.crop(_[0], *self.prediction_cutter.cut_coords_downsampled[i, 2:]))
                mask.append(TF.crop(_[2], *self.prediction_cutter.cut_coords_downsampled[i, 2:]))
            pred = torch.cat(pred, dim=2)
            mask = torch.cat(mask, dim=2)
        elif self.predict_mode == self.prediction_cutter.PredictMode.PATCH:
            orig_imgs = self.prediction_cutter.cut_patch(orig_imgs_roi)
            pred = []
            mask = []
            for i in range(self.patches_shape[0]):
                pred_row = []
                mask_row = []
                for j in range(self.patches_shape[1]):
                    _ = self(orig_imgs[i][j])
                    pred_row.append(TF.crop(_[0], *self.prediction_cutter.cut_coords_downsampled[i, j, 2:]))
                    mask_row.append(TF.crop(_[2], *self.prediction_cutter.cut_coords_downsampled[i, j, 2:]))
                pred.append(torch.cat(pred_row, dim=3))
                mask.append(torch.cat(mask_row, dim=3))
            pred = torch.cat(pred, dim=2)
            mask = torch.cat(mask, dim=2)

        orig_imgs = F.interpolate(orig_imgs_roi, scale_factor=1 / self.model.target_downsample_factor, mode='bilinear')
        orig_imgs = self.trainer.predict_dataloaders[0].dataset.denormalizer(orig_imgs)
        pred = self.trainer.predict_dataloaders[0].dataset.denormalizer(pred)
        mask = mask = torch.cat([mask for i in range(pred.shape[1])], dim=1)
        pred = pred * mask + orig_imgs * (1. - mask)
        orig_masked = orig_imgs * (1. - mask)

        if hasattr(self, 'inference_example_fld'):
            return (orig_imgs, pred, orig_masked), img_names

        return (pred,), img_names

    def on_predict_end(self):
        if self.featuremap_visualizer is not None:
            for img_idx, img in enumerate(self.predict_step_outputs):
                fmaps = self.model.forward_featuremaps(img, self.featuremap_mask_ratio)
                paths = self.featuremap_visualizer(fmaps, img_idx)
                if self.mlflow_logger_available:
                    for path in paths.values():
                        self.logger.experiment.log_artifact(
                            self.logger._run_id,
                            path,
                            f"feature_map"
                        )

# EOF