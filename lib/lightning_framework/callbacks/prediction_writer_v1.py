import os
import warnings
from pathlib import Path
from collections.abc import Sequence
from PIL import Image as PILImage
import torch
import torchvision.transforms.functional as TF
from lib.pytorch_framework.utils import CustomCfgNode as CN
from lightning.pytorch.callbacks import BasePredictionWriter


__all__ = [
    'build_prediction_writer',
    'PredictionWriter',
    'DEFAULT_CONFIG',
]


DEFAULT_CONFIG = CN(visible=False)
DEFAULT_CONFIG.output_dir = 'predictions'
DEFAULT_CONFIG.image_suffixes = ''
DEFAULT_CONFIG.omits = []
DEFAULT_CONFIG.image_format = 'png'
DEFAULT_CONFIG.concatenate = False
DEFAULT_CONFIG.log_prediction = False
DEFAULT_CONFIG.log_folder = ''
DEFAULT_CONFIG.write_interval = 'batch'
DEFAULT_CONFIG.set_typecheck_exclude_keys(['concatenate'])


def build_prediction_writer(node=DEFAULT_CONFIG):
    return PredictionWriter(**node)


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, image_suffixes='', omits=[], image_format='png', concatenate=False, log_prediction=False, log_folder='', write_interval='batch'):
        super(PredictionWriter, self).__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.image_suffixes = image_suffixes if isinstance(image_suffixes, Sequence) else [image_suffixes]
        self.omits = omits
        self.image_format = image_format
        if isinstance(concatenate, int) and concatenate == 0:
            warnings.warn('Concatenate in dimension 0 means to concatenate image channels, which is confusing. The concatenation will not be applied.', UserWarning)
        elif concatenate > 2:
            raise ValueError('Concatenate in dimension higher than 2 is not allowed.')
        self.concatenate = concatenate
        self.log_prediction = log_prediction
        self.log_folder = log_folder

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        pred, img_names = prediction
        self.write_prediction(pred, img_names, self.output_dir, self.image_suffixes, self.omits, self.image_format,
                              self.concatenate, self.log_prediction, self.log_folder, pl_module)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for (pred, img_names) in predictions:
            self.write_prediction(pred, img_names, self.output_dir, self.image_suffixes, self.omits, self.image_format,
                                  self.concatenate, self.log_prediction, self.log_folder, pl_module)

    @staticmethod
    def write_prediction(pred, img_names, output_dir, image_suffixes, omits, image_format, concatenate, log_prediction,
                         log_folder, pl_module):
        if not output_dir.exists():
            os.makedirs(output_dir)
        if not concatenate and len(pred) != len(image_suffixes):
            assert len(pred) == len(image_suffixes) + len(omits), 'Number of predictions must match the length of image suffixes plus the length of omits'
            for i in sorted(omits):
                image_suffixes.insert(i, 'dummy')
        # img_names: list of length B(batch_size)
        # pred: list of tensors of length N(number of predict output), tensors are of shape B, C, H, W
        paths = PredictionWriter.get_paths(img_names, output_dir, image_suffixes, omits, image_format, concatenate)
        # paths: nested list of paths (N x B, if not concatenated) or list of paths (length B, otherwise)
        pred = PredictionWriter.transform(pred, image_suffixes, omits, concatenate)
        # transformed pred: nested list of PIL images(N x B, if not concatenated) or list of PIL images(length B, otherwise)
        image_paths = PredictionWriter.save_image(pred, paths, concatenate, log_prediction, log_folder, pl_module)

        return image_paths

    @staticmethod
    def get_paths(img_names, output_dir, image_suffixes, omits, image_format, concatenate):
        paths = []
        if not concatenate:
            for i in range(len(image_suffixes)):
                if i not in omits:
                    suffix = image_suffixes[i]
                    suffix_paths = []
                    for j in range(len(img_names)):
                        suffix_paths.append(output_dir / '{img}{suf}.{fmt}'.format(
                            img=img_names[j],
                            suf='_' + suffix if suffix != '' else '',
                            fmt=image_format
                        ))
                    paths.append(suffix_paths)
        else:
            for i in range(len(img_names)):
                paths.append(output_dir / '{img}_concat.{fmt}'.format(
                        img=img_names[i],
                        fmt=image_format
                    ))

        return paths

    @staticmethod
    def transform(pred, image_suffixes, omits, concatenate):
        ret = []
        if not concatenate:
            for i in range(len(image_suffixes)):
                if i not in omits:
                    pred_img = pred[i]
                    mode = 'RGB' if pred_img.shape[1] == 3 else 'L'
                    suffix_imgs = []
                    for j in range(pred_img.shape[0]):
                        suffix_imgs.append(TF.to_pil_image(pred_img[j], mode=mode))
                    ret.append(suffix_imgs)
        else:
            _pred = []
            for i in range(len(pred)):
                if i not in omits:
                    _pred.append(pred[i])
            pred = torch.cat(_pred, dim=concatenate + 1)
            mode = 'RGB' if pred.shape[1] == 3 else 'L'
            for j in range(pred.shape[0]):
                ret.append(TF.to_pil_image(pred[j], mode=mode))

        return ret

    @staticmethod
    def save_image(pred, paths, concatenate, log_prediction, log_folder, pl_module):
        image_paths = []
        if not concatenate:
            for i in range(len(paths)):
                suffix_paths = paths[i]
                for j in range(len(suffix_paths)):
                    pred[i][j].save(suffix_paths[j])
                    image_paths.append(suffix_paths[j])
                    if log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
                        pl_module.logger.experiment.log_artifact(
                            pl_module.logger._run_id,
                            suffix_paths[j],
                            log_folder,
                        )
        else:
            for j in range(len(paths)):
                pred[j].save(paths[j])
                image_paths.append(paths[j])
                if log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
                    pl_module.logger.experiment.log_artifact(
                        pl_module.logger._run_id,
                        paths[j],
                        log_folder,
                    )

        return image_paths
