import os
import warnings
from pathlib import Path
from collections.abc import Sequence
from PIL import Image as PILImage
import torch
import torchvision.transforms.functional as TF
from lib.pytorch_framework.utils import CustomCfgNode as CN
from lightning.pytorch.callbacks import BasePredictionWriter
from lib.cv_utils import torch_merge_alpha, torch_align_channels


__all__ = [
    'build_prediction_writer',
    'PredictionWriter',
    'DEFAULT_CONFIG',
]


DEFAULT_CONFIG = CN(visible=False)
DEFAULT_CONFIG.output_dir = 'predictions'
DEFAULT_CONFIG.image_format = 'auto'
DEFAULT_CONFIG.concatenate = False
DEFAULT_CONFIG.log_prediction = False
DEFAULT_CONFIG.log_folder = ''
DEFAULT_CONFIG.write_interval = 'batch'
DEFAULT_CONFIG.set_typecheck_exclude_keys(['concatenate'])


def build_prediction_writer(node=DEFAULT_CONFIG):
    return PredictionWriter(**node)


class PredictionWriter(BasePredictionWriter):
    AVAILABLE_FORMATS = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'auto']


    def __init__(self, output_dir, image_format='auto', concatenate=False, log_prediction=False, log_folder='', write_interval='batch'):
        super(PredictionWriter, self).__init__(write_interval)
        self.output_dir = Path(output_dir)
        assert image_format.lower() in PredictionWriter.AVAILABLE_FORMATS, "Invalid image format. The available formats are 'jpg', 'jpeg', 'png', 'bmp', 'gif' and 'auto'"
        self.image_format = image_format.lower()
        if isinstance(concatenate, int) and concatenate == 0:
            warnings.warn('Concatenate in dimension 0 means to concatenate image channels, which is confusing. The concatenation will not be applied.', UserWarning)
        elif concatenate > 2:
            raise ValueError('Concatenate in dimension higher than 2 is not allowed.')
        self.concatenate = concatenate
        self.log_prediction = log_prediction
        self.log_folder = log_folder

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        pred, img_names = prediction
        self.write_prediction(pred, img_names, self.output_dir, self.concatenate,
                              self.log_prediction, self.log_folder, pl_module, self.image_format)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        for (pred, img_names) in predictions:
            self.write_prediction(pred, img_names, self.output_dir, self.concatenate,
                                  self.log_prediction, self.log_folder, pl_module, self.image_format)

    @staticmethod
    def write_prediction(pred, img_names, output_dir, concatenate, log_prediction,
                         log_folder, pl_module, image_format='auto'):
        # img_names: list of length B(batch_size)
        # pred: dict of string-tensor pairs of length N(number of predict output), tensors are of shape B, C, H, W
        if not output_dir.exists():
            os.makedirs(output_dir)

        _logging_available = False
        if log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
            _logging_available = True

        image_paths = []
        if not concatenate:
            for suffix, suffix_imgs in pred.items():
                assert isinstance(suffix_imgs, torch.Tensor) and suffix_imgs.ndim == 4, 'Images in prediction must be torch.Tensor with ndim = 4'
                assert suffix_imgs.shape[0] == len(img_names), 'Unmatched number of image names.'
                if not suffix.startswith('.'):
                    if image_format.lower() == 'auto':
                        suffix, fmt, mode = PredictionWriter.infer_format_and_mode(suffix, suffix_imgs)
                    else:
                        suffix, _ = PredictionWriter.infer_format(suffix)
                        fmt = image_format.lower() if image_format.lower() in PredictionWriter.AVAILABLE_FORMATS else 'png'
                        mode, suffix_imgs = PredictionWriter.infer_mode(suffix_imgs, fmt)

                    for i in range(len(img_names)):
                        img_path = output_dir / '{img}{suf}.{fmt}'.format(
                            img=img_names[i],
                            suf='_' + suffix if suffix != '' else '',
                            fmt=fmt
                        )
                        img = TF.to_pil_image(suffix_imgs[i], mode=mode)
                        img.save(img_path)
                        image_paths.append(img_path)
                        if _logging_available:
                            pl_module.logger.experiment.log_artifact(
                                pl_module.logger._run_id,
                                img_path,
                                log_folder,
                            )
        else:
            _is_alpha_exist = False
            mode = 'L'
            for suffix, suffix_imgs in pred.items():
                assert isinstance(suffix_imgs, torch.Tensor) and suffix_imgs.ndim == 4, 'Images in prediction must be torch.Tensor with ndim = 4'
                assert suffix_imgs.shape[0] == len(img_names), 'Unmatched number of image names.'
                if not suffix.startswith('.'):
                    suffix, _fmt, _mode = PredictionWriter.infer_format_and_mode(suffix, suffix_imgs)
                    if _fmt in ['png', 'gif']:
                        _is_alpha_exist = True
                    if _mode == 'RGBA':
                        mode = _mode
                    elif _mode == 'RGB':
                        mode = {'L': 'RGB', 'RGB': 'RGB', 'RGBA': 'RGBA'}[mode]
            if image_format.lower() == 'auto':
                fmt = 'png' if _is_alpha_exist else 'jpg'
            else:
                fmt = image_format.lower() if image_format.lower() in PredictionWriter.AVAILABLE_FORMATS else 'png'
                if fmt in ['jpg', 'jpeg', 'bmp']:
                    mode = {'L': 'L', 'RGB': 'RGB', 'RGBA': 'RGB'}[mode]

            _pred = []
            for suffix, suffix_imgs in pred.items():
                if not suffix.startswith('.'):
                    suffix_imgs = torch_align_channels(suffix_imgs, mode)
                    _pred.append(suffix_imgs)
            concat_pred = torch.cat(_pred, dim=concatenate + 1)
            for i in range(len(img_names)):
                concat_path = output_dir / '{img}_concat.{fmt}'.format(
                        img=img_names[i],
                        fmt=fmt
                    )
                concat_img = TF.to_pil_image(concat_pred[i], mode=mode)
                concat_img.save(concat_path)
                image_paths.append(concat_path)
                if _logging_available:
                    pl_module.logger.experiment.log_artifact(
                        pl_module.logger._run_id,
                        concat_path,
                        log_folder,
                    )

        return image_paths


    # @staticmethod
    # def get_paths(pred, img_names, output_dir, image_format, concatenate):
    #     paths = {}
    #     if not concatenate:
    #         for k in pred.keys():
    #             if not k.startswith('.'):
    #                 if image_format.lower() == 'auto':
    #                     suffix, fmt = PredictionWriter.infer_format(k)
    #                 elif image_format.lower() in PredictionWriter.AVAILABLE_FORMATS:
    #                     suffix = k
    #                     fmt = image_format.lower()
    #                 else:
    #                     warnings.warn('Invalid image format, use PNG format instead')
    #                     suffix = k
    #                     fmt = 'png'
    #                 suffix_paths = []
    #                 for j in range(len(img_names)):
    #                     suffix_paths.append(output_dir / '{img}{suf}.{fmt}'.format(
    #                         img=img_names[j],
    #                         suf='_' + suffix if suffix != '' else '',
    #                         fmt=fmt
    #                     ))
    #                 paths[k] = suffix_paths
    #     else:
    #         concat_paths = []
    #         if image_format.lower() == 'auto':
    #             _, fmt = PredictionWriter.infer_format(list(pred.keys())[0])
    #         elif image_format.lower() in PredictionWriter.AVAILABLE_FORMATS:
    #             fmt = image_format.lower()
    #         else:
    #             warnings.warn('Invalid image format, use PNG format instead')
    #             fmt = 'png'
    #         for i in range(len(img_names)):
    #             concat_paths.append(output_dir / '{img}_concat.{fmt}'.format(
    #                 img=img_names[i],
    #                 fmt=fmt
    #             ))
    #         paths['concat'] = concat_paths
    #
    #     return paths
    #
    # @staticmethod
    # def transform(pred, concatenate):
    #     ret = {}
    #     if not concatenate:
    #         for k in pred.keys():
    #             if not k.startswith('.'):
    #                 _, fmt = PredictionWriter.infer_format(k)
    #                 pred_img = pred[k]
    #                 mode, pred_img = PredictionWriter.infer_mode(pred_img, fmt)
    #                 suffix_imgs = []
    #                 for j in range(pred_img.shape[0]):
    #                     suffix_imgs.append(TF.to_pil_image(pred_img[j], mode=mode))
    #                 ret[k] = suffix_imgs
    #     else:
    #         concat_imgs = []
    #         _pred = []
    #         for k in pred.keys():
    #             if not k.startswith('.'):
    #                 _pred.append(pred[k])
    #         concat_pred = torch.cat(_pred, dim=concatenate + 1)
    #
    #         mode = 'RGB' if pred.shape[1] == 3 else 'L'
    #         for j in range(concat_pred.shape[0]):
    #             concat_imgs.append(TF.to_pil_image(concat_pred[j], mode=mode))
    #         ret['concat'] = concat_imgs
    #
    #     return ret
    #
    #
    #
    # @staticmethod
    # def save_image(pred, paths, log_prediction, log_folder, pl_module):
    #     image_paths = []
    #     assert set(pred.keys()) == set(paths.keys()), 'The keys of predictions and paths must be the same'
    #     for suffix, suffix_imgs in pred.items():
    #         for i in range(len(suffix_imgs)):
    #             suffix_imgs[i].save(paths[suffix][i])
    #             image_paths.append(paths[suffix][i])
    #             if log_prediction and hasattr(pl_module.logger.experiment, 'log_artifact'):
    #                 pl_module.logger.experiment.log_artifact(
    #                     pl_module.logger._run_id,
    #                     paths[suffix][i],
    #                     log_folder,
    #                 )
    #
    #     return image_paths

    @staticmethod
    def infer_format(name):
        name_parts = name.split('.')
        if len(name_parts) == 1:
            name_main = name
            name_fmt = 'png'
        else:
            name_main = '.'.join(name_parts[:-1])
            name_fmt = name_parts[-1]
            if name_fmt not in ['jpg', 'jpeg', 'bmp', 'png', 'gif']:
                warnings.warn('Cannot infer format from name {}, use PNG format instead'.format(name))
                name_fmt = 'png'

        return name_main, name_fmt

    @staticmethod
    def infer_mode(img_tensor, fmt):
        assert isinstance(img_tensor, torch.Tensor) and img_tensor.ndim == 4, 'Image must be torch.Tensor with ndim = 4'
        img_nchannel = img_tensor.shape[1]
        if img_nchannel == 4:
            if fmt in ['jpg', 'jpeg', 'bmp']:
                warnings.warn('The image has alpha channel which is not compatible with its format. The conversion will be applied.')
                img_tensor = torch_merge_alpha(img_tensor)
                mode = 'RGB'
            elif fmt in ['png', 'gif']:
                mode = 'RGBA'
        elif img_tensor.shape[1] == 3:
            mode = 'RGB'
        elif img_tensor.shape[1] == 1:
            mode = 'L'
        else:
            raise ValueError('The number of channels of the image must be 1, 3 or 4.')

        return mode, img_tensor

    @staticmethod
    def infer_format_and_mode(name, img_tensor):
        assert isinstance(img_tensor, torch.Tensor) and img_tensor.ndim == 4, 'Image must be torch.Tensor with ndim = 4'
        img_nchannel = img_tensor.shape[1]

        name_main, name_fmt = PredictionWriter.infer_format(name)

        if img_nchannel == 4:
            if name_fmt in ['jpg', 'jpeg', 'bmp']:
                warnings.warn('The image has alpha channel which is not compatible with its format. Use PNG format instead.')
                name_fmt = 'png'
            mode = 'RGBA'
        elif img_nchannel == 3:
            mode = 'RGB'
        elif img_nchannel == 1:
            mode = 'L'
        else:
            raise ValueError('The number of channels of the image must be 1, 3 or 4.')

        return name_main, name_fmt, mode


if __name__ == '__main__':
    from lightning import LightningModule
    module = LightningModule()
    output_dir = Path(r"d:\pythondata\test")
    writer = PredictionWriter(output_dir)
    pred = {'result.png': torch.rand((4, 3, 32, 32)),
            'result2.png': torch.rand((4, 1, 32, 32)),
            'result3.png': torch.rand((4, 4, 32, 32)),
            'result4.jpg': torch.rand((4, 3, 32, 32)),
            'result5.jpg': torch.rand((4, 4, 32, 32)),
            'result6.gif': torch.rand((4, 4, 32, 32)),
            'result7.bmp': torch.rand((4, 4, 32, 32))}
    img_names = ['A', 'B', 'C', 'D']
    writer.write_prediction(pred, img_names, output_dir, 1, False, '', module, 'auto')


    ii = 0