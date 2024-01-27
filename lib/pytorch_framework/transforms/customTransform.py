import random
import math
import numbers
from typing import Union, Optional, Tuple
from collections.abc import Sequence
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import _setup_angle, _check_sequence_input


__all__ = [
    'RandomHorizontalFlip',
    'RandomVerticalFlip',
    'RandomRotation',
    'RandomResizedCrop',
    'RandomResizedCrop_Relative',
    'RandomAffine',
    'Resize',
    'Crop',
    'CenterCrop',
    'RandomPerspective',
    'RandomAutoContrast',
    'RandomEqualize',
    'ColorJitter',
]


class BaseBatchTransform(nn.Module):
    def __init__(self):
        super(BaseBatchTransform, self).__init__()

    def forward(self, imgs: Union[Image.Image, torch.Tensor, Sequence[Union[Image.Image, torch.Tensor]]])\
            -> Union[Image.Image, torch.Tensor, Sequence[Union[Image.Image, torch.Tensor]]]:
        if isinstance(imgs, Sequence):
            result = []
            params = self.get_transform_params(imgs[0])
            for img in imgs:
                result.append(self.make_transform(img, params))
        else:
            params = self.get_transform_params(imgs)
            result = self.make_transform(imgs, params)
        return result

    def get_transform_params(self, img):
        pass

    def make_transform(self, img, params):
        pass

class BaseBatchTransform_Mask(BaseBatchTransform):
    def __init__(self, mask: Optional[Sequence[int]] = []):
        super(BaseBatchTransform_Mask, self).__init__()
        self.mask = mask

    def forward(self,imgs: Union[Image.Image, torch.Tensor, Sequence[Union[Image.Image, torch.Tensor]]])\
            -> Union[Image.Image, torch.Tensor, Sequence[Union[Image.Image, torch.Tensor]]]:
        if isinstance(imgs, Sequence):
            result = []
            params = self.get_transform_params(imgs[0])
            for i in range(len(imgs)):
                result.append(self.make_transform(imgs[i], params, True if i in self.mask else False))
        else:
            params = self.get_transform_params(imgs)
            result = self.make_transform(imgs, params, False)
        return result

    def make_transform(self, img, params, _mask):
        pass


class BatchTransform_MaskFill(BaseBatchTransform_Mask):
    def __init__(
            self,
            fill: Union[str, int, Sequence[int]] = 0,
            fill_mask: Union[str, int, Sequence[int]] = 0,
            mask: Optional[Sequence[int]] = []
    ):
        super(BatchTransform_MaskFill, self).__init__(mask)
        self.fill = fill
        if type(fill) is str and fill.lower() == 'auto':
            self.fill_mask = 'auto'
        else:
            self.fill_mask = fill_mask

    def calculate_fill(self, img, _mask):
        def autofill(img):
            img_hist = np.array(img.histogram())
            img_hist = img_hist.reshape(3, 256)
            fill = tuple([np.argmax(img_hist[i]) for i in (0, 1, 2)])
            return fill
        if not _mask:
            if type(self.fill) is str and self.fill.lower() == 'auto':
                # Auto calculate fill for image, set fill for ground truth to 0.
                fill = autofill(img)
            elif type(self.fill) is int or isinstance(self.fill, Sequence):
                fill = self.fill
            else:
                raise NotImplementedError('Only support str, int, Sequence[int] for fill')
        else:
            if type(self.fill_mask) is str and self.fill_mask.lower() == 'auto':
                # Auto calculate fill for image, set fill for ground truth to 0.
                fill = 0
            elif type(self.fill_mask) is int or isinstance(self.fill_mask, Sequence):
                fill = self.fill_mask
        return fill


class Resize(BaseBatchTransform):
    def __init__(
            self,
            size,
            interpolation=TF.InterpolationMode.BILINEAR,
            max_size=None,
            antialias='warn'
    ):
        super(Resize, self).__init__()
        self.transform = T.Resize(size, interpolation, max_size, antialias)

    def make_transform(self, img, params):
        return self.transform(img)


class Crop(BaseBatchTransform):
    def __init__(
            self,
            top,
            left,
            height,
            width,
    ):
        super(Crop, self).__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def make_transform(self, img, params):
        return TF.crop(img, self.top, self.left, self.height, self.width)


class CenterCrop(BaseBatchTransform):
    def __init__(self, size):
        super(CenterCrop, self).__init__()
        self.transform = T.CenterCrop(size)

    def make_transform(self, img, params):
        return self.transform(img)


class RandomHorizontalFlip(BaseBatchTransform):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.flip_prob = p

    def get_transform_params(self, img):
        return random.random() < self.flip_prob

    def make_transform(self, img, params):
        if params:
            return TF.hflip(img)
        else:
            return img


class RandomVerticalFlip(BaseBatchTransform):
    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.flip_prob = p

    def get_transform_params(self, img):
        return random.random() < self.flip_prob

    def make_transform(self, img, params):
        if params:
            return TF.vflip(img)
        else:
            return img


class RandomResizedCrop(BaseBatchTransform):
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias: Optional[Union[str, bool]] = 'warn'
    ):
        super(RandomResizedCrop, self).__init__()
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.antialias = antialias

    def get_transform_params(self, img):
        return T.RandomResizedCrop.get_params(img, self.scale, self.ratio)

    def make_transform(self, img, params):
        return TF.resized_crop(img, *params, self.size, self.interpolation, self.antialias)


class RandomResizedCrop_Relative(RandomResizedCrop):
    def __init__(
            self,
            size,
            scale_relative=(0.8, 1.2),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias: Optional[Union[str, bool]] = 'warn'
    ):
        super(RandomResizedCrop_Relative, self).__init__(size=size, ratio=ratio, interpolation=interpolation, antialias=antialias)
        self.scale_relative = scale_relative
        self.target_area = [self.size[0] * self.size[1] * self.scale_relative[i] for i in range(2)]

    def get_transform_params(self, img):
        image_size = TF.get_image_size(img)
        scale = [self.target_area[i] / (image_size[0] * image_size[1]) for i in range(2)]
        return T.RandomResizedCrop.get_params(img, scale, self.ratio)


class RandomRotation(BatchTransform_MaskFill):
    def __init__(
            self,
            degrees,
            interpolation=TF.InterpolationMode.NEAREST,
            expand = False,
            center = None,
            fill: Union[str, int, Sequence[int]] = 0,
            fill_mask: Union[str, int, Sequence[int]] = 0,
            mask: Optional[Sequence[int]] = []
    ):
        super(RandomRotation, self).__init__(fill, fill_mask, mask)
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        self.interpolation = interpolation
        self.expand = expand
        self.center = center

    def get_transform_params(self, img):
        return T.RandomRotation.get_params(self.degrees)

    def make_transform(self, img, params, _mask):
        fill = self.calculate_fill(img, _mask)
        return TF.rotate(img, params, self.interpolation, self.expand, self.center, fill)


class RandomPerspective(BatchTransform_MaskFill):
    def __init__(
            self,
            distortion_scale=0.5,
            p=0.5,
            interpolation=TF.InterpolationMode.BILINEAR,
            fill: Union[str, int, Sequence[int]] = 0,
            fill_mask: Union[str, int, Sequence[int]] = 0,
            mask: Optional[Sequence[int]] = []
    ):
        super(RandomPerspective, self).__init__(fill, fill_mask, mask)
        self.distortion_scale = distortion_scale
        self.p = p
        self.interpolation = interpolation

    def get_transform_params(self, img):
        # if isinstance(img, Image.Image):
        #     image_size = (img.size[0], img.size[1])
        # elif isinstance(img, torch.Tensor) and img.dim() == 3:
        #     image_size = (img.size(1), img.size(2))
        # else:
        #     raise NotImplementedError('Only support PIL.Image.Image or torch.Tensor with dim=3')
        image_size = TF.get_image_size(img)
        return random.random() < self.p, T.RandomPerspective.get_params(*image_size, self.distortion_scale)

    def make_transform(self, img, params, _mask):
        if params[0]:
            fill = self.calculate_fill(img, _mask)
            return TF.perspective(img, *params[1], self.interpolation, fill)
        else:
            return img


class RandomAffine(BatchTransform_MaskFill):
    def __init__(
            self,
            degrees,
            translate=None,
            scale=None,
            shear=None,
            interpolation=TF.InterpolationMode.NEAREST,
            fill: Union[str, int, Sequence[int]] = 0,
            center=None,
            fill_mask: Union[str, int, Sequence[int]] = 0,
            mask: Optional[Sequence[int]] = []
    ):
        super(RandomAffine, self).__init__(fill, fill_mask, mask)
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.interpolation = interpolation

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    def get_transform_params(self, img):
        if isinstance(img, Image.Image):
            image_size = (img.size[0], img.size[1])
        elif isinstance(img, torch.Tensor) and img.dim() == 3:
            image_size = (img.size(1), img.size(2))
        else:
            raise NotImplementedError('Only support PIL.Image.Image or torch.Tensor with dim=3')
        return T.RandomAffine.get_params(self.degrees, self.translate, self.scale, self.shear, image_size)

    def make_transform(self, img, params, _mask):
        fill = self.calculate_fill(img, _mask)
        return TF.affine(img, *params, self.interpolation, fill, self.center)


class RandomAutoContrast(BaseBatchTransform):
    def __init__(self, p=0.5):
        super(RandomAutoContrast, self).__init__()
        self.p = p

    def get_transform_params(self, img):
        return random.random() < self.p

    def make_transform(self, img, params):
        if params:
            return TF.autocontrast(img)
        else:
            return img


class RandomEqualize(BaseBatchTransform):
    def __init__(self, p=0.5):
        super(RandomEqualize, self).__init__()
        self.p = p

    def get_transform_params(self, img):
        return random.random() < self.p

    def make_transform(self, img, params):
        if params:
            return TF.equalize(img)
        else:
            return img


class ColorJitter(BaseBatchTransform):
    def __init__(
            self,
            brightness: Union[float, Tuple[float, float]] = 0,
            contrast: Union[float, Tuple[float, float]] = 0,
            saturation: Union[float, Tuple[float, float]] = 0,
            hue: Union[float, Tuple[float, float]] = 0,
    ):
        super(ColorJitter, self).__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(f"If {name} is a single number, it must be non negative.")
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            value = [float(value[0]), float(value[1])]
        else:
            raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

        if not bound[0] <= value[0] <= value[1] <= bound[1]:
            raise ValueError(f"{name} values should be between {bound}, but got {value}.")

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            return None
        else:
            return tuple(value)

    def get_transform_params(self, img):
        return T.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)

    def make_transform(self, img, params):
        for fn_id in params[0]:
            if fn_id == 0 and params[1] is not None:
                img = TF.adjust_brightness(img, params[1])
            elif fn_id == 1 and params[2] is not None:
                img = TF.adjust_contrast(img, params[2])
            elif fn_id == 2 and params[3] is not None:
                img = TF.adjust_saturation(img, params[3])
            elif fn_id == 3 and params[4] is not None:
                img = TF.adjust_hue(img, params[4])
        return img


if __name__ == '__main__':
    pass