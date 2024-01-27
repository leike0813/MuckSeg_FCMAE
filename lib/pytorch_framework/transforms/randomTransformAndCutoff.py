from collections.abc import Sequence
import random
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.transforms import _setup_angle
from .customTransform import BaseBatchTransform_Mask
from .maxROICutter import ImageWithIrregularROI


__all__ = ['RandomTransformAndCutoff']


class RandomTransformAndCutoff(BaseBatchTransform_Mask):
    def __init__(
            self,
            degrees,
            distortion_scale=0.5,
            p_perspective=0.5,
            p_hflip=0.5,
            p_vflip=0.5,
            interpolation=TF.InterpolationMode.BILINEAR,
            mask=[]
    ):
        super(RandomTransformAndCutoff, self).__init__(mask)
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))
        self.distortion_scale = distortion_scale
        self.p_perspective = p_perspective
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.interpolation = interpolation

    def get_transform_params(self, img):
        # if isinstance(img, Image.Image):
        #     image_size = (img.size[0], img.size[1])
        # elif isinstance(img, torch.Tensor) and img.dim() == 3:
        #     image_size = (img.size(1), img.size(2))
        # else:
        #     raise NotImplementedError('Only support PIL.Image.Image or torch.Tensor with dim=3')
        image_size = TF.get_image_size(img)
        perspective_params = T.RandomPerspective.get_params(*image_size, self.distortion_scale)
        rotation_params = T.RandomRotation.get_params(self.degrees)
        _do_random_perspective = random.random() < self.p_perspective
        _do_hflip = random.random() < self.p_hflip
        _do_vflip = random.random() < self.p_vflip
        if _do_random_perspective:
            image_ROI = perspective_params[1]
        else:
            image_ROI = [[0, 0], [image_size[0] - 1, 0], [image_size[0] - 1, image_size[1] - 1], [0, image_size[1] - 1]]
        cutter = ImageWithIrregularROI(*image_size, image_ROI)
        cutter.rotate(-rotation_params)
        if _do_hflip:
            cutter.hflip()
        if _do_vflip:
            cutter.vflip()
        orig_ROI = np.array(cutter.list_points)
        cutter.cut_max_ROI()
        cut_params = (cutter.p1.y, cutter.p1.x, cutter.p4.y - cutter.p1.y, cutter.p2.x - cutter.p1.x)
        return (perspective_params, rotation_params, cut_params, orig_ROI,
                _do_random_perspective, _do_hflip, _do_vflip)

    def make_transform(self, img, params, _mask):
        if params[4]:
            img = TF.perspective(img, *params[0], interpolation=self.interpolation)
        img = TF.rotate(img, params[1], interpolation=self.interpolation, expand=True)
        if params[5]:
            img = TF.hflip(img)
        if params[6]:
            img = TF.vflip(img)
        if _mask:
            import cv2
            validator_channels = TF.get_image_num_channels(img)
            if isinstance(img, torch.Tensor):
                validator_type = 'tensor'
                if validator_channels == 3:
                    img = img.permute(1, 2, 0)
            elif TF.F_pil._is_pil_image(img):
                validator_type = 'pil'
            else:
                raise TypeError("Unexpected type {}".format(type(img)))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_GRAY2BGR if validator_channels == 1 else cv2.COLOR_RGB2BGR)
            cv2.polylines(img, [np.array(params[3])], True, (0, 0, 255), 3)
            cv2.polylines(img, [np.array([
                [params[2][1], params[2][0]],
                [params[2][1] + params[2][3], params[2][0]],
                [params[2][1] + params[2][3], params[2][0] + params[2][2]],
                [params[2][1], params[2][0] + params[2][2]],
            ], dtype=np.int64)], True, (0, 255, 0), 3)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if validator_type == 'tensor':
                img = torch.Tensor(img).permute(2, 0, 1)
            elif validator_type == 'pil':
                img = Image.fromarray(img, "RGB")
        else:
            img = TF.crop(img, *params[2])
        return img

