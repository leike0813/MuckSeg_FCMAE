import os
from pathlib import Path
from collections.abc import Sequence
import torch
from torchvision.transforms import ToPILImage
from lib.pytorch_framework.transforms import make_grid
from lib.pytorch_framework.utils import CustomCfgNode as CN
from lib.pytorch_framework.visualization import FeatureMapVisualizer
from lib.pytorch_framework.visualization import DEFAULT_CONFIG as DEFAULT_CONFIG_FEATUREMAP


DEFAULT_CONFIG = CN(visible=False)

DEFAULT_CONFIG.BASE_FLD = 'results'
DEFAULT_CONFIG.IMAGE_CHANNELS = 1
DEFAULT_CONFIG.IMAGE_MEAN = [0.6]
DEFAULT_CONFIG.IMAGE_STD = [0.2]
DEFAULT_CONFIG.EXPORT_FEATUREMAP = True
DEFAULT_CONFIG.NUM_FEATUREMAP_SAMPLES = 3
DEFAULT_CONFIG.FEATUREMAP = DEFAULT_CONFIG_FEATUREMAP.FEATUREMAP.clone()


def build_visualizer(node=DEFAULT_CONFIG):
    test_visualizer = TestVisualizer(
        basefld=node.BASE_FLD,
        image_channels=node.IMAGE_CHANNELS,
        mean=node.IMAGE_MEAN,
        std=node.IMAGE_STD,
    )
    if node.EXPORT_FEATUREMAP:
        feature_visualizer = FeatureMapVisualizer(
            basefld=node.BASE_FLD,
            **node.FEATUREMAP
        )
    else:
        feature_visualizer = None

    return test_visualizer, feature_visualizer


class TestVisualizer:
    def __init__(self, basefld, image_channels, mean=None, std=None,):
        self.basefld = Path(basefld)
        self.image_channels = image_channels
        self.converter = ToPILImage(mode='L')
        if mean is not None and std is not None:
            assert isinstance(mean, Sequence) and isinstance(std, Sequence), 'The mean and std must be sequences'
            assert len(mean) == len(std), 'The mean and std must be sequences with same length'
            assert len(mean) == self.image_channels, 'The mean and std must be sequences with length of {length}'.format(
                length=self.image_channels
            )
            self.mean = mean
            self.std = std
        else:
            self.mean = [0.0 for i in range(self.image_channels)]
            self.std = [1.0 for i in range(self.image_channels)]

    def __call__(self, image, pred, image_idx, mask=None, name_suffix=None):
        image_paths = self.get_image_path(image_idx, name_suffix)
        image, pred = self.pre_process(image, pred)
        result = self.draw_image(image, pred, mask)
        self.save_image(result, image_paths['result'])

        return [v for v in image_paths.values()]

    def get_image_path(self, image_idx, name_suffix=None):
        if not self.basefld.exists():
            os.makedirs(self.basefld)
        image_paths = {}
        image_paths['result'] = self.basefld / '{idx}{suffix}.png'.format(
            idx=image_idx,
            suffix='_' + name_suffix if name_suffix else ''
        )

        return image_paths

    def pre_process(self, image, pred):
        # Dimensionality check
        assert image.dim() == pred.dim(), 'The dimensionality of image and prediction must be the same'
        if image.dim() == 3:
            self.batched_input = False
        elif image.dim() == 4:
            self.batched_input = True
        else:
            raise NotImplementedError('The dimensionality of input image cannot be recognized')
        assert image.shape[
                   1 if self.batched_input else 0
               ] == self.image_channels, "The channels of input image do not match this visualizer's setting"
        assert pred.shape[
                   1 if self.batched_input else 0
               ] == self.image_channels, "The channels of prediction do not match this visualizer's setting"

        return image, pred

    def draw_image(self, image, pred, mask):
        if mask is not None:
            mask = torch.cat([mask for i in range(self.image_channels)], dim=1 if self.batched_input else 0)
            pred = pred * mask + image * (1. - mask)
            masked_image = image * (1. - mask)

            target = torch.cat([image, pred, masked_image], dim=2 if self.batched_input else 1)
        else:
            target = torch.cat([image, pred], dim=2 if self.batched_input else 1)

        if self.batched_input:
            target = make_grid(target)

        # Denormalize
        target = torch.cat([
            (target[i] * self.std[i] + self.mean[i]).unsqueeze(0)
            for i in range(self.image_channels)
        ], dim=0)

        target = self.converter(target)

        images = []
        images.append(target)

        return images

    def save_image(self, image, path):
        if len(image) > 1:
            for i in range(len(image)):
                image[i].save((path.parent / '{img}_Class{ind}{suf}'.format(
                    img=path.stem, ind=i, suf=path.suffix
                )).as_posix())
        else:
            image[0].save(path.as_posix())

# EOF