import warnings
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
from ..transforms import make_grid


class FeatureMapVisualizer:
    colormap = cv2.COLORMAP_OCEAN

    def __init__(self, basefld, fmap_size, result_width, retain_size=True, colormap=None):
        self.basefld = Path(basefld)
        self.fmap_size = fmap_size
        if result_width % fmap_size != 0:
            warnings.warn('Result image width is not a multiple of feature map size, this will results in undesirable margin in the result image', UserWarning)
        self.result_width = result_width
        self.retain_size = retain_size
        if colormap is not None:
            self.colormap = getattr(cv2, colormap, self.colormap)

    def __call__(self, fmaps, image_idx):
        paths = self.get_image_path(fmaps, image_idx)
        for layer, fmap in fmaps.items():
            img = self.draw_featuremap(fmap)
            self.save_image(img, paths[layer])
        return paths

    def get_image_path(self, fmaps, image_idx):
        if not self.basefld.exists():
            os.makedirs(self.basefld)
        paths = {}
        for layer in fmaps.keys():
            paths[layer] = self.basefld / '{idx}_{lay}_featuremap.png'.format(idx=image_idx, lay=layer)
        return paths

    def draw_featuremap(self, fmap):
        B, C, H, W = fmap.shape
        assert B == 1, 'Only support single-batched input.'
        fmap = fmap.permute(1, 0, 2, 3).sigmoid()
        if self.retain_size:
            nrow = self.result_width // W
        else:
            nrow = self.result_width // self.fmap_size
            fmap = F.interpolate(fmap, size=(self.fmap_size, self.fmap_size), mode='bilinear')
        fmap = make_grid(fmap, nrow=nrow, padding=0)
        fmap = fmap.mul(255).add_(0.5).clamp_(0, 255).permute((1, 2, 0)).to("cpu", torch.uint8).numpy()
        fmap = cv2.applyColorMap(fmap, self.colormap)

        return fmap

    def save_image(self, image, path):
        cv2.imwrite(path.as_posix(), image)