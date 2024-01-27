import warnings
import os
from pathlib import Path
from collections.abc import Sequence
import torch
import cv2
import numpy as np
from ..transforms import make_grid


class TestVisualizer:
    palette = {
        'rendergraph-pred': (0, 0, 255),
        'rendergraph-GT': (0, 255, 0),
        'errormap-FN': (255, 0, 0), # Note: the color of FN and FP have been swapped since 2023.10.22
        'errormap-FP': (0, 255, 0),
        'errormap-TP': (255, 255, 255),
    }
    def __init__(self, basefld, image_channels, num_classes, export_rendergraph=True, export_errormap=True,
                 mean=None, std=None, palette=None, ignore_background=True):
        self.basefld = Path(basefld)
        self.image_channels = image_channels
        self.num_classes = num_classes
        self.export_rendergraph = export_rendergraph
        self.export_errormap = export_errormap
        self.ignore_background = ignore_background
        if palette:
            try:
                if np.all([k in palette.keys() for k in self.palette.keys()]):
                    self.palette = palette
                else:
                    warnings.warn("Missing key(s) in palette, use default instead.")
            except Exception:
                warnings.warn("Invalid palette, use default instead.")
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

    def __call__(self, image, pred, GT, image_idx, name_suffix=None):
        image_paths = self.get_image_path(image_idx, name_suffix)
        image, pred, GT = self.pre_process(image, pred, GT)
        if self.export_rendergraph:
            rendergraph = self.draw_image(image, pred, GT)
            self.save_image(rendergraph, image_paths['rendergraph'])
        if self.export_errormap:
            errormap = self.draw_errormap(pred, GT)
            self.save_image(errormap, image_paths['errormap'])

        return [v for v in image_paths.values()]

    def get_image_path(self, image_idx, name_suffix=None):
        if not self.basefld.exists():
            os.makedirs(self.basefld)
        image_paths = {}
        if self.export_rendergraph:
            image_paths['rendergraph'] = self.basefld / '{idx}{suffix}.png'.format(
                idx=image_idx,
                suffix='_' + name_suffix if name_suffix else ''
            )
        if self.export_errormap:
            image_paths['errormap'] = self.basefld / '{idx}{suffix}_errormap.png'.format(
                idx=image_idx,
                suffix='_' + name_suffix if name_suffix else ''
            )

        return image_paths

    def pre_process(self, image, pred, GT):
        def pred_inference(pred):
            if torch.max(pred) > 1 or torch.min(pred) < 0:
                # Logits
                if self.num_classes == 1:
                    pred = torch.sigmoid(pred)
                elif self.num_classes > 1:
                    pred = torch.softmax(pred, dim=1 if self.batched_input else 0)
            else:
                # Probabilities
                pass
            return pred

        def GT_inference(GT, pred):
            if GT.dim() == image.dim() - 1 \
                    and torch.max(GT) <= self.num_classes - 1 \
                    and torch.min(GT) >= 0:
                # Class index format, transform to one-hot format
                GT = torch.nn.functional.one_hot(GT, num_classes=pred.shape[1]).to(torch.float32)
                GT = GT.permute((0, 3, 1, 2) if self.batched_input else (2, 0, 1))
            elif GT.dim() == image.dim() and torch.max(GT) <= 1:
                # One-hot format or probabilities
                pass
            else:
                raise NotImplementedError('The format of ground truth cannot be recognized')
            return GT

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
               ] == self.num_classes, "The channels of prediction do not match this visualizer's setting"
        # Prediction format inference
        pred = pred_inference(pred)
        # Ground truth format inference
        GT = GT_inference(GT, pred)
        # Make grid for batched input
        if self.batched_input:
            image = make_grid(image)
            pred = make_grid(pred)
            GT = make_grid(GT)
        # Denormalize
        image = torch.cat([
            (image[i] * self.std[i] + self.mean[i]).unsqueeze(0) for i in range(self.image_channels)
        ], dim=0)
        # Convert to opencv format
        image = image.mul(255).add_(0.5).clamp_(0, 255).permute((1, 2, 0)).to("cpu", torch.uint8).numpy()
        pred = pred.mul(255).add_(0.5).clamp_(0, 255).permute((1, 2, 0)).to("cpu", torch.uint8).numpy()
        GT = GT.mul(255).add_(0.5).clamp_(0, 255).permute((1, 2, 0)).to("cpu", torch.uint8).numpy()

        return image, pred, GT

    def draw_image(self, image, pred, GT):
        if self.image_channels == 1:
            color_format = cv2.COLOR_GRAY2BGR
        elif self.image_channels == 3:
            color_format = cv2.COLOR_RGB2BGR
        else:
            raise NotImplementedError
        image = cv2.cvtColor(image, color_format)
        images = []
        for i in range(
                1 if self.num_classes > 1 and self.ignore_background else 0,
                self.num_classes):
            pred_class = pred[:, :, i]
            GT_class = GT[:, :, i]

            img = image.copy()

            pred_mask = np.stack([
                (pred_class * (self.palette['rendergraph-pred'][2] / 255)).astype(np.uint8),
                (pred_class * (self.palette['rendergraph-pred'][1] / 255)).astype(np.uint8),
                (pred_class * (self.palette['rendergraph-pred'][0] / 255)).astype(np.uint8)
            ], axis=2)
            GT_mask = np.stack([
                (GT_class * (self.palette['rendergraph-GT'][2] / 255)).astype(np.uint8),
                (GT_class * (self.palette['rendergraph-GT'][1] / 255)).astype(np.uint8),
                (GT_class * (self.palette['rendergraph-GT'][0] / 255)).astype(np.uint8)
            ], axis=2)

            img = cv2.addWeighted(img, 1, GT_mask, 0.8, 0)
            img = cv2.addWeighted(img, 1, pred_mask, 0.8, 0)
            images.append(img)

        return images

    def draw_errormap(self, pred, GT):
        errormaps = []
        for i in range(
                1 if self.num_classes > 1 and self.ignore_background else 0,
                self.num_classes):
            pred_class = pred[:, :, i]
            GT_class = GT[:, :, i]

            pred_mask = pred_class > 127
            GT_mask = GT_class > 127

            fn_mask = GT_mask & ~pred_mask
            fp_mask = ~GT_mask & pred_mask
            tp_mask = GT_mask & pred_mask

            fn_mask = np.stack([
                fn_mask.astype(np.uint8) * self.palette['errormap-FN'][2],
                fn_mask.astype(np.uint8) * self.palette['errormap-FN'][1],
                fn_mask.astype(np.uint8) * self.palette['errormap-FN'][0],
            ], axis=2)
            fp_mask = np.stack([
                fp_mask.astype(np.uint8) * self.palette['errormap-FP'][2],
                fp_mask.astype(np.uint8) * self.palette['errormap-FP'][1],
                fp_mask.astype(np.uint8) * self.palette['errormap-FP'][0],
            ], axis=2)
            tp_mask = np.stack([
                tp_mask.astype(np.uint8) * self.palette['errormap-TP'][2],
                tp_mask.astype(np.uint8) * self.palette['errormap-TP'][1],
                tp_mask.astype(np.uint8) * self.palette['errormap-TP'][0],
            ], axis=2)

            emap = cv2.addWeighted(tp_mask, 1, fn_mask, 1, 0)
            emap = cv2.addWeighted(emap, 1, fp_mask, 1, 0)
            errormaps.append(emap)

        return errormaps

    def save_image(self, image, path):
        if len(image) > 1:
            for i in range(len(image)):
                cv2.imwrite((path.parent / '{img}_Class{ind}{suf}'.format(
                    img=path.stem, ind=i, suf=path.suffix
                )).as_posix(), image[i])
        else:
            cv2.imwrite(path.as_posix(), image[0])