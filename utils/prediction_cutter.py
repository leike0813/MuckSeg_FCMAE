from enum import IntEnum
import math

import torch
import torchvision.transforms.functional as TF
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error


class PredictionCutter:
    class PredictMode(IntEnum):
        FULL_SIZE = 1
        THREE_FOLD = 2
        PATCH = 3


    def __init__(self, pl_module, config):
        self.pl_module = pl_module
        self.config = config
        self.calculate_roi_cropbox()

    def calculate_roi_cropbox(self):
        self.image_roi = self.config.PREDICT.IMAGE_ROI
        self.image_size = self.config.PREDICT.IMAGE_SIZE
        zero_mask = torch.zeros((
            self.config.PREDICT.DATAMODULE.batch_size,
            self.config.MODEL.OUT_CHANS,
            self.image_size[1],
            self.image_size[0]
        ), dtype=torch.float32)

        if self.image_roi[0] > 0:
            self.left_mask = TF.crop(zero_mask, 0, 0, self.image_size[1], self.image_roi[0])
        else:
            self.left_mask = None
        if self.image_roi[0] + self.image_roi[2] < self.image_size[0]:
            self.right_mask = TF.crop(zero_mask, 0, self.image_roi[0] + self.image_roi[2],
                                  self.image_size[1], self.image_size[0] - self.image_roi[0] - self.image_roi[2])
        else:
            self.right_mask = None
        if self.image_roi[1] > 0:
            self.top_mask = TF.crop(zero_mask, 0, self.image_roi[0], self.image_roi[1], self.image_roi[2])
        else:
            self.top_mask = None
        if self.image_roi[1] + self.image_roi[3] < self.image_size[1]:
            self.bottom_mask = TF.crop(zero_mask, self.image_roi[1] + self.image_roi[3], self.image_roi[0],
                                   self.image_size[1] - self.image_roi[1] - self.image_roi[3], self.image_roi[2])
        else:
            self.bottom_mask = None

        self.cropbox_roi = [self.image_roi[1], self.image_roi[0], self.image_roi[3], self.image_roi[2]]

    def calculate_3fold_cut_coords(self):
        self.fold_size = math.ceil((self.image_roi[3] / 3)
                                   / self.pl_module.model.size_modulus) * self.pl_module.model.size_modulus
        self.fold_margin = math.ceil(self.fold_size * self.config.PREDICT.THREEFOLD_MARGIN_RATE
                                     / self.pl_module.model.size_modulus) * self.pl_module.model.size_modulus
        self.cut_coords = torch.zeros((3, 6), dtype=int)
        self.cut_coords[0, 0] = 0
        self.cut_coords[0, 2] = 0
        self.cut_coords[0, 4] = self.fold_size
        self.cut_coords[0, 5] = self.image_roi[2]

        self.cut_coords[1, 0] = math.floor((self.image_roi[3] - self.fold_size - self.fold_margin) / 2)
        self.cut_coords[1, 2] = math.ceil((3 * self.fold_size - self.image_roi[3] + self.fold_margin) / 2)
        self.cut_coords[1, 4] = self.image_roi[3] - 2 * self.fold_size
        self.cut_coords[1, 5] = self.image_roi[2]

        self.cut_coords[2, 0] = self.image_roi[3] - self.fold_size - self.fold_margin
        self.cut_coords[2, 2] = self.fold_margin
        self.cut_coords[2, 4] = self.fold_size
        self.cut_coords[2, 5] = self.image_roi[2]

    def calculate_patch_cut_coords(self):
        self.window_size = self.config.MODEL.IMAGE_SIZE

        ROI_margin = [self.image_roi[3 - i] - (self.image_roi[3 - i] // self.window_size) * self.window_size for i in range(2)]
        num_cuts = [(self.image_roi[3 - i] - ROI_margin[i] - 3 * self.window_size // 2)
                    // (self.window_size // 2)
                    + 2
                    for i in range(2)]

        self.cut_coords = torch.zeros((
            num_cuts[0] + (0 if ROI_margin[0] == 0 else 1),
            num_cuts[1] + (0 if ROI_margin[1] == 0 else 1),
            6
        ), dtype=int)

        for i in range(num_cuts[0]):
            for j in range(num_cuts[1]):
                self.cut_coords[i, j, 0] = i * self.window_size // 2
                self.cut_coords[i, j, 1] = j * self.window_size // 2
                self.cut_coords[i, j, 2] = 0 if i == 0 else self.window_size // 4
                self.cut_coords[i, j, 3] = 0 if j == 0 else self.window_size // 4
                self.cut_coords[i, j, 4] = self.window_size // 2 if i > 0 and i < num_cuts[0] - 1 else 3 * self.window_size // 4
                self.cut_coords[i, j, 5] = self.window_size // 2 if j > 0 and j < num_cuts[1] - 1 else 3 * self.window_size // 4
            if ROI_margin[1] > 0:
                self.cut_coords[i, num_cuts[1], 0] = i * self.window_size // 2
                self.cut_coords[i, num_cuts[1], 1] = (num_cuts[1] - 1) * self.window_size // 2 + ROI_margin[1]
                self.cut_coords[i, num_cuts[1], 2] = 0 if i == 0 else self.window_size // 4
                self.cut_coords[i, num_cuts[1], 3] = self.window_size - ROI_margin[1]
                self.cut_coords[i, num_cuts[1], 4] = self.window_size // 2 if i > 0 and i < num_cuts[0] - 1 else 3 * self.window_size // 4
                self.cut_coords[i, num_cuts[1], 5] = ROI_margin[1]
        if ROI_margin[0] > 0:
            for j in range(num_cuts[1]):
                self.cut_coords[num_cuts[0], j, 0] = (num_cuts[0] - 1) * self.window_size // 2 + ROI_margin[0]
                self.cut_coords[num_cuts[0], j, 1] = j * self.window_size // 2
                self.cut_coords[num_cuts[0], j, 2] = self.window_size - ROI_margin[0]
                self.cut_coords[num_cuts[0], j, 3] = 0 if j == 0 else self.window_size // 4
                self.cut_coords[num_cuts[0], j, 4] = ROI_margin[0]
                self.cut_coords[num_cuts[0], j, 5] = self.window_size // 2 if j > 0 and j < num_cuts[1] - 1 else 3 * self.window_size // 4
            if ROI_margin[1] > 0:
                self.cut_coords[num_cuts[0], num_cuts[1], 0] = (num_cuts[0] - 1) * self.window_size // 2 + ROI_margin[0]
                self.cut_coords[num_cuts[0], num_cuts[1], 1] = (num_cuts[1] - 1) * self.window_size // 2 + ROI_margin[1]
                self.cut_coords[num_cuts[0], num_cuts[1], 2] = self.window_size - ROI_margin[0]
                self.cut_coords[num_cuts[0], num_cuts[1], 3] = self.window_size - ROI_margin[1]
                self.cut_coords[num_cuts[0], num_cuts[1], 4] = ROI_margin[0]
                self.cut_coords[num_cuts[0], num_cuts[1], 5] = ROI_margin[1]

        self.cut_coords_downsampled = self.cut_coords // 4

        return self.cut_coords.shape

    def cut_roi(self, img):
        return TF.crop(img, *self.cropbox_roi)

    def cut_fold(self, roi):
        folds = []
        folds.append(TF.crop(roi, self.cut_coords[0, 0], self.cut_coords[0, 1],
                             self.fold_size + self.fold_margin, self.image_roi[2]))
        folds.append(TF.crop(roi, self.cut_coords[1, 0], self.cut_coords[1, 1],
                             self.fold_size + self.fold_margin, self.image_roi[2]))
        folds.append(TF.crop(roi, self.cut_coords[2, 0], self.cut_coords[2, 1],
                             self.fold_size + self.fold_margin, self.image_roi[2]))

        return folds

    def cut_patch(self, roi):
        patches = []
        for i in range(self.cut_coords.shape[0]):
            patches.append([])
            for j in range(self.cut_coords.shape[1]):
                patches[i].append(TF.crop(roi, self.cut_coords[i, j, 0], self.cut_coords[i, j, 1], self.window_size, self.window_size))

        return patches

    def concatenate(self, roi):
        cat_list = []
        if self.top_mask is not None:
            cat_list.append(self.top_mask.to(roi.device))
        cat_list.append(roi)
        if self.bottom_mask is not None:
            cat_list.append(self.bottom_mask.to(roi.device))
        roi = torch.cat(cat_list, dim=2)
        cat_list = []
        if self.left_mask is not None:
            cat_list.append(self.left_mask.to(roi.device))
        cat_list.append(roi)
        if self.right_mask is not None:
            cat_list.append(self.right_mask.to(roi.device))
        result = torch.cat(cat_list, dim=3)

        return result

    def find_predict_mode(self):
        if not self.pl_module.model.is_auto_scalable:
            print('Model is not auto-scalable, force to use patch mode for inference.')
            patches_shape = self.calculate_patch_cut_coords()
            return self.PredictMode.PATCH, patches_shape
        if not self.config.PREDICT.IMAGE_ROI[2] % self.pl_module.model.size_modulus \
                and not self.config.PREDICT.IMAGE_ROI[3] % self.pl_module.model.size_modulus:
            test_tensor = torch.rand((
                self.config.PREDICT.DATAMODULE.batch_size,
                self.config.MODEL.IN_CHANS,
                self.config.PREDICT.IMAGE_ROI[3],
                self.config.PREDICT.IMAGE_ROI[2]
            ), device=self.pl_module.device)
            try: # try full-size(roi) prediction
                _ = self.pl_module(test_tensor)
                print('Memory test passed, use full-size mode for inference.')
                del test_tensor
                garbage_collection_cuda()
                return self.PredictMode.FULL_SIZE, ()
            except RuntimeError as e:
                if is_oom_error(e):
                    # If we fail in full-size mode, try the 3-fold mode
                    del test_tensor
                    garbage_collection_cuda()
                else:
                    raise e
        print('Failed to make inference at full-size, attempting to use 3-fold mode.')
        self.calculate_3fold_cut_coords()
        test_tensor = torch.rand((
            self.config.PREDICT.DATAMODULE.batch_size,
            self.config.MODEL.IN_CHANS,
            self.fold_size + self.fold_margin,
            self.config.PREDICT.IMAGE_ROI[2]
        ), device=self.pl_module.device)
        try:  # try 3-fold(roi) prediction
            _ = self.pl_module(test_tensor)
            print('Memory test passed, use 3-fold mode for inference.')
            del test_tensor
            garbage_collection_cuda()
            return self.PredictMode.THREE_FOLD, ()
        except RuntimeError as e:
            if is_oom_error(e):
                # If we fail in 3-fold mode, use the patch mode
                del test_tensor
                garbage_collection_cuda()
                print('Failed to make inference by 3-fold mode, falling back to patch mode.')
                patches_shape = self.calculate_patch_cut_coords()
                return self.PredictMode.PATCH, patches_shape
            else:
                raise e

# EOF