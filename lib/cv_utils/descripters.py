import numpy as np
import math
import cv2
from collections.abc import Sequence
from enum import IntEnum


def check_param(window_size, block_size, block_stride, cell_size):
    for i, param in enumerate([window_size, block_size, block_stride, cell_size]):
        if isinstance(param, int):
            if i == 0:
                windos_size = (param, param)
            elif i == 1:
                block_size = (param, param)
            elif i == 2:
                block_stride = (param, param)
            else:
                cell_size = (param, param)
        elif isinstance(param, Sequence) and len(param) == 2:
            pass
        else:
            raise ValueError("{} must be integer or Sequence of length 2, but got {}".format(
                {0: 'window_size', 1: 'block_size', 2: 'block_stride', 3: 'cell_size'}[i], param))

    assert window_size[0] % cell_size[0] == 0, "window_size must be a multiple of cell_size"
    assert window_size[1] % cell_size[1] == 0, "window_size must be a multiple of cell_size"
    assert block_size[0] % cell_size[0] == 0, "block_size must be a multiple of cell_size"
    assert block_size[1] % cell_size[1] == 0, "block_size must be a multiple of cell_size"
    assert block_stride[0] % cell_size[0] == 0, "block_stride must be a multiple of cell_size"
    assert block_stride[1] % cell_size[1] == 0, "block_stride must be a multiple of cell_size"
    assert (window_size[0] - block_size[0]) % block_stride[0] == 0, \
        "Invalid window_size, please check the compatibility of the parameters"
    assert (window_size[1] - block_size[1]) % block_stride[1] == 0, \
        "Invalid window_size, please check the compatibility of the parameters"

    return window_size, block_size, block_stride, cell_size


def rearrange_HOG(hog_features, window_size, block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8),
                  fill=False, renormalize=False, squeeze=False):
    """
    Rearrange flat HOG descripter vector to cell-wise form
    :param hog_features:
    :param window_size:
    :param block_size:
    :param block_stride:
    :param cell_size:
    :return:
    """
    window_size, block_size, block_stride, cell_size = check_param(window_size, block_size, block_stride, cell_size)
    
    cells_per_block = (block_size[0] // cell_size[0], block_size[1] // cell_size[1])
    cells_per_stride = (block_stride[0] // cell_size[0], block_stride[1] // cell_size[1])
    max_overlap = (math.ceil(block_size[0] / block_stride[0]), math.ceil(block_size[1] / block_stride[1]))
    block_resolution = (
        (window_size[0] - block_size[0]) // block_stride[0] + 1,
        (window_size[1] - block_size[1]) // block_stride[1] + 1
    )
    cell_resolution = (window_size[0] // cell_size[0], window_size[1] // cell_size[1])
    overlap_matr = np.zeros(cell_resolution, dtype=np.uint8)

    assert (hog_features.shape[0] // (block_resolution[0] * block_resolution[1])) \
           % (cells_per_block[0] * cells_per_block[1]) == 0,\
        "Cannot infer number of HOG orientations, please check the compatibility of input feature vector and the parameters"
    num_orientations = hog_features.shape[0] // (block_resolution[0] * block_resolution[1]) \
                       // (cells_per_block[0] * cells_per_block[1])
    
    cell_hog_array = np.zeros((cell_resolution[0], cell_resolution[1], max_overlap[0] * max_overlap[1], num_orientations))

    hog_features = hog_features.reshape(
        block_resolution[0],
        block_resolution[1],
        (cells_per_block[0] * cells_per_block[1]),
        num_orientations
    )

    for x in range(hog_features.shape[0]):
        for y in range(hog_features.shape[1]):
            block_hogs = hog_features[x, y, :, :]
            for i in range(cells_per_block[0]):
                for j in range(cells_per_block[1]):
                    cell_idx = (x * cells_per_stride[0] + i, y * cells_per_stride[1] + j)
                    cell_hogs = block_hogs[i * cells_per_block[1] + j, :]
                    if renormalize:
                        cell_hogs = cell_hogs / np.linalg.norm(cell_hogs)
                        cell_hogs = np.where(np.isnan(cell_hogs), 0.0, cell_hogs)
                    cell_hog_array[
                        cell_idx[0],
                        cell_idx[1],
                        overlap_matr[cell_idx[0], cell_idx[1]],
                    ] = cell_hogs
                    overlap_matr[cell_idx[0], cell_idx[1]] += 1
    if fill:
        for i in range(cell_resolution[0]):
            for j in range(cell_resolution[1]):
                _pre = overlap_matr[i, j]
                _cnt = 0
                while _pre < max_overlap[0] * max_overlap[1]:
                    cell_hog_array[i, j, _pre, :] = cell_hog_array[i, j, _cnt % overlap_matr[i, j], :]
                    _pre += 1
                    _cnt += 1

    if squeeze:
        cell_hog_array = cell_hog_array.reshape(cell_hog_array.shape[0], cell_hog_array.shape[1], -1)

    return cell_hog_array


def visualize_HOG(cell_hogs, visual_cell_size=(64, 64), length_reduce_factor=1, thickness=1):
    if isinstance(thickness, int):
        thickness = (thickness, thickness)
    elif isinstance(thickness, Sequence) and len(thickness) == 2 and isinstance(thickness[0], int) and isinstance(thickness[1], int):
        pass
    else:
        raise ValueError('thickness must be integer or 2-sequence of integers')
    win_size = (
        cell_hogs.shape[0] * visual_cell_size[0],
        cell_hogs.shape[1] * visual_cell_size[1]
    )
    num_orientations = cell_hogs.shape[2]
    img = np.zeros((win_size[0], win_size[1], 1), dtype=np.uint8)
    max_len = min(visual_cell_size) // length_reduce_factor
    max_val = cell_hogs.max()

    max_orientation = np.argmax(cell_hogs, axis=2)
    for x in range(cell_hogs.shape[0]):
        for y in range(cell_hogs.shape[1]):
            cx = x * visual_cell_size[0] + visual_cell_size[0] // 2
            cy = y * visual_cell_size[1] + visual_cell_size[1] // 2
            for o in range(num_orientations):
                angle = o * (np.pi / num_orientations)
                val = cell_hogs[x, y, o] / max_val
                _emphasis = o == max_orientation[x, y] and val >= 0.5
                dx = int(np.cos(angle) * val * max_len)
                dy = int(np.sin(angle) * val * max_len)
                color = 255 if _emphasis else int(255 * math.sqrt(val))
                _thickness = thickness[1] if _emphasis else thickness[0]
                cv2.line(img, (cy - dy, cx - dx), (cy + dy, cx + dx), color, _thickness)

    return img


def render_HOG(cell_hogs, img, visual_cell_size=(64, 64), color=(0, 0, 255), background_alpha=1.0, length_reduce_factor=1, thickness=1):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, dsize=(cell_hogs.shape[0] * visual_cell_size[0], cell_hogs.shape[1] * visual_cell_size[1]), interpolation=cv2.INTER_CUBIC)
    img = (img.astype(float) * background_alpha).astype(np.uint8)
    cell_hog_img = visualize_HOG(cell_hogs, visual_cell_size, length_reduce_factor, thickness)
    _, mask_array = cv2.threshold(cell_hog_img, 1, 255, cv2.THRESH_BINARY)
    mask_array = mask_array == 255
    cell_hog_img = np.concatenate([
        ((cell_hog_img / 255) * color[i]).astype(np.uint8) for i in [0, 1, 2]
    ], axis=2)

    img[:, :, :][mask_array] = cell_hog_img[:, :, :][mask_array]

    return img


class NormalizationMethod(IntEnum):
    NONE = -1
    L2 = 1
    L2_HYS = 2
    L1 = 3
    L1_SQRT = 4


class HOGExtractor:
    NormalizationMethod = NormalizationMethod

    def __init__(self, block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=8, sobel_ksize=1, l2_hys_threshold=0.2):
        _, self.block_size, self.block_stride, self.cell_size = check_param((0, 0), block_size, block_stride, cell_size)
        self.cells_per_block = (self.block_size[0] // self.cell_size[0], self.block_size[1] // self.cell_size[1])
        self.cells_per_stride = (self.block_stride[0] // self.cell_size[0], self.block_stride[1] // self.cell_size[1])
        assert isinstance(nbins, int), 'nbins must be integer'
        self.nbins = nbins
        self.angle_unit = 180 / self.nbins
        assert sobel_ksize in [-1, 1, 3, 5, 7], 'sobel_ksize must be 1, 3, 5, 7, or -1 (Scharr filter)'
        self.sobel_ksize = sobel_ksize
        assert l2_hys_threshold > 0.0 and l2_hys_threshold < 1.0, 'l2_hys_threshold must between 0 and 1'
        self.l2_hys_threshold = l2_hys_threshold

    def compute_gradient(self, image):
        assert isinstance(image, np.ndarray), 'image must be a numpy array'
        assert image.ndim == 3 or image.ndim == 2, 'image must be a 3-dimensional or 2-dimensional numpy array'
        if image.ndim == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        else:
            assert image.shape[2] in [1, 3, 4], 'Invalid channel numbers. image must have 1, 3, or 4 channel(s)'
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        gradient_values_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
        gradient_values_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.sobel_ksize)
        gradient_magnitude = cv2.addWeighted(cv2.pow(gradient_values_x, 2), 1.0, cv2.pow(gradient_values_y, 2), 1.0, 0)
        gradient_magnitude = cv2.pow(gradient_magnitude, 0.5)
        gradient_angle = cv2.phase(gradient_values_x, gradient_values_y, angleInDegrees=True)
        gradient_angle = np.where(gradient_angle > 180, gradient_angle - 180, gradient_angle)
        return gradient_magnitude, gradient_angle

    def cell_histogram(self, cell_magnitude, cell_angle, normalize=False):
        orientation_centers = [0.01] * self.nbins
        for i in range(cell_magnitude.shape[0]):
            for j in range(cell_magnitude.shape[1]):
                gradient_strength = cell_magnitude[i][j]
                gradient_angle = cell_angle[i][j]
                min_angle, max_angle, mod = self.get_closest_bins(gradient_angle)
                orientation_centers[min_angle] += (gradient_strength * (1 - (mod / self.angle_unit)))
                orientation_centers[max_angle] += (gradient_strength * (mod / self.angle_unit))
        if normalize:
            orientation_norm = np.linalg.norm(orientation_centers)
            orientation_centers = [c / orientation_norm for c in orientation_centers]
        return orientation_centers

    def get_closest_bins(self, gradient_angle):
        idx = int(gradient_angle / self.angle_unit)
        mod = gradient_angle % self.angle_unit
        if idx == self.nbins:
            return idx - 1, (idx) % self.nbins, mod
        return idx, (idx + 1) % self.nbins, mod

    def block_normalize(self, block_features, normalization):
        assert block_features.ndim == 1, 'block_features must be a 1-dimensional array'
        assert normalization in NormalizationMethod, 'Invalid normalization method'
        if normalization in [NormalizationMethod.L2, NormalizationMethod.L2_HYS]:
            if normalization == NormalizationMethod.L2_HYS:
                block_features = np.clip(block_features, 0.0, self.l2_hys_threshold)
            norm = np.linalg.norm(block_features)
            block_features = block_features / math.sqrt(norm ** 2 + 1e-12)
        elif normalization in [NormalizationMethod.L1, NormalizationMethod.L1_SQRT]:
            norm = np.linalg.norm(block_features, ord=1)
            block_features = block_features / (norm + 1e-6)
            if normalization == NormalizationMethod.L1_SQRT:
                block_features = np.sqrt(block_features)
        return block_features

    def extract_cells(self, image, normalize=True):
        height = image.shape[0]
        width = image.shape[1]
        assert height % self.cell_size[0] == 0, \
            "Image height must be a multiple of vertical cell size {}".format(self.cell_size[0])
        assert width % self.cell_size[1] == 0, \
            "Image width must be a multiple of horizontal cell size {}".format(self.cell_size[1])
        gradient_magnitude, gradient_angle = self.compute_gradient(image)
        cell_histogram_vector = np.zeros((height // self.cell_size[0], width // self.cell_size[1], self.nbins))
        for i in range(cell_histogram_vector.shape[0]):
            for j in range(cell_histogram_vector.shape[1]):
                cell_magnitude = gradient_magnitude[i * self.cell_size[0]:(i + 1) * self.cell_size[0],
                                 j * self.cell_size[0]:(j + 1) * self.cell_size[0]]
                cell_angle = gradient_angle[i * self.cell_size[1]:(i + 1) * self.cell_size[1],
                             j * self.cell_size[1]:(j + 1) * self.cell_size[1]]
                cell_histogram_vector[i][j] = self.cell_histogram(cell_magnitude, cell_angle, normalize)

        cell_histogram_vector = np.where(np.isnan(cell_histogram_vector), 0.0, cell_histogram_vector)
        return cell_histogram_vector

    def extract_blocks(self, image, normalization=NormalizationMethod.L2_HYS):
        height = image.shape[0]
        width = image.shape[1]
        assert (height - self.block_size[0]) % self.block_stride[0] == 0, \
            "Invalid image height, (height - vertical_block_size) must be a multiple of vertical block stride"
        assert (width - self.block_size[1]) % self.block_stride[1] == 0, \
            "Invalid image width, (width - horizontal_block_size) must be a multiple of horizontal block stride"
        cell_histogram_vector = self.extract_cells(image, normalize=False)
        block_features = []
        for i in range((height - self.block_size[0]) // self.block_stride[0] + 1):
            row_features = []
            for j in range((width - self.block_size[1]) // self.block_stride[1] + 1):
                blk_feat = []
                for y in range(self.cells_per_block[0]):
                    for x in range(self.cells_per_block[1]):
                        blk_feat.append(cell_histogram_vector[
                            i * self.cells_per_stride[0] + y,
                            j * self.cells_per_stride[1] + x,
                            :
                        ])
                blk_feat = self.block_normalize(np.concatenate(blk_feat, axis=0), normalization)
                row_features.append(blk_feat)
            block_features.append(np.stack(row_features, axis=0))
        return np.stack(block_features, axis=0)

    def compute(self, img, normalization=NormalizationMethod.L2_HYS):
        block_features = self.extract_blocks(img, normalization)
        block_features = block_features.reshape(-1)
        return block_features


if __name__ == '__main__':
    from pathlib import Path
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    import cv2
    from PIL import Image
    import numpy as np
    from format_convert import torch2cv, cv2torch

    img_path = Path(r"D:\pythondata\datasets\slag\stage2_output_512x512\2023-06-09T08-56-26.988_S2_R1_019.jpg")
    img_torch = TF.to_tensor(Image.open(img_path).convert('L'))
    img = torch2cv(img_torch)

    # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # img = np.stack([img], axis=2)
    # img = img[127:383, 127:383, :]

    # HOGGen = cv2.HOGDescriptor((256, 256), (8, 8), (4, 4), (4, 4), 8)
    # hog = HOGGen.compute(img)
    # hog_ = rearrange_HOG(hog, (256, 256), (8, 8), (4, 4), (4, 4), fill=True, renormalize=True)
    # # hog_img1 = visualize_HOG(hog.reshape(64, 64, 8))
    # # hog_render_img1 = render_HOG(hog.reshape(64, 64, 8), img)
    # # cv2.imwrite(r"C:\Users\leike\.cache\img1.png", hog_render_img1)
    # hog_img1_1 = visualize_HOG(hog_[:, :, 0, :])
    # hog_render_img1_1 = render_HOG(hog_[:, :, 0, :], img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img1_1.png", hog_render_img1_1)
    # hog_render_img1_2 = render_HOG(hog_[:, :, 1, :], img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img1_2.png", hog_render_img1_2)
    # hog_render_img1_3 = render_HOG(hog_[:, :, 2, :], img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img1_3.png", hog_render_img1_3)
    # hog_render_img1_4 = render_HOG(hog_[:, :, 3, :], img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img1_4.png", hog_render_img1_4)

    CHOGGen = HOGExtractor(8, 4, 4, 8, sobel_ksize=5, l2_hys_threshold=0.4)
    # cell_hog = CHOGGen.extract_cells(img)
    # hog_img2 = visualize_HOG(cell_hog)
    # hog_render_img2 = render_HOG(cell_hog, img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img2.png", hog_render_img2)
    # # block_hog = CHOGGen.extract_blocks(img)
    # hogdesc = CHOGGen.compute(img, normalization=NormalizationMethod.L2_HYS)
    # hog2_ = rearrange_HOG(hogdesc, (256, 256), (8, 8), (4, 4), (4, 4), fill=False, renormalize=True)
    # hog_img2_1 = visualize_HOG(hog2_[:, :, 0, :])
    # hog_render_img2_1 = render_HOG(hog2_[:, :, 0, :], img, color=(0, 223, 192), background_alpha=0.5, thickness=(2, 8))
    # cv2.imwrite(r"C:\Users\leike\.cache\img2_1.png", hog_render_img2_1)

    hog3_ = rearrange_HOG(CHOGGen.compute(img), (512, 512), (8, 8), (4, 4), (4, 4), fill=True, squeeze=True)
    # hog_render_img2_2 = render_HOG(hog2_[:, :, 1, :], img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img2_2.png", hog_render_img2_2)
    # hog_render_img2_3 = render_HOG(hog2_[:, :, 2, :], img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img2_3.png", hog_render_img2_3)
    # hog_render_img2_4 = render_HOG(hog2_[:, :, 3, :], img)
    # cv2.imwrite(r"C:\Users\leike\.cache\img2_4.png", hog_render_img2_4)


    ii = 0