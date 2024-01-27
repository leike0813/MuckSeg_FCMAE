import numpy as np
from numpy import ndarray
from typing import List, Union, Optional


def _is_array_a_torch_image(x: ndarray) -> bool:
    return x.ndim >= 2


def _assert_image_array(img: ndarray) -> None:
    if not _is_array_a_torch_image(img):
        raise TypeError("Array is not a torch image")


def get_dimensions(img: ndarray) -> List[int]:
    _assert_image_array(img)
    channels = 1 if img.ndim == 2 else img.shape[-3]
    height, width = img.shape[-2:]
    return [channels, height, width]


def get_image_size(img: ndarray) -> List[int]:
    _assert_image_array(img)
    return [img.shape[-1], img.shape[-2]]


def get_image_num_channels(img: ndarray) -> int:
    _assert_image_array(img)
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]


def _pad_symmetric(img: ndarray, padding: List[int]) -> ndarray:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        neg_min_padding = [-min(x, 0) for x in padding]
        crop_left, crop_right, crop_top, crop_bottom = neg_min_padding
        img = img[..., crop_top : img.shape[-2] - crop_bottom, crop_left : img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.shape

    _x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
    right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]
    x_indices = np.array(left_indices + _x_indices + right_indices)

    _y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2] - 1, -1, -1)]
    bottom_indices = [-(i + 1) for i in range(padding[3])]
    y_indices = np.array(top_indices + _y_indices + bottom_indices)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]].copy()
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]].copy()
    else:
        raise RuntimeError("Symmetric padding of N-D arrays are not supported yet")


def _pad_reflect(img: ndarray, padding: List[int]) -> ndarray:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        neg_min_padding = [-min(x, 0) for x in padding]
        crop_left, crop_right, crop_top, crop_bottom = neg_min_padding
        img = img[..., crop_top : img.shape[-2] - crop_bottom, crop_left : img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.shape

    _x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [i for i in range(padding[0], 0, -1)]  # e.g. [4, 3, 2, 1]
    right_indices = [-(i + 2) for i in range(padding[1])]  # e.g. [-2, -3, -4]
    x_indices = np.array(left_indices + _x_indices + right_indices)

    _y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [i for i in range(padding[2], 0, -1)]
    bottom_indices = [-(i + 2) for i in range(padding[3])]
    y_indices = np.array(top_indices + _y_indices + bottom_indices)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]].copy()
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]].copy()
    else:
        raise RuntimeError("Symmetric padding of N-D arrays are not supported yet")


def _pad_edge(img: ndarray, padding: List[int]) -> ndarray:
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        neg_min_padding = [-min(x, 0) for x in padding]
        crop_left, crop_right, crop_top, crop_bottom = neg_min_padding
        img = img[..., crop_top : img.shape[-2] - crop_bottom, crop_left : img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.shape

    _x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
    left_indices = [0 for i in range(padding[0])]  # e.g. [0, 0, 0, 0]
    right_indices = [-1 for i in range(padding[1])]  # e.g. [-1, -1, -1]
    x_indices = np.array(left_indices + _x_indices + right_indices)

    _y_indices = [i for i in range(in_sizes[-2])]
    top_indices = [0 for i in range(padding[2])]
    bottom_indices = [-1 for i in range(padding[3])]
    y_indices = np.array(top_indices + _y_indices + bottom_indices)

    ndim = img.ndim
    if ndim == 3:
        return img[:, y_indices[:, None], x_indices[None, :]].copy()
    elif ndim == 4:
        return img[:, :, y_indices[:, None], x_indices[None, :]].copy()
    else:
        raise RuntimeError("Symmetric padding of N-D arrays are not supported yet")


def _parse_pad_padding(padding: Union[int, List[int]]) -> List[int]:
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif len(padding) == 1:
        pad_left = pad_right = pad_top = pad_bottom = padding[0]
    elif len(padding) == 2:
        pad_left = pad_right = padding[0]
        pad_top = pad_bottom = padding[1]
    else:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]

    return [pad_left, pad_right, pad_top, pad_bottom]


def pad(
    img: ndarray, padding: Union[int, List[int]], fill: Optional[Union[int, float]] = 0, padding_mode: str = "constant"
) -> ndarray:
    _assert_image_array(img)

    if fill is None:
        fill = 0

    if not isinstance(padding, (int, tuple, list)):
        raise TypeError("Got inappropriate padding arg")
    if not isinstance(fill, (int, float)):
        raise TypeError("Got inappropriate fill arg")
    if not isinstance(padding_mode, str):
        raise TypeError("Got inappropriate padding_mode arg")

    if isinstance(padding, tuple):
        padding = list(padding)

    if isinstance(padding, list):
        if len(padding) not in [1, 2, 4]:
            raise ValueError(
                f"Padding must be an int or a 1, 2, or 4 element tuple, not a {len(padding)} element tuple"
            )

    if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
        raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

    padding = _parse_pad_padding(padding)
    # padding is left, right, top, bottom

    # crop if needed
    if padding[0] < 0 or padding[1] < 0 or padding[2] < 0 or padding[3] < 0:
        neg_min_padding = [-min(x, 0) for x in padding]
        crop_left, crop_right, crop_top, crop_bottom = neg_min_padding
        img = img[..., crop_top: img.shape[-2] - crop_bottom, crop_left: img.shape[-1] - crop_right]
        padding = [max(x, 0) for x in padding]

    in_sizes = img.shape
    ndim = img.ndim

    if padding_mode in ["edge", "reflect", "symmetric"]:
        _x_indices = [i for i in range(in_sizes[-1])]  # [0, 1, 2, 3, ...]
        _y_indices = [i for i in range(in_sizes[-2])]
        if padding_mode == "edge":
            left_indices = [0 for i in range(padding[0])]  # e.g. [0, 0, 0, 0]
            right_indices = [-1 for i in range(padding[1])]  # e.g. [-1, -1, -1]

            top_indices = [0 for i in range(padding[2])]
            bottom_indices = [-1 for i in range(padding[3])]
        elif padding_mode == "reflect":
            left_indices = [i for i in range(padding[0], 0, -1)]  # e.g. [4, 3, 2, 1]
            right_indices = [-(i + 2) for i in range(padding[1])]  # e.g. [-2, -3, -4]

            top_indices = [i for i in range(padding[2], 0, -1)]
            bottom_indices = [-(i + 2) for i in range(padding[3])]
        elif padding_mode == "symmetric":
            left_indices = [i for i in range(padding[0] - 1, -1, -1)]  # e.g. [3, 2, 1, 0]
            right_indices = [-(i + 1) for i in range(padding[1])]  # e.g. [-1, -2, -3]

            top_indices = [i for i in range(padding[2] - 1, -1, -1)]
            bottom_indices = [-(i + 1) for i in range(padding[3])]

        x_indices = np.array(left_indices + _x_indices + right_indices)
        y_indices = np.array(top_indices + _y_indices + bottom_indices)

        if ndim == 3:
            return img[:, y_indices[:, None], x_indices[None, :]].copy()
        elif ndim == 4:
            return img[:, :, y_indices[:, None], x_indices[None, :]].copy()
        else:
            raise RuntimeError("Symmetric padding of N-D arrays are not supported yet")
    else: # "constant"
        if ndim == 3:
            padded_img = np.full((
                in_sizes[0],
                padding[2] + in_sizes[1] + padding[3],
                padding[0] + in_sizes[2] + padding[1]
            ), fill, dtype=img.dtype)
            padded_img[:, padding[2]: padding[2] + in_sizes[1], padding[0]: padding[0] + in_sizes[2]] = img
        if ndim == 4:
            padded_img = np.full((
                in_sizes[0],
                in_sizes[1],
                padding[2] + in_sizes[2] + padding[3],
                padding[0] + in_sizes[3] + padding[1]
            ), fill, dtype=img.dtype)
            padded_img[:, :, padding[2]: padding[2] + in_sizes[2], padding[0]: padding[0] + in_sizes[3]] = img

    return padded_img


def crop(img: ndarray, top: int, left: int, height: int, width: int) -> ndarray:
    _assert_image_array(img)

    _, h, w = get_dimensions(img)
    right = left + width
    bottom = top + height

    if left < 0 or top < 0 or right > w or bottom > h:
        padding_ltrb = [
            max(-left + min(0, right), 0),
            max(-top + min(0, bottom), 0),
            max(right - max(w, left), 0),
            max(bottom - max(h, top), 0),
        ]
        return pad(img[..., max(top, 0) : bottom, max(left, 0) : right], padding_ltrb, fill=0)
    return img[..., top:bottom, left:right].copy()


def normalize(array: ndarray, mean: List[float], std: List[float], inplace: bool = False) -> ndarray:
    _assert_image_array(array)

    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"Input array should be a float tensor. Got {array.dtype}.")

    if array.ndim < 3:
        raise ValueError(
            f"Expected array to be a tensor image of size (..., C, H, W). Got array.shape = {array.shape}"
        )

    if not inplace:
        array = array.copy()

    dtype = array.dtype
    mean = np.array(mean, dtype=dtype)
    std = np.array(std, dtype=dtype)
    if np.any(std == 0):
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.reshape(-1, 1, 1)
    if std.ndim == 1:
        std = std.reshape(-1, 1, 1)
    array[..., :, :] = (array[..., :, :] - mean) / std
    return array