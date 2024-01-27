import torch
import numpy as np
from collections.abc import Sequence
import cv2


def filter_lowerbound(tensor_or_array, lowerbound, include_lowerbound=True):
    _ = tensor_or_array - lowerbound
    mask = _ >= 0 if include_lowerbound else _ > 0
    if isinstance(tensor_or_array, torch.Tensor):
        return _.clip(0, torch.max(_)) + lowerbound * mask
    elif isinstance(tensor_or_array, np.ndarray):
        return np.clip(_, 0, np.max(_)) + lowerbound * mask
    else:
        raise ValueError('Input must be torch.Tensor or numpy.ndarray instance')


def id2hsv(_id, hue_interval=5, saturation_interval=5, start_id=1):
    num_hue_level = 180 // hue_interval
    num_saturation_level = 255 // saturation_interval
    hue = ((_id - start_id) % num_hue_level) * hue_interval if _id > 0 else 0
    saturation = 255 - ((_id - start_id) // num_hue_level) * saturation_interval if _id > 0 else 0
    value = 255 if _id > 0 else 0
    return hue, saturation, value

id2hsv_ufunc = np.frompyfunc(id2hsv, 1, 3)


def replace_background(image, color, threshold=10):
    if not isinstance(color, np.ndarray) and isinstance(color, Sequence):
        color = np.asarray(color, dtype=np.uint8)
    _, mask_array = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
    mask_array = mask_array == 0
    new_background = np.zeros_like(image)
    new_background[:, :] = new_background[:, :] + color
    image[mask_array] = new_background[mask_array]

    return image

def make_background_transparent(image, threshold=10):
    _, alpha = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    output = cv2.merge([b, g, r, alpha], 4)
    return output
