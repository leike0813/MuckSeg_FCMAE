import torch
from functools import wraps
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms.functional as TF


def torch2cv(tensor, convert_value=True, swap_channels=True):
    if tensor.ndim == 3:
        if convert_value:
            array = tensor.mul(255).add_(0.5).clamp_(0, 255).permute((1, 2, 0)).to("cpu", torch.uint8).numpy()
        else:
            array = tensor.permute((1, 2, 0)).to("cpu").numpy()
        if swap_channels:
            if array.shape[2] == 3:
                array = np.stack([array[:, :, 2], array[:, :, 1], array[:, :, 0]], axis=2)
            elif array.shape[2] == 4:
                array = np.stack([array[:, :, 2], array[:, :, 1], array[:, :, 0], array[:, :, 3]], axis=2)
            elif array.shape[2] == 1:
                pass
            else:
                raise NotImplementedError('Image must be of 1-channel, 3-channels or 4-channels')
        return array
    elif tensor.ndim == 2:
        return tensor.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy() \
            if convert_value else tensor.to("cpu").numpy()
    else:
        raise NotImplementedError('Can only convert 2-dimensional or 3-dimensional torch.Tensor')


def cv2torch(array, convert_value=True, swap_channels=True):
    if array.ndim == 3:
        if array.shape[2] == 3:
            if swap_channels:
                array = np.stack([array[:, :, 2], array[:, :, 1], array[:, :, 0]], axis=2)
            return TF.to_tensor(PILImage.fromarray(array)) \
                if convert_value else TF.pil_to_tensor(PILImage.fromarray(array))
        elif array.shape[2] == 4:
            if swap_channels:
                array = np.stack([array[:, :, 2], array[:, :, 1], array[:, :, 0], array[:, :, 3]], axis=2)
            return TF.to_tensor(PILImage.fromarray(array)) \
                if convert_value else TF.pil_to_tensor(PILImage.fromarray(array))
        elif array.shape[2] == 1:
            return TF.to_tensor(PILImage.fromarray(array[:, :, 0])) \
                if convert_value else TF.pil_to_tensor(PILImage.fromarray(array[:, :, 0]))
        else:
            raise NotImplementedError('Image must be of 1-channel, 3-channels or 4-channels')
    elif array.ndim == 2:
        return TF.to_tensor(PILImage.fromarray(array)).squeeze(0) \
            if convert_value else TF.pil_to_tensor(PILImage.fromarray(array)).squeeze(0)


def batchize(base_ndim):
    def batch_wrapper(func):
        @wraps(func)
        def batchized_func(tensor, *args, **kwargs):
            if tensor.ndim == base_ndim + 1:
                _ = []
                for i in range(tensor.shape[0]):
                    _.append(func(tensor[i], *args, **kwargs).unsqueeze(0))
                return torch.cat(_, dim=0)
            elif tensor.ndim <= base_ndim:
                return func(tensor, *args, **kwargs)
            else:
                raise ValueError('Input tensor has too many dimensions')
        return batchized_func
    return batch_wrapper


@batchize(3)
def torch_RGB2GRAY(tensor):
    assert tensor.ndim == 3 and tensor.shape[0] == 3, 'Invalid input tensor'
    return (0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]).unsqueeze(0)


@batchize(3)
def torch_GRAY2RGB(tensor):
    assert (tensor.ndim == 3 and tensor.shape[0] == 1) or tensor.ndim == 2, 'Invalid input tensor'
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return torch.cat([tensor, tensor, tensor], dim=0)


@batchize(3)
def torch_add_alpha(tensor, alpha=1.):
    """
    Append alpha channel to the input tensor which representing an image
    Support grayscale and RGB images
    :param tensor: Can be 2-dimensional (H, W) or 3-dimensional (C, H, W)
    :param alpha: Pixel value of alpha channel
    :returns tensor: 3-dimensional (C, H, W)
    """
    assert tensor.ndim in [2, 3], 'Input tensor must have 2 or 3 dimensions'
    if tensor.ndim == 3: # (C, H, W)
        assert tensor.shape[0] in [1, 3], 'Input tensor must be of 1 or 3 channel(s)'
        alpha = torch.full((1, tensor.shape[1], tensor.shape[2]), alpha, dtype=tensor.dtype)
        return torch.cat([tensor, alpha], dim=0)
    elif tensor.ndim == 2: # (H, W)
        alpha = torch.full((1, tensor.shape[0], tensor.shape[1]), alpha, dtype=tensor.dtype)
        return torch.cat([tensor.unsqueeze(0), alpha], dim=0)


@batchize(3)
def torch_merge_alpha(tensor, ignore_alpha=False):
    """
    Merge alpha channel of the input tensor which representing an image
    Support grayscale+alpha and RGBA images
    :param tensor: 3-dimensional (C, H, W)
    :param ignore_alpha: If true, simply remove the alpha channel, else make an conversion
    :return: tensor: 3-dimensional (C, H, W)
    """
    assert tensor.ndim == 3, 'Input tensor must have 3 dimensions'
    assert tensor.shape[0] in [2, 4], 'Input tensor must be of 2 or 4 channels'
    new_tensor = tensor[:-1, :, :]
    if ignore_alpha:
        return new_tensor
    alpha = tensor[-1, :, :]
    new_tensor *= alpha
    return new_tensor


@batchize(3)
def torch_RGB2RGBA(tensor, alpha=1.):
    assert tensor.ndim == 3 and tensor.shape[0] == 3, 'Invalid input tensor'
    return torch_add_alpha(tensor, alpha)

@batchize(3)
def torch_RGBA2RGB(tensor, ignore_alpha=False):
    assert tensor.ndim == 3 and tensor.shape[0] == 4, 'Invalid input tensor'
    return torch_merge_alpha(tensor, ignore_alpha)


@batchize(3)
def torch_align_channels(tensor, mode):
    assert (tensor.ndim == 3 and tensor.shape[0] <= 4) or tensor.ndim == 2, 'Invalid input tensor'
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    if mode == 'L':
        if tensor.shape[0] == 1:
            return tensor
        elif tensor.shape[0]== 3:
            return torch_RGB2GRAY(tensor)
        elif tensor.shape[0] == 4:
            return torch_RGB2GRAY(torch_RGBA2RGB(tensor))
    elif mode == 'RGB':
        if tensor.shape[0] == 1:
            return torch_GRAY2RGB(tensor)
        elif tensor.shape[0] == 3:
            return tensor
        elif tensor.shape[0] == 4:
            return torch_RGBA2RGB(tensor)
    elif mode == 'RGBA':
        if tensor.shape[0] == 1:
            return torch_RGB2RGBA(torch_GRAY2RGB(tensor))
        elif tensor.shape[0] == 3:
            return torch_RGB2RGBA(tensor)
        elif tensor.shape[0] == 4:
            return tensor
    else:
        "Only support 'RGB', 'RGBA' and 'L' modes"