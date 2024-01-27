import torch
import numpy as np


__all__ = [
    'patchify',
    'unpatchify',
    'gen_random_mask',
    'upsample_mask',
]


def patchify(imgs, patch_size):
    """
    imgs: (B, C, H, W)
    x: (B, h, w, C*patch_size**2)
    """
    p = patch_size
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h, w, p ** 2 * imgs.shape[1]))
    return x


def unpatchify(x, patch_size):
    """
    x: (B, h, w, C*patch_size**2)
    imgs: (B, C, H, W)
    """
    p = patch_size
    H = x.shape[1] * patch_size
    W = x.shape[2] * patch_size

    x = x.reshape(shape=(x.shape[0], x.shape[1], x.shape[2], p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, H, W))
    return imgs


def gen_random_mask(x, patch_size, mask_ratio):
    N = x.shape[0]
    h = x.shape[2] // patch_size
    w = x.shape[3] // patch_size
    L = h * w
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.randn(N, L, device=x.device)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask = mask.reshape((N, h, w))
    return mask


def upsample_mask(mask, scale):
    assert len(mask.shape) == 3
    if scale == 1:
        return mask
    return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)


if __name__ == '__main__':
    img = torch.rand((4, 1, 512, 512))
    mask = gen_random_mask(img, 16, 0.7)
    mask_add = upsample_mask(mask, 16)
    mask_add = mask_add.unsqueeze(1).type_as(img)
    img_masked = img * (1. - mask_add)
    img_patchified = patchify(img, 16)
    img_unpatchified = unpatchify(img, 16)

    ii = 0