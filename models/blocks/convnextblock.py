import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .auxiliaries import GRN


class ConvNeXtBlock(nn.Module):
    """ This module is derived from ''

        All copyrights reserved by the original author and Joshua Reed.

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Kernel size used for depthwise convolution, padding is calculated automatically to preserve the data shape. Default: 7
        mlp_ratio (float): Expansion ratio of hidden feature dimension in MLP layer. Default: 4.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, kernel_size=7, mlp_ratio=4., drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ConvNeXtBlock_WithConcate(nn.Module):
    def __init__(self, in_dim, cat_dim, out_dim, kernel_size=7, mlp_ratio=4., layer_scale_init_value=1e-6):
        super().__init__()
        cated_dim = in_dim + cat_dim
        self.dwconv = nn.Conv2d(cated_dim, cated_dim, kernel_size=kernel_size,
                                padding=(kernel_size - 1) // 2, groups=cated_dim)  # depthwise conv
        self.norm = nn.LayerNorm(cated_dim, eps=1e-6)
        self.pwconv1 = nn.Linear(cated_dim, int(mlp_ratio * out_dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * out_dim), out_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, x_cat):
        x = torch.cat((x, x_cat), dim=1)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, 2C, H, W) -> (N, H, W, 2C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)  # (N, H, W, 2C) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x


class ConvNeXtV2Block(nn.Module):
    """ This module is derived from ''

        All copyrights reserved by the original author and Joshua Reed.

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Kernel size used for depthwise convolution, padding is calculated automatically to preserve the data shape. Default: 7
        mlp_ratio (float): Expansion ratio of hidden feature dimension in MLP layer. Default: 4.0
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, kernel_size=7, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(int(mlp_ratio * dim))
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x