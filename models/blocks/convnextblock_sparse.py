import torch
import torch.nn as nn

from .auxiliaries_sparse import (
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)
from MinkowskiEngine import (
    MinkowskiChannelwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU
)


class ConvNeXtBlock_Sparse(nn.Module):
    """ This module is derived from ''

        All copyrights reserved by the original author and Joshua Reed.

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Kernel size used for depthwise convolution. Default: 7
        mlp_ratio (float): Expansion ratio of hidden feature dimension in MLP layer. Default: 4.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        D (int): Number of dimensions of input sparse tensor(w/o batch dimension and channel dimension). Default: 2
    """

    def __init__(self, dim, kernel_size=7, mlp_ratio=4., drop_path=0., layer_scale_init_value=1e-6, D=2):
        super().__init__()
        self.dwconv = MinkowskiChannelwiseConvolution(dim, kernel_size=kernel_size, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, int(mlp_ratio * dim))
        self.act = MinkowskiGELU()
        self.pwconv2 = MinkowskiLinear(int(mlp_ratio * dim), dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x.F
        x = input + self.drop_path(x)
        return x


class ConvNeXtV2Block_Sparse(nn.Module):
    """ This module is derived from ''

        All copyrights reserved by the original author and Joshua Reed.

    Args:
        dim (int): Number of input channels.
        kernel_size (int): Kernel size used for depthwise convolution. Default: 7
        mlp_ratio (float): Expansion ratio of hidden feature dimension in MLP layer. Default: 4.0
        drop_path (float): Stochastic depth rate. Default: 0.0
        D (int): Number of dimensions of input sparse tensor(w/o batch dimension and channel dimension). Default: 2
    """

    def __init__(self, dim, kernel_size=7, mlp_ratio=4., drop_path=0., D=2):
        super().__init__()
        self.dwconv = MinkowskiChannelwiseConvolution(dim, kernel_size=kernel_size, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, int(mlp_ratio * dim))
        self.act = MinkowskiGELU()
        self.grn = MinkowskiGRN(int(mlp_ratio * dim))
        self.pwconv2 = MinkowskiLinear(int(mlp_ratio * dim), dim)
        self.drop_path = MinkowskiDropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x


if __name__ == "__main__":
    from models.blocks.functional import *
    from MinkowskiOps import to_sparse


    block = ConvNeXtBlock_Sparse(3).to('cuda')
    example_tensor = torch.rand((5, 3, 224, 224)).to('cuda')
    mask = gen_random_mask(example_tensor, 16, 0.6)
    mask = upsample_mask(mask, 16)
    mask = mask.unsqueeze(1).type_as(example_tensor)
    example_tensor *= (1. - mask)
    example_tensor = to_sparse(example_tensor)
    out = block(example_tensor)
    out = out.dense()[0]

    ii = 0