import torch
import torch.nn as nn
import torch.nn.functional as F
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiChannelwiseConvolution,
    MinkowskiLinear,
    SparseTensor,
)
from MinkowskiOps import to_sparse
from timm.models.layers import trunc_normal_
from models.blocks import StemBlock, MinkowskiLayerNorm, get_ConvNeXtBlock_Sparse
from models.blocks.functional import upsample_mask


class MuckSeg_Encoder_Sparse(nn.Module):
    """
    Args:
        in_chans (int): Number of input image channels. Default: 3
        kernel_sizes (list): Kernel sizes of each encoder stage. Default: [3, 3, 3, 3]
        depths (tuple(int)): Number of blocks in each stage. Default: [2, 2, 2, 2]
        dim (int): Initial feature dimension after Stem block. Default: 32
        mlp_ratio (int): Expansion ratio of hidden feature dimension in MLP layers in each ConvNeXt block.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        multi_scale_input (bool): Whether to use multi-scale input. Default: False
        use_convnext_v2 (bool): Whether to use ConvNeXt-V2 block instead of ConvNeXt block. Default: True

    Pipeline:
        <input> B, C, H, W -(stem)-> B, D, H, W -(ConvBlock0)-> B, D, H, W
        -(downsample0)-> B, 2*D, H/2, W/2 -(ConvBlock1)-> B, 2*D, H/2, W/2
        -(downsample1)-> B, 4*D, H/4, W/4 -(ConvBlock2)-> B, 4*D, H/4, W/4
        -(downsample2)-> B, 8*D, H/8, W/8 -(ConvBlock3)-> B, 8*D, H/8, W/8
        -(downsample3)-> B, 16*D, H/16, W/16
    """
    _NUM_STAGES = 4


    def __init__(self, in_chans=1, kernel_sizes=[7, 7, 7, 7], depths=[3, 9, 3, 3],
                 dim=32, stem_routes=['3CONV', '5CONV', '7CONV', '9CONV', 'D-3CONV', 'D-5CONV'],
                 mlp_ratio=4., drop_path_rate=0., D=2, multi_scale_input=False, use_convnext_v2=False):
        super().__init__()
        ConvNeXtBlock_Sparse = get_ConvNeXtBlock_Sparse(use_convnext_v2)
        assert dim % 4 == 0, 'dim must be a multiple of 4'
        self._check_inputs(kernel_sizes, depths)
        self.multi_scale_input = multi_scale_input

        self.stem = StemBlock(in_chans, dim, stem_routes)

        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(
                MinkowskiLayerNorm(dim * 2 ** i, eps=1e-6),
                MinkowskiConvolution(dim * 2 ** i, dim * 2 ** (i + 1), kernel_size=2, stride=2, bias=True, dimension=D),
            ) for i in range(self._NUM_STAGES)]
        ) # downsampling conv layers

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self._NUM_STAGES):
            stage = nn.Sequential(
                *[ConvNeXtBlock_Sparse(
                    dim=dim * 2 ** i, kernel_size=kernel_sizes[i], mlp_ratio=mlp_ratio, drop_path=dp_rates[cur + j], D=D
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    def _check_inputs(self, *args):
        parameter_lengths = set([len(arg) for arg in args])
        if len(parameter_lengths) > 1:
            raise ValueError('kernel_sizes and depths must be length 4 sequence.')
        if parameter_lengths.pop() != self._NUM_STAGES:
            raise ValueError('kernel_sizes and depths must be length 4 sequence.')

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (MinkowskiConvolution, MinkowskiChannelwiseConvolution)):
            trunc_normal_(m.kernel, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            if m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)

    def forward(self, x, mask):
        if self.multi_scale_input:
            x_add = F.interpolate(x, scale_factor=1 / 2)
            mask_add = upsample_mask(mask, 2 ** (self._NUM_STAGES - 1))
            mask_add = mask_add.unsqueeze(1).type_as(x_add)
            x_add_masked = to_sparse(x_add * (1. - mask_add))
        x = self.stem(x)
        mask_add = upsample_mask(mask, 2 ** self._NUM_STAGES)
        mask_add = mask_add.unsqueeze(1).type_as(x)
        x *= (1. - mask_add)
        x = to_sparse(x)

        for i in range(self._NUM_STAGES):
            x = self.stages[i](x)
            x = self.downsample_layers[i](x)
            if self.multi_scale_input and i <= 2:
                x += SparseTensor(
                    features=x_add_masked.F,
                    coordinate_manager=x.coordinate_manager,
                    coordinate_map_key=x.coordinate_map_key
                )
                if i < 2:
                    x_add = F.interpolate(x_add, scale_factor=1 / 2)
                    mask_add = upsample_mask(mask, 2 ** (self._NUM_STAGES - i - 2))
                    mask_add = mask_add.unsqueeze(1).type_as(x_add)
                    x_add_masked = to_sparse(x_add * (1. - mask_add))

        return x.dense()[0]

    def forward_featuremaps(self, x, mask):
        fmaps = {}
        if self.multi_scale_input:
            x_add = F.interpolate(x, scale_factor=1 / 2)
            mask_add = upsample_mask(mask, 2 ** (self._NUM_STAGES - 1))
            mask_add = mask_add.unsqueeze(1).type_as(x_add)
            x_add_masked = to_sparse(x_add * (1. - mask_add))
        x = self.stem(x)
        mask_add = upsample_mask(mask, 2 ** self._NUM_STAGES)
        mask_add = mask_add.unsqueeze(1).type_as(x)
        x *= (1. - mask_add)
        x = to_sparse(x)

        for i in range(self._NUM_STAGES):
            x = self.stages[i](x)
            fmaps['Encoder-Stage{si}'.format(si=i)] = x.dense()[0]
            x = self.downsample_layers[i](x)
            if self.multi_scale_input and i <= 2:
                x += SparseTensor(
                    features=x_add_masked.F,
                    coordinate_manager=x.coordinate_manager,
                    coordinate_map_key=x.coordinate_map_key
                )
                if i < 2:
                    x_add = F.interpolate(x_add, scale_factor=1 / 2)
                    mask_add = upsample_mask(mask, 2 ** (self._NUM_STAGES - i - 2))
                    mask_add = mask_add.unsqueeze(1).type_as(x_add)
                    x_add_masked = to_sparse(x_add * (1. - mask_add))

        return x.dense()[0], fmaps

# EOF