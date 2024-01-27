import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from MinkowskiEngine import MinkowskiConvolution
from MinkowskiOps import to_sparse
from timm.models.layers import trunc_normal_
from models.blocks.functional import upsample_mask
from models.blocks import get_ConvNeXtBlock, get_ConvNeXtBlock_Sparse


class MuckSeg_FCMAE_Decoder(L.LightningModule):
    """
    Args:
        kernel_sizes (list): Kernel sizes of each decoder stage. Default: [7, 7]
        depths (tuple(int)): Number of blocks in each stage. Default: [2, 2]
        dim (int): Input feature dimension. Default: 512
        mlp_ratio (int): Expansion ratio of hidden feature dimension in MLP layers in each ConvNeXt block.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        neck_kernel_size (int): Kernel sizes of neck. Default: 7.
        neck_depth (int): Number of blocks in neck. Default: 8.
        use_convnext_v2 (bool): Whether to use ConvNeXt-V2 block instead of ConvNeXt block. Default: True

    Pipeline:
        <input> B, D, H/16, W/16 -(neck)-> B, D, H/16, W/16
             -(upsample0)-> B, D, H/8, W/8 -(D-ConvBlock0)-> B, D/2, H/8, W/8
             -(upsample1)-> B, D/2, H/4, W/4 -(D-ConvBlock1)-> B, D/4, H/4, W/4
             -(upsample2)-> B, D/4, H/2, W/2 -(D-ConvBlock2)-> B, D/8, H/2, W/2
             -(upsample3)-> B, D/8, H, W -(D-ConvBlock3)-> B, D/16, H, W <output>
    """
    _NUM_STAGES = 2


    def __init__(self, kernel_sizes=[7, 7], depths=[2, 2], dim=512, mlp_ratio=4., drop_path_rate=0.,
                 neck_kernel_size=7, neck_depth=8, use_convnext_v2=True):
        super().__init__()
        ConvNeXtBlock = get_ConvNeXtBlock(use_convnext_v2)
        ConvNeXtBlock_Sparse = get_ConvNeXtBlock_Sparse(use_convnext_v2)
        assert dim % 4 == 0, 'dim must be a multiple of 4'
        self._check_inputs(kernel_sizes, depths)

        self.extra_neck = nn.Sequential(
            *[ConvNeXtBlock(
                dim=dim, kernel_size=neck_kernel_size, mlp_ratio=mlp_ratio, drop_path=0.
            ) for i in range(neck_depth)]
        )

        self.decode_stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self._NUM_STAGES):
            stage_modules = [MinkowskiConvolution(in_channels=dim // (2 ** i), out_channels=dim // (2 ** (i + 1)), kernel_size=1, dimension=2)]
            for j in range(depths[i]):
                stage_modules.append(ConvNeXtBlock_Sparse(
                    dim=dim // (2 ** (i + 1)), kernel_size=kernel_sizes[i], mlp_ratio=mlp_ratio, drop_path=dp_rates[cur + j]
                ))
            self.decode_stages.append(nn.Sequential(*stage_modules))
            cur += depths[i]

        self.mask_token = nn.Parameter(torch.zeros(1, dim, 1, 1))

        self.apply(self._init_weights)

    def _check_inputs(self, *args):
        parameter_lengths = set([len(arg) for arg in args])
        if len(parameter_lengths) > 1:
            raise ValueError('kernel_sizes and depths must be sequence or equal length.')
        return parameter_lengths.pop()

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, mask):
        if bool(torch.all(mask == 0)):
            x = self.extra_neck(x)
            for i in range(self._NUM_STAGES):
                x = F.interpolate(x, scale_factor=2, mode='bilinear')
                x = to_sparse(x)
                x = self.decode_stages[i](x)
                x = x.dense()[0]
        else:
            _mask = mask.unsqueeze(1).type_as(x)
            mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
            x = x * (1. - _mask) + mask_token * _mask
            x = self.extra_neck(x)
            for i in range(self._NUM_STAGES):
                x = F.interpolate(x, scale_factor=2, mode='bilinear')
                _mask = upsample_mask(mask, 2 ** (i + 1))
                _mask = _mask.unsqueeze(1).type_as(x)
                x_masked = x * _mask
                x_masked = to_sparse(x_masked)
                x_masked = self.decode_stages[i](x_masked)
                x_masked = x_masked.dense()[0]
                x_unmasked = x * (1. - _mask)
                x_unmasked = to_sparse(x_unmasked)
                x_unmasked = self.decode_stages[i](x_unmasked)
                x_unmasked = x_unmasked.dense()[0]
                x = x_masked + x_unmasked

        return x

    def forward_featuremaps(self, x, mask):
        fmaps = {}
        if bool(torch.all(mask == 0)):
            x = self.extra_neck(x)
            fmaps['Extra_Neck'] = x
            for i in range(self._NUM_STAGES):
                x = F.interpolate(x, scale_factor=2, mode='bilinear')
                x = to_sparse(x)
                x = self.decode_stages[i](x)
                x = x.dense()[0]
                fmaps['Decoder-Stage{si}'.format(si=i)] = x
        else:
            _mask = mask.unsqueeze(1).type_as(x)
            mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
            x = x * (1. - _mask) + mask_token * _mask
            x = self.extra_neck(x)
            fmaps['Extra_Neck'] = x
            for i in range(self._NUM_STAGES):
                x = F.interpolate(x, scale_factor=2, mode='bilinear')
                _mask = upsample_mask(mask, 2 ** (i + 1))
                _mask = _mask.unsqueeze(1).type_as(x)
                x_masked = x * _mask
                x_masked = to_sparse(x_masked)
                x_masked = self.decode_stages[i](x_masked)
                x_masked = x_masked.dense()[0]
                x_unmasked = x * (1. - _mask)
                x_unmasked = to_sparse(x_unmasked)
                x_unmasked = self.decode_stages[i](x_unmasked)
                x_unmasked = x_unmasked.dense()[0]
                x = x_masked + x_unmasked
                fmaps['Decoder-Stage{si}'.format(si=i)] = x

        return fmaps

# EOF