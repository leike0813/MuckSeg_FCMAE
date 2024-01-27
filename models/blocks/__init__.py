from .auxiliaries import LayerNorm, GRN
from .auxiliaries_sparse import MinkowskiLayerNorm, MinkowskiDropPath, MinkowskiGRN
from .stem import StemBlock
from .convnextblock_sparse import ConvNeXtBlock_Sparse, ConvNeXtV2Block_Sparse
from .convnextblock import ConvNeXtBlock, ConvNeXtBlock_WithConcate, ConvNeXtV2Block


__all__ = [
    'LayerNorm',
    'GRN',
    'MinkowskiLayerNorm',
    'MinkowskiDropPath',
    'MinkowskiGRN',
    'StemBlock',
    'ConvNeXtBlock_Sparse',
    'ConvNeXtV2Block_Sparse',
    'ConvNeXtBlock',
    'ConvNeXtBlock_WithConcate',
    'ConvNeXtV2Block',
    'get_ConvNeXtBlock',
    'get_ConvNeXtBlock_Sparse',
]


def get_ConvNeXtBlock(use_convnext_v2):
    if use_convnext_v2:
        return ConvNeXtV2Block
    else:
        return ConvNeXtBlock


def get_ConvNeXtBlock_Sparse(use_convnext_v2):
    if use_convnext_v2:
        return ConvNeXtV2Block_Sparse
    else:
        return ConvNeXtBlock_Sparse