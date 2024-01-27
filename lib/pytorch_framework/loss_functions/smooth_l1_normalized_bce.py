import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


__all__ = [
    'SmoothL1NormalizedBCELoss',
    'SmoothL1NormalizedBCEWithLogitsLoss',
]


class BaseSmoothL1NormalizedBCELoss(nn.Module):
    """
    Loss function used for regression problems within [0, 1] region.
    """
    def __init__(self, beta=0.05, label_smoothing=0.01, pos_weight=None, reduction='mean'):
        super().__init__()
        self.register_buffer('pos_weight', pos_weight)
        self.reduction = reduction
        assert beta >= 0, 'beta must be non-negative'
        self.beta = beta
        if label_smoothing <= 0 or label_smoothing >= 1.0:
            raise ValueError('smooth value should be in (0,1)')
        self.label_smoothing = label_smoothing
        self.bce_max = - (1. - label_smoothing) * math.log(label_smoothing) - label_smoothing * math.log(1. - label_smoothing)
        self.sl1_max = (1. - 0.5 * beta) if beta < 1 else 0.5 / beta


    def forward_bce_loss(self, input: Tensor, target: Tensor) -> Tensor:
        pass

    def forward(self, input, target):
        target = torch.clamp(
            target,
            self.label_smoothing,
            1.0 - self.label_smoothing
        )
        bce_loss = self.forward_bce_loss(input, target)
        sl1_loss = F.smooth_l1_loss(input, target, beta=self.beta, reduction='none')
        loss = (bce_loss / self.bce_max) + (sl1_loss / self.sl1_max)
        if self.reduction in ['sum', 'mean']:
            loss = loss.sum()
        if self.reduction == 'mean':
            loss  = loss.mean()
        return loss


class SmoothL1NormalizedBCELoss(BaseSmoothL1NormalizedBCELoss):
    def forward_bce_loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, reduction='none')


class SmoothL1NormalizedBCEWithLogitsLoss(BaseSmoothL1NormalizedBCELoss):
    def forward_bce_loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, pos_weight=self.pos_weight, reduction='none')

# EOF