import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


__all__ = [
    'BinaryFocalLoss',
    'BinaryFocalWithLogitsLoss',
    'GeneralizedBinaryFocalLoss',
    'FocalLoss',
    'GeneralizedFocalLoss'
]


class BaseBinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(BaseBinaryFocalLoss, self).__init__()
        if alpha is None:
            _alpha = torch.ones(2)
        elif isinstance(alpha, float):
            _alpha = torch.tensor([alpha, 1 - alpha])
        self.register_buffer('_alpha', _alpha)
        self.alpha = alpha
        self.gamma = gamma
        if reduction in ('mean', 'sum', 'none'):
            self.reduction = reduction
        else:
            raise ValueError(f'Invalid reduction method {reduction}')

    def forward_bce_loss(self, input: Tensor, target: Tensor) -> Tensor:
        pass

    def reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        BCE_loss = self.forward_bce_loss(input, target)
        alpha = self._alpha.to(input.device)
        # target = target.long()
        # at = alpha.gather(0, target.view(-1)).reshape(target.shape)
        at = alpha[target.round().to(torch.long)]
        pt = torch.exp(-BCE_loss)
        loss = at * torch.pow(1. - pt, self.gamma) * BCE_loss
        return self.reduce_loss(loss)


class BinaryFocalWithLogitsLoss(BaseBinaryFocalLoss):
    def forward_bce_loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, reduction='none')


class BinaryFocalLoss(BaseBinaryFocalLoss):
    def forward_bce_loss(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, reduction='none')


class GeneralizedBinaryFocalLoss(BaseBinaryFocalLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        alpha = self._alpha.to(input.device)
        gamma_pt = torch.pow(torch.pow(target - input, 2), self.gamma / 2)
        foregrond = input.log().clamp(-100.0, 0.0) * alpha[0]
        backgroud = (1. - input).log().clamp(-100.0, 0.0) * alpha[1]
        loss = -1 * (gamma_pt * foregrond * target + gamma_pt * backgroud * (1. - target))
        return self.reduce_loss(loss)


class BaseFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=-100, balance_index=0, label_smoothing=0.0):
        super(BaseFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.balance_index = balance_index
        if label_smoothing < 0 or label_smoothing > 1.0:
            raise ValueError('smooth value should be in [0,1]')
        else:
            self.label_smoothing = label_smoothing
        if reduction in ('mean', 'sum', 'none'):
            self.reduction = reduction
        else:
            raise ValueError(f'Invalid reduction method {reduction}')

    def forward_tensors(self, input: Tensor, target: Tensor):
        in_shape = input.shape
        num_class = input.shape[1]
        if input.dim() - target.dim() == 1:
            # Assuming input is tensor with shape (N, C,...), target is tensor with shape (N, ...), make one-hot transformation
            """***For further reference***
            #Before one-hot, use original target to threat tensor alpha
            at = alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(input.shape).gather(
                1,
                target.unsqueeze(1).expand(input.shape).long()
            )
            """
            target = F.one_hot(target, num_classes=input.size(1)).permute(
                tuple([0, -1] + [i for i in range(1, target.dim())])).float()
        elif input.dim() == target.dim():
            # Assuming input and target are both tensors with same shape of (N, C, ...), which target is one-hot encoded
            target = target.float()
        else:
            raise ValueError(f'input.dim()!= target.dim() or input.dim() - target.dim() != 1')
        assert input.shape == target.shape
        # N, C, d1, d2,... -> N, C, m (m=d1*d2*...)
        input = F.softmax(input, dim=1)
        input = input.view(input.size(0), input.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        # N, C, m -> Nxm, C
        input = input.permute(0, 2, 1).contiguous()
        input = input.view(-1, input.size(-1))
        target = target.permute(0, 2, 1).contiguous()
        target = target.view(-1, target.size(-1))

        if self.label_smoothing > 0:
            target = torch.clamp(
                target,
                self.label_smoothing / (num_class - 1),
                1.0 - self.label_smoothing
            )

        if self.alpha is None:
            alpha = torch.ones(num_class)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == num_class
            alpha = torch.FloatTensor(self.alpha).view(num_class)
            alpha = alpha / alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(num_class)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError('Not support alpha type')
        alpha = alpha.to(input.device)

        return input, target, alpha, in_shape

    def reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(BaseFocalLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input, target, alpha, in_shape = self.forward_tensors(input, target)
        alpha = alpha[torch.argmax(target, dim=1)]
        pt = torch.sum((input * target), axis=1) + self.label_smoothing
        logpt = pt.log()
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
        return self.reduce_loss(loss)


class GeneralizedFocalLoss(BaseFocalLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input, target, alpha, in_shape = self.forward_tensors(input, target)
        gamma_pt = torch.pow(torch.pow(target - input, 2), self.gamma / 2)
        input = input.log().clamp(-100.0, 0.0) * alpha.unsqueeze(0)
        loss = -1 * torch.sum(gamma_pt * input * target, dim=1)
        return self.reduce_loss(loss)

# EOF