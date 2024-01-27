import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, task='binary', mode='soft', average='batch', channel_reduction='mean', threshold=0.5, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        if mode in ('soft', 'hard'):
            self.mode = mode
        else:
            raise ValueError(f'Invalid mode {mode}')
        if task in ('binary', 'multiclass'):
            self.task = task
        else:
            raise ValueError(f'Invalid task {task}')
        if average in ('batch', 'sample'):
            self.average = average
        else:
            raise ValueError(f'Invalid average method {average}')
        if channel_reduction in ('mean', 'sum'):
            self.channel_reduction = channel_reduction
        else:
            raise ValueError(f'Invalid channel_reduction method {channel_reduction}')
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.task == 'binary':
            # Assuming input and target are both tensors with same shape of (N, C, ...)
            pass
        elif self.task == 'multiclass':
            if input.dim() - target.dim() == 1:
                # Assuming input is tensor with shape (N, C,...), target is tensor with shape (N, ...), make one-hot transformation
                target = F.one_hot(target, num_classes=input.size(1)).permute(tuple([0, -1] + [i for i in range(1, target.dim())])).float()
            elif input.dim() == target.dim():
                # Assuming input and target are both tensors with same shape of (N, C, ...), which target is one-hot encoded
                target = target.float()
            else:
                raise ValueError(f'input.dim()!= target.dim() or input.dim() - target.dim() != 1')
        if self.mode == 'hard':
            input = torch.ceil(input - self.threshold)
        return 1 - self.dice_coeff(input, target)

    def dice_coeff(self, input: Tensor, target: Tensor):
        assert input.shape == target.shape
        # N, C, d1, d2,... -> N, C, m (m=d1*d2*...)
        input = input.view(input.size(0), input.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        dice = 0
        if self.average == 'batch':
            # N, C, m -> Nxm, C
            input = input.permute(0, 2, 1).contiguous()
            input = input.view(-1, input.size(-1))
            target = target.permute(0, 2, 1).contiguous()
            target = target.view(-1, target.size(-1))
            for channel in range(input.size(-1)):
                dice += self._dice_compute(input[:, channel], target[:, channel])
            return (dice / input.size(-1)) if self.channel_reduction == 'mean' else dice
        elif self.average =='sample':
            for sample in range(input.size(0)):
                dice_sample = 0
                for channel in range(input.size(1)):
                    dice_sample += self._dice_compute(input[sample, channel], target[sample, channel])
                dice += (dice_sample / input.size(1)) if self.channel_reduction == 'mean' else dice_sample
            return dice / input.size(0)

    def _dice_compute(self, input: Tensor, target: Tensor):
        numerator = 2 * torch.dot(input, target)
        if self.mode == 'soft':
            denominator = torch.sum(input ** 2) + torch.sum(target ** 2)
        else:  # mode == 'hard
            denominator = torch.sum(input) + torch.sum(target)
        return (numerator + self.epsilon) / (denominator + self.epsilon)


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, task='binary', average='batch', epsilon=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        if task in ('binary', 'multiclass'):
            self.task = task
        else:
            raise ValueError(f'Invalid task {task}')
        if average in ('batch', 'sample'):
            self.average = average
        else:
            raise ValueError(f'Invalid average method {average}')
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.task == 'binary':
            # Assuming input and target are both tensors with same shape of (N, C, ...)
            pass
        elif self.task == 'multiclass':
            if input.dim() - target.dim() == 1:
                # Assuming input is tensor with shape (N, C,...), target is tensor with shape (N, ...), make one-hot transformation
                target = F.one_hot(target, num_classes=input.size(1)).permute(tuple([0, -1] + [i for i in range(1, target.dim())])).float()
            elif input.dim() == target.dim():
                # Assuming input and target are both tensors with same shape of (N, C, ...), which target is one-hot encoded
                target = target.float()
            else:
                raise ValueError(f'input.dim()!= target.dim() or input.dim() - target.dim() != 1')

        return 1 - self.gen_dice_coeff(input, target)

    def gen_dice_coeff(self, input: Tensor, target: Tensor):
        assert input.shape == target.shape
        # N, C, d1, d2,... -> N, C, m (m=d1*d2*...)
        input = input.view(input.size(0), input.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        if self.task == 'binary':
            input = torch.cat([1 - input, input], dim=1)
            target = torch.cat([1 - target, target], dim=1)
        if self.average == 'batch':
            # N, C, m -> C, Nxm
            input = input.permute(1, 0, 2).contiguous()
            input = input.view(input.size(0), -1)
            target = target.permute(1, 0, 2).contiguous()
            target = target.view(target.size(0), -1)
            return self._dice_compute(input, target)
        elif self.average =='sample':
            dice = 0
            for sample in range(input.size(0)):
                dice += self._dice_compute(input[sample], target[sample])
            return dice / input.size(0)

    def _dice_compute(self, input: Tensor, target: Tensor):
        weight = 1 / ((torch.sum(target, dim=1) + self.epsilon) ** 2)
        numerator = 2 * torch.sum(weight * torch.sum(input * target, dim=1))
        denominator = torch.sum(weight * (torch.sum(input, dim=1) + torch.sum(target, dim=1)))
        return (numerator + self.epsilon) / (denominator + self.epsilon)

# EOF