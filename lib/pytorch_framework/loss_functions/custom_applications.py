import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        if reduction in ('mean', 'sum', 'none'):
            self.reduction = reduction
        else:
            raise ValueError(f'Invalid reduction method {reduction}')

    def forward(self, input, target):
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
        input = input.view(input.size(0), input.size(1), -1)
        target = target.view(target.size(0), target.size(1), -1)
        # N, C, m -> Nxm, C
        input = input.permute(0, 2, 1).contiguous()
        input = input.view(-1, input.size(-1))
        target = target.permute(0, 2, 1).contiguous()
        target = target.view(-1, target.size(-1))

        if self.weight is None:
            weight = torch.ones(num_class)
        else:
            weight = self.weight

        input = F.softmax(input, dim=1).log() * weight.unsqueeze(0)
        loss = -1 * torch.sum(input * target, dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.view(in_shape[0], *in_shape[2:])