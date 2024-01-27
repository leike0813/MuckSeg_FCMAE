from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings


EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    def __init__(self, optimizer, mode='min', start_factor=1.0 / 3,
                 end_factor=1.0, total_iters=5, last_epoch=-1,
                 reduce_factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False):
        super(ReduceLROnPlateauWithWarmup, self).__init__(optimizer, mode, reduce_factor, patience,
                                                         threshold, threshold_mode, cooldown,
                                                         min_lr, eps, verbose)
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError('Starting multiplicative factor expected to be greater than 0 and less or equal to 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError('Ending multiplicative factor expected to be between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self._step_count = 0
        self.step_warmup()

    def step_warmup(self, epoch=None):
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1
            values = self.get_lr()
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_lr"):
                values = self._get_closed_form_lr()
            else:
                values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr
            if self.verbose:
                if epoch is None:
                    print('Adjusting learning rate'
                          ' of group {} to {:.4e}.'.format(i, lr))
                else:
                    epoch_str = ("%.2f" if isinstance(epoch, float) else
                                 "%.5d") % epoch
                    print('Epoch {}: adjusting learning rate'
                          ' of group {} to {:.4e}.'.format(epoch_str, i, lr))
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]

        if self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.total_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * (self.start_factor +
                (self.end_factor - self.start_factor) * min(self.total_iters, self.last_epoch) / self.total_iters)
                for base_lr in self.base_lrs]

    def step(self, metrics, epoch=None):
        if self.last_epoch < self.total_iters:
            self.step_warmup(epoch)
        else:
            super(ReduceLROnPlateauWithWarmup, self).step(metrics, epoch)