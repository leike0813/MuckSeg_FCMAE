from torch.optim.lr_scheduler import LRScheduler
import math


# TODO: Complete Damped Cosine LR Scheduler
class DampedCosineLR(LRScheduler):


    def __init__(self, optimizer, min_lr, step_period, damp_factor, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch, verbose)


    @staticmethod
    def damped_cosine_func(t, alpha, beta):
        return math.exp(-alpha * t) * math.cos(beta * t)