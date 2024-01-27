from torch.optim.lr_scheduler import CyclicLR
from .reduce_plateau_with_warmup import ReduceLROnPlateauWithWarmup
from .default_config import DEFAULT_CONFIG


def build_scheduler(optimizer, node=DEFAULT_CONFIG):
    hparams = {'lr-scheduler': node.NAME}
    if node.NAME == 'ReduceLROnPlateauWithWarmup':
        scheduler = ReduceLROnPlateauWithWarmup(
            optimizer,
            start_factor=node.WARMUP.START_FACTOR,
            end_factor=node.WARMUP.END_FACTOR,
            total_iters=node.WARMUP.WARMUP_EPOCHS,
            mode=node.MONITOR_MODE,
            verbose=node.VERBOSE,
            **node.REDUCEONPLATEAU
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': node.INTERVAL,
            'frequency': node.FREQUENCY,
            'monitor': node.MONITOR,
        }
        hparams.update({
            'lr-reduce-factor': node.REDUCEONPLATEAU.reduce_factor,
            'lr-reduce-patience': node.REDUCEONPLATEAU.patience,
            'lr-reduce-cooldown': node.REDUCEONPLATEAU.cooldown,
            'min-lr': node.REDUCEONPLATEAU.min_lr,
            'init-lr-factor': node.WARMUP.START_FACTOR,
            'lr-warmup-epochs': node.WARMUP.WARMUP_EPOCHS
        })
        node.REDUCEONPLATEAU._visible(True)
        node.WARMUP._visible(True)
    elif node.NAME == 'CyclicLR':
        scheduler = CyclicLR(
            optimizer,
            **node.CYCLICLR
        )
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': node.FREQUENCY,
        }
        hparams.update({
            'max_lr': node.CYCLICLR.max_lr,
            'lr-cycle-steps': [node.CYCLICLR.step_size_up, node.CYCLICLR.step_size_down],
            'lr-cycle-mode': node.CYCLICLR.mode,
            'cycle-momentum': node.CYCLICLR.cycle_momentum,
            'base_momentum': node.CYCLICLR.base_momentum,
            'max_momentum': node.CYCLICLR.max_momentum
        })
        if node.CYCLICLR.mode == 'exp_range':
            hparams['lr-scale-gamma'] = node.CYCLICLR.gamma
        node.CYCLICLR._visible(True)
    else:
        raise NotImplementedError

    return scheduler_config, hparams

