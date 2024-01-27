import warnings
from torch import optim as optim
from .default_config import DEFAULT_CONFIG
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
except Exception:
    DeepSpeedCPUAdam = optim.AdamW
    print('Failed to import DeepSpeedCPUAdam, please check your DeepSpeed installation. The default AdamW optimizer will be used instead')


def build_optimizer(model, node=DEFAULT_CONFIG):
    """
    Build optimizer.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = node.NAME.lower()
    hparams = {
        'optimizer': node.NAME,
        'base-lr': node.BASE_LR,
    }
    if opt_lower == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=node.ADAMW.eps,
            betas=node.ADAMW.betas,
            lr=node.BASE_LR,
            weight_decay=node.WEIGHT_DECAY,
        )
        hparams.update({
                'adamw-betas': node.ADAMW.betas,
                'weight-decay': node.WEIGHT_DECAY
            })
        node.ADAMW._visible(True)
    elif opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters,
            lr=node.BASE_LR,
            momentum=node.SGD.momentum,
            dampening=node.SGD.dampening,
            weight_decay=node.WEIGHT_DECAY,
        )
        hparams.update({
            'sgd-momentum': node.SGD.momentum,
            'sgd-dampening': node.SGD.dampening,
            'weight-decay': node.WEIGHT_DECAY
        })
        node.SGD._visible(True)
    elif opt_lower == 'cpu_adamw':
        warnings.warn('CPU_AdamW must be used in conjunction with DeepSpeed training strategy, make sure to have your trainer set properly', UserWarning)
        optimizer = DeepSpeedCPUAdam(
            parameters,
            eps=node.ADAMW.eps,
            betas=node.ADAMW.betas,
            lr=node.BASE_LR,
            weight_decay=node.WEIGHT_DECAY,
        )
        hparams.update({
            'adamw-betas': node.ADAMW.betas,
            'weight-decay': node.WEIGHT_DECAY
        })
        node.ADAMW._visible(True)
    else:
        return (None, {})

    return optimizer, hparams


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
