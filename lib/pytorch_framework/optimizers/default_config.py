from ..utils import CustomCfgNode as CN

DEFAULT_CONFIG = CN()
# Optimizer to be used
# could be overwritten by command line argument
DEFAULT_CONFIG.NAME = 'AdamW'
DEFAULT_CONFIG.BASE_LR = 5e-4
DEFAULT_CONFIG.WEIGHT_DECAY = 0.01
# AdamW optimizer parameters
DEFAULT_CONFIG.ADAMW = CN(visible=False)
DEFAULT_CONFIG.ADAMW.eps = 1e-8
DEFAULT_CONFIG.ADAMW.betas = (0.9, 0.999)

# SGD optimizer parameters
DEFAULT_CONFIG.SGD = CN(visible=False)
DEFAULT_CONFIG.SGD.momentum = 0
DEFAULT_CONFIG.SGD.dampening = 0