from lib.lightning_framework import DEFAULT_ENVIRONMENT
from lib.lightning_framework.trainer import DEFAULT_CONFIG as DEFAULT_CONFIG_TRAIN, EXAMPLE_MODEL_CONFIG
from lib.pytorch_framework.utils import CustomCfgNode as CN
from lib.pytorch_framework.utils import update_config
from lib.pytorch_framework.optimizers.default_config import DEFAULT_CONFIG as DEFAULT_CONFIG_OPTIMIZER
from lib.pytorch_framework.lr_schedulers import DEFAULT_CONFIG as DEFAULT_CONFIG_LR_SCHEDULER
from lib.lightning_framework.callbacks import DEFAULT_CONFIG_PREDICTION_WRITER
from utils import DEFAULT_CONFIG_VISUALIZATION


_C = CN()

_C.BASE = ['']

_C.ENVIRONMENT = DEFAULT_ENVIRONMENT.clone()
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = EXAMPLE_MODEL_CONFIG.clone()
_C.MODEL.ABBR.MuckSeg_FCMAE = 'MS_FCMAE'
# Model type
# could be overwritten by command line argument
_C.MODEL.TYPE = 'MuckSeg_FCMAE'
# Model name
_C.MODEL.SPEC_NAME = '512_D32'
_C.MODEL.FILE_PATHS = [
    'models/__init__.py',
    'models/build.py',
    'models/MuckSeg_Encoder_Sparse.py',
    'models/MuckSeg_FCMAE_Decoder.py',
    'models/MuckSeg_FCMAE.py',
]

_C.MODEL.IMAGE_SIZE = 512
_C.MODEL.IN_CHANS = 1
_C.MODEL.OUT_CHANS = 1
_C.MODEL.DIM = 32
_C.MODEL.MLP_RATIO = 4.
_C.MODEL.DROP_PATH_RATE = 0.
_C.MODEL.PATCH_SIZE_FACTOR = 2
_C.MODEL.TASK = 'binary'
_C.MODEL.USE_CONVNEXT_V2 = True

_C.MODEL.set_invisible_keys(['IN_CHANS', 'TASK'])

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.kernel_sizes = [7, 7, 7, 7]
_C.MODEL.ENCODER.depths = [3, 9, 3, 3]
_C.MODEL.ENCODER.stem_routes = ['3CONV', '5CONV', '7CONV', '9CONV', 'D-3CONV', 'D-5CONV']
_C.MODEL.ENCODER.multi_scale_input = False

_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.dim = 512
_C.MODEL.DECODER.neck_kernel_size = 7
_C.MODEL.DECODER.neck_depth = 8
_C.MODEL.DECODER.kernel_sizes = [7, 7]
_C.MODEL.DECODER.depths = [2, 2]

_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.norm_pix_loss = False

_C.MODEL.MASK_RATIO = 0.6
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = DEFAULT_CONFIG_TRAIN.clone()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.USE_BATCHSIZE_FINDER = False
_C.TRAIN.MONITOR = 'valid/epoch/loss'
_C.TRAIN.MONITOR_MODE = 'min'
# Trainer settings
_C.TRAIN.TRAINER = CN(new_allowed=True)
_C.TRAIN.TRAINER.accelerator = 'gpu'
_C.TRAIN.TRAINER.precision = 16
_C.TRAIN.TRAINER.min_steps = 5000
_C.TRAIN.TRAINER.max_steps = 500000
_C.TRAIN.TRAINER.overfit_batches = 0

_C.TRAIN.TRAINER.set_typecheck_exclude_keys(['precision', 'overfit_batches'])
_C.TRAIN.TRAINER.set_invisible_keys(['accelerator'])


# Callbacks
# Whether to use custom checkpointing callback
# could be overwritten by command line argument
_C.TRAIN.USE_CUSTOM_CHECKPOINTING = True
_C.TRAIN.CHECKPOINT_TOPK = 3
_C.TRAIN.CHECKPOINT_SAVELAST = True
# Whether to use earlystopping
# could be overwritten by command line argument
_C.TRAIN.USE_EARLYSTOPPING = False
_C.TRAIN.EARLYSTOPPING_PATIENCE = 10

# Experiment name to be logged
# could be overwritten by command line argument
_C.TRAIN.EXPERIMENT_NAME = 'MuckSeg_FCMAE'
# Experiment tag to be used as part of run name
# could be overwritten by command line argument
_C.TRAIN.TAG = ''

# Optimizer
_C.TRAIN.OPTIMIZER = DEFAULT_CONFIG_OPTIMIZER.clone()
_C.TRAIN.OPTIMIZER.BASE_LR = 1e-4

# LR scheduler
_C.TRAIN.LR_SCHEDULER = DEFAULT_CONFIG_LR_SCHEDULER.clone()
_C.TRAIN.LR_SCHEDULER.NAME = 'CyclicLR'
_C.TRAIN.LR_SCHEDULER.CYCLICLR.base_lr = _C.TRAIN.OPTIMIZER.BASE_LR
_C.TRAIN.LR_SCHEDULER.CYCLICLR.max_lr = 1e-5
_C.TRAIN.LR_SCHEDULER.CYCLICLR.mode = 'exp_range'
_C.TRAIN.LR_SCHEDULER.CYCLICLR.gamma = 0.9999
_C.TRAIN.LR_SCHEDULER.CYCLICLR.cycle_momentum = False
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Dataset name
_C.DATA.DATA_PATH = ''
# Image data crop anchor (where the top-left crop starts with) in (left, top) format (opencv fashion)
_C.DATA.CROP_ANCHOR = (256, 0)
# Number of crops in (row, column) direction
_C.DATA.CROP_GRIDSHAPE = (8, 3)
_C.DATA.IMAGE_MEAN = [0.618]
_C.DATA.IMAGE_STD = [0.229]
_C.DATA.DATAMODULE = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.DATAMODULE.batch_size = _C.TRAIN.BATCH_SIZE
_C.DATA.DATAMODULE.train_volume = 20000
_C.DATA.DATAMODULE.val_volume = 100
_C.DATA.DATAMODULE.test_volume = 1000
# Path to dataset, could be overwritten by command line argument
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.DATAMODULE.pin_memory = True
# Number of data loading threads
_C.DATA.DATAMODULE.num_workers = 0
# If explicitly given, the partition of dataset for each experiment instance will be fixed
_C.DATA.DATAMODULE.split_seed = 33

_C.DATA.DATAMODULE.set_invisible_keys(['pin_memory', 'split_seed'])
# -----------------------------------------------------------------------------
# Inference settings
# -----------------------------------------------------------------------------
_C.PREDICT = CN()
# Path to output results, could be overwritten by command line argument
_C.PREDICT.DATA_PATH = ''
_C.PREDICT.RESULT_PATH = ''
_C.PREDICT.CKPT_PATH = ''
# Original image size in W x H format (opencv fashion)
_C.PREDICT.IMAGE_SIZE = (2048, 4096)
# Original image ROI in (left, top, width, height) format (opencv fashion)
_C.PREDICT.IMAGE_ROI = (256, 0, 1536, 4096)
_C.PREDICT.THREEFOLD_MARGIN_RATE = 0.1
_C.PREDICT.IMAGE_MEAN = [0.618]
_C.PREDICT.IMAGE_STD = [0.229]

_C.PREDICT.set_invisible_keys(['CKPT_PATH', 'THREEFOLD_MARGIN_RATE'])

_C.PREDICT.DATAMODULE = CN(visible=False)
_C.PREDICT.DATAMODULE.batch_size = 1
_C.PREDICT.DATAMODULE.pin_memory = True
# Number of data loading threads
_C.PREDICT.DATAMODULE.num_workers = 0

_C.PREDICT.DATAMODULE.set_invisible_keys(['pin_memory'])

_C.PREDICT.WRITER = DEFAULT_CONFIG_PREDICTION_WRITER.clone()
_C.PREDICT.WRITER.concatenate = 2
_C.PREDICT.WRITER.log_prediction = True
_C.PREDICT.WRITER.log_folder = 'inference_example'
# -----------------------------------------------------------------------------
# Visualization settings
# -----------------------------------------------------------------------------
_C.VISUALIZATION = DEFAULT_CONFIG_VISUALIZATION.clone()
_C.VISUALIZATION.IMAGE_CHANNELS = _C.MODEL.IN_CHANS
_C.VISUALIZATION.IMAGE_MEAN = _C.DATA.IMAGE_MEAN
_C.VISUALIZATION.IMAGE_STD = _C.DATA.IMAGE_STD
_C.VISUALIZATION.FEATUREMAP_MASK_RATIO = 0.0
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder, overwritten by command line argument
_C.CONFIG_OUTPUT_PATH = 'configs'
_C.FULL_DUMP = False

_C.set_invisible_keys(['FULL_DUMP'])


def get_config(args, arg_mapper):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args, arg_mapper)

    return config
