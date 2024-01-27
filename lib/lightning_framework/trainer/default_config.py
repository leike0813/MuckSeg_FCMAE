from lib.pytorch_framework.utils import CustomCfgNode as CN


DEFAULT_CONFIG = CN()

DEFAULT_CONFIG.CKPT_PATH = ''
DEFAULT_CONFIG.TRAINER_ROOT_DIR = 'train'
DEFAULT_CONFIG.BATCH_SIZE = 4
DEFAULT_CONFIG.MONITOR = 'valid/epoch/loss'
DEFAULT_CONFIG.MONITOR_MODE = 'min'
# Trainer settings
DEFAULT_CONFIG.TRAINER = CN(new_allowed=True)
DEFAULT_CONFIG.TRAINER.accelerator = 'gpu'
DEFAULT_CONFIG.TRAINER.devices = 1
DEFAULT_CONFIG.TRAINER.strategy = 'single_device'
DEFAULT_CONFIG.TRAINER.precision = '16-mixed'
DEFAULT_CONFIG.TRAINER.min_epochs = 1
DEFAULT_CONFIG.TRAINER.max_epochs = 50
DEFAULT_CONFIG.TRAINER.overfit_batches = 0

DEFAULT_CONFIG.TRAINER.set_typecheck_exclude_keys(['precision', 'overfit_batches'])
DEFAULT_CONFIG.TRAINER.set_invisible_keys(['accelerator'])

# Callbacks
# Whether to use custom checkpointing callback
DEFAULT_CONFIG.USE_CUSTOM_CHECKPOINTING = False
DEFAULT_CONFIG.CHECKPOINT_TOPK = 1
DEFAULT_CONFIG.CHECKPOINT_SAVELAST = False
# Whether to log model checkpoints on exceptions
DEFAULT_CONFIG.LOG_CHECKPOINT_ON_EXCEPTION = ['KeyboardInterrupt']
# Whether to use earlystopping
DEFAULT_CONFIG.USE_EARLYSTOPPING = False
DEFAULT_CONFIG.EARLYSTOPPING_PATIENCE = 10

DEFAULT_CONFIG.LOG_LOSS = True
DEFAULT_CONFIG.LOG_LEARNINGRATE = True

# Experiment name to be logged
DEFAULT_CONFIG.EXPERIMENT_NAME = 'Default'
# Experiment tag to be used as part of run name
DEFAULT_CONFIG.TAG = 'N/A'
# -----------------------------------------------------------------------------
# Logger settings
# -----------------------------------------------------------------------------
DEFAULT_CONFIG.LOGGER = CN()
# Logger to be used
DEFAULT_CONFIG.LOGGER.NAME = 'MLFlowLogger'
DEFAULT_CONFIG.LOGGER.MLFLOW = CN()
DEFAULT_CONFIG.LOGGER.MLFLOW.tracking_uri = 'http://192.168.13.111:5000'
DEFAULT_CONFIG.LOGGER.MLFLOW.artifact_location = 'mlflow'

DEFAULT_CONFIG.LOGGER.set_invisible_keys(['NAME'])

# Experiment tag to be logged
DEFAULT_CONFIG.LOGGER.TAGS = CN(visible=False, new_allowed=True)
DEFAULT_CONFIG.LOGGER.TAGS.TASK = 'SlagSegmentation'
DEFAULT_CONFIG.LOGGER.TAGS.TYPE = 'test'