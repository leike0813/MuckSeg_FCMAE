from lib.pytorch_framework.utils import CustomCfgNode as CN


DEFAULT_CONFIG = CN(visible=False)

DEFAULT_CONFIG.CATEGORY = 'default'
DEFAULT_CONFIG.APPEND_CATEGORY_NAME = False
DEFAULT_CONFIG.TRAINING = CN(visible=False)

DEFAULT_CONFIG.TRAINING.Accuracy = CN(visible=False)
DEFAULT_CONFIG.TRAINING.Accuracy.task = 'binary'
DEFAULT_CONFIG.TRAINING.Accuracy.num_classes = 1
DEFAULT_CONFIG.TRAINING.Accuracy.LEVEL = 3

DEFAULT_CONFIG.VALIDATION = CN(visible=False)

DEFAULT_CONFIG.VALIDATION.Accuracy = CN(visible=False)
DEFAULT_CONFIG.VALIDATION.Accuracy.task = 'binary'
DEFAULT_CONFIG.VALIDATION.Accuracy.num_classes = 1
DEFAULT_CONFIG.VALIDATION.Accuracy.LEVEL = 2

DEFAULT_CONFIG.TEST = CN(visible=False)

DEFAULT_CONFIG.TEST.Accuracy = CN(visible=False)
DEFAULT_CONFIG.TEST.Accuracy.task = 'binary'
DEFAULT_CONFIG.TEST.Accuracy.num_classes = 1
DEFAULT_CONFIG.TEST.Accuracy.LEVEL = 2
DEFAULT_CONFIG.TEST.Recall = CN(visible=False)
DEFAULT_CONFIG.TEST.Recall.task = 'binary'
DEFAULT_CONFIG.TEST.Recall.num_classes = 1
DEFAULT_CONFIG.TEST.Recall.LEVEL = 2
DEFAULT_CONFIG.TEST.Specificity = CN(visible=False)
DEFAULT_CONFIG.TEST.Specificity.task = 'binary'
DEFAULT_CONFIG.TEST.Specificity.num_classes = 1
DEFAULT_CONFIG.TEST.Specificity.LEVEL = 2
DEFAULT_CONFIG.TEST.Precision = CN(visible=False)
DEFAULT_CONFIG.TEST.Precision.task = 'binary'
DEFAULT_CONFIG.TEST.Precision.num_classes = 1
DEFAULT_CONFIG.TEST.Precision.LEVEL = 2