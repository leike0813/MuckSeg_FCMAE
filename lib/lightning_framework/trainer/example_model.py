from lib.pytorch_framework.utils import CustomCfgNode as CN


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
EXAMPLE_MODEL_CONFIG = CN()
EXAMPLE_MODEL_CONFIG.TYPE = 'Model'
EXAMPLE_MODEL_CONFIG.SPEC_NAME = 'Base'
EXAMPLE_MODEL_CONFIG.ABBR = CN(visible=False)
EXAMPLE_MODEL_CONFIG.ABBR.Model = 'MDL'