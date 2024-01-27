import os
import warnings
from pathlib import Path
from collections.abc import Sequence
from PIL import Image as PILImage
import torch
import torchvision.transforms.functional as TF
from lightning.pytorch.callbacks import Callback
from lib.pytorch_framework.utils import CustomCfgNode as CN


# TODO: Write callback version of visualizers