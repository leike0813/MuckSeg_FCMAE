from collections.abc import Sequence
from .metric_hook import MetricHook
from .metric_callback import MetricCallback
import lightning as L


def build_metric_callback(node):
    return MetricCallback(
        training_metric_dict=node.get('TRAINING', {}), validation_metric_dict=node.get('VALIDATION', {}),
        test_metric_dict=node.get('TEST', {}), category=node.get('CATEGORY', 'default'),
        append_category_name=node.get('APPEND_CATEGORY_NAME', False)
    )


def install_metric_hook(lightningModule):
    if not issubclass(lightningModule, L.LightningModule):
        raise TypeError('Can only be used to lightning.LightningModule')

    class _LightningModule(lightningModule, MetricHook):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    _LightningModule.__name__ = lightningModule.__name__

    return _LightningModule
