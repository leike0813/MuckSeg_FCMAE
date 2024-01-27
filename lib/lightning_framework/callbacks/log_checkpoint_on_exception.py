from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger


class LogCheckpointOnException(Callback):
    def __init__(self, exception_types=[Exception]):
        super().__init__()
        self.exception_types = exception_types

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, tuple(self.exception_types)):
            if isinstance(trainer.logger, MLFlowLogger):
                if trainer.logger._checkpoint_callback:
                    trainer.logger._scan_and_log_checkpoints(trainer.logger._checkpoint_callback)