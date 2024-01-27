import torch
from lightning.pytorch.callbacks import Callback


class LogLossCallback(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        if stage == 'fit':
            self.training_step_outputs = []
            self.validation_step_outputs = []
        if stage == 'validate':
            if not hasattr(self, 'validation_step_outputs'):
                self.validation_step_outputs = []
        if stage == 'test':
            self.test_step_outputs = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = self.fetch_data_for_calc(outputs)
        self.log('train/batch/loss', loss)
        self.training_step_outputs.append(loss)

    def on_train_epoch_end(self, trainer, pl_module):
        loss = sum(self.training_step_outputs) / len(self.training_step_outputs)
        self.log('train/epoch/loss', loss)
        self.training_step_outputs.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = self.fetch_data_for_calc(outputs)
        # self.log('valid/batch/loss', loss)
        self.validation_step_outputs.append(loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = sum(self.validation_step_outputs) / len(self.validation_step_outputs)
        self.log('valid/epoch/loss', loss)
        self.validation_step_outputs.clear()

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = self.fetch_data_for_calc(outputs)
        # self.log('test/batch/loss', loss)
        self.test_step_outputs.append(loss)

    def on_test_epoch_end(self, trainer, pl_module):
        loss = sum(self.test_step_outputs) / len(self.test_step_outputs)
        self.log('test/epoch/loss', loss)
        self.test_step_outputs.clear()

    def fetch_data_for_calc(self, outputs):
        if isinstance(outputs, dict):
            return outputs['loss'].item()
        elif isinstance(outputs, torch.Tensor):
            return outputs.item()
        else:
            return None