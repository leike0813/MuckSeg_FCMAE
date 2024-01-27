from enum import IntEnum, IntFlag
from torch import nn
import torchmetrics as metrics
from lightning.pytorch.callbacks import Callback


class EvaluationLevel(IntFlag):
    BATCH = 1
    EPOCH = 2


class MetricCallback(Callback):
    stage_conversion = {'fit': 'train', 'validate': 'valid', 'test': 'test'}

    def __init__(self, training_metric_dict=None, validation_metric_dict=None, test_metric_dict=None, category='defualt',
                 append_category_name=False):
        super().__init__()
        self.training_metric_dict = training_metric_dict
        self.validation_metric_dict = validation_metric_dict
        self.test_metric_dict = test_metric_dict
        self.category = category
        self.append_category_name = append_category_name
        self.__TRAIN_METRICS_MOVED = False
        self.__VALID_METRICS_MOVED = False
        self.__TEST_METRICS_MOVED = False

    @property
    def state_key(self):
        return self.category

    def state_dict(self):
        return {
            'training_metric_dict': self.training_metric_dict,
            'validation_metric_dict': self.validation_metric_dict,
            'test_metric_dict': self.test_metric_dict,
            'category': self.category,
            'append_category_name': self.append_category_name,
        }


    def setup(self, trainer, pl_module, stage):
        if stage == 'fit':
            self.training_batch_metrics = nn.ModuleDict()
            self.training_epoch_metrics = nn.ModuleDict()
            self.setup_matrics(self.training_batch_metrics, self.training_epoch_metrics,
                               self.training_metric_dict, self.stage_conversion[stage])

            self.validation_batch_metrics = nn.ModuleDict()
            self.validation_epoch_metrics = nn.ModuleDict()
            self.setup_matrics(self.validation_batch_metrics, self.validation_epoch_metrics,
                               self.validation_metric_dict, 'valid')
        if stage == 'validate':
            if not hasattr(self, 'validation_batch_metrics'):
                self.validation_batch_metrics = nn.ModuleDict()
                self.validation_epoch_metrics = nn.ModuleDict()
                self.setup_matrics(self.validation_batch_metrics, self.validation_epoch_metrics,
                                   self.validation_metric_dict, self.stage_conversion[stage])
        if stage == 'test':
            self.test_batch_metrics = nn.ModuleDict()
            self.test_epoch_metrics = nn.ModuleDict()
            self.setup_matrics(self.test_batch_metrics, self.test_epoch_metrics,
                               self.test_metric_dict, self.stage_conversion[stage])

    def on_train_start(self, trainer, pl_module):
        self._data_fetched = False
        if not self.__TRAIN_METRICS_MOVED:
            for metric_func in self.training_batch_metrics.values():
                metric_func.to(pl_module.device)
            for metric_func in self.training_epoch_metrics.values():
                metric_func.to(pl_module.device)

    def on_validation_start(self, trainer, pl_module):
        self._data_fetched = False
        if not self.__VALID_METRICS_MOVED:
            for metric_func in self.validation_batch_metrics.values():
                metric_func.to(pl_module.device)
            for metric_func in self.validation_epoch_metrics.values():
                metric_func.to(pl_module.device)

    def on_test_start(self, trainer, pl_module):
        self._data_fetched = False
        if not self.__TEST_METRICS_MOVED:
            for metric_func in self.test_batch_metrics.values():
                metric_func.to(pl_module.device)
            for metric_func in self.test_epoch_metrics.values():
                metric_func.to(pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.calc_batch_metrics(outputs, self.training_batch_metrics, self.training_epoch_metrics, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self.calc_epoch_metrics(self.training_epoch_metrics, pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.calc_batch_metrics(outputs, self.validation_batch_metrics, self.validation_epoch_metrics, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.calc_epoch_metrics(self.validation_epoch_metrics, pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.calc_batch_metrics(outputs, self.test_batch_metrics, self.test_epoch_metrics, pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        self.calc_epoch_metrics(self.test_epoch_metrics, pl_module)

    def setup_matrics(self, metrics_batch_container, metrics_epoch_container, metric_dict, stage):
        containers = {'batch': metrics_batch_container, 'epoch': metrics_epoch_container}
        for metric_type, params in metric_dict.items():
            level = EvaluationLevel(params.pop('LEVEL', 2))
            alias = params.pop('ALIAS', None)
            if not isinstance(level, (int, EvaluationLevel)) or level < 1 or level > 3:
                raise ValueError('level must be valid metrics.EvaluationLevel flag or integer')
            _levels = []
            if EvaluationLevel.BATCH in level:
                _levels.append('batch')
            if EvaluationLevel.EPOCH in level:
                _levels.append('epoch')

            metric_generator = getattr(metrics, metric_type)
            metric_func = metric_generator(**params)

            for lvl in _levels:
                reg_name = '{stg}/{lvl}/{nme}{cat}'.format(
                    stg=stage,
                    lvl=lvl,
                    nme=alias if alias else metric_type,
                    cat='_' + self.category if self.append_category_name else ''
                )
                containers[lvl][reg_name] = metric_func

    def fetch_data_for_calc(self, outputs):
        keys = ['metric_{cat}'.format(cat=self.category), 'metrics_{cat}'.format(cat=self.category), 'metric_src_{cat}'.format(cat=self.category), self.category]
        if self.category == 'default':
            keys.extend(['metric', 'metrics', 'metric_src'])
        for key in keys:
            data = outputs.get(key)
            if data is not None:
                return data
        return None

    def calc_batch_metrics(self, outputs, metrics_batch_container, metrics_epoch_container, pl_module):
        # input = self.collate_fn(input)
        input = self.fetch_data_for_calc(outputs)
        if input is not None:
            self._data_fetched = True
            for metric_name, metric_fn in metrics_batch_container.items():
                metric_value = metric_fn(*input)
                self.log(metric_name, metric_value)
            for _, metric_fn in metrics_epoch_container.items():
                if metric_fn not in metrics_batch_container.values():
                    metric_fn(*input)

    def calc_epoch_metrics(self, metrics_epoch_container, pl_module):
        if self._data_fetched:
            for metric_name, metric_fn in metrics_epoch_container.items():
                metric_value = metric_fn.compute()
                self.log(metric_name, metric_value)
                metric_fn.reset()
            self._data_fetched = False
