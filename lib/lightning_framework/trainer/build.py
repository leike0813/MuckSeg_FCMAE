import os
import datetime
import torch
from lightning import Trainer
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from ..callbacks.loss_logger import LogLossCallback
from ..callbacks.log_checkpoint_on_exception import LogCheckpointOnException
from .default_config import DEFAULT_CONFIG
from ..default_env import DEFAULT_ENVIRONMENT
from .example_model import EXAMPLE_MODEL_CONFIG


def build_trainer(train_node=DEFAULT_CONFIG, model_node=EXAMPLE_MODEL_CONFIG, env_node=DEFAULT_ENVIRONMENT, extra_callbacks=[]):
    try:
        run_name = '{abbr}{spec}{tag}_{time}'.format(
            abbr=model_node.ABBR[model_node.TYPE] + '_' if hasattr(model_node.ABBR, model_node.TYPE) else '',
            spec=model_node.SPEC_NAME + '_' if model_node.SPEC_NAME != '' else '',
            tag=train_node.TAG,
            time=datetime.datetime.now().strftime('%d%H%M')
        )
    except Exception:
        run_name = None

    callbacks = []
    hparams = {
        'precision': train_node.TRAINER.get('precision', None),
        'min-epochs': train_node.TRAINER.get('min_epochs', None),
        'max-epochs': train_node.TRAINER.get('max_epochs', None),
        'min-steps': train_node.TRAINER.get('min_steps', None),
        'max-steps': train_node.TRAINER.get('max_steps', None),
        'overfit-batches': train_node.TRAINER.get('overfit_batches', None),
        'early-stopping': False,
        'checkpointing': False,
    }
    if train_node.TRAINER.precision in ['16-mixed', 'bf16-mixed', '16', 'bf16', 16]:
        torch.set_float32_matmul_precision('high')
    # build logger
    if train_node.LOGGER.NAME.lower() == 'mlflowlogger':
        logger = MLFlowLogger(
            tracking_uri=train_node.LOGGER.MLFLOW.tracking_uri,
            artifact_location=os.path.join(env_node.MLFLOW_BASE_PATH, train_node.LOGGER.MLFLOW.artifact_location),
            experiment_name=train_node.EXPERIMENT_NAME,
            run_name=run_name,
            tags=train_node.LOGGER.TAGS,
            log_model=True,
        )
    else:
        logger = None
    # build callbacks
    if train_node.LOG_LEARNINGRATE:
        callbacks.append(LearningRateMonitor())
    if train_node.LOG_LOSS:
        callbacks.append(LogLossCallback())

    if train_node.USE_EARLYSTOPPING:
        callbacks.append(EarlyStopping(
            monitor=train_node.MONITOR,
            mode=train_node.MONITOR_MODE,
            patience=train_node.EARLYSTOPPING_PATIENCE,
        ))
        hparams['early-stopping'] = True
        hparams.update({
            'early-stopping-patience': train_node.EARLYSTOPPING_PATIENCE,
            'train-monitor': train_node.MONITOR
        })

    if train_node.USE_CUSTOM_CHECKPOINTING:
        callbacks.append(ModelCheckpoint(
            monitor=train_node.MONITOR,
            mode=train_node.MONITOR_MODE,
            save_top_k=train_node.CHECKPOINT_TOPK,
            save_last=train_node.CHECKPOINT_SAVELAST,
        ))
        hparams['checkpointing'] = True
        hparams.update({
            'train-monitor': train_node.MONITOR
        })

    if len(train_node.LOG_CHECKPOINT_ON_EXCEPTION) > 0:
        exception_types = []
        for exception_name in train_node.LOG_CHECKPOINT_ON_EXCEPTION:
            try:
                exception_types.append(__builtins__.get(exception_name))
            except AttributeError:
                continue
        if len(exception_types) == 0:
            exception_types.append(Exception)
        callbacks.append(LogCheckpointOnException(exception_types))
        
    if os.path.isfile(os.path.join(env_node.DATA_BASE_PATH, train_node.CKPT_PATH)):
        hparams.update({
            'checkpoint': os.path.join(env_node.DATA_BASE_PATH, train_node.CKPT_PATH)
        })

    callbacks.extend(extra_callbacks)

    return (
        Trainer(
            logger=logger,
            default_root_dir=os.path.join(env_node.RESULT_BASE_PATH, train_node.TRAINER_ROOT_DIR),
            callbacks=callbacks,
            **train_node.TRAINER
        ),
        hparams
    )
