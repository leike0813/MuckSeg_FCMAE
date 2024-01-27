import warnings
import argparse
from pathlib import Path
from config import get_config
import torch
from lightning_module import build_lightning_module
from data import build_datamodule, build_inference_dataloader
from lib.lightning_framework.trainer import build_trainer
from lib.lightning_framework.callbacks import build_prediction_writer


def parse_option():
    parser = argparse.ArgumentParser('MuckSeg Fully Convolutional Mask Auto Encoder training and evaluation script')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    group.add_argument('--resume-from-run-path', type=str,
                       help="path to folder which saves model configuration and checkpoint to resume training")

    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--max-steps', type=int, help="maximum training steps")
    parser.add_argument('--min-steps', type=int, help="minimum training steps")
    parser.add_argument('--base-lr', type=float, help="base learning rate for training")
    parser.add_argument('--overfit-batches', type=int, help="overfit batches for testing or finetuning")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--data-volume', type=int, help='approximate volume of data used for training')
    parser.add_argument('--experiment', type=str, help='experiment name')
    parser.add_argument('--spec-name', type=str, help='model spec name')
    parser.add_argument('--use-earlystopping', action='store_true',
                        help="whether to use early stopping to prevent overfitting")
    parser.add_argument('--use-custom-checkpointing', action='store_true',
                        help="whether to use custom checkpointing callback")
    parser.add_argument('--device', type=str, help='accelerate device to be used')
    parser.add_argument('--config-output-path', type=str, metavar='PATH',
                        help='root of output folder, the full path is <config-output-path>/<model_type>_<model_spec>_<tag>.yaml')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--logger', type=str,
                        help='overwrite logger if provided, can be MLFlowLogger only for now.')
    parser.add_argument('--full-dump', action='store_true',
                        help="whether to dump full config file")
    parser.add_argument('--ignore-warnings', action='store_true',
                        help="whether to ignore warnings during execution")

    args, unparsed = parser.parse_known_args()

    arg_mapper = {
        'batch_size': {'TRAIN.BATCH_SIZE': None, 'TRAIN.USE_BATCHSIZE_FINDER': False},
        'max_steps': {'TRAIN.TRAINER.max_steps': None},
        'min_steps': {'TRAIN.TRAINER.min_steps': None},
        'overfit_batches': {'TRAIN.TRAINER.overfit_batches': None},
        'base_lr': {'TRAIN.OPTIMIZER.BASE_LR': None},
        'data_path': {'DATA.DATA_PATH': None},
        'data_volume': {'DATA.DATAMODULE.train_volume': None},
        'experiment': {'TRAIN.EXPERIMENT_NAME': None},
        'spec_name': {'MODEL.SPEC_NAME': None},
        'use_earlystopping': {'TRAIN.USE_EARLYSTOPPING': None},
        'use_custom_checkpointing': {'TRAIN.USE_CUSTOM_CHECKPOINTING': None},
        'device': {'TRAIN.TRAINER.accelerator': None},
        'config_output_path': {'CONFIG_OUTPUT_PATH': None},
        'tag': {'TRAIN.TAG': None},
        'logger': {'TRAIN.LOGGER.NAME': None},
        'full_dump': {'FULL_DUMP': None},
        'checkpoint_path': {'TRAIN.CKPT_PATH': None},
    }

    if args.resume_from_run_path is not None:
        try:
            import yaml
            with open('configs/environment._local.yaml', 'r') as f:
                yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
                env_settings = [
                    'ENVIRONMENT.DATA_BASE_PATH',
                    yaml_cfg['ENVIRONMENT']['DATA_BASE_PATH'],
                    'ENVIRONMENT.RESULT_BASE_PATH',
                    yaml_cfg['ENVIRONMENT']['RESULT_BASE_PATH'],
                    'ENVIRONMENT.PROJECT_PATH',
                    yaml_cfg['ENVIRONMENT']['PROJECT_PATH'],
                    'ENVIRONMENT.MLFLOW_BASE_PATH',
                    yaml_cfg['ENVIRONMENT']['MLFLOW_BASE_PATH'],
                ]
                if args.opts is None:
                    args.opts = env_settings
                else:
                    args.opts.extend(env_settings)

            run_path = Path(args.resume_from_run_path)
            args.cfg = next(iter(run_path.rglob('*Recent.yaml'))).as_posix()
            checkpoint_fld = next(iter(run_path.rglob('checkpoints')))
            _checkpoint_found = False
            for checkpoint in checkpoint_fld.glob('*'):
                if checkpoint.name == 'last':
                    checkpoint_path = next(iter(checkpoint.glob('*.ckpt'))).as_posix()
                    args.checkpoint_path = checkpoint_path
                    _checkpoint_found = True
            if not _checkpoint_found:
                warnings.warn('Cannot find model checkpoint, the training will start from scratch')
        except Exception as e:
            warnings.warn('Error occurred while loading settings from run folder:\n{msg}'.format(msg=e.__repr__()))
            if args.cfg is None:
                raise FileNotFoundError('Cannot load configuration file')

    config = get_config(args, arg_mapper)

    return args, config


def main(config, ignore_warnings=False):
    if ignore_warnings:
        warnings.filterwarnings('ignore')

    if config.TRAIN.TRAINER.accelerator == 'cpu':
        config.defrost()
        config.DATA.DATAMODULE.pin_memory = False
        config.PREDICT.DATAMODULE.pin_memory = False
        config.freeze()

    datamodule = build_datamodule(config)
    dataloader = build_inference_dataloader(config)

    print('Initializing training...')
    lightningmodule, hparams = build_lightning_module(config)

    prediction_writer = build_prediction_writer(config.PREDICT.WRITER)
    trainer, train_hparams = build_trainer(
        train_node=config.TRAIN,
        model_node=config.MODEL,
        env_node=config.ENVIRONMENT,
        extra_callbacks=[prediction_writer],
    )
    checkpoint_path = train_hparams.get('checkpoint', None)
    if checkpoint_path is not None:
        print('Loading model parameters from checkpoint: {ckpt}'.format(ckpt=checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        lightningmodule.load_state_dict(checkpoint['state_dict'])
    hparams.update(train_hparams)

    if config.TRAIN.USE_BATCHSIZE_FINDER:
        import os
        from pytorch_lightning.tuner.tuning import Tuner
        from pytorch_lightning import Trainer
        temp_trainer = Trainer(
            accelerator=config.TRAIN.TRAINER.accelerator, precision=config.TRAIN.TRAINER.precision,
            default_root_dir=os.path.join(config.ENVIRONMENT.RESULT_BASE_PATH, 'tuner'), logger=False,
        )
        tuner = Tuner(temp_trainer)
        batch_size = tuner.scale_batch_size(lightningmodule, datamodule=datamodule, mode='binsearch')
    else:
        batch_size = config.TRAIN.BATCH_SIZE
        datamodule.batch_size = batch_size
    hparams['batch_size'] = batch_size
    trainer.logger.log_hyperparams(hparams)

    print('Start fitting procedure...')
    trainer.fit(model=lightningmodule, datamodule=datamodule)
    print('Start test procedure...')
    trainer.test(model=lightningmodule, datamodule=datamodule)
    print('Start inference test...')
    trainer.predict(model=lightningmodule, dataloaders=dataloader, return_predictions=False)


if __name__ == '__main__':
    args, config = parse_option()
    main(config, args.ignore_warnings)

# EOF