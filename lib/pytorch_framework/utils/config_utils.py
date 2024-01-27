import os
import yaml


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    base_configs = yaml_cfg.setdefault('BASE', [''])
    for cfg in base_configs:
        if cfg:
            cfg_path = os.path.join(os.path.dirname(cfg_file), cfg)
            if os.path.exists(cfg_path):
                _update_config_from_file(
                    config, cfg_path
                )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    for base in base_configs:
        config.BASE.remove(base)
    config.BASE.append('')
    config.freeze()


def update_config(config, args, arg_mapper):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and ((eval(f'args.{name}') == 0 and eval(f'args.{name}') is not False) or eval(f'args.{name}')):
            return True
        return False

    # merge from specific arguments
    check_list = []
    for arg_name, arg_dict in arg_mapper.items():
        if _check_args(arg_name):
            for k, v in arg_dict.items():
                check_list.append(k)
                if v is None:
                    check_list.append(getattr(args, arg_name))
                else:
                    check_list.append(v)
    config.merge_from_list(check_list)

    config.freeze()