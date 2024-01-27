import math
from collections import OrderedDict


def config_validator(config, fcmae_config):
    return config.MODEL.ENCODER == fcmae_config.MODEL.ENCODER \
        and config.MODEL.DIM == fcmae_config.MODEL.DIM \
        and config.MODEL.IN_CHANS == fcmae_config.MODEL.IN_CHANS \
        and config.MODEL.MLP_RATIO == fcmae_config.MODEL.MLP_RATIO


def extract_encoder_state_dict(state_dict):
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        k_parts = k.split('.')
        if k.startswith('model.encoder'):
            k_parts.pop(0)
            k_parts.pop(0)
            new_k = '.'.join(k_parts)
            new_dict[new_k] = v

    return new_dict


def map_sparse_to_dense(state_dict):
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        k_parts = k.split('.')
        if k.startswith('downsample_layers'):
            if k_parts[2] == '0': # LayerNorm
                k_parts.pop(-2) # remove 'ln'
            if k_parts[2] == '1': # Plain Convolution with bias
                if k_parts[-1] == 'kernel':
                    k_parts[-1] = 'weight' # rename 'kernel' to 'weight'
                    kv, in_dim, out_dim = v.shape # reshape sparse weights: {ks^2, in_dim, out_dim} to dense weights: {out_dim, in_dim, ks, ks}
                    ks = int(math.sqrt(kv))
                    v = v.permute(2, 1, 0).reshape(out_dim, in_dim, ks, ks).transpose(3, 2)
                elif k_parts[-1] == 'bias':
                    v = v.squeeze(0) # reshape sparse bias: {1, out_dim} to dense bias: {out_dim}
        elif k.startswith('stages'):
            if k_parts[3] == 'dwconv': # Depthwise Convolution
                if k_parts[-1] == 'kernel':
                    k_parts[-1] = 'weight'
                    kv, dim = v.shape # reshape sparse weights: {ks^2, dim} to dense weights: {dim, 1, ks, ks}
                    ks = int(math.sqrt(kv))
                    v = v.permute(1, 0).reshape(dim, 1, ks, ks).transpose(3, 2)
                elif k_parts[-1] == 'bias':
                    v = v.squeeze(0)  # reshape sparse bias: {1, dim} to dense bias: {dim}
            elif k_parts[3] == 'norm': # LayerNorm
                k_parts.pop(-2)
            elif k_parts[3] == 'pwconv1' or k_parts[3] == 'pwconv2': # Pointwise Convolution
                k_parts.pop(-2) # remove 'linear'
            elif k_parts[3] == 'grn':
                v = v.unsqueeze(0).unsqueeze(0)
        new_k = '.'.join(k_parts)
        new_dict[new_k] = v

    return new_dict


def map_dense_to_sparse(state_dict):
    new_dict = OrderedDict()
    for k, v in state_dict.items():
        k_parts = k.split('.')
        if k.startswith('downsample_layers'):
            if k_parts[2] == '0':  # LayerNorm
                k_parts.insert(-1, 'ln')  # insert 'ln'
            if k_parts[2] == '1':  # Plain Convolution with bias
                if k_parts[-1] == 'weight':
                    k_parts[-1] = 'kernel'  # rename 'weight' to 'kernel'
                    out_dim, in_dim, ks, _ks = v.shape  # reshape dense weights: {out_dim, in_dim, ks, ks} to sparse weights: {ks^2, in_dim, out_dim}
                    kv = ks * _ks
                    v = v.transpose(3, 2).reshape(out_dim, in_dim, kv).permute(2, 1, 0)
                elif k_parts[-1] == 'bias':
                    v = v.unsqueeze(0)  # reshape dense bias: {out_dim} to sparse bias: {1, out_dim}
        elif k.startswith('stages'):
            if k_parts[3] == 'dwconv':  # Depthwise Convolution
                if k_parts[-1] == 'weight':
                    k_parts[-1] = 'kernel'
                    dim, _, ks, _ks = v.shape  # reshape dense weights: {dim, 1, ks, ks} to sparse weights: {ks^2, dim}
                    kv = ks * _ks
                    v = v.transpose(3, 2).reshape(dim, kv).permute(1, 0)
                elif k_parts[-1] == 'bias':
                    v = v.unsqueeze(0)  # reshape dense bias: {dim} to sparse bias: {1, dim}
            elif k_parts[3] == 'norm':  # LayerNorm
                k_parts.insert(-1, 'ln')
            elif k_parts[3] == 'pwconv1' or k_parts[3] == 'pwconv2':  # Pointwise Convolution
                k_parts.insert(-1, 'linear')  # insert 'linear'
            elif k_parts[3] == 'grn':
                v = v.squeeze(0).squeeze(0)
        new_k = '.'.join(k_parts)
        new_dict[new_k] = v

    return new_dict