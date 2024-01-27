from pathlib import Path
from .MuckSeg_FCMAE import MuckSeg_FCMAE


__all__ = ['build_model']


def build_model(node):
    node.defrost()
    cwd = Path.cwd()
    node.FILE_PATHS = [(cwd / path).as_posix() for path in node.FILE_PATHS]
    node.freeze()

    hparams = {
        'image-size': node.IMAGE_SIZE,
        'input-channel': node.IN_CHANS,
        'embedding-dim': node.DIM,
        'mlp-ratio': node.MLP_RATIO,
        'encoder-kernel-sizes': node.ENCODER.kernel_sizes,
        'encoder-depths': node.ENCODER.depths,
        'multi-scale-input': node.ENCODER.multi_scale_input,
        'decoder-embedding-dim': node.DECODER.dim,
        'decoder-kernel-sizes': node.DECODER.kernel_sizes,
        'decoder-depths': node.DECODER.depths,
        'neck-kernel-size': node.DECODER.neck_kernel_size,
        'neck-depth': node.DECODER.neck_depth,
        'mask-ratio': node.MASK_RATIO,
    }
    model = MuckSeg_FCMAE(
        in_chans=node.IN_CHANS,
        dim=node.DIM,
        mlp_ratio=node.MLP_RATIO,
        drop_path_rate=node.DROP_PATH_RATE,
        decoder_dim=node.DECODER.dim,
        decoder_depths=node.DECODER.depths,
        decoder_kernel_sizes=node.DECODER.kernel_sizes,
        decoder_drop_path_rate=node.DROP_PATH_RATE,
        neck_kernel_size=node.DECODER.neck_kernel_size,
        neck_depth=node.DECODER.neck_depth,
        norm_pix_loss=node.LOSS.norm_pix_loss,
        mask_ratio=node.MASK_RATIO,
        patch_size_factor=node.PATCH_SIZE_FACTOR,
        use_convnext_v2=node.USE_CONVNEXT_V2,
        **node.ENCODER,
    )

    return model, hparams
