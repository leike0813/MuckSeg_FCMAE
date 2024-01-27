import torch
import torch.nn as nn
import torch.nn.functional as F
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiChannelwiseConvolution,
    MinkowskiLinear
)
from timm.models.layers import trunc_normal_
from models.blocks.functional import upsample_mask, gen_random_mask, patchify
from models.MuckSeg_Encoder_Sparse import MuckSeg_Encoder_Sparse
from models.MuckSeg_FCMAE_Decoder import MuckSeg_FCMAE_Decoder


class MuckSeg_FCMAE(nn.Module):
    _PATCH_SIZE_MODULUS = 16
    is_auto_scalable = True
    support_side_output = True
    size_modulus = 16

    def __init__(self, in_chans=1, kernel_sizes=[7, 7, 7, 7], depths=[3, 9, 3, 3],
                 dim=32, stem_routes=['3CONV', '5CONV', '7CONV', '9CONV', 'D-3CONV', 'D-5CONV'],
                 mlp_ratio=4., drop_path_rate=0., decoder_depths=[2, 2],
                 decoder_dim=512, decoder_kernel_sizes=[7, 7], decoder_drop_path_rate=0.,
                 neck_kernel_size=3, neck_depth=1, mask_ratio=0.6,
                 patch_size_factor=2, multi_scale_input=False, norm_pix_loss=False,
                 use_convnext_v2=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size_factor = patch_size_factor
        self.norm_pix_loss = norm_pix_loss

        self.encoder = MuckSeg_Encoder_Sparse(
            in_chans=in_chans, kernel_sizes=kernel_sizes, depths=depths, dim=dim, stem_routes=stem_routes,
            mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate, multi_scale_input=multi_scale_input,
            use_convnext_v2=use_convnext_v2
        )
        self.decoder = MuckSeg_FCMAE_Decoder(
            depths=decoder_depths, dim=decoder_dim, kernel_sizes=decoder_kernel_sizes, mlp_ratio=mlp_ratio,
            drop_path_rate=decoder_drop_path_rate, neck_kernel_size=neck_kernel_size, neck_depth=neck_depth,
            use_convnext_v2=use_convnext_v2
        )
        self.enc_dec_adaptor = nn.Conv2d(in_channels=dim * self._PATCH_SIZE_MODULUS, out_channels=decoder_dim, kernel_size=1)
        self.head = nn.Conv2d(in_channels=decoder_dim // (self._PATCH_SIZE_MODULUS // self.target_downsample_factor), out_channels=in_chans, kernel_size=1)
        self.lap_head = nn.Conv2d(in_channels=decoder_dim // (self._PATCH_SIZE_MODULUS // self.target_downsample_factor), out_channels=in_chans, kernel_size=1)
        self.hog_head = nn.Conv2d(in_channels=decoder_dim // (self._PATCH_SIZE_MODULUS // 4), out_channels=32, kernel_size=1) # TODO: Parameterize this

        self.apply(self._init_weights)
        self.encoder.stem.init_maxpool_route_weight()

    def _init_weights(self, m):
        if isinstance(m, (MinkowskiConvolution, MinkowskiChannelwiseConvolution)):
            trunc_normal_(m.kernel, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            if m.linear.bias is not None:
                nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(m, 'mask_token'):
            torch.nn.init.normal_(m.mask_token, std=.02)

    def patchify(self, imgs):
        """
        imgs: (B, C, H, W)
        x: (B, L, patch_size**2 *C)
        """
        return patchify(imgs, self._PATCH_SIZE_MODULUS // 4)

    def gen_random_mask(self, x, mask_ratio):
        return gen_random_mask(x, self._PATCH_SIZE_MODULUS * self.patch_size_factor, mask_ratio)

    def upsample_mask(self, mask, scale):
        return upsample_mask(mask, scale)

    def loss_pixel(self, pred, target, mask):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # mean loss per patch: B, H/16, W/16

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def loss_lap(self, pred, target, mask):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def loss_hog(self, pred, target, mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def loss_func(self, pred, target, mask):
        return 0.5 * self.loss_pixel(pred[0], target[0], mask) \
            + 0.2 * self.loss_lap(pred[1], target[1], mask) \
            + 0.3 * self.loss_hog(pred[2], target[2], mask)

    def forward_features(self, x, mask=None):
        if mask is None:
            mask = self.gen_random_mask(x, self.mask_ratio)  # mask: B, H/P, W/P
        mask = upsample_mask(mask, self.patch_size_factor) # B, H/P, W/P -> B, H/Pm, W/Pm (Pm=16)
        x = self.encoder(x, mask)  # B, C, H, W -> B, 16*De, H/16, W/16
        x = self.enc_dec_adaptor(x)  # B, 16*De, H/16, W/16 -> B, Dd, H/16, W/16
        x = self.decoder(x, mask)  # B, Dd, H/16, W/16 -> B, Dd/(16/St), H/St, W/St
        x_pic = self.head(x)  # B, Dd/(16/St), H/St, W/St -> B, C, H/St, W/St
        x_lap = self.lap_head(x)
        x_hog = self.hog_head(x)

        return x_pic, x_lap, x_hog, mask

    def forward(self, x):
        pic, lap, hog = x
        target_pic = F.interpolate(pic, scale_factor=1 / 4, mode="bilinear") # B, C, H, W -> B, C, H/St, W/St
        target_lap = F.interpolate(lap, scale_factor=1 / 4, mode="bilinear")
        target_pic = self.patchify(target_pic) # B, C, H/St, W/St -> B, H/Pm, W/Pm, C*(Pm/St)**2 (Pm=16)
        target_lap = self.patchify(target_lap)
        target_hog = self.patchify(hog)
        x_pic, x_lap, x_hog, mask = self.forward_features(pic) # B, C, H, W -> B, C, H/St, W/St
        x_pic = self.patchify(x_pic) # B, C, H/St, W/St -> B, H/Pm, W/Pm, C*(Pm/St)**2
        x_lap = self.patchify(x_lap)
        x_hog = self.patchify(x_hog)

        return self.loss_func((x_pic, x_lap, x_hog), (target_pic, target_lap, target_hog), mask)

    def forward_featuremaps(self, x, mask_ratio=0.):
        x = x[0]
        mask = self.gen_random_mask(x, mask_ratio)
        mask = upsample_mask(mask, self.patch_size_factor)
        x, fmaps = self.encoder.forward_featuremaps(x, mask)
        x = self.enc_dec_adaptor(x)
        fmaps['Neck'] = x
        fmaps_decoder = self.decoder.forward_featuremaps(x, mask)
        fmaps.update(fmaps_decoder)

        return fmaps

# EOF