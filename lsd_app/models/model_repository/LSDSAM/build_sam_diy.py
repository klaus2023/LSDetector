import torch
import torch.nn as nn
from .SAM_blocks import *
from functools import partial
import math
import warnings

edge_channels=3
edge_tune_reduction=4
class Sam(nn.Module):
    def __init__(self, image_encoder,mff,mask_decoder,seg_head):
        super(Sam, self).__init__()
        self.image_encoder = image_encoder
        self.mff = mff
        self.mask_decoder = mask_decoder
        self.seg_head=seg_head
        self.apply(initialize_weights)
    def forward(self, x, return_feats=False):
        # Encode image features.
        features = self.image_encoder(x) 
        
        # Fuse features.
        mask = self.mff(features)  # (B, C_out, H, W)
        mff=mask
        # Decode mask.
        mask = self.mask_decoder(features[-1],mask)  # (B, 1, H, W) or (B, num_classes, H, W)
        mask=self.seg_head(features[:-1],mask)
        mask=F.interpolate(
            mask,
            size=self.image_encoder.img_size,
            mode="bilinear",
            align_corners=False
        )
        if return_feats:
            return mask,mff
        else:
            return mask
    


h_params={
    "encoder_embed_dim":1280,
    "encoder_depth":32,
    "encoder_num_heads":16,
    "encoder_global_attn_indexes":[7, 15, 23, 31],
    "checkpoint": None
}

l_params={
    "encoder_embed_dim":1024,
    "encoder_depth":24,
    "encoder_num_heads":16,
    "encoder_global_attn_indexes":[5, 11, 17, 23],
    "checkpoint": None
    
}


b_params={
    "encoder_embed_dim":768,
    "encoder_depth":12,
    "encoder_num_heads":12,
    "encoder_global_attn_indexes":[2, 5, 8, 11],
    "checkpoint": None
    
}

out_chans=256
img_size=1024
vit_patch_size = 16
prompt_embedding_dim=256

def build_sam_h(checkpoint=False, map_location='cuda'):
    sam_h=Sam(  image_encoder=ImageEncoderViT(img_size=img_size,
                            mlp_ratio=4,
                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                            depth=h_params['encoder_depth'],
                            embed_dim=h_params['encoder_embed_dim'],
                            num_heads=h_params['encoder_num_heads'],
                            edge_chans=edge_channels,
                            edge_tune_reduction=edge_tune_reduction,
                            patch_size=vit_patch_size,
                            global_attn_indexes=h_params['encoder_global_attn_indexes'],
                            use_rel_pos=True,
                            qkv_bias=True,
                            window_size=14,
                            out_chans=out_chans,
                            adapter_reduction=4,
                            fusion_reduction=4
                            ),
                mff=MFF(channels_out=256),
                seg_head=Segmentation_head(dim=256),
                mask_decoder=MaskDecoder(
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embedding_dim,
                    mlp_dim=2048,
                    num_heads=8),
                    transformer_dim=prompt_embedding_dim,
                    MFF_Enhancement=MFF_Enhancement(dim=prompt_embedding_dim,num_heads=8)))
    if checkpoint:
        if not h_params.get('checkpoint'):
            raise FileNotFoundError(
                "LSDSAM-H bundled SAM initialization checkpoint is disabled. "
                "Load weights through the LSD_APP weight registry base_weight instead."
            )
        sam_h=load_sam_checkpoint(sam_model=sam_h,checkpoint_path=h_params['checkpoint'],map_location=map_location)
    
    return sam_h

def build_sam_l(checkpoint=False, map_location='cuda'):
    sam_l=Sam(  image_encoder=ImageEncoderViT(img_size=img_size,
                            mlp_ratio=4,
                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                            depth=l_params['encoder_depth'],
                            embed_dim=l_params['encoder_embed_dim'],
                            num_heads=l_params['encoder_num_heads'],
                            edge_chans=edge_channels,
                            edge_tune_reduction=edge_tune_reduction,
                            patch_size=vit_patch_size,
                            global_attn_indexes=l_params['encoder_global_attn_indexes'],
                            use_rel_pos=True,
                            qkv_bias=True,
                            window_size=14,
                            out_chans=out_chans,
                            adapter_reduction=4,
                            fusion_reduction=4
                            ),
                mff=MFF(channels_out=256),
                seg_head=Segmentation_head(dim=256),
                mask_decoder=MaskDecoder(
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embedding_dim,
                    mlp_dim=2048,
                    num_heads=8),
                    transformer_dim=prompt_embedding_dim,
                    MFF_Enhancement=MFF_Enhancement(dim=prompt_embedding_dim,num_heads=8)))
    if checkpoint:
        if not l_params.get('checkpoint'):
            raise FileNotFoundError(
                "LSDSAM-L bundled SAM initialization checkpoint is disabled. "
                "Load weights through the LSD_APP weight registry base_weight instead."
            )
        sam_l=load_sam_checkpoint(sam_model=sam_l,checkpoint_path=l_params['checkpoint'],map_location=map_location)
    return sam_l


def build_sam_b(checkpoint,map_location='cuda'):
    sam_b=Sam(  image_encoder=ImageEncoderViT(img_size=img_size,
                            mlp_ratio=4,
                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                            depth=b_params['encoder_depth'],
                            embed_dim=b_params['encoder_embed_dim'],
                            edge_chans=edge_channels,
                            edge_tune_reduction=edge_tune_reduction,
                            num_heads=b_params['encoder_num_heads'],
                            patch_size=vit_patch_size,
                            global_attn_indexes=b_params['encoder_global_attn_indexes'],
                            use_rel_pos=True,
                            qkv_bias=True,
                            window_size=14,
                            out_chans=out_chans,
                            adapter_reduction=4,
                            fusion_reduction=4
                            ),
                mff=MFF(channels_out=256),
                seg_head=Segmentation_head(dim=256),
                mask_decoder=MaskDecoder(
                    transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embedding_dim,
                    mlp_dim=2048,
                    num_heads=8),
                    transformer_dim=prompt_embedding_dim,
                    MFF_Enhancement=MFF_Enhancement(dim=prompt_embedding_dim,num_heads=8)))
    if checkpoint:
        if not b_params.get('checkpoint'):
            raise FileNotFoundError(
                "LSDSAM-B bundled SAM initialization checkpoint is disabled. "
                "Load weights through the LSD_APP weight registry base_weight instead."
            )
        sam_b=load_sam_checkpoint(sam_model=sam_b,checkpoint_path=b_params['checkpoint'],map_location=map_location)
    return sam_b


def load_sam_checkpoint(sam_model, checkpoint_path,map_location='cuda'):
    # 1. Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # 2. Extract state dict
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # 3. Trim mask_tokens weights if necessary
    if 'mask_decoder.mask_tokens.weight' in state_dict:
        mask_tokens_weight = state_dict['mask_decoder.mask_tokens.weight']
        if mask_tokens_weight.size(0) > 2:
            state_dict['mask_decoder.mask_tokens.weight'] = mask_tokens_weight[:2, :]

    # 4. Load parameters.
    sam_model.load_state_dict(state_dict, strict=False)
    sam_model = sam_model.cuda()

    # 5. Get model parameter names.
    model_params = set(sam_model.state_dict().keys())
    checkpoint_params = set(state_dict.keys())

    # 6. Compute matched parameters.
    matching_params = model_params.intersection(checkpoint_params)

    # # 7. Print successfully matched parameter names.
    # print("Successfully matched parameter names:")
    # for param in matching_params:
    #     print(param)

    # 8. Freeze successfully loaded image_encoder parameters and count frozen parameters.
    frozen_param_total = 0
    for name, param in sam_model.named_parameters():
        if name in matching_params and name.startswith("image_encoder"):
            param.requires_grad = False
            frozen_param_total += param.numel()
    print(f"Frozen parameter total: {frozen_param_total / 1e6:.2f} M")

    # 9. Check unmatched parameters.
    missing_params = model_params - matching_params
    # if missing_params:
    #     print("\nMissing model parameters:")
    #     for param in missing_params:
    #         print(param)
    # else:
    #     print("\nAll model parameters loaded successfully")

    # 10. Count matched parameters.
    matched_param_count = sum(sam_model.state_dict()[k].numel() for k in matching_params if k in sam_model.state_dict())
    missing_param_count = sum(sam_model.state_dict()[k].numel() for k in missing_params if k in sam_model.state_dict())

    print(f"\nSuccessfully matched parameter total: {matched_param_count / 1e6:.2f} M")
    print(f"Unloaded or missing parameter total: {missing_param_count / 1e6:.2f} M")

    return sam_model

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""
    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
