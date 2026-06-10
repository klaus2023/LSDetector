import math

import torch
from torch import nn
import torch.nn.functional as F


def _gelu_tanh():
    """PyTorch GELU variant matching the original FFN implementation."""
    return nn.GELU(approximate="tanh")


def init_lsdformer_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
    elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()


class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, reduction=4, group_channels=16):
        super().__init__()
        if channels % group_channels != 0:
            raise ValueError("channels must be divisible by group_channels")

        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.reduction_ratio = reduction
        self.group_channels = group_channels
        self.groups = channels // group_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.BatchNorm2d(channels // reduction),
            nn.GELU(),
        )
        self.conv2 = nn.Conv2d(channels // reduction, kernel_size**2 * self.groups, kernel_size=1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        x_for_weight = self.avgpool(x) if self.stride > 1 else x
        weight = self.conv2(self.conv1(x_for_weight))
        b, _, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
        unfolded = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
        return (weight * unfolded).sum(dim=3).view(b, self.channels, h, w)


class MISA(nn.Module):
    """Multi-level involution-based spatial attention from LSDFormer."""

    def __init__(self):
        super().__init__()
        self.involution_scale1 = Involution(channels=2, kernel_size=3, stride=1, reduction=1, group_channels=1)
        self.involution_scale2 = Involution(channels=2, kernel_size=3, stride=1, reduction=1, group_channels=1)
        self.involution_scale4 = Involution(channels=2, kernel_size=3, stride=1, reduction=1, group_channels=1)
        self.involution_scale8 = Involution(channels=2, kernel_size=3, stride=1, reduction=1, group_channels=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value, _ = torch.max(x, dim=1, keepdim=True)
        attention_seed = torch.cat([avg, max_value], dim=1)
        height, width = attention_seed.shape[-2:]

        scale1 = self.involution_scale1(attention_seed)
        scale2 = F.adaptive_avg_pool2d(attention_seed, (max(1, height // 2), max(1, width // 2)))
        scale4 = F.adaptive_avg_pool2d(attention_seed, (max(1, height // 4), max(1, width // 4)))
        scale8 = F.adaptive_avg_pool2d(attention_seed, (max(1, height // 8), max(1, width // 8)))

        scale2 = self.involution_scale2(scale2)
        scale4 = self.involution_scale4(scale4)
        scale8 = self.involution_scale8(scale8)

        scale2 = F.interpolate(scale2, size=(height, width), mode="bilinear")
        scale4 = F.interpolate(scale4, size=(height, width), mode="bilinear")
        scale8 = F.interpolate(scale8, size=(height, width), mode="bilinear")
        attention = self.fuse(torch.cat([scale1, scale2, scale4, scale8], dim=1))
        return x * self.sigmoid(attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_value, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.conv1(torch.cat([avg, max_value], dim=1))
        return x * self.sigmoid(attention)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden_channels = channels // ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_value = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg + max_value)


class CBAM(nn.Module):
    def __init__(self, channels, kernel_size=7, ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class OverlapPatchEmbedding(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(init_lsdformer_weights)

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), height, width


class EMSA(nn.Module):
    """Efficient multi-head self-attention used by Efficient-CT and Enhanced-CT."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divided by num_heads {num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.spatial_reduction = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.key_value = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(init_lsdformer_weights)

    def forward(self, x, height, width):
        batch_size, token_count, channels = x.shape
        query = self.query(x).reshape(batch_size, token_count, self.num_heads, channels // self.num_heads)
        query = query.permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            reduced = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
            reduced = self.spatial_reduction(reduced).reshape(batch_size, channels, -1).permute(0, 2, 1)
            reduced = self.norm(reduced)
            key_value = self.key_value(reduced)
        else:
            key_value = self.key_value(x)

        key_value = key_value.reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads)
        key_value = key_value.permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = (attention @ value).transpose(1, 2).reshape(batch_size, token_count, channels)
        return self.proj_drop(self.proj(x))


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class DepthwiseConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, height, width):
        batch_size, _, channels = x.shape
        x = x.transpose(1, 2).view(batch_size, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=_gelu_tanh, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthwiseConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(init_lsdformer_weights)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MB3Block(nn.Module):
    """Modified MobileNetV3 block: expansion, depthwise convolution, CBAM, projection."""

    def __init__(self, kernel_size, in_size, expand_size, out_size, act_layer, stride, attention=True, attention_kernel=7):
        super().__init__()
        self.stride = stride
        self.expand_conv = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_size)
        self.expand_act = act_layer()
        self.depthwise_conv = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(expand_size)
        self.depthwise_act = act_layer()
        self.cbam = CBAM(channels=expand_size, kernel_size=attention_kernel) if attention else nn.Identity()
        self.project_conv = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.project_norm = nn.BatchNorm2d(out_size)
        self.project_act = act_layer()

    def forward(self, x):
        x = self.expand_act(self.expand_norm(self.expand_conv(x)))
        x = self.depthwise_act(self.depthwise_norm(self.depthwise_conv(x)))
        x = self.cbam(x)
        x = self.project_norm(self.project_conv(x))
        return self.project_act(x)


class EfficientCTBlock(nn.Module):
    """Parallel MB3 and EMSA branches for local-global feature extraction."""

    def __init__(
        self,
        conv_expand,
        embed_dim,
        sr_ratio,
        num_head,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        attention_kernel=7,
    ):
        super().__init__()
        drop_path = float(drop_path or 0.0)
        self.norm1 = norm_layer(embed_dim)
        self.emsa = EMSA(
            embed_dim,
            num_heads=num_head,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            sr_ratio=sr_ratio,
        )
        self.norm2 = norm_layer(embed_dim)
        self.ffn = MixFFN(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mb3_branch = MB3Block(
            kernel_size=3,
            in_size=embed_dim,
            expand_size=conv_expand * embed_dim,
            out_size=embed_dim,
            act_layer=act_layer,
            stride=1,
            attention=True,
            attention_kernel=attention_kernel,
        )
        self.apply(init_lsdformer_weights)

    def forward(self, x, height, width):
        batch_size = x.shape[0]
        cnn_features = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        cnn_features = self.mb3_branch(cnn_features).flatten(2).transpose(1, 2)

        # Efficient-CT fuses local MB3 features with global EMSA features before the FFN.
        fused = x + cnn_features + self.drop_path(self.emsa(self.norm1(x), height, width))
        return fused + self.drop_path(self.ffn(self.norm2(fused), height, width))


class DecoderEfficientCTBlock(nn.Module):
    """Efficient-CT variant that consumes image-format decoder features."""

    def __init__(
        self,
        conv_expand,
        embed_dim,
        sr_ratio,
        num_head,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        attention_kernel=7,
    ):
        super().__init__()
        drop_path = float(drop_path or 0.0)
        self.norm1 = norm_layer(embed_dim)
        self.emsa = EMSA(
            embed_dim,
            num_heads=num_head,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            sr_ratio=sr_ratio,
        )
        self.norm2 = norm_layer(embed_dim)
        self.ffn = MixFFN(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mb3_branch = MB3Block(
            kernel_size=3,
            in_size=embed_dim,
            expand_size=conv_expand * embed_dim,
            out_size=embed_dim,
            act_layer=act_layer,
            stride=1,
            attention=True,
            attention_kernel=attention_kernel,
        )
        self.apply(init_lsdformer_weights)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        sequence = x.flatten(2).transpose(1, 2)
        cnn_features = self.mb3_branch(x).flatten(2).transpose(1, 2)
        fused = sequence + cnn_features + self.drop_path(self.emsa(self.norm1(sequence), height, width))
        x = fused + self.drop_path(self.ffn(self.norm2(fused), height, width))
        return x.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()


class EnhancedCTBlock(nn.Module):
    """Enhanced-CT adds involution, channel attention, and MISA before EMSA."""

    def __init__(
        self,
        conv_expand,
        embed_dim,
        sr_ratio,
        num_head,
        mlp_ratio,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        attention_kernel=7,
        reduction=4,
        group_channels=16,
    ):
        super().__init__()
        drop_path = float(drop_path or 0.0)
        self.norm1 = norm_layer(embed_dim)
        self.emsa = EMSA(
            embed_dim,
            num_heads=num_head,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            sr_ratio=sr_ratio,
        )
        self.norm2 = norm_layer(embed_dim)
        self.ffn = MixFFN(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), drop=drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.involution1 = Involution(embed_dim, kernel_size=3, reduction=reduction, group_channels=group_channels)
        self.involution2 = Involution(embed_dim, kernel_size=3, reduction=reduction, group_channels=group_channels)
        self.channel_attention = ChannelAttention(channels=embed_dim, ratio=16)
        self.mb3_branch = MB3Block(
            kernel_size=3,
            in_size=embed_dim,
            expand_size=conv_expand * embed_dim,
            out_size=embed_dim,
            act_layer=act_layer,
            stride=1,
            attention=True,
            attention_kernel=attention_kernel,
        )
        self.misa = MISA()
        self.norm3 = nn.Sequential(nn.BatchNorm2d(embed_dim), nn.GELU())
        self.norm4 = nn.Sequential(nn.BatchNorm2d(embed_dim), nn.GELU())
        self.apply(init_lsdformer_weights)

    def forward(self, x, height, width):
        batch_size = x.shape[0]
        image_features = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        mb3_features = self.mb3_branch(image_features).flatten(2).transpose(1, 2)

        # Enhanced-CT prepares the attention branch with spatial-specific involution,
        # channel attention, and multi-level involution-based spatial attention.
        enhanced = self.involution1(image_features)
        enhanced = self.norm3(enhanced)
        enhanced = self.involution2(enhanced)
        enhanced = self.norm4(enhanced)
        enhanced = self.channel_attention(enhanced)
        enhanced = self.misa(enhanced).flatten(2).transpose(1, 2)

        sequence = image_features.flatten(2).transpose(1, 2)
        fused = sequence + mb3_features + self.drop_path(self.emsa(self.norm1(enhanced), height, width))
        return fused + self.drop_path(self.ffn(self.norm2(fused), height, width))


__all__ = [
    "CBAM",
    "ChannelAttention",
    "DecoderEfficientCTBlock",
    "EfficientCTBlock",
    "EMSA",
    "EnhancedCTBlock",
    "Involution",
    "MB3Block",
    "MISA",
    "OverlapPatchEmbedding",
    "SpatialAttention",
    "init_lsdformer_weights",
]
