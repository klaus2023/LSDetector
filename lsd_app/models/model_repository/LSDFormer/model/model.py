import torch
from torch import nn
import torch.nn.functional as F

from .modules import (
    DecoderEfficientCTBlock,
    EfficientCTBlock,
    EnhancedCTBlock,
    OverlapPatchEmbedding,
    init_lsdformer_weights,
)


def _resize_logits(logits, size):
    return F.interpolate(logits, size=size, mode="bilinear")


class CNNStem(nn.Module):
    """CNN stem used before the four LSDFormer encoder stages."""

    def __init__(self, in_chans, stem_embed, stem_out):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(in_chans, stem_embed, kernel_size=3, stride=2, padding=1),
                ResidualConv(stem_embed, stem_embed, stride=1, padding=1),
                ResidualConv(stem_embed, stem_out, stride=1, padding=1),
                nn.BatchNorm2d(stem_out),
                nn.GELU(),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LSDFormerEncoderStage(nn.Module):
    def __init__(self, downsample, efficient_ct_blocks, enhanced_ct_blocks):
        super().__init__()
        self.efficient_ct_blocks = efficient_ct_blocks
        self.enhanced_ct_blocks = enhanced_ct_blocks
        self.downsample = downsample

    def forward(self, x):
        batch_size = x.shape[0]
        x, height, width = self.downsample(x)
        for block in self.efficient_ct_blocks:
            x = block(x, height, width)
        for block in self.enhanced_ct_blocks:
            x = block(x, height, width)
        return x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()


class LSDFormerEncoder(nn.Module):
    def __init__(
        self,
        in_chans=3,
        stem_embed=16,
        stem_out=32,
        embed_dims=None,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.01,
        attn_drop_rate=0.01,
        drop_path_rate=0.05,
        norm_layer=nn.LayerNorm,
        efficient_ct_depths=None,
        sr_ratios=None,
        conv_expands=None,
        enhanced_ct_depths=None,
        involution_reductions=None,
        involution_group_channels=None,
    ):
        super().__init__()
        embed_dims = embed_dims or [64, 128, 320, 512]
        num_heads = num_heads or [1, 2, 5, 8]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        efficient_ct_depths = efficient_ct_depths or [2, 2, 2, 2]
        sr_ratios = sr_ratios or [8, 4, 2, 1]
        conv_expands = conv_expands or [4, 4, 4, 4]
        enhanced_ct_depths = enhanced_ct_depths or [1, 1, 1, 1]
        involution_reductions = involution_reductions or [4, 4, 4, 4]
        involution_group_channels = involution_group_channels or [16, 16, 16, 16]

        drop_paths = None
        total_depth = sum(efficient_ct_depths) + sum(enhanced_ct_depths)
        if drop_path_rate > 0:
            drop_paths = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self._drop_path_cursor = 0
        self.stem = CNNStem(in_chans, stem_embed, stem_out)
        self.stages = nn.ModuleList()

        for stage_index in range(4):
            stage_depth = efficient_ct_depths[stage_index] + enhanced_ct_depths[stage_index]
            stage_drop_paths = None
            if drop_paths is not None:
                start = self._drop_path_cursor
                stage_drop_paths = drop_paths[start : start + stage_depth]

            efficient_ct_blocks = nn.ModuleList(
                [
                    EfficientCTBlock(
                        conv_expand=conv_expands[stage_index],
                        embed_dim=embed_dims[stage_index],
                        sr_ratio=sr_ratios[stage_index],
                        num_head=num_heads[stage_index],
                        mlp_ratio=mlp_ratios[stage_index],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path=stage_drop_paths[block_index] if stage_drop_paths else None,
                        norm_layer=norm_layer,
                        act_layer=nn.GELU,
                        attention_kernel=7 if stage_index != 3 else 3,
                    )
                    for block_index in range(efficient_ct_depths[stage_index])
                ]
            )

            enhanced_ct_blocks = nn.ModuleList(
                [
                    EnhancedCTBlock(
                        conv_expand=conv_expands[stage_index],
                        embed_dim=embed_dims[stage_index],
                        sr_ratio=sr_ratios[stage_index],
                        num_head=num_heads[stage_index],
                        mlp_ratio=mlp_ratios[stage_index],
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path=stage_drop_paths[efficient_ct_depths[stage_index] + block_index]
                        if stage_drop_paths
                        else None,
                        norm_layer=nn.LayerNorm,
                        act_layer=nn.GELU,
                        attention_kernel=7 if stage_index != 3 else 3,
                        reduction=involution_reductions[stage_index],
                        group_channels=involution_group_channels[stage_index],
                    )
                    for block_index in range(enhanced_ct_depths[stage_index])
                ]
            )

            downsample = OverlapPatchEmbedding(
                patch_size=3,
                stride=2,
                in_chans=stem_out if stage_index == 0 else embed_dims[stage_index - 1],
                embed_dim=embed_dims[stage_index],
            )

            self.stages.append(LSDFormerEncoderStage(downsample, efficient_ct_blocks, enhanced_ct_blocks))
            self._drop_path_cursor += stage_depth

    def forward(self, x):
        encoder_features = []
        x = self.stem(x)
        encoder_features.append(x)
        for stage in self.stages:
            x = stage(x)
            encoder_features.append(x)
        return encoder_features


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.GELU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class EnhancedSkipConnection(nn.Module):
    """Residual skip fusion used before decoder Efficient-CT blocks."""

    def __init__(self, embed_dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, encoder_feature, decoder_feature):
        fused = self.down(torch.cat([encoder_feature, decoder_feature], dim=1))
        return fused + encoder_feature + decoder_feature


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv="ResidualConv",
        conv_expand=None,
        embed_dim=None,
        sr_ratio=None,
        num_head=None,
        mlp_ratio=None,
        qkv_bias=None,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.conv_type = conv
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        if conv == "ResidualConv":
            self.conv = ResidualConv(2 * out_channels, out_channels, stride=1, padding=1)
        elif conv == "EfficientCT":
            self.conv = DecoderEfficientCTBlock(
                conv_expand=conv_expand,
                embed_dim=embed_dim,
                sr_ratio=sr_ratio,
                num_head=num_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU,
                attention_kernel=7,
            )
            self.skip_fusion = EnhancedSkipConnection(embed_dim=embed_dim)
        else:
            raise ValueError(f"Unsupported decoder conv type: {conv}")

    def forward(self, decoder_feature, encoder_feature):
        decoder_feature = self.up(decoder_feature)
        diff_y = encoder_feature.size(2) - decoder_feature.size(2)
        diff_x = encoder_feature.size(3) - decoder_feature.size(3)
        decoder_feature = F.pad(
            decoder_feature,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        if self.conv_type == "EfficientCT":
            x = self.skip_fusion(encoder_feature, decoder_feature)
        else:
            x = torch.cat([encoder_feature, decoder_feature], dim=1)
        return self.conv(x)


class LSDFormerDecoder(nn.Module):
    def __init__(
        self,
        num_classes,
        encoder_channels,
        stem_out,
        stem_embed,
        features=1024,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=False,
        qk_scale=None,
        sr_ratios=None,
        conv_expands=None,
    ):
        super().__init__()
        num_heads = num_heads or [1, 2, 5, 8]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        sr_ratios = sr_ratios or [8, 4, 2, 1]
        conv_expands = conv_expands or [4, 4, 4, 4]

        self.upsample_layers = nn.ModuleList(
            [
                DecoderStage(
                    features,
                    encoder_channels[-2],
                    conv="EfficientCT",
                    conv_expand=conv_expands[-2],
                    embed_dim=encoder_channels[-2],
                    sr_ratio=sr_ratios[-2],
                    num_head=num_heads[-2],
                    mlp_ratio=mlp_ratios[-2],
                    qk_scale=qk_scale,
                    qkv_bias=qkv_bias,
                ),
                DecoderStage(
                    encoder_channels[-2],
                    encoder_channels[-3],
                    conv="EfficientCT",
                    conv_expand=conv_expands[-3],
                    embed_dim=encoder_channels[-3],
                    sr_ratio=sr_ratios[-3],
                    num_head=num_heads[-3],
                    mlp_ratio=mlp_ratios[-3],
                    qk_scale=qk_scale,
                    qkv_bias=qkv_bias,
                ),
                DecoderStage(
                    encoder_channels[-3],
                    encoder_channels[-4],
                    conv="EfficientCT",
                    conv_expand=conv_expands[-4],
                    embed_dim=encoder_channels[-4],
                    sr_ratio=sr_ratios[-4],
                    num_head=num_heads[-4],
                    mlp_ratio=mlp_ratios[-4],
                    qk_scale=qk_scale,
                    qkv_bias=qkv_bias,
                ),
                DecoderStage(encoder_channels[-4], encoder_channels[-5], conv="ResidualConv"),
            ]
        )
        self.final_up = nn.ConvTranspose2d(stem_out, stem_embed, kernel_size=2, stride=2)
        self.final_residualconv = ResidualConv(stem_embed, stem_embed, stride=1, padding=1)

        self.head1 = nn.Conv2d(stem_embed, num_classes, kernel_size=3, padding=1)
        self.head2 = nn.Conv2d(encoder_channels[-5], num_classes, kernel_size=3, padding=1)
        self.head3 = nn.Conv2d(encoder_channels[-4], num_classes, kernel_size=3, padding=1)
        self.head4 = nn.Conv2d(encoder_channels[-3], num_classes, kernel_size=3, padding=1)
        self.head5 = nn.Conv2d(encoder_channels[-2], num_classes, kernel_size=3, padding=1)
        self.head6 = nn.Conv2d(encoder_channels[-1], num_classes, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_classes)
        self.bn2 = nn.BatchNorm2d(num_classes)
        self.bn3 = nn.BatchNorm2d(num_classes)
        self.bn4 = nn.BatchNorm2d(num_classes)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.bn6 = nn.BatchNorm2d(num_classes)
        self.out_act = nn.GELU()
        self.outconv = nn.Conv2d(6 * num_classes, num_classes, kernel_size=1)

    def forward(self, encoder_features, output_size):
        x = encoder_features[-1]
        deep_outputs = [None] * 6
        deep_outputs[5] = _resize_logits(self.bn6(self.head6(x)), output_size)

        for index, upsample_layer in enumerate(self.upsample_layers):
            x = upsample_layer(x, encoder_features[-(index + 2)])
            if index == 0:
                deep_outputs[4] = _resize_logits(self.bn5(self.head5(x)), output_size)
            elif index == 1:
                deep_outputs[3] = _resize_logits(self.bn4(self.head4(x)), output_size)
            elif index == 2:
                deep_outputs[2] = _resize_logits(self.bn3(self.head3(x)), output_size)
            elif index == 3:
                deep_outputs[1] = _resize_logits(self.bn2(self.head2(x)), output_size)

        x = self.final_up(x)
        x = self.final_residualconv(x)
        deep_outputs[0] = _resize_logits(self.bn1(self.head1(x)), output_size)

        # Deep supervision concatenates logits from all decoder scales before the final head.
        return self.outconv(self.out_act(torch.cat(deep_outputs, dim=1)))


class LSDFormer(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_chans=3,
        stem_embed=16,
        stem_out=32,
        embed_dims=None,
        num_heads=None,
        mlp_ratios=None,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.01,
        attn_drop_rate=0.01,
        drop_path_rate=0.05,
        norm_layer=nn.LayerNorm,
        efficient_ct_depths=None,
        sr_ratios=None,
        conv_expands=None,
        enhanced_ct_depths=None,
        involution_reductions=None,
        involution_group_channels=None,
    ):
        super().__init__()
        embed_dims = embed_dims or [64, 128, 320, 512]
        num_heads = num_heads or [1, 2, 5, 8]
        mlp_ratios = mlp_ratios or [4, 4, 4, 4]
        efficient_ct_depths = efficient_ct_depths or [2, 2, 2, 2]
        sr_ratios = sr_ratios or [8, 4, 2, 1]
        conv_expands = conv_expands or [4, 4, 4, 4]
        enhanced_ct_depths = enhanced_ct_depths or [1, 1, 1, 1]
        involution_reductions = involution_reductions or [4, 4, 4, 4]
        involution_group_channels = involution_group_channels or [16, 16, 16, 16]

        self.encoder = LSDFormerEncoder(
            in_chans=in_chans,
            stem_embed=stem_embed,
            stem_out=stem_out,
            embed_dims=embed_dims,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            efficient_ct_depths=efficient_ct_depths,
            sr_ratios=sr_ratios,
            conv_expands=conv_expands,
            enhanced_ct_depths=enhanced_ct_depths,
            involution_reductions=involution_reductions,
            involution_group_channels=involution_group_channels,
        )
        self.decoder = LSDFormerDecoder(
            num_classes,
            encoder_channels=[stem_out] + embed_dims,
            features=embed_dims[-1],
            stem_out=stem_out,
            stem_embed=stem_embed,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            qkv_bias=False,
            qk_scale=None,
            sr_ratios=sr_ratios,
            conv_expands=conv_expands,
        )
        initialize_weights(self)

    def forward(self, x, return_feats=False):
        output_size = x.shape[-2:]
        features = self.encoder(x)
        logits = self.decoder(features, output_size)
        if return_feats:
            return logits, features[-1]
        return logits


def initialize_weights(model):
    model.apply(init_lsdformer_weights)


__all__ = [
    "CNNStem",
    "EnhancedSkipConnection",
    "LSDFormer",
    "LSDFormerDecoder",
    "LSDFormerEncoder",
    "LSDFormerEncoderStage",
    "ResidualConv",
]
