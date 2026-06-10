# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from .modules import *
from .common import LayerNorm2d, MLPBlock
from .modeling.image_encoder import Attention, window_partition,window_unpartition,add_decomposed_rel_pos,get_rel_pos,PatchEmbed
# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        edge_chans:int=3,
        edge_tune_reduction:int=8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        adapter_reduction=4,
        fusion_reduction=4,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size=patch_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
        
        self.global_attn_indexes=global_attn_indexes

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                reduction=adapter_reduction
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        
        self.CNN_branch=MobileNetV3_Large()

        self.alignments=nn.ModuleList(
            [
            Alignment(in_channels=24,out_channels=embed_dim//fusion_reduction),
            Alignment(in_channels=40,out_channels=embed_dim//fusion_reduction),
            Alignment(in_channels=112,out_channels=embed_dim//fusion_reduction),
            Alignment(in_channels=160,out_channels=embed_dim//fusion_reduction)
            ]
        )
        
        self.fusions=nn.ModuleList(
            [
            cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction),
            cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction),
            cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction),
            cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction)
            # twoway_cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction),
            # twoway_cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction),
            # twoway_cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction),
            # twoway_cross_branch_fusion(dim=embed_dim,num_heads=num_heads,dr_ratio=fusion_reduction)
            ]
        )


        self.tune_MLPs=nn.ModuleList(
            [
            nn.Sequential(nn.Linear(embed_dim//edge_tune_reduction,embed_dim//edge_tune_reduction),
                                nn.GELU()) for _ in range(depth)
            ]
        )

        self.shared_MLP=  nn.Sequential(
            nn.Linear(embed_dim//edge_tune_reduction,embed_dim)
        )

        self.pe_tune=nn.Sequential(
            nn.Linear(embed_dim,embed_dim//edge_tune_reduction),
            nn.LayerNorm(embed_dim//edge_tune_reduction),
            nn.GELU()
            )
        self.hfc_tune = HFCTune(edge_chans, embed_dim // edge_tune_reduction)
        
    def resize_pos_embed(self, new_img_size: int):
        assert self.pos_embed is not None, "pos_embed has not been initialized"
        old_gs = self.img_size // self.patch_size
        new_gs = new_img_size // self.patch_size

        # Original pos_embed [1, old_gs, old_gs, C] -> [1, new_gs, new_gs, C]
        pe = self.pos_embed  # nn.Parameter
        pe = pe.permute(0, 3, 1, 2)  # [1, C, old_gs, old_gs]
        pe = F.interpolate(pe, size=(new_gs, new_gs),
                        mode='bicubic', align_corners=False)
        pe = pe.permute(0, 2, 3, 1).contiguous()  # [1, new_gs, new_gs, C]
        self.pos_embed = nn.Parameter(pe)

        # Update image size.
        self.img_size = new_img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge=x[:,3:,:,:]
        x=x[:,:3,:,:]
        CNN_feats=self.CNN_branch(x)
        results=[]
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        hpc=self.hfc_tune(edge).permute(0,2,3,1)+self.pe_tune(x)
        j=0
        for i,blk in enumerate(self.blocks):
            x=x+self.shared_MLP(self.tune_MLPs[i](hpc))
            if i in self.global_attn_indexes:
                x=self.fusions[j](x,self.alignments[j](CNN_feats[j]))
                j=j+1
            x = blk(x)
            if i in self.global_attn_indexes and i != self.global_attn_indexes[-1]:
                results.append(self.neck(x.permute(0,3,1,2))) 
        x = self.neck(x.permute(0, 3, 1, 2))
        results.append(x)
        return results


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        reduction=4
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size


        # self.Adapter1=MLPAdapter(input_dim=dim,reduction_ratio=reduction,output_dim=dim)
        # self.Adapter2=MLPAdapter(input_dim=dim,reduction_ratio=reduction,output_dim=dim)        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        # print(f'before attention {x.shape}')
        x = self.attn(x)
        # print(f'after attn size:{x.shape}')
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        
        # x=self.Adapter1(x)

        x = x + self.mlp(self.norm2(x))

        # x=self.Adapter2(x)
        

        return x


class MaskDecoder(nn.Module):
    def __init__(self,
               transformer_dim: int,
               transformer:nn.Module,
               MFF_Enhancement:nn.Module,
               activation: Type[nn.Module]=nn.GELU,
               num_classes: int = 2
               ):
        super().__init__()
        self.pe_layer=PositionEmbeddingRandom(128)
        self.img_pe=self.pe_layer([32, 32]).unsqueeze(0)
        self.transformer_dim=transformer_dim
        self.transformer=transformer
        self.num_classes=num_classes
        self.mask_tokens=nn.Embedding(self.num_classes,transformer_dim)
        self.MFF_enhance=MFF_Enhancement
        self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                activation(),
            )
        self.output_hypernetworks_mlps=nn.ModuleList(
        [
        MLP(transformer_dim, transformer_dim,transformer_dim//8,3)
        for _ in range(num_classes)
        ]
        )

    def forward(self,img_embeddings:torch.Tensor,mff:torch.Tensor):
       masks=self.predict_masks(img_embeddings=img_embeddings,img_pe=self.img_pe,mff=mff)
       return masks
    
    def predict_masks(self,
                      img_embeddings:torch.Tensor,
                      img_pe:torch.Tensor,
                      mff:torch.Tensor):
        #tokens shape: (2,256) -> (1,2,256)->(B,2,256)
        tokens=self.mask_tokens.weight.unsqueeze(0).repeat(img_embeddings.size(0),1,1) 
        src=img_embeddings
        # Shape: (B, 256, 64, 64).
        pos_src=img_pe.repeat(src.shape[0],1,1,1)
        b,c,h,w=src.shape

        #input shape: src(B,256,64,64)  pos(B,256,64,64)  tokens(B,2,256)
        hs,src=self.transformer(src,pos_src,tokens)
        mask_tokens_out=hs
        src=src.transpose(1,2).view(b,c,h,w)
        #(B,32,256,256)
        upscaled_embedding=self.output_upscaling(self.MFF_enhance(src,mff))
        hyper_in_list:List[torch.Tensor]=[]
        for i in range(self.num_classes):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:,i,:]))
        #hyper_in (B,2,32)
        hyper_in=torch.stack(hyper_in_list,dim=1)
        b,c,h,w=upscaled_embedding.shape
        # (B,2,32) @ (B,32,256*256)   
        masks=(hyper_in @ upscaled_embedding.view(b,c,h*w)).view(b,-1,h,w)
        return masks
    


class MFF(nn.Module):
    def __init__(self, channels_out):
        super(MFF, self).__init__()
        # self.fusion=nn.Sequential(
        #     nn.Conv2d(channels_out*4,channels_out*4,3,1,padding=1),
        #     LayerNorm2d(channels_out*4),
        #     nn.GELU())
        
        self.down=nn.Sequential(
            nn.Conv2d(channels_out*4,channels_out,1,1),
            LayerNorm2d(channels_out),
            nn.GELU()
        )
        
        self.conv1=nn.Sequential(
            nn.Conv2d(channels_out,channels_out,3,1,padding=1),
            LayerNorm2d(channels_out),
            nn.GELU()
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(channels_out,channels_out,3,1,padding=1),
            LayerNorm2d(channels_out),
            nn.GELU()
        )
    def forward(self,features):
        # f1, f2, f3, f4:(B, channels_out, H, W)

        B, C,H, W, = features[0].shape
        
        fusions=self.down(torch.concat([features[0],features[1],features[2],features[3]],dim=1))+features[0]+features[1]+features[2]+features[3]
        fusions=self.conv1(fusions)+fusions
        fusions=self.conv2(fusions)+fusions
        return fusions


class MFF_Enhancement(nn.Module):
   
    def __init__(self, dim: int, num_heads: int):
        super(MFF_Enhancement, self).__init__()
        # Apply two fusion submodules in sequence with the default twoway_cross_fusion parameters.
        self.crossfusion_stage1 = twoway_cross_fusion(dim=dim, num_heads=num_heads)
        self.crossfusion_stage2 = twoway_cross_fusion(dim=dim, num_heads=num_heads)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        # Stage 1 fusion.
        x1,x2 = self.crossfusion_stage1(x1, x2)
        # Stage 2 fusion.
        x1,x2 = self.crossfusion_stage2(x1, x2)

        return x1+x2


class SingleSegHead(nn.Module):
    def __init__(self,dim, num_classes=2):
        super().__init__()
        
        self.pre_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
        )
        self.up1 = nn.ConvTranspose2d(dim, dim//4, kernel_size=2, stride=2)
        self.ln1 = LayerNorm2d(dim//4)
        self.act1 = nn.GELU()
        
        self.mid_conv = nn.Sequential(
            nn.Conv2d(dim//4,  dim//4, kernel_size=3, padding=1, bias=False),
            LayerNorm2d( dim//4),
            nn.GELU(),
        )
        self.up2 = nn.ConvTranspose2d( dim//4,  dim//8, kernel_size=2, stride=2)
        self.act2 = nn.GELU()
        
        self.pred_conv = nn.Conv2d( dim//8, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.pre_conv(x)+x        
        x = self.up1(x)                 
        x = self.ln1(x)
        x = self.act1(x)
        x = self.mid_conv(x)+x           
        x = self.up2(x)                
        x = self.act2(x)
        x = self.pred_conv(x)
        return x

class Segmentation_head(nn.Module):
    def __init__(self,dim,num_classes=2,num_heads=3):
        super().__init__()
        self.heads=nn.ModuleList([
            SingleSegHead(dim,num_classes=num_classes) for _ in range(num_heads)
        ])
        
        self.fusion=nn.Conv2d(num_classes*(num_heads+1),num_classes,kernel_size=1,stride=1)
    def forward(self, feats, maskdecoder_out):

        seg_logits = []
        for i, feat in enumerate(feats):
            seg_logits.append(self.heads[i](feat))  
        x = torch.cat([maskdecoder_out] + seg_logits, dim=1)  
        fused = self.fusion(x)  
        return fused
