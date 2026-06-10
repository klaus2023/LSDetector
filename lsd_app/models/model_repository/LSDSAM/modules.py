import torch
import torch.nn as nn
from .common import LayerNorm2d
from .modeling.prompt_encoder import PositionEmbeddingRandom
from .modeling.mask_decoder import MLP
from typing import List, Tuple, Type
import torch.nn.functional as F
from .modeling.transformer import TwoWayTransformer

class twoway_cross_fusion(nn.Module):
    #Input shape: x1:x2:[B,H,W,C]
    #Output shape: [B,H,W,C]
    def __init__(self,dim, num_heads,dr_ratio=4,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 CBAM_kernel=3,mlp_ratio=8,act_layer=nn.GELU,drop=0.,drop_path_rate=0.):
        super(twoway_cross_fusion,self).__init__()



        self.norm11=nn.LayerNorm(dim)
        self.norm12=nn.LayerNorm(dim)
        self.attention1=Efficient_Multihead_Crossattention(dim=dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=sr_ratio)
        self.attention2=Efficient_Multihead_Crossattention(dim=dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=sr_ratio)
        self.CBAM1=channel_and_normal_spatial_attention(in_planes=dim,kernel_size=CBAM_kernel)
        self.CBAM2=channel_and_normal_spatial_attention(in_planes=dim,kernel_size=CBAM_kernel)
        self.norm21=nn.LayerNorm(dim)
        self.norm22=nn.LayerNorm(dim)
        self.mlp1=Mlp_2(in_features=dim,hidden_features=int(dim*mlp_ratio),act_layer=act_layer,drop=drop)
        self.mlp2=Mlp_2(in_features=dim,hidden_features=int(dim*mlp_ratio),act_layer=act_layer,drop=drop)
        self.drop_path  = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self,x1,x2):
        B,C,H,W=x1.shape
        x1=x1.view(B,C,H*W).permute(0,2,1)
        residual_x1=x1
        x2=x2.view(B,C,H*W).permute(0,2,1)
        residual_x2=x2
        x1=x1+self.CBAM1(self.drop_path(self.attention1(self.norm11(residual_x1),self.norm12(residual_x2),H,W)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()).flatten(2).transpose(1,2)
        x1=x1+self.drop_path(self.mlp1(self.norm21(x1),H,W))

        x2=x2+self.CBAM2(self.drop_path(self.attention2(self.norm12(residual_x2),self.norm11(residual_x1),H,W)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()).flatten(2).transpose(1,2)
        x2=x2+self.drop_path(self.mlp2(self.norm22(x2),H,W))

        
        return (x1+residual_x1).reshape(B,H,W,C).permute(0,3,1,2).contiguous(), (x2+residual_x2).reshape(B,H,W,C).permute(0,3,1,2).contiguous()

class MLPAdapter(nn.Module):
    def __init__(self, input_dim, reduction_ratio, output_dim,act=nn.GELU()):
        super(MLPAdapter, self).__init__()
        # The adapter consists of two fully connected layers.
        self.input_dim=input_dim
        self.reduction_ratio=reduction_ratio
        self.hidden_dim=self.input_dim//self.reduction_ratio
        self.output_dim=output_dim
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)  # Input to hidden layer.
        self.act = act                      # Activation function.
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim)  # Hidden layer to output layer.
        self.residual_proj=(
            nn.Identity() if input_dim == output_dim 
            else nn.Linear(input_dim,output_dim) 
        )


    def forward(self, x):
        x=self.fc2(self.act(self.fc1(x)))+self.residual_proj(x)
        return x


class Mona(nn.Module):
    def __init__(self,input_dim,reduction_ratio,output_dim,act=nn.GELU()):
        super(Mona,self).__init__()
        self.input_dim=input_dim
        self.reduction_ratio=reduction_ratio
        self.hidden_dim=input_dim//self.reduction_ratio
        self.output_dim=output_dim
        self.ln=nn.LayerNorm(self.input_dim)
        self.down_proj=nn.Linear(self.input_dim,self.hidden_dim)
        self.up_proj=nn.Linear(self.hidden_dim,self.output_dim)
        self.s1=nn.Parameter(torch.tensor(1.0))
        self.s2=nn.Parameter(torch.tensor(1.0))
        self.act=act
        self.dw3=nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=3,padding=1,groups=self.hidden_dim)
        self.dw5=nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=5,padding=2,groups=self.hidden_dim)
        self.dw7=nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=7,padding=3,groups=self.hidden_dim)
        self.pw=nn.Conv2d(self.hidden_dim,self.hidden_dim,kernel_size=1)

    def forward(self,x):
        residual=x
        x=self.down_proj(self.ln(x)*self.s1+x*self.s2)
        x = x.permute(0, 3, 1, 2)  
        x=x+(self.dw3(x)+self.dw5(x)+self.dw7(x))/3
        x=x+self.pw(x)
        x = x.permute(0, 2, 3, 1) 
        x=self.up_proj(self.act(x))+residual
        return x


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            LayerNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class Mobile_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, se, stride,act=nn.GELU):
        super(Mobile_Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)  # Expansion convolution.
        self.norm1 = LayerNorm2d(expand_size)
        self.act1 = act()

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False) # Depthwise separable convolution.
        self.norm2 = LayerNorm2d(expand_size)
        self.act2 = act()
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)# Output 1x1 convolution.
        self.norm3= LayerNorm2d(out_size)
        self.act3 = act()

        self.skip = None
        if stride == 1 and in_size != out_size:  # Non-downsampling case.
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                LayerNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size: # Downsampling stage.
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False), # Downsampling convolution.
                LayerNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),  # Feature-channel adjustment.
                LayerNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                LayerNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.norm1(self.conv1(x)))
        out = self.act2(self.norm2(self.conv2(out)))
        out = self.se(out)
        out = self.norm3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)
    

class MobileNetV3_Large(nn.Module):
    def __init__(self, act=nn.GELU):
        super(MobileNetV3_Large, self).__init__()

        self.stage1=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
             LayerNorm2d(16),
             act(),
             Mobile_Block(3, 16, 64, 16, True, 1),
            Mobile_Block(3, 16, 64, 24, True, 1)
        )
        self.stage2=nn.Sequential(
            Mobile_Block(3, 24, 96, 24, True, 2),
            Mobile_Block(3, 24, 96, 40,  True, 1),
            Mobile_Block(3, 40, 160, 40, True, 1),
            Mobile_Block(3, 40, 160, 40,  True, 1)
        )
        self.stage3=nn.Sequential(
            Mobile_Block(3, 40, 160, 80,  False, 2),
            Mobile_Block(3, 80, 320, 80,  False, 1),
            Mobile_Block(3, 80, 320, 80, False, 1),
            Mobile_Block(3, 80, 320, 80, False, 1),
            Mobile_Block(3, 80, 320, 112, True, 1),
            Mobile_Block(3, 112, 448, 112, True, 1)
        )

        self.stage4=nn.Sequential(
            Mobile_Block(3, 112, 448, 160,  True, 2),
            Mobile_Block(3, 160, 640, 160,  True, 1),
            Mobile_Block(3, 160, 640, 160,  True, 1)
        )


    def forward(self, x):
        out1=self.stage1(x)
        out2=self.stage2(out1)
        out3=self.stage3(out2)
        out4=self.stage4(out3)
        return [out1,out2,out3,out4]

class Alignment(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels,out_channels)  # Map channels to 1280.
        self.act=nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((32, 32))  # Downsample to 64x64.
    def forward(self, x):
        # x: B, C, H, W
        x = self.pool(x)                     # B, C, 64, 64
        x = x.permute(0, 2, 3, 1).contiguous()         # B, 64, 64, C
        x = self.act(self.linear(x))                   # B, 64, 64, 1280
        return x



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob       = 1 - drop_prob
    shape           = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor   = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class Mlp_2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1    = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act    = act_layer()
        
        self.fc2    = nn.Linear(hidden_features, out_features)
        
        self.drop   = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

   
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        a = self.conv1(a)
        return x*self.sigmoid(a)
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # Input format: [batch_size, C, H, W].
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x*self.sigmoid(out)
 
# Spatial attention was removed for the basic benchmark.
class channel_and_normal_spatial_attention(nn.Module):
    def __init__(self, in_planes, kernel_size=7,ratio=16):
        super(channel_and_normal_spatial_attention, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        x=self.sa(self.ca(x))
        return x

class Efficient_Multihead_Crossattention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim        = dim
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q          = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr     = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm   = nn.LayerNorm(dim)
        self.kv         = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim)
        self.proj_drop  = nn.Dropout(proj_drop)

    def forward(self,x1,x2,H,W): #x1——>q, x2——>k,v
        B,N,C=x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x2.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = x_[0], x_[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class cross_branch_fusion(nn.Module):
    #Input shape: x1:x2:[B,H,W,C]
    #Output shape: [B,H,W,C]
    def __init__(self,dim, num_heads,dr_ratio=4,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 CBAM_kernel=3,mlp_ratio=4,act_layer=nn.GELU,drop=0.,drop_path_rate=0.):
        super(cross_branch_fusion,self).__init__()
        down_dim=dim//dr_ratio
        self.down1=nn.Sequential(
            nn.Linear(dim,down_dim),
            nn.GELU()
        )
        self.up= nn.Sequential(
            nn.Linear(down_dim,dim),
            nn.GELU()
        )


        self.norm1=nn.LayerNorm(down_dim)
        self.norm12=nn.LayerNorm(down_dim)
        self.attention=Efficient_Multihead_Crossattention(dim=down_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=sr_ratio)
        self.CBAM=channel_and_normal_spatial_attention(in_planes=down_dim,kernel_size=CBAM_kernel)
        self.norm2=nn.LayerNorm(down_dim)
        self.mlp=Mlp_2(in_features=down_dim,hidden_features=int(down_dim*mlp_ratio),act_layer=act_layer,drop=drop)
        self.drop_path  = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self,x1,x2):
        residual=x1
        x1=self.down1(x1)
        B,H,W,C=x1.shape
        x1=x1.view(B,H*W,C)+x2.view(B,H*W,C)+self.CBAM(self.drop_path(self.attention(self.norm1(x1).view(B,H*W,C),self.norm12(x2).view(B,H*W,C),H,W)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()).flatten(2).transpose(1,2)
        x1=x1+self.drop_path(self.mlp(self.norm2(x1),H,W))
        return residual + self.up(x1).reshape(B,H,W,-1).contiguous()
    

class twoway_cross_branch_fusion(nn.Module):
    #Input shape: x1:x2:[B,H,W,C]
    #Output shape: [B,H,W,C]
    def __init__(self,dim, num_heads,dr_ratio=4,qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 CBAM_kernel=3,mlp_ratio=4,act_layer=nn.GELU,drop=0.,drop_path_rate=0.):
        super(twoway_cross_branch_fusion,self).__init__()
        down_dim=dim//dr_ratio
        # self.down2=nn.Linear(dim,down_dim)
        self.down1=nn.Sequential(
            nn.Linear(dim,down_dim),
            nn.GELU()
        )
        self.up= nn.Sequential(
            nn.Linear(down_dim,dim),
            nn.GELU()
        )


        self.norm11=nn.LayerNorm(down_dim)
        self.norm12=nn.LayerNorm(down_dim)
        self.attention1=Efficient_Multihead_Crossattention(dim=down_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=sr_ratio)
        self.attention2=Efficient_Multihead_Crossattention(dim=down_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,
                    attn_drop=attn_drop,proj_drop=proj_drop,sr_ratio=sr_ratio)
        self.CBAM1=channel_and_normal_spatial_attention(in_planes=down_dim,kernel_size=CBAM_kernel)
        self.CBAM2=channel_and_normal_spatial_attention(in_planes=down_dim,kernel_size=CBAM_kernel)
        self.norm21=nn.LayerNorm(down_dim)
        self.norm22=nn.LayerNorm(down_dim)
        self.mlp1=Mlp_2(in_features=down_dim,hidden_features=int(dim*mlp_ratio),act_layer=act_layer,drop=drop)
        self.mlp2=Mlp_2(in_features=down_dim,hidden_features=int(dim*mlp_ratio),act_layer=act_layer,drop=drop)
        self.drop_path  = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self,x1,x2):
        x1=self.down1(x1)
        # x2=self.down2(x2)
        B,H,W,C=x1.shape
        x1=x1.view(B,H*W,C)+self.CBAM1(self.drop_path(self.attention1(self.norm11(x1).view(B,H*W,C),self.norm12(x2).view(B,H*W,C),H,W)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()).flatten(2).transpose(1,2)
        x1=x1+self.drop_path(self.mlp1(self.norm21(x1),H,W))

        x2=x2.view(B,H*W,C)+self.CBAM2(self.drop_path(self.attention2(self.norm12(x2).view(B,H*W,C),self.norm11(x1).view(B,H*W,C),H,W)).reshape(B,H,W,-1).permute(0,3,1,2).contiguous()).flatten(2).transpose(1,2)
        x2=x2+self.drop_path(self.mlp2(self.norm22(x2),H,W))


        return (self.up(x1+x2)).reshape(B,H,W,-1).contiguous()


class HFCTune(nn.Module):
    def __init__(self, edge_chans, target_dim):
        super().__init__()
        self.target_dim = target_dim
        self.proj = nn.Sequential(
            nn.Conv2d(edge_chans, self.target_dim, kernel_size=1),  # Equivalent to an MLP.
            LayerNorm2d(self.target_dim),  # Needs reshape afterward.
            nn.GELU()
        )

    def forward(self, x):
        # Assume input size is B x C x H x W.
        B, C, H, W = x.shape
        # Downsample to H/16 x W/16.
        x = F.interpolate(x, size=(H // 16, W // 16), mode='bilinear', align_corners=False)
        x = self.proj(x)  # Project channels.
        return x
