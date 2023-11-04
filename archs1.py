import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch

import torchvision.transforms.functional as TF
import os
import matplotlib.pyplot as plt
from utils import *
__all__ = ['mcafnet']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb

##### Double Convs
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


###### Attention Block
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
            x_shift = [ torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
            x_cat = torch.cat(x_shift, 1)
            x_cat = torch.narrow(x_cat, 2, self.pad, H)
            x_cat = torch.narrow(x_cat, 3, self.pad, W)
            return x_cat

class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5, shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
#     def shift(x, dim):
#         x = F.pad(x, "constant", 0)
#         x = torch.chunk(x, shift_size, 1)
#         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
#         x = torch.cat(x, 1)
#         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)


        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_r = x_s.transpose(1,2)


        x = self.fc1(x_shift_r)

        x = self.dwconv(x, H, W)
        x = self.act(x) 
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad) , "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad+1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = x_s.reshape(B,C,H*W).contiguous()
        x_shift_c = x_s.transpose(1,2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x



class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.5, attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()


        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
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

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MCAFNet(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.5,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.nb_filter = nb_filter

        self.conv1_0 = DoubleConv(nb_filter[0]*2, nb_filter[0])
        self.conv2_0 = DoubleConv(nb_filter[1]*2, nb_filter[1])
        self.conv3_0 = DoubleConv(nb_filter[2]*2, nb_filter[2])
        self.conv4_0 = DoubleConv(nb_filter[3]*2, nb_filter[3])

        self.conv11_0 = DoubleConv(in_chans, nb_filter[0])
        self.conv12_0 = DoubleConv(in_chans, nb_filter[0])
        self.conv13_0 = DoubleConv(in_chans, nb_filter[0])
        self.conv14_0 = DoubleConv(in_chans, nb_filter[0])

        self.conv22_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv23_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv24_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv33_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv34_0 = DoubleConv(nb_filter[1], nb_filter[2])

        self.conv44_0 = DoubleConv(nb_filter[2], nb_filter[3])

        self.Attdw1 = Attention_block(F_g=nb_filter[0], F_l=nb_filter[0], F_int=int(nb_filter[0] / 2))
        self.Attdw2 = Attention_block(F_g=nb_filter[1], F_l=nb_filter[1], F_int=nb_filter[0])
        self.Attdw3 = Attention_block(F_g= nb_filter[2], F_l=nb_filter[2], F_int=nb_filter[1])
        self.Attdw4 = Attention_block(F_g= nb_filter[3], F_l=nb_filter[3], F_int=nb_filter[2])

        
        self.encoder1 = DoubleConv(3, 32)
        self.encoder2 = DoubleConv(32, 64)
        self.encoder3 = DoubleConv(64, 128)
        self.encoder4 = DoubleConv(128, 256)
        self.encoder5 = DoubleConv(256, 512)

        self.ebn1 = nn.BatchNorm2d(32)
        self.ebn2 = nn.BatchNorm2d(64)
        self.ebn3 = nn.BatchNorm2d(128)
        self.ebn4 = nn.BatchNorm2d(256)
        self.ebn5 = nn.BatchNorm2d(512)
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(256)
        self.dnorm4 = norm_layer(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = DoubleConv(512, 256)
        self.decoder2 = DoubleConv(256, 128)
        self.decoder3 = DoubleConv(128, 64)
        self.decoder4 = DoubleConv(64, 32)
        self.decoder5 = DoubleConv(32, 32)

        self.dbn1 = nn.BatchNorm2d(256)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(64)
        self.dbn4 = nn.BatchNorm2d(32)

        self.a1_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.a1_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.a1_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.a1_conv4 = BasicConv2d(64, 1, kernel_size=1)

        self.a2_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.a2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.a2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.a2_conv4 = BasicConv2d(64, 1, kernel_size=1)

        self.a3_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.a3_conv2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.a3_conv3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.a3_conv4 = BasicConv2d(32, 1, kernel_size=1)

        self.a4_conv1 = BasicConv2d(32, 16, kernel_size=1)
        self.a4_conv2 = BasicConv2d(16, 16, kernel_size=3, padding=1)
        self.a4_conv3 = BasicConv2d(16, 16, kernel_size=3, padding=1)
        self.a4_conv4 = BasicConv2d(16, 1, kernel_size=1)
        
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        ### Images
        image_size = x.shape
        Images = []
        divsize = [2,4,8,16]
        for i in range(len(self.nb_filter)-1):
            Images.append(TF.resize(x, size=[int(image_size[2]/divsize[i]) , int(image_size[3]/divsize[i])]))

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2)) #c=32
        t1 = out
        ### MI1
        x11_0 = self.conv11_0(Images[0]) #32
        t1 = self.conv1_0(torch.cat((x11_0, t1), dim=1)) #32
        out = t1

        ### Stage 2
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2)) #c=64
        t2 = out
        ### MI2
        x12_0 = self.conv12_0(Images[1]) #32
        x22_0 = self.conv22_0(x12_0)   #64
        t2 = self.conv2_0(torch.cat((x22_0, t2), dim=1)) #64
        out = t2


        ### Stage 3
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2)) #c=128
        t3 = out
        ### MI3
        x13_0 = self.conv13_0(Images[2])  # 32
        x23_0 = self.conv23_0(x13_0)  # 64
        x33_0 = self.conv33_0(x23_0)  # 128
        t3 = self.conv3_0(torch.cat((x33_0, t3), dim=1))  # 128
        out = t3

        ### Tokenized MLP Stage
        ### Stage 4

        out,H,W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out

        ### MI4
        x14_0 = self.conv14_0(Images[3])  #32
        x24_0 = self.conv24_0(x14_0)    #64
        x34_0 = self.conv34_0(x24_0)    #128
        x44_0 = self.conv44_0(x34_0)    #256
        t4 = self.conv4_0(torch.cat((x44_0, t4),dim=1))   #256
        out = t4


        ### Bottleneck

        out ,H,W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        s1 = out
        #attention 1
        out = torch.sigmoid(out)
        out = out.expand(-1, 256, -1, -1).mul(t4)
        out = self.a1_conv1(out)
        out = F.relu(self.a1_conv2(out))
        out = F.relu(self.a1_conv3(out))
        out = self.a1_conv4(out)
        out = out + s1

        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        s2 = out
        #attention 2
        out = torch.sigmoid(out)
        out = out.expand(-1, 128, -1, -1).mul(t3)
        out = self.a2_conv1(out)
        out = F.relu(self.a2_conv2(out))
        out = F.relu(self.a2_conv3(out))
        out = self.a2_conv4(out)
        out = out + s2

        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        s3 = out
        #attention 3
        out = torch.sigmoid(out)
        out = out.expand(-1, 64, -1, -1).mul(t2)
        out = self.a3_conv1(out)
        out = F.relu(self.a3_conv2(out))
        out = F.relu(self.a3_conv3(out))
        out = self.a3_conv4(out)
        out = out + s3

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        s4 = out
        #attention 4
        out = torch.sigmoid(out)
        out = out.expand(-1, 32, -1, -1).mul(t1)
        out = self.a4_conv1(out)
        out = F.relu(self.a4_conv2(out))
        out = F.relu(self.a4_conv3(out))
        out = self.a4_conv4(out)
        out = out + s4

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        return self.final(out)

#EOF
