import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Sequence, Tuple, Union
from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks import UnetOutBlock, UnetBasicBlock, UnetUpBlock
# from models.ecb import ECB, ECB_sobel, ECB_lap

class UpBlock_light(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.norm = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )


    def forward(self, inp):
        # number of channels for skip should equals to out_channels
        inp = self.norm(inp)
        out = self.transp_conv(inp)
        return out


class UnetrUpBlock_light(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.norm = get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=in_channels)
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        self.conv_block = get_conv_layer(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_only=True,
            is_transposed=False,
        )


    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        inp = self.norm(inp)
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class UNet_encoder(nn.Module):
    def __init__(self, in_chans, base_channels=32, norm_name='Batch', act_name='relu'):
        super().__init__()
        self.in_channels = in_chans
        self.n_channels = base_channels

        self.enc0 = UnetBasicBlock(spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.n_channels,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            act_name=act_name) # (4, 128, 128, 128) --> (32, 64, 64, 64)
        self.enc1 = UnetBasicBlock(spatial_dims=3,
            in_channels=self.n_channels,
            out_channels=2 * self.n_channels,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            act_name=act_name) # (32, 64, 64, 64) --> (64, 32, 32, 32)
        self.enc2 = UnetBasicBlock(spatial_dims=3,
            in_channels=2 * self.n_channels,
            out_channels=4 * self.n_channels,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            act_name=act_name) # (64, 32, 32, 32) --> (128, 16, 16, 16)
        self.enc3 = UnetBasicBlock(spatial_dims=3,
            in_channels=4 * self.n_channels,
            out_channels=8 * self.n_channels,
            kernel_size=3,
            stride=2,
            norm_name=norm_name,
            act_name=act_name) # (128, 16, 16, 16) --> (256, 8, 8, 8)
 
    def forward(self, x):
        
        intmd_output = {}
        x0 = self.enc0(x)
        intmd_output[0] = x0
        x1 = self.enc1(x0)
        intmd_output[1] = x1
        x2 = self.enc2(x1)
        intmd_output[2] = x2
        x3 = self.enc3(x2)
        intmd_output[3] = x3
        
        return x3, intmd_output

    
class UNet_decoder(nn.Module):
    def __init__(self, spatial_dims, out_channels, base_channels=32, norm_name='Batch', act_name='relu'):
        super().__init__()
        self.n_classes = out_channels
        self.n_channels = base_channels
        
        
        self.dec0 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * self.n_channels,
            # out_channels=4 * self.n_channels,
            out_channels=320,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name
        )
        self.dec1 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=320,
            # in_channels=4 * self.n_channels,
            out_channels=2 * self.n_channels,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name
        )
        self.dec2 = UnetUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * self.n_channels,
            out_channels=1 * self.n_channels,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            act_name=act_name
        )
        
        self.up = UpBlock_light(
            spatial_dims=spatial_dims,
            in_channels=1 * self.n_channels,
            out_channels=self.n_channels // 2,
            kernel_size=7, 
            upsample_kernel_size=4,
            norm_name=norm_name
        )
        
    def forward(self, intmd_output):
        
        x = intmd_output[3]
        # print("x shape:", x.shape)
        # print("intmd_output shape:", intmd_output[2].shape)
        x = self.dec0(x, intmd_output[2])
        x = self.dec1(x, intmd_output[1])
        x = self.dec2(x, intmd_output[0])
        x = self.up(x)

        return x


if __name__ == "__main__":

    enc = UNet_encoder(in_chans=4)
    
    enc = enc.cuda()

    pytorch_total_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total parameters count: {pytorch_total_params:.2f}M.")

    img = torch.rand([1, 4, 128, 128, 128]).cuda()
    output, intmd_output= enc(img)
    
    dec = UNet_decoder(out_channels=3)
    dec = dec.cuda()
    out = dec(intmd_output)

    print(out.shape)