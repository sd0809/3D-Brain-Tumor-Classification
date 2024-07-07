import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.metaformer_2d import convformer_s8, convformer_s12, caformer_s8, caformer_s12, caformer_s18, caformer_s18_384
from models.metaformer_3d import convformer_encoder, caformer_encoder

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from models.unet import UNet_encoder, UNet_decoder


class MetaFormerUnet(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, drop_path, base_channels=32, spatial_dims=3, norm_name=("instance")):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = out_channels
        self.n_channels = base_channels
        
        if spatial_dims == 2:
            # self.encoder = convformer_s8(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path)
            # self.encoder = convformer_s12(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path)
            # self.encoder = caformer_s8(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path)
            # self.encoder = caformer_s12(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path)
            self.encoder = caformer_s18(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path, pretrained=True)
            # self.encoder = caformer_s18_384(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path, pretrained=True)
        
        else:
            # self.encoder = UNet_encoder(in_chans=self.in_channels, num_classes=num_classes)
            # self.encoder = convformer_encoder(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path)
            self.encoder = caformer_encoder(in_chans=self.in_channels, num_classes=num_classes, drop_path_rate=drop_path)
        
        self.decoder = UNet_decoder(spatial_dims=spatial_dims, out_channels=self.n_classes, base_channels=64)
        
        self.out = UnetOutBlock(
            spatial_dims=spatial_dims, 
            # in_channels=self.n_channels, 
            # out_channels=self.n_classes
            in_channels=self.n_channels, 
            out_channels=self.n_classes
        )

    def forward(self, x):
        encout_x, intmd_output = self.encoder(x)
        dec_out = self.decoder(intmd_output)
        # print("dec_out shape: ", dec_out.shape)
        out = self.out(dec_out)
        return encout_x, out


if __name__ == "__main__":

    model = MetaFormerUnet_v2(in_channels=4, out_channels=3, drop_path=0.1)
    
    model = model.cuda()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total parameters count: {pytorch_total_params:.2f}M.")

    img = torch.rand([1, 4, 128, 128, 128]).cuda()
    start = time.time()
    output= model(img)
    end = time.time()
    print(output.shape)
    print(f'time consuming {(end - start)}s.')