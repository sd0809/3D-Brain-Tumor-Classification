from grpc import insecure_channel
import torch
from torch import nn
from models import resnet


def generate_model(opt, channel):
    assert opt.model in [
        'resnet'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        
        if opt.model_depth == 10:
            model = resnet.resnet3d_10(
                in_channel= channel,
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 18:
            model = resnet.resnet3d_18(
                in_channel= channel,
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 34:
            model = resnet.resnet3d_34(
                in_channel= channel,
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 50:
            model = resnet.resnet3d_50(
                in_channel= channel,
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 101:
            model = resnet.resnet3d_101(
                in_channel= channel,
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 152:
            model = resnet.resnet3d_152(
                in_channel= channel,
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
        elif opt.model_depth == 200:
            model = resnet.resnet3d_200(
                in_channel= channel,
                sample_input_W=opt.crop_W,
                sample_input_H=opt.crop_H,
                sample_input_D=opt.crop_D,
                shortcut_type=opt.resnet_shortcut,
                no_cuda=opt.no_cuda,
                num_seg_classes=opt.n_seg_classes)
    
    return model