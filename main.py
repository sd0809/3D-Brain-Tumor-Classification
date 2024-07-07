    #!/usr/bin/env python3
import os
import csv
import json
import math
import random
import argparse
import numpy as np
import pandas as pd
from settings import parse_opts
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import models
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from thop import profile

from models.resnet2d import resnet
from models.metaformer_unet import MetaFormerUnet

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from utils.utils import visualize_data_3mod, visualize_data_4mod
from dataset.dataset_hdf5 import BrainDataset, get_dataset_hdf5, get_dataset_hdf5_molecular
from trainer import train, evaluate, evaluate_and_plot

from monai.losses import FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss
from monai.transforms import Resize, CenterSpatialCrop, RandSpatialCrop, RandGaussianNoise, RandFlip, RandAxisFlip, Rotate, RandShiftIntensity, Randomizable, Compose
from monai.utils import set_determinism
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    ToTensord,
    RandGaussianNoised
)

def init_seeds(manual_seed):
    # 实验可重复
    set_determinism(seed=manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.cuda.manual_seed(manual_seed)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(manual_seed)   # 为所有GPU设置随机种子（多块GPU)  
    

def main(args):
    
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group('nccl', world_size=len(args.gpu_id), rank=local_rank)
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(args.manual_seed+local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(args.manual_seed)
    
    # 读取分子亚型信息
    df = pd.read_excel('./molecular.xlsx',sheet_name='Sheet1')
    molecular_info_dict={} # patientID : molecular
    for i in range(0, len(df.index)):
        if isinstance(df.values[i, 1], str) and '/' not in df.values[i, 1]:
            if df.values[i, 1].startswith('S') or df.values[i, 1].startswith('W') or df.values[i, 1].startswith('G'):
                if (df.values[i, 1] == 'SHH' or df.values[i, 1] == 'WNT' or df.values[i, 1] == 'G3' or df.values[i, 1] == 'G4'):
                    molecular_info_dict[str(df.values[i, 0])] = df.values[i, 1]
        else:
            continue

    # 读取肿瘤切片信息(nnUnet_mask)
    with open ('./max_slice_info_annotations.txt', 'r') as file:
        max_slice_info = file.read()
        max_slice_info_dict = eval(max_slice_info) # patientID : tumor_max_area_slice
    file.close()
    
    # 读取肿瘤切片信息(nnUnet_mask)
    # with open ('./slices_info.txt', 'r') as file:
    with open ('./max_slice_info_annotations.txt', 'r') as file:
        max_slice_info = file.read()
        max_slice_info_dict = eval(max_slice_info)
    file.close()
    
    test_seed = args.test_seed
    
    # 建立保存文件目录
    save_folder = args.save_folder
    log_save_folder = save_folder + f'/fold{test_seed}/logs/'
    weight_save_folder = save_folder + f'/fold{test_seed}/weights/'
    if args.distributed:
        if local_rank == 0:
            if not os.path.exists(log_save_folder):
                os.makedirs(log_save_folder)
            if not os.path.exists(weight_save_folder):
                os.makedirs(weight_save_folder)
    else:
        if not os.path.exists(log_save_folder):
            os.makedirs(log_save_folder)
        if not os.path.exists(weight_save_folder):
            os.makedirs(weight_save_folder)
            
    modality_lst = sorted(args.modality)
    modality_num = len(modality_lst)
    
    if args.model_dim == '2d': # 2d model H W, D 
        train_crop_size = (args.crop_H, args.crop_W)
        k_divisible_ = [128, 128]
    elif args.model_dim == '3d': # 3d model H W D
        train_crop_size = (args.crop_H, args.crop_W, args.crop_D)
        k_divisible_ = [32, 32, 1]
        
    transform_tiantan = {
        'train': Compose([
            EnsureTyped(keys=["image", "mask"], allow_missing_keys=True),
            CropForegroundd(keys=["image", "mask"], source_key="image", k_divisible=k_divisible_, allow_missing_keys=True), # 裁剪出image有像素信息的区域
            RandSpatialCropd(keys=["image", "mask"], roi_size=train_crop_size, random_size=False, allow_missing_keys=True), # D, H, W
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=1, allow_missing_keys=True),
            # RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=2, allow_missing_keys=True),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandGaussianNoised(keys="image", prob=0.5, mean=0.0, std=0.1),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5), # 通过 v = v * (1 + 因子) 随机缩放输入图像的强度，其中因子是随机选取的 img*factor。
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5), # 使用随机选择的偏移量随机改变强度 img+rand_value
            # ToTensord(keys=["image", "mask"]),
            ]),
        
        'valid': Compose([
            EnsureTyped(keys=["image", "mask"], allow_missing_keys=True),
            CenterSpatialCropd(keys=["image", "mask"], roi_size=train_crop_size, allow_missing_keys=True), # D, H, W
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            # ToTensord(keys=["image", "label"]),
            ]),
        }
    
#     if args.num_classes == 2:
#         print('肿瘤二分类任务')
#         train_dataset, valid_dataset, test_dataset, train_list, valid_list, test_list = get_dataset_hdf5(
#             args,
#             args.hdf5_path,
#             args.mask_path,
#             slice_info = max_slice_info_dict,
#             modality=modality_lst,
#             transform=transform_tiantan,
#             test_seed = test_seed,
#             seed=args.manual_seed)

#         train_label = []
#         mb_train=0
#         not_mb_train=0
#         for i in range(len(train_list)):
#             train_label.append(train_list[i][2])
#             if train_list[i][2] == 0:
#                 mb_train += 1
#             else:
#                 not_mb_train +=1
#         mb_percent = mb_train / (mb_train+not_mb_train)
#         prob = [mb_percent, 1-mb_percent]
#         print(f'mb in the train set:{mb_percent:.4f} , other tumor in the train set:{1- mb_percent:.4f}.')

    if args.task == "tumor_classification":
        print('肿瘤四分类')
        labels_name = ['mb', 'choroid', 'ependymoma', 'glioma']
        train_dataset, valid_dataset, test_dataset, train_list, valid_list, test_list = get_dataset_hdf5(
            args,
            args.hdf5_path,
            args.mask_path,
            slice_info = max_slice_info_dict,
            modality=modality_lst,
            transform=transform_tiantan,
            test_seed = test_seed,
            seed=args.manual_seed)

        train_label = []
        mb_train=0
        choroid_train=0
        ependymoma_train=0
        glioma_train=0
        for i in range(len(train_list)):
            train_label.append(train_list[i][2])
            if train_list[i][2] == 0:
                mb_train += 1
            elif train_list[i][2] == 1:
                choroid_train +=1
            elif train_list[i][2] == 2:
                ependymoma_train +=1
            elif train_list[i][2] == 3:
                glioma_train +=1
        mb_percent = mb_train / (mb_train+choroid_train+ependymoma_train+glioma_train)
        choroid_percent = choroid_train / (mb_train+choroid_train+ependymoma_train+glioma_train)
        ependymoma_percent = ependymoma_train / (mb_train+choroid_train+ependymoma_train+glioma_train)
        glioma_percent = glioma_train / (mb_train+choroid_train+ependymoma_train+glioma_train)
        prob = [mb_percent, choroid_percent, ependymoma_percent, glioma_percent]
        print(f'mb in the train set:{mb_percent:.4f}, choroid in the train set:{choroid_percent:.4f}, ependymoma in the train set:{ependymoma_percent:.4f}, glioma in the train set:{glioma_percent:.4f}.')
        
    elif args.task == "molecular_classification":
        print('分子亚型四分类')
        labels_name = ['shh', 'wnt', 'g3', 'g4']
        train_dataset, valid_dataset, test_dataset, train_list, valid_list, test_list = get_dataset_hdf5_molecular(
            args.hdf5_path,
            args.mask_path,
            slice_info = max_slice_info_dict,
            molecular_info = molecular_info_dict,
            modality=modality_lst,
            molecular = ['SHH', 'WNT', 'G3', 'G4'],
            transform=transform_tiantan,
            test_seed = test_seed,
            seed=args.manual_seed)
        train_label = []
        shh_train=0
        wnt_train=0
        g3_train=0
        g4_train=0
        for i in range(len(train_list)):
            train_label.append(train_list[i][2])
            if train_list[i][2] == 0:
                shh_train += 1
            elif train_list[i][2] == 1:
                wnt_train += 1
            elif train_list[i][2] == 2:
                g3_train +=1
            else:
                g4_train += 1

        shh_percent = shh_train / len(train_label)
        wnt_percent = wnt_train / len(train_label)
        g3_percent = g3_train / len(train_label)
        g4_percent = g4_train / len(train_label)
        prob = [shh_percent, wnt_percent, g3_percent, g4_percent]
        print(f'SHH in the train set:{shh_percent*100:.2f}%, WNT in the train set:{wnt_percent*100:.2f}%, G3 in the train set:{g3_percent*100:.2f}%, G4 in the train set:{g4_percent*100:.2f}%.')


    if args.distributed:
        if local_rank == 0:
            sub0 = train_dataset[10]
            print(f'current train input', sub0['ID'], 'shape:', sub0['image'].shape)
            sub1 = valid_dataset[20]
            print(f'current valid input', sub1['ID'], 'shape:', sub1['image'].shape)
            sub2 = test_dataset[30]
            print(f'current test input', sub2['ID'], 'shape:', sub2['image'].shape)

            if modality_num == 3:
                visualize_data_3mod(sub0)
                visualize_data_3mod(sub1)
                visualize_data_3mod(sub2)
            elif modality_num == 4:
                visualize_data_4mod(sub0)
                visualize_data_4mod(sub1)
                visualize_data_4mod(sub2)
    else:
        sub0 = train_dataset[10]
        print(f'current train input', sub0['ID'], 'shape:', sub0['image'].shape)
        sub1 = valid_dataset[10]
        print(f'current valid input', sub1['ID'], 'shape:', sub1['image'].shape)
        sub2 = test_dataset[10]
        print(f'current test input', sub2['ID'], 'shape:', sub2['image'].shape)

        if modality_num == 3:
            visualize_data_3mod(sub0)
            visualize_data_3mod(sub1)
            visualize_data_3mod(sub2)
        elif modality_num == 4:
            visualize_data_4mod(sub0)
            visualize_data_4mod(sub1)
            visualize_data_4mod(sub2)

    if args.distributed:
        if local_rank == 0:
            print(' - - -'*20)
            print('device:', device)
            print(args,'\n')
            print('Start Tensorboard with "tensorboard --logdir=/runs --port=6011" ')
            tb_writer = SummaryWriter(f'{save_folder}/tb/fold{test_seed}') # tb_path保存tensorboard记录
    else:
        print(' - - -'*20)
        print('device:', device)
        print(args,'\n')
        print('Start Tensorboard with "tensorboard --logdir=/runs --port=6011" ')
        tb_writer = SummaryWriter(f'{save_folder}/tb/fold{test_seed}') # tb_path保存tensorboard记录

#     if args.resample:
#         print('resample!!!')
#         reciprocal_weights = [0 for i in range(len(train_dataset))]
#         for index in range(len(train_dataset)):
#             reciprocal_weights[index] = prob[train_label[index]]
#         weights = (1 / torch.Tensor(reciprocal_weights))
#         train_sampler = WeightedRandomSampler(weights, len(train_dataset))

#         train_loader = DataLoader(train_dataset,
#                                 batch_size=args.batch_size,
#                                 sampler=train_sampler,
#                                 pin_memory=True,
#                                 num_workers=args.num_workers,
#                                 prefetch_factor=4
#                                 )
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            sampler=train_sampler,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=4)

    else:
        train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                prefetch_factor = 4
                                )

    valid_loader = DataLoader(valid_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=4
                            )

    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=4
                            )

    if args.model_dim == '2d':
        if args.model == 'resnet':
            if args.model_depth == 18:
                model = resnet.resnet18(pretrained=True, num_classes=args.num_classes, in_channels= modality_num * args.crop_D)
            elif args.model_depth == 34:
                model = resnet.resnet34(pretrained=True, num_classes=args.num_classes, in_channels= modality_num * args.crop_D)
            elif args.model_depth == 50:
                model = resnet.resnet50(pretrained=True, num_classes=args.num_classes, in_channels= modality_num * args.crop_D)
            elif args.model_depth == 101:
                model = resnet.resnet101(pretrained=True, num_classes=args.num_classes, in_channels= modality_num * args.crop_D)
        elif args.model == 'metaformer':
            model = MetaFormerUnet(spatial_dims=2, in_channels=modality_num, out_channels=args.n_seg_classes, num_classes=args.num_classes, drop_path=args.drop_path_rate)

    elif args.model_dim == '3d':
        if args.model == 'metaformer':
            model = MetaFormerUnet(spatial_dims=3, in_channels=modality_num, out_channels=args.n_seg_classes, num_classes=args.num_classes, drop_path=args.drop_path_rate)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank])
    else:
        model = nn.DataParallel(model.to(device), device_ids = args.gpu_id)

    pg = [p for p in model.parameters() if p.requires_grad]

    # # optimizer selection
    if args.optim == 'sgd':
        optimizer =  optim.SGD(pg, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer =  optim.Adam(pg, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(pg, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    # lr scheduler selection
    if args.sched == 'LambdaLR':
        # cosine  最终降到初始学习率的learning_rate_fate倍
        lambda_cosine = lambda epoch_x: ((1 + math.cos(epoch_x * math.pi / args.epochs)) / 2) * (1 - args.learning_rate_fate) + 1e-3
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)
    elif args.sched == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.sched == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif args.sched == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', patience=10, factor=0.2)
    elif args.sched == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs)
    elif args.sched == "cosine":
        scheduler, num_epochs = create_scheduler(args, optimizer)

    # if args.loss_weight:
    #     loss_weight=torch.HalfTensor([1.8, 4, 1]).to(device)
    #     print(f'Using loss weight: {loss_weight}')
    #     criterion = nn.CrossEntropyLoss(weight = loss_weight)
    # else:
    cla_loss_fn = nn.CrossEntropyLoss()

    # loss function selection
    if args.seg_loss_fn == 'CE':
        seg_loss_fn = nn.CrossEntropyLoss()
    elif args.seg_loss_fn == 'Dice':
        seg_loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)
    elif args.seg_loss_fn == 'DiceCE':
        seg_loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
    elif args.seg_loss_fn == 'Focal':
        seg_loss_fn = FocalLoss()
    elif args.seg_loss_fn == 'DiceFocal':
        seg_loss_fn = DiceFocalLoss(to_onehot_y=False, sigmoid=True)


    start_epoch = args.start_epoch
    # 断点续训
    if args.resume:
        print('load checkpoint: ', weight_save_folder+ f'/checkpoint.pth')
        if os.path.isfile(weight_save_folder+ f'/checkpoint.pth'):
            checkpoint = torch.load(weight_save_folder+ f'/checkpoint.pth')
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['acc']
            best_auc = checkpoint['auc']
            best_f1 = checkpoint['f1']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint path: {weight_save_folder+ '/checkpoint.pth'}, (epoch {checkpoint['epoch']})")
        else:
            print('=>checkpoint not exists!')

    else:
        if args.distributed:
            if local_rank == 0:
                print("=> no checkpoint found\n")
        else:
            print("=> no checkpoint found\n")
        best_acc = 0.3 if args.epochs > 50 else 0
        best_auc = 0.3 if args.epochs > 50 else 0
        best_f1 = 0.3 if args.epochs > 50 else 0

    if args.distributed:
        if local_rank == 0:
            print(f'fold{test_seed} start!\n')
    else:
        print(f'fold{test_seed} start!\n')
    min_epoch = 0
    best_acc_epoch = 0
    best_auc_epoch = 0
    best_f1_epoch = 0

    for epoch in range(start_epoch, args.epochs+1):
        # train
        if args.distributed:
            train_sampler.set_epoch(epoch) # 为了让每张卡在每个周期中得到的数据是随机的
            train_loss, train_loss_cla, train_loss_seg, train_acc, train_f1 = train(model,  optimizer, cla_loss_fn, seg_loss_fn, train_loader, device=local_rank)  # 混合精度训练
            if local_rank == 0:
                print(f"epoch {epoch} train loss: {train_loss:.4f}, cla loss: {train_loss_cla:.4f}, seg loss: {train_loss_seg:.4f}, acc: {train_acc:.4f} and f1 score: {train_f1:.4f}.")
        else:
            train_loss, train_loss_cla, train_loss_seg, train_acc, train_f1 = train(model,  optimizer, cla_loss_fn, seg_loss_fn, train_loader, device=device)  # 混合精度训练
            print(f"epoch {epoch} train loss: {train_loss:.4f}, cla loss: {train_loss_cla:.4f}, seg loss: {train_loss_seg:.4f}, acc: {train_acc:.4f} and f1 score: {train_f1:.4f}.")
        scheduler.step()

        # validate
        if epoch % args.val_every == 0:
            if args.distributed:
                if local_rank == 0:
                    valid_loss, valid_loss_cla, valid_loss_seg, valid_acc, valid_auc, valid_f1, valid_dice = evaluate(model, cla_loss_fn, seg_loss_fn, valid_loader, labels_name=labels_name, device=local_rank)
                    print(f"epoch {epoch} valid loss: {valid_loss:.4f}, cla loss: {valid_loss_cla:.4f}, seg loss: {valid_loss_seg:.4f}, acc: {valid_acc:.4f}, dice: {valid_dice:.4f} , auc: {valid_auc:.4f} and f1 score: {valid_f1:.4f} \n")
                    if epoch > min_epoch and valid_acc > best_acc:  # 保存在验证集上当前最佳acc模型
                        best_acc = valid_acc
                        best_acc_epoch = epoch
                        valid_acc, valid_auc, valid_f1, valid_dice = evaluate_and_plot(model, valid_loader, labels_name=labels_name, device=local_rank, phase='valid', mod_num=modality_num, title= f'epoch' + str(epoch), save_path=log_save_folder)
                        torch.save({
                                    'epoch': epoch,
                                    'acc': valid_acc,
                                    'dice': valid_dice,
                                    'auc': valid_auc,
                                    'f1': valid_f1,
                                    'state_dict': model.module.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    weight_save_folder+ f'/best_acc_model.pth')
                    if epoch > min_epoch and valid_auc > best_auc:  # 保存在验证集上当前最佳auc模型
                        best_auc = valid_auc
                        best_auc_epoch = epoch
                        valid_acc, valid_auc, valid_f1, valid_dice = evaluate_and_plot(model, valid_loader, labels_name=labels_name, phase='valid', mod_num=modality_num, device=local_rank, title= f'epoch' + str(epoch),save_path=log_save_folder)
                        torch.save({
                                    'epoch': epoch,
                                    'acc': valid_acc,
                                    'dice': valid_dice,
                                    'auc': valid_auc,
                                    'f1': valid_f1,
                                    'state_dict': model.module.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    weight_save_folder+f'/best_auc_model.pth')
                    if epoch > min_epoch and valid_f1 > best_f1:  # 保存在验证集上最佳f1 socre模型
                        best_f1 = valid_f1
                        best_f1_epoch = epoch
                        valid_acc, valid_auc, valid_f1, valid_dice = evaluate_and_plot(model, valid_loader, labels_name=labels_name, phase='valid', mod_num=modality_num, device=local_rank, title= f'epoch' + str(epoch),save_path=log_save_folder)
                        torch.save({
                                    'epoch': epoch,
                                    'acc': valid_acc,
                                    'dice': valid_dice,
                                    'auc': valid_auc,
                                    'f1': valid_f1,
                                    'state_dict': model.module.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    weight_save_folder+ f'/best_f1_model.pth')
                # torch.distributed.barrier()
            else:
                valid_loss, valid_loss_cla, valid_loss_seg, valid_acc, valid_auc, valid_f1, valid_dice = evaluate(model, cla_loss_fn, seg_loss_fn, valid_loader, device=device, labels_name=labels_name)
                print(f"epoch {epoch} valid loss: {valid_loss:.4f}, cla loss: {valid_loss_cla:.4f}, seg loss: {valid_loss_seg:.4f}, acc: {valid_acc:.4f}, dice: {valid_dice:.4f} , auc: {valid_auc:.4f} and f1 score: {valid_f1:.4f} \n")
                if epoch > min_epoch and valid_acc > best_acc:  # 保存在验证集上当前最佳acc模型
                        best_acc = valid_acc
                        best_acc_epoch = epoch
                        valid_acc, valid_auc, valid_f1, valid_dice = evaluate_and_plot(model, valid_loader, labels_name=labels_name, device=device, phase='valid', mod_num=modality_num, title= f'epoch' + str(epoch), save_path=log_save_folder)
                        torch.save({
                                    'epoch': epoch,
                                    'acc': valid_acc,
                                    'dice': valid_dice,
                                    'auc': valid_auc,
                                    'f1': valid_f1,
                                    'state_dict': model.module.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    weight_save_folder+ f'/best_acc_model.pth')
                if epoch > min_epoch and valid_auc > best_auc:  # 保存在验证集上当前最佳auc模型
                    best_auc = valid_auc
                    best_auc_epoch = epoch
                    valid_acc, valid_auc, valid_f1, valid_dice = evaluate_and_plot(model, valid_loader, labels_name=labels_name, phase='valid', device=device, mod_num=modality_num, title= f'epoch' + str(epoch),save_path=log_save_folder)
                    torch.save({
                                'epoch': epoch,
                                'acc': valid_acc,
                                'dice': valid_dice,
                                'auc': valid_auc,
                                'f1': valid_f1,
                                'state_dict': model.module.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                weight_save_folder+f'/best_auc_model.pth')
                if epoch > min_epoch and valid_f1 > best_f1:  # 保存在验证集上最佳f1 socre模型
                    best_f1 = valid_f1
                    best_f1_epoch = epoch
                    valid_acc, valid_auc, valid_f1, valid_dice = evaluate_and_plot(model, valid_loader, labels_name=labels_name, phase='valid', device=device, mod_num=modality_num, title= f'epoch' + str(epoch),save_path=log_save_folder)
                    torch.save({
                                'epoch': epoch,
                                'acc': valid_acc,
                                'dice': valid_dice,
                                'auc': valid_auc,
                                'f1': valid_f1,
                                'state_dict': model.module.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                weight_save_folder+ f'/best_f1_model.pth')

            tags = ["train loss", "valid loss", "train acc" , "valid acc", "train f1 score", "valid f1 score",  "valid auc",  "learning_rate"]

            if args.distributed:
                if local_rank == 0:
                    tb_writer.add_scalar('Loss/'+tags[0], train_loss, epoch)
                    tb_writer.add_scalar('Loss/'+tags[1], valid_loss, epoch)
                    tb_writer.add_scalar('Accuracy/'+tags[2], train_acc, epoch)
                    tb_writer.add_scalar('Accuracy/'+tags[3], valid_acc, epoch)
                    tb_writer.add_scalar('F1 Score/'+tags[4], train_f1, epoch)
                    tb_writer.add_scalar('F1 Score/'+tags[5], valid_f1, epoch)
                    tb_writer.add_scalar('AUC/'+tags[6], valid_auc, epoch)
                    tb_writer.add_scalar(tags[7], optimizer.param_groups[0]["lr"], epoch)
                # torch.distributed.barrier()
            else:
                tb_writer.add_scalar('Loss/'+tags[0], train_loss, epoch)
                tb_writer.add_scalar('Loss/'+tags[1], valid_loss, epoch)
                tb_writer.add_scalar('Accuracy/'+tags[2], train_acc, epoch)
                tb_writer.add_scalar('Accuracy/'+tags[3], valid_acc, epoch)
                tb_writer.add_scalar('F1 Score/'+tags[4], train_f1, epoch)
                tb_writer.add_scalar('F1 Score/'+tags[5], valid_f1, epoch)
                tb_writer.add_scalar('AUC/'+tags[6], valid_auc, epoch)
                tb_writer.add_scalar(tags[7], optimizer.param_groups[0]["lr"], epoch)
    
    if args.distributed:
        if local_rank == 0:
            print(f'best acc in validation set:{best_acc:.4f}, in epoch: {best_acc_epoch}, best auc in validation set:{best_auc:.4f}, in epoch: {best_auc_epoch}, best f1 in validation set:{best_f1:.4f}, in epoch: {best_f1_epoch}\n')
            print(' - - - - - test phase - - - - - ')
        torch.distributed.barrier()
    else:
        print(f'best acc in validation set:{best_acc:.4f}, in epoch: {best_acc_epoch}, best auc in validation set:{best_auc:.4f}, in epoch: {best_auc_epoch}, best f1 in validation set:{best_f1:.4f}, in epoch: {best_f1_epoch}\n')
        print(' - - - - - test phase - - - - - ')
    
    
    
    best_acc_model_path = weight_save_folder+ f'/best_acc_model.pth'
    best_auc_model_path = weight_save_folder+ f'/best_auc_model.pth'
    best_f1_model_path = weight_save_folder+ f'/best_f1_model.pth'
    assert os.path.exists(best_acc_model_path), "cannot find {} file".format(best_acc_model_path)
    assert os.path.exists(best_auc_model_path), "cannot find {} file".format(best_auc_model_path)
    assert os.path.exists(best_f1_model_path), "cannot find {} file".format(best_f1_model_path)
    
    if args.distributed:
        if local_rank==0:
            with open(f"{save_folder}/log.csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                model.module.load_state_dict(torch.load(best_acc_model_path)['state_dict'], strict=False)
                test_acc1, test_auc1, test_f1_1, test_dice_1 = evaluate_and_plot(model, test_loader, labels_name=labels_name, phase='test', device=local_rank, mod_num=modality_num, title= f'test best acc model',save_path=f'{log_save_folder}')
                print(f'best acc model performance in test set: test acc:{test_acc1:.4f}, test dice:{test_dice_1:.4f}, test auc:{test_auc1:.4f}, test_f1:{test_f1_1:.4f}\n')

                model.module.load_state_dict(torch.load(best_auc_model_path)['state_dict'], strict=False)
                test_acc2, test_auc2, test_f1_2, test_dice_2 = evaluate_and_plot(model, test_loader, labels_name=labels_name, phase='test', device=local_rank, mod_num=modality_num, title= f'test best auc model',save_path=f'{log_save_folder}')
                print(f'best auc model performance in test set: test acc:{test_acc2:.4f}, test dice:{test_dice_2:.4f}, test auc:{test_auc2:.4f}, test_f1:{test_f1_2:.4f}\n')

                model.module.load_state_dict(torch.load(best_f1_model_path)['state_dict'], strict=False)
                test_acc3, test_auc3, test_f1_3, test_dice_3 = evaluate_and_plot(model, test_loader, labels_name=labels_name, phase='test', device=local_rank, mod_num=modality_num,title= f'test best f1 model',save_path=f'{log_save_folder}')
                print(f'best f1 model performance in test set: test acc:{test_acc3:.4f}, test dice:{test_dice_3:.4f}, test auc:{test_auc3:.4f}, test_f1:{test_f1_3:.4f}\n')
                
                writer.writerow(['best valid acc', ' ', 'acc model', ' ', ' ', ' ', 'auc model', ' ', ' ', ' ', 'f1 model', ' ', ' ' ])
                writer.writerow([best_acc, ' ', test_acc1, test_auc1, test_f1_1, ' ', test_acc2, test_auc2, test_f1_2, ' ', test_acc3, test_auc3, test_f1_3])
                csvfile.close()
        torch.distributed.barrier()
    else:
        with open(f"{save_folder}/log.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            model.module.load_state_dict(torch.load(best_acc_model_path)['state_dict'], strict=False)
            test_acc1, test_auc1, test_f1_1, test_dice_1 = evaluate_and_plot(model, test_loader, labels_name=labels_name, phase='test', device=device, mod_num=modality_num, title= f'test best acc model',save_path=f'{log_save_folder}')
            print(f'best acc model performance in test set: test acc:{test_acc1:.4f}, test dice:{test_dice_1:.4f}, test auc:{test_auc1:.4f}, test_f1:{test_f1_1:.4f}\n')

            model.module.load_state_dict(torch.load(best_auc_model_path)['state_dict'], strict=False)
            test_acc2, test_auc2, test_f1_2, test_dice_2 = evaluate_and_plot(model, test_loader, labels_name=labels_name, phase='test', device=device, mod_num=modality_num, title= f'test best auc model',save_path=f'{log_save_folder}')
            print(f'best auc model performance in test set: test acc:{test_acc2:.4f}, test dice:{test_dice_2:.4f}, test auc:{test_auc2:.4f}, test_f1:{test_f1_2:.4f}\n')

            model.module.load_state_dict(torch.load(best_f1_model_path)['state_dict'], strict=False)
            test_acc3, test_auc3, test_f1_3, test_dice_3 = evaluate_and_plot(model, test_loader, labels_name=labels_name, phase='test', device=device, mod_num=modality_num,title= f'test best f1 model',save_path=f'{log_save_folder}')
            print(f'best f1 model performance in test set: test acc:{test_acc3:.4f}, test dice:{test_dice_3:.4f}, test auc:{test_auc3:.4f}, test_f1:{test_f1_3:.4f}\n')

            writer.writerow(['best valid acc', ' ', 'acc model', ' ', ' ', ' ', 'auc model', ' ', ' ', ' ', 'f1 model', ' ', ' ' ])
            writer.writerow([best_acc, ' ', test_acc1, test_auc1, test_f1_1, ' ', test_acc2, test_auc2, test_f1_2, ' ', test_acc3, test_auc3, test_f1_3])
            csvfile.close()
        

if __name__ == '__main__':

    opt = parse_opts()

    # 实验可重复
    seed = opt.manual_seed
    set_determinism(seed=seed)  # monai reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)      # 为当前GPU设置随机种子（只用一块GPU)
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子（多块GPU)

    main(opt)