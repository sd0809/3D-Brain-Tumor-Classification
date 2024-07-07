'''
Configs for training & testing
Written by Whalechen
'''

import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="DDP")
    
    parser.add_argument(
        '--task',
        type=str,
        help='(tumor_classification | molecular_classification)'
    )
    
    parser.add_argument(
        '--data_path',
        default="/data/sd0809/TianTanData/data",
        type=str,
        help='Root directory path of data')
    
    parser.add_argument(
        '--mask_path',
        default='/data/sd0809/TianTanData/nnUNet_mask',
        type=str,
        help='Root directory path of data')
    
    parser.add_argument(
        '--hdf5_path',
        default="/data/sd0809/TianTanData/data_skull_unreg.hdf5",
        type=str,
        help='Root directory path of data')
    
    parser.add_argument(
        '--pretrain_path',
        # default= '/data/sd0809/Pretrain_Weight/ResNet3D/resnet_10_23dataset.pth',
        type=str,
        help='Path for pretrained model.')
    
    parser.add_argument(
        '--seg_loss_fn',
        default= 'Dice',
        type=str,
        help='loss fn for segmentation.')
    
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Run or not.")
    
    parser.add_argument(
        '--modality',
        nargs='+',
        type=str,
        default='T2_Ax',
        help='modality needed')
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续进")
    
    parser.add_argument(
        '--drop_path_rate',  # set to 0.001 when finetune
        default=0.1,
        type=float,
        help='drop path rate')

    parser.add_argument(
        '--crop_H',
        default=256,
        type=int,
        help='Input size of depth')

    parser.add_argument(
        '--crop_W',
        default=256,
        type=int,
        help='Input size of height')

    parser.add_argument(
        '--crop_D',
        default=24,
        type=int,
        help='Input size of width')

    parser.add_argument(
        '--start_epoch',
        default=1,
        type=int,
        help='start epoch')
    
    parser.add_argument(
        '--val_every',
        default=5,
        type=int,
        help='val every epoch')
    
    parser.add_argument(
        '--warmup_epochs',
        default=30,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument(
        '--cooldown_epochs',
        default=10,
        type=int,
        help='Number of total epochs to run')

    parser.add_argument(
        "--loss_weight",
        action="store_true",
        help="Run or not.")
    
    parser.add_argument(
        '--optim',  # 分割任务的输出类别 WT, TC, ET
        default= 'adam',
        type=str,
        help='( sgd | adam | ...) '
    )

    parser.add_argument(
        '--sched',  # 分割任务的输出类别 WT, TC, ET
        default= 'ExponentialLR',
        type=str,
        help='( LambdaLR | StepLR | ExponentialLR | ReduceLROnPlateau ) '
    )

    parser.add_argument(
        '--learning_rate',  # set to 0.001 when finetune
        default=1e-3,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    
    parser.add_argument(
        '--warmup_lr',  # set to 0.001 when finetune
        default=1e-7,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    
    parser.add_argument(
        '--min_lr',  # set to 0.001 when finetune
        default=1e-6,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    
    parser.add_argument(
        '--learning_rate_fate', 
        type=float,
        default=1e-2)
    
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=1e-3)
    
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int,
        help='Number of jobs')
    
    parser.add_argument(
        '--batch_size', 
        default=64,
        type=int, 
        help='Batch Size')

    parser.add_argument(
        '--epochs',
        default=200,
        type=int,
        help='Number of total epochs to run')

    # model para
    # parser.add_argument(
    #     "--cbam",
    #     action="store_true",
    #     help="Run or not.")
    
    # parser.add_argument(
    #     '--cbam_layer',
    #     nargs='+',
    #     type=str,
    #     help='layers needs cbam block')
    
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')

    parser.add_argument(
        '--model_dim',
        default='2d',
        type=str,
        help='(2d | 3d')

    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (18 | 34 | 50 | 101)')

    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')

    parser.add_argument(
        '--num_classes',
        default=4,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument(
        '--n_seg_classes',
        default=1,
        type=int,
        help='Number of total epochs to run')
    
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)

    parser.add_argument(
        '--gpu_id',
        nargs='+',
        type=int,              
        help='Gpu id lists')

    parser.add_argument(
        '--save_folder', default="/data1/sd0809/output/cla/test", type=str, help='path to save model')

    parser.add_argument(
        '--valid_seed', default=1, type=int, help='Manually set random seed')
    
    parser.add_argument(
        '--test_seed', default=1, type=int, help='Manually set random seed')

    parser.add_argument(
        '--manual_seed', default=42, type=int, help='Manually set random seed')

    args = parser.parse_args()
    
    return args