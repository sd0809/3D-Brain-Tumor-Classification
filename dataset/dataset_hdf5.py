import os
import h5py
import numpy as np
import random
from PIL import Image
import logging
import cv2
import torch 
from torch.utils.data import Dataset

from typing import List, Tuple, Dict
from collections import defaultdict


from monai.transforms import Resize, Spacing
from utils.utils import get_array_from_nifti, get_max_slice_index
from skimage import transform

_logger = logging.getLogger('vis')

resize_size = (256, 256)  # input shape must be (depth, H, W)
# resize_size = (512, 512)  # input shape must be (depth, H, W)
monai_resize = Resize(spatial_size=resize_size)
monai_spacing = Spacing(pixdim=(1.0, 1.0, 1.0))

class BrainDataset(Dataset):
    """ define dataset """
    def __init__(self, 
                 data_list: List[Tuple[np.ndarray, int]],
                 mean,
                 std,
                 train: bool = False,
                 transform=None,
                 ):
        super().__init__()
        self.data_list = data_list
        self.train = train
        self.transform=transform
        self.mean = mean
        self.std = std
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        ID, array, label, mask = self.data_list[index]  # [H, W, C]
        # 3d
        # mask = np.expand_dims(mask, axis=0)
        if self.train:
            if self.transform is not None:
                dict_ = self.transform({'image': array, 'mask': mask})
                # dict_ = self.transform({'image': array})
        else:
            if self.transform is not None:
                dict_ = self.transform({'image': array, 'mask': mask})
                # dict_ = self.transform({'image': array})
        
        dict_['ID'] = ID
        dict_['label'] = label
        dict_['image'] = dict_['image'].to(torch.float32)
        dict_['mask'] = dict_['mask'].to(torch.float32)
        return dict_


def get_dataset_hdf5(
                  args,
                  hdf5_path: str, 
                  mask_dir: str,
                  slice_info,
                  modality: List[str] = ['T1_Ax' , 'T1_E_Ax', 'T2_Ax', 'T2_Flair_Ax'], 
                  tumor: List[str] = ['choroid','ependymoma','glioma', 'mb'],
                  test_ratio: float = 0.2,
                  transform=None,
                  seed: int=42,
                  test_seed: int=1,
                  size=(256,256)
                 )-> Tuple[Dataset, Dataset]:
    
    assert hdf5_path.endswith('.hdf5'), '{} does not ends with .hdf5'.format(hdf5_path)
    _logger.info('Load file {}'.format(hdf5_path))
    hdf5_file = h5py.File(hdf5_path, 'r')
    doc = hdf5_file.attrs['doc']
    _logger.info('Infos: {}'.format(doc))
    available_modalities = hdf5_file.attrs['modalities'].split('+')
    for m in modality:
        assert m in available_modalities, 'modality {} is not supported, maybe {}'.format(m, available_modalities)
    ''' load mean and std '''
    mean_per_modality = {m: hdf5_file.attrs['mean_{}'.format(m)].astype(np.float16) for m in modality}
    std_per_modality = {m: hdf5_file.attrs['std_{}'.format(m)].astype(np.float16) for m in modality}
    _logger.info('Pixel mean of each modality: {}'.format(mean_per_modality))
    _logger.info('Pixel std of each modality: {}'.format(std_per_modality))
    mean = []
    std = []
    for m in modality:
        mean.append(hdf5_file.attrs['mean_{}'.format(m)].astype(np.float16))
        std.append(hdf5_file.attrs['std_{}'.format(m)].astype(np.float16))
    print(f'mean per modality: {mean_per_modality}, mean value: {mean}' )
    print(f'std per modality: {std_per_modality}, std value: {std}' )
    
    ''' create info list '''
    info_list = []
    tumor_label = -1
    patient_num = defaultdict(int)
    tumor_idx = []
    info_list = defaultdict(list)
    for tumor_type, tumor_grp in hdf5_file.items():
        if args.num_classes == 2:
            if tumor_type == 'mb':
                tumor_label = 0
            else:
                tumor_label = 1
        elif args.num_classes == 4:
            if tumor_type in tumor:
                if tumor_type == 'mb':
                    tumor_label = 0
                elif tumor_type == 'choroid':
                    tumor_label = 1
                elif tumor_type == 'ependymoma':
                    tumor_label = 2
                elif tumor_type == 'glioma':
                    tumor_label = 3
        
        print(f'tumor type: {tumor_type}, tumor label: {tumor_label}')
        tumor_idx.append(tumor_type)

        for patient, patient_grp in tumor_grp.items():
            patient_ID = tumor_type + '_' + patient
            patient_num[tumor_type] +=1
            # 取出需要的mask切片
            mask_path = mask_dir + '/' + tumor_type + '_' + patient + '_t1_wt.nii.gz'
            # 2d
            max_slice = slice_info[patient_ID]
            mask_array = np.expand_dims(get_array_from_nifti(mask_path)[:, ::-1, :][max_slice, ...], axis = 0) # 取一帧
            # mask_array = get_array_from_nifti(mask_path)[:, ::-1, :][max_slice-1: max_slice+2, ...] if max_slice!=0 else get_array_from_nifti(mask_path)[:, ::-1, :][0:3, ...]
            mask_array = monai_spacing(mask_array)
            mask_array = monai_resize(mask_array)
            
            # 3d
            # mask_array = get_array_from_nifti(mask_path)[0:18, ::-1, :] # 取18帧
            # mask_array = monai_spacing(mask_array)
            # mask_array = monai_resize(mask_array)
            # mask_array = np.transpose(mask_array, (1, 2, 0)) # [D, H, W] --> [H, W, D]
            
            
            
            array_list = []
            for _, _array in patient_grp.items():
                array_shape = _array.shape
                break

            for mod in modality:
                if mod in patient_grp:
                    # 2d
                    mod_array = np.expand_dims(patient_grp[mod][max_slice, ...], axis = 0) # np.float16
                    # mod_array = patient_grp[mod][max_slice-1: max_slice+2, ...] if max_slice!=0 else patient_grp[mod][0:3, ...] # np.float16
                    # 3d
                    # mod_array = patient_grp[mod][0:18,...] # np.float16
                    
                else:
                    # 2d
                    mod_array = np.expand_dims(np.zeros(array_shape)[mod][max_slice, ...], axis = 0) # np.float16
                    # mod_array = np.zeros(array_shape)[max_slice-1: max_slice+2, ...] if max_slice!=0 else np.zeros(array_shape)[0:3, ...]
                    # 3d
                    # mod_array = np.zeros(array_shape)[0:18,...]
                    

                # resize 到相同的尺寸
                mod_array = monai_spacing(mod_array)
                mod_array = monai_resize(mod_array)
                # 3d
                # mod_array = np.transpose(mod_array, (1, 2, 0)) # [D, H, W] --> [H, W, D]
                array_list.append(mod_array)
            
            # 2d
            concated_array = np.concatenate(array_list, axis=0)  # [ Modality * D, H, W]
            # 3d
            # stacked_array = np.stack(array_list, axis=0)  # [ Modality, H, W, D]
            
            if ( not np.any(concated_array) ): # 当前patient没有任何所需的模态
            # if ( not np.any(stacked_array) ): # 当前patient没有任何所需的模态
                _logger.warn('Tumor({}) - Patient({}) does not have any modality in {}, so it will be removed in this exp! '.format(tumor_type, patient, modality))
                continue
            else:
                info_list[tumor_type].append((patient_ID, concated_array, tumor_label, mask_array))
                # info_list[tumor_type].append((patient_ID, stacked_array, tumor_label, mask_array))
    
    # Tumor count
    other_tumor = 0
    for tumor_type, tumor_grp in hdf5_file.items():
        if tumor_type != 'mb':
            other_tumor += patient_num[tumor_type]
        print('Tumor({})----num({})'.format(tumor_type, patient_num[tumor_type]))
    
    # split train and test, the split plan is fixed once the seed is fixed
    # 5-fold validation
    
    num_test = dict()
    tumor = sorted(tumor)
    for tumor_type in tumor:
        random.seed(seed) #seed 1-10
        random.shuffle(info_list[tumor_type])
        num_test[tumor_type] = int(test_ratio * len(info_list[tumor_type]) + 0.5)

    print('sample num in the test set:', num_test)
    split_list = defaultdict(list)
    for i in range(1,11):
        if i == 10:
            for tumor_type in tumor:
                split_list[i] += info_list[tumor_type][(i-1)*num_test[tumor_type]:]
        else:
            for tumor_type in tumor:
                split_list[i] += info_list[tumor_type][(i-1)*num_test[tumor_type]:i*num_test[tumor_type]]

    train_list = []
    assert test_seed<11 and test_seed>0 and type(test_seed)==int, 'test_seed should : test_seed<11 and test_seed>0 and test_seed==int' 
    valid_seed = -1
    if test_seed == 1:
        valid_seed = 5
    else:
        valid_seed = test_seed - 1
    print('valid_seed', valid_seed)
    print('test_seed', test_seed)

    test_list = split_list[test_seed]  #test_seed 1-5
    valid_list = split_list[valid_seed]    #val_seed 1-5 and != test_seed

    for i in range(1,6):
        if i != test_seed and i != valid_seed:
            train_list += split_list[i]
    
    label_num_train = defaultdict(int)
    label_num_valid = defaultdict(int)
    label_num_test = defaultdict(int)
    
    for data,num,label,mask in train_list:
        label_num_train[label] +=1
    print('the num of the train set: ', label_num_train)
    
    for data,num,label,mask in valid_list:
        label_num_valid[label] +=1
    print('the num of the validation set: ', label_num_valid)

    for data,num,label,mask in test_list:
        label_num_test[label] +=1
    print('the num of the test set: ', label_num_test)


    train_set = BrainDataset(train_list,
                             train=True,
                             transform=transform['train'],
                             mean = mean,
                             std = std)
    
    valid_set = BrainDataset(valid_list,
                             train=False,
                             transform=transform['valid'],
                             mean = mean,
                             std = std)
 
    test_set = BrainDataset(test_list,
                            train=False,
                            transform=transform['valid'],
                            mean = mean,
                            std = std)

    return train_set, valid_set, test_set, train_list, valid_list, test_list


def get_dataset_hdf5_molecular(hdf5_path: str,
                  mask_dir: str,
                  slice_info,
                  molecular_info,
                  modality: List[str] = ['T1_Ax' , 'T1_E_Ax', 'T2_Ax'],
                  molecular: List[str] = ['SHH', 'WNT', 'G3', 'G4'],
                  test_ratio: float = 0.2,
                  transform=None,
                  seed: int=42,
                  test_seed: int=1,
                  size=(256,256)
                 )-> Tuple[Dataset, Dataset]:
    
    assert hdf5_path.endswith('.hdf5'), '{} does not ends with .hdf5'.format(hdf5_path)
    _logger.info('Load file {}'.format(hdf5_path))
    hdf5_file = h5py.File(hdf5_path, 'r')
    doc = hdf5_file.attrs['doc']
    _logger.info('Infos: {}'.format(doc))
    available_modalities = hdf5_file.attrs['modalities'].split('+')
    for m in modality:
        assert m in available_modalities, 'modality {} is not supported, maybe {}'.format(m, available_modalities)
    ''' load mean and std '''
    mean_per_modality = {m: hdf5_file.attrs['mean_{}'.format(m)].astype(np.float16) for m in modality}
    std_per_modality = {m: hdf5_file.attrs['std_{}'.format(m)].astype(np.float16) for m in modality}
    _logger.info('Pixel mean of each modality: {}'.format(mean_per_modality))
    _logger.info('Pixel std of each modality: {}'.format(std_per_modality))
    mean = []
    std = []
    for m in modality:
        mean.append(hdf5_file.attrs['mean_{}'.format(m)].astype(np.float16))
        std.append(hdf5_file.attrs['std_{}'.format(m)].astype(np.float16))
    print(f'mean per modality: {mean_per_modality}, mean value: {mean}' )
    print(f'std per modality: {std_per_modality}, std value: {std}' )
    
    ''' create info list '''
    info_list = []
    patient_num = defaultdict(int)
    info_list = defaultdict(list)
    molecular_dict = {'SHH': 0, 'WNT': 1, 'G3': 2, 'G4':3}
    for tumor_type, tumor_grp in hdf5_file.items():
        if tumor_type != 'mb':
            continue
        for patient, patient_grp in tumor_grp.items():
            if patient not in sorted(list(molecular_info.keys())):
                continue
            patient_ID = tumor_type + '_' + patient
            # slices_lst = slices_info[patientID]
            mask_path = mask_dir + '/' + tumor_type + '_' + patient + '_t1_wt.nii.gz'
            max_slice = slice_info[patient_ID]
            mask_array = np.expand_dims(get_array_from_nifti(mask_path)[:, ::-1, :][max_slice, ...], axis = 0) # 取一帧
            # mask_array = get_array_from_nifti(mask_path)[:, ::-1, :][max_slice-1: max_slice+2, ...] if max_slice!=0 else get_array_from_nifti(mask_path)[:, ::-1, :][0:3, ...]
            mask_array = monai_spacing(mask_array)
            mask_array = monai_resize(mask_array)

            molecular_type = molecular_info[patient]
            molecular_label = molecular_dict[molecular_type]
            patient_num[molecular_type] +=1
            array_list = []
            for _, _array in patient_grp.items():
                array_shape = _array.shape
                break

            for mod in modality:
                if mod in patient_grp:
                    # mod_array = patient_grp[mod][1:13,...] # np.float16
                    mod_array = np.expand_dims(patient_grp[mod][max_slice, ...], axis = 0) # np.float16
                    # mod_array = patient_grp[mod][max_slice-1: max_slice+2, ...] if max_slice!=0 else patient_grp[mod][0:3, ...] # np.float16
                else:
                    # mod_array = np.zeros(array_shape)[1:13,...]
                    mod_array = np.expand_dims(np.zeros(array_shape)[mod][max_slice, ...], axis = 0) # np.float16
                    # mod_array = np.zeros(array_shape)[max_slice-1: max_slice+2, ...] if max_slice!=0 else np.zeros(array_shape)[0:3, ...]

                # resize 到相同的尺寸
                mod_array = monai_spacing(mod_array)
                mod_array = monai_resize(mod_array)
                array_list.append(mod_array)
                
            # stacked_array = np.stack(array_list, axis=0)  # [ Modality, C, H, W]
            concated_array = np.concatenate(array_list, axis=0)  # [ Modality * D , H, W]
            
            if ( not np.any(concated_array) ): # 当前patient没有任何所需的模态
                _logger.warn('Tumor({}) - Patient({}) does not have any modality in {}, so it will be removed in this exp! '.format(tumor_type, patient, modality))
                continue
            else:
                info_list[molecular_type].append((patient_ID, concated_array, molecular_label, mask_array))

    # Tumor count
    for molecular_type in molecular_dict.keys():
        print('Molecular({})----num({})'.format(molecular_type, patient_num[molecular_type]))
    
    # split train and test, the split plan is fixed once the seed is fixed
    # 5-fold validation
    
    num_test = dict()
    for molecular_type in molecular:
        random.seed(seed) # seed 1-10
        random.shuffle(info_list[molecular_type])
        num_test[molecular_type] = int(test_ratio * len(info_list[molecular_type]) + 0.5)

    print('sample num in the test set:', num_test)
    split_list = defaultdict(list)
    for i in range(1,6):
        if i == 5:
            for molecular_type in molecular:
                split_list[i] += info_list[molecular_type][(i-1)*num_test[molecular_type]:]
        else:
            for molecular_type in molecular:
                split_list[i] += info_list[molecular_type][(i-1)*num_test[molecular_type]:i*num_test[molecular_type]]
    
    train_list = []
    assert test_seed<6 and test_seed>0 and type(test_seed)==int, 'test_seed should : test_seed<6 and test_seed>0 and test_seed==int' 
    valid_seed = -1
    if test_seed == 1:
        valid_seed = 5
    else:
        valid_seed = test_seed - 1
    print('valid_seed', valid_seed)
    print('test_seed', test_seed)
    test_list = split_list[test_seed]  #test_seed 1-5
    valid_list = split_list[valid_seed]    #val_seed 1-5 and != test_seed

    for i in range(1,6):
        if i != test_seed and i != valid_seed:
            train_list += split_list[i]
    
    label_num_train = defaultdict(int)
    label_num_valid = defaultdict(int)
    label_num_test = defaultdict(int)
    
    for data,num,label,mask in train_list:
        label_num_train[label] +=1
    print('the num of the train set: ', label_num_train)
    
    for data,num,label,mask in valid_list:
        label_num_valid[label] +=1
    print('the num of the validation set: ', label_num_valid)

    for data,num,label,mask in test_list:
        label_num_test[label] +=1
    print('the num of the test set: ', label_num_test)

    train_set = BrainDataset(train_list,
                             train=True,
                             transform=transform['train'],
                             mean = mean,
                             std = std)
    
    valid_set = BrainDataset(valid_list,
                             train=False,
                             transform=transform['valid'],
                             mean = mean,
                             std = std)
 
    test_set = BrainDataset(test_list,
                            train=False,
                            transform=transform['valid'],
                            mean = mean,
                            std = std)
    
    return train_set, valid_set, test_set, train_list, valid_list, test_list
