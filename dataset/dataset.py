from PIL import Image

import os
import numpy as np
from scipy import ndimage

import torch 
from torch.utils.data import Dataset
import nibabel as nib

class BrainDataset_sample(Dataset):
    """ define dataset """
    def __init__(self, sample_path_lst: list, sample_class_lst: list, shape_dict, phase, modility_lst,  transform=None):
        self.sample_path_lst = sample_path_lst
        self.sample_class_lst = sample_class_lst
        self.input_D = shape_dict['input_D']
        self.input_H = shape_dict['input_H']
        self.input_W = shape_dict['input_W']
        self.phase = phase
        self.modility_lst = modility_lst
        self.transform = transform
    
    def __load_nii__(self, nii_path):  #读取nii数据, 逆时针旋转90° 并将Depth移至第0维
        if os.path.exists(nii_path):
            nii = nib.load(nii_path)
            nii_np = np.array(nii.dataobj)
            nii_np = np.rot90(nii_np, k=2) 
            nii_torch = torch.from_numpy(nii_np.copy())
            nii_torch= nii_torch.permute(2,1,0)
            nii_np =  np.array(nii_torch)
            return True, nii_np
        # else:
        #     return False, np.zeros((self.input_D, self.input_H, self.input_W))

    def __nii2tensorarray__(self, data):
        [D, H, W] = data.shape
        new_data = np.reshape(data, [1, D, H, W])
        new_data = new_data.astype("float32")
        return new_data

    def __len__(self):
        return len(self.sample_path_lst)  # len 就是 sample的个数，因为len对应的是item的编号，比如len为100，那么item 就是0-99，然后在下面get item,如果写成img的长度，在get item的时候会报错
    
    # 使用配准后的nii数据
    def __getitem__(self,item):
        sample_path = self.sample_path_lst[item]
        sample_ID = sample_path.split('/')[-2] + '_' + sample_path.split('/')[-1]

        # read and choose modility
        N4Bias_sample_path = os.path.join(sample_path, 'N4BiasFieldCorrection')
        nii_path_lst = []
        for modility in self.modility_lst: 
            modility_path = os.path.join(N4Bias_sample_path, modility + '_reg.nii.gz')
            nii_path_lst.append(modility_path)
        label = self.sample_class_lst[item]

        if self.phase == 'train':
            
            for idx in range(len(nii_path_lst)):
                exist, nii_array = self.__load_nii__(nii_path_lst[idx]) # output: (182, 218, 182) output: D, H, W   unreg_nii: (23, 512, 512) or (24, 512, 512)
                if exist:
                    nii_array = self.__training_data_process__(nii_array) # output: (182, 218, 182) output: D, H, W
                    nii_array = self.__nii2tensorarray__(nii_array)  #(1, 182, 218, 182)  C, H, W, D
                else:
                    nii_array = self.__nii2tensorarray__(nii_array)  #(1, 182, 218, 182)  C, H, W, D
                if idx == 0:
                    nii_cat_array = nii_array
                else:
                    nii_cat_array = np.concatenate([nii_cat_array, nii_array], axis=0)

            if self.transform is not None :
                nii_cat_array = self.transform(nii_cat_array)
            
            return sample_ID, nii_cat_array.copy(), label
            
        elif self.phase == 'valid':
            
            for idx in range(len(nii_path_lst)):
                exist, nii_array = self.__load_nii__(nii_path_lst[idx]) # output: (182, 218, 182) output: D, H, W   unreg_nii: (23, 512, 512) or (24, 512, 512)
                if exist:
                    nii_array = self.__testing_data_process__(nii_array)
                    nii_array = self.__nii2tensorarray__(nii_array)
                else:
                    nii_array = self.__nii2tensorarray__(nii_array)  #(1, 182, 218, 182)  C, H, W, D
                if idx == 0:
                    nii_cat_array = nii_array
                else:
                    nii_cat_array = np.concatenate([nii_cat_array, nii_array], axis=0)

            return sample_ID, nii_cat_array.copy(), label


    def __drop_invalid_range__(self, volume):
        """
        Cut off the invalid area
        """
        zero_value = volume[0, 0, 0]
        non_zeros_idx = np.where(volume != zero_value)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        return volume[min_z:max_z, min_h:max_h, min_w:max_w]

    # def __crop_data__(self, data, label):
    #     """
    #     Random crop with different methods:
    #     """ 
    #     # random center crop
    #     data, label = self.__random_center_crop__ (data, label)
        
        # return data, label

    def __itensity_normalize_one_volume__(self, volume):
        """
        normalize the itensity of an nd volume based on the mean and std of nonzeor region
        inputs:
            volume: the input nd volume
        outputs:
            out: the normalized nd volume
        """
        
        pixels = volume[volume > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (volume - mean)/std
        # out_random = np.random.normal(0, 1, size = volume.shape)
        # out[volume == 0] = out_random[volume == 0]
        return out
    
    def __resize_data__(self, data):
        """
        Resize the data to the input size
        """ 
        [depth, height, width] = data.shape
        scale = [self.input_D*1.0/depth, self.input_H*1.0/height, self.input_W*1.0/width]  
        data = ndimage.interpolation.zoom(data, scale, order=0)

        return data

    def __training_data_process__(self, data): 
        
        # drop out the invalid range
        data = self.__drop_invalid_range__(data)
        
        # # crop data
        # data, label = self.__crop_data__(data, label) 

        # resize data
        data = self.__resize_data__(data)

        # normalization datas
        data = self.__itensity_normalize_one_volume__(data)

        return data

    def __testing_data_process__(self, data): 

        # resize data
        data = self.__resize_data__(data)

        # normalization data
        data = self.__itensity_normalize_one_volume__(data)

        return data