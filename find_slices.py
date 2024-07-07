from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import math
import json
import os


def get_array_from_nifti(input_nii_path: str) -> np.ndarray:
    """Get np.ndarray from nifti file

    Parameters
    ----------
    input_nii_path : str
        path of the nifti file

    Returns
    -------
    np.ndarray

    Reference
    ---------
    https://github.com/SimpleITK/SimpleITK
    """
    if not ( input_nii_path.endswith('nii.gz') or input_nii_path.endswith('nii')):
        raise ValueError('{} does not ends with nii.gz or nii'.format(input_nii_path))
    nii_image = sitk.ReadImage(input_nii_path)
    array = sitk.GetArrayFromImage(nii_image)
    return array

def find_slices(npImage_mask):
    depth = npImage_mask.shape[0]
    num_list = sorted(list(set(list(np.where(npImage_mask==1)[0]))))
    if num_list[0] >=2:
        start = num_list[0] - 2
    else:
        start = 0
    if num_list[-1] + 2 > depth:
        end = depth
    else:
        end = num_list[-1] + 2
    new_list = [i for i in range(start, end)]
    return new_list

def get_max_slice_index(array):  # 可视化 实例化之后的dataset中的数值
    D, H, W = array.shape
    index = 0
    max_sum = 0
    for i in range(D):
        slice_sum = array[i,...].sum()
        if slice_sum > max_sum:
            index = i
            max_sum = slice_sum
    return index


mask_dir = '/data/sd0809/TianTanData/Student_nnUNet_mask'

dict = {}

for sub in sorted(os.listdir(mask_dir)):
    tumor_type, patient, mod, type = sub.split('_')
    id = tumor_type + '_' + patient
    mask_path = os.path.join(mask_dir, sub)
    mask_array = get_array_from_nifti(mask_path)
    index = get_max_slice_index(mask_array)
    dict[id] = index

print(dict)