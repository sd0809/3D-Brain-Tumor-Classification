import os
import math
import os.path as osp
import h5py
import numpy as np
from collections import defaultdict
import SimpleITK as sitk

import statistics

sitk.SobelEdgeDetectionImageFilter()
# 包含mask


def create_hdf5_file(inputdir: str,
                    output_path: str,
                    interval=[1, 100, 181, 211, 0, 60]):
    """Create HDF5 dataset from nifti files (v1)

    Parameters
    ----------
    inputdir : str
    output_path : str
        the path name of the output hdf5 file
    interval : list
        [xmin, ymin, xmax, ymax, cmin, cmax]

    Hierarchy of the nifti files
    ----------------------------
    inputdir
        - med (tumor)
            - 507024 (patient)
                - T1_Ax.nii.gz
                - T1_E_Ax.nii.gz
                - T2_Ax.nii.gz
                - T2_Flair_Ax.nii.gz
            ...

    Attributes of HDF5 file
    -----------------------
    doc: summarized details of the dataset
    angles: of the MRIs
    modalities: of the MRIs
    registration: whether registrated to the normalized human brain
    num_arrays: total number of the MRI sequence (nifti file)
    Reference
    ---------
    https://docs.h5py.org/en/stable/quick.html
    """
    assert osp.exists(inputdir), 'dataset dir {} does not exist.'.format(
        inputdir)
    num_arrays = 0
    num_arrays_per_modality = defaultdict(int)
    mean_per_modality = defaultdict(int)
    std_per_modality = defaultdict(int)
    print('Loading {}...'.format(inputdir))
    with h5py.File(output_path, 'a') as dset:
        dset.attrs['doc'] = 'angles: Ax , modalities: T1 + T1_E + T2 + T2_Flair, registration: True'
        print('Doc: {}'.format(dset.attrs['doc']))
        dset.attrs['angles'] = 'Ax'
        dset.attrs['modalities'] = 'T1_Ax+T1_E_Ax+T2_Ax+T2_Flair_Ax'
        dset.attrs['registration'] = True
        num_tumors = -1
        for tumor_type in os.listdir(inputdir):
            num_tumors += 1
            tumor_dir = osp.join(inputdir, tumor_type)
            tumor_grp = dset.create_group(tumor_type)
            num_patients = -1
            for patient in os.listdir(tumor_dir):
                ''' Use the after-registration nifti file, usually have more 180 channels'''
                num_patients += 1
                print('\nProcessing #{} tumor `{}`, #{} patient `{}`...'.format(
                    num_tumors, tumor_type, num_patients, patient))
                patient_dir = osp.join(tumor_dir, patient)
                patient_grp = tumor_grp.create_group(patient)
                for mod in os.listdir(patient_dir):
                    mod = os.path.join(patient_dir, mod)
                    if not mod.endswith('nii.gz'):
                        continue
                    mod_name = mod.split('/')[-1]
                    # if mod_name.startswith('T') and not mod_name.startswith('T2_Flair'):
                    if mod_name.startswith('T'):
                        if 'Ax' in mod_name:
                            nii_file_path = mod
                            array = get_array_from_nifti(nii_file_path)
                            array = process_array(array)
                            array_to_write = array.astype(np.float16)   # 要保存的数据类型
                            patient_grp.create_dataset(mod_name[:-7], data=array_to_write)
                            num_arrays += 1
                            num_arrays_per_modality[mod_name[:-7]] += 1
                            mean_per_modality[mod_name[:-7]] += array_to_write.mean(dtype=np.float64).astype(np.float16)
                            std_per_modality[mod_name[:-7]] += array_to_write.std(dtype=np.float64).astype(np.float16)
                            print('num_arrays: ', num_arrays)
                            print('num_arrays of {}: {}'.format(mod_name[:-7], num_arrays_per_modality[mod_name[:-7]]))
                            print('array shape: {}'.format(array_to_write.shape))
                            print('mean of {}: {}'.format(mod_name[:-7], array_to_write.mean(dtype=np.float64).astype(np.float16)))
                            print('std of {}: {}'.format(mod_name[:-7], array_to_write.std(dtype=np.float64).astype(np.float16)))
                    
        # write doc string in the attributes
        dset.attrs['num_arrays'] = num_arrays
        for mod in num_arrays_per_modality:
            dset.attrs['num_arrays_{}'.format(mod)] = num_arrays_per_modality[mod]
            dset.attrs['mean_{}'.format(mod)] = mean_per_modality[mod] / num_arrays_per_modality[mod]
            dset.attrs['std_{}'.format(mod)] = std_per_modality[mod] / num_arrays_per_modality[mod]
            print('Pixel mean of {}: {}'.format(mod, dset.attrs['mean_{}'.format(mod)]))
            print('Pixel std of {}: {}'.format(mod, dset.attrs['std_{}'.format(mod)]))



def create_hdf5_file_v2(inputdir: str,
                        output_path: str,
                        interval=[1, 100, 181, 211, 0, 60]):
    """Create HDF5 dataset from nifti files (v1)

    Parameters
    ----------
    inputdir : str
    output_path : str
        the path name of the output hdf5 file
    interval : list
        [xmin, ymin, xmax, ymax, cmin, cmax]

    Hierarchy of the nifti files
    ----------------------------
    inputdir
        - med (tumor)
            - 507024 (patient)
                - T1_Ax_reg.nii.gz
                - T1_E_Ax.nii.gz
                - T2_Ax.nii.gz
                - T2_Flair_Ax.nii.gz
                - Seg_wt.nii.gz
                - Seg_wt.nii.gz
            ...

    Attributes of HDF5 file
    -----------------------
    doc: summarized details of the dataset
    angles: of the MRIs
    modalities: of the MRIs
    registration: whether registrated to the normalized human brain
    num_arrays: total number of the MRI sequence (nifti file)
    Reference
    ---------
    https://docs.h5py.org/en/stable/quick.html
    """
    assert osp.exists(inputdir), 'dataset dir {} does not exist.'.format(
        inputdir)
    num_arrays = 0
    num_arrays_per_modality = defaultdict(int)
    mean_per_modality = defaultdict(int)
    std_per_modality = defaultdict(int)
    print('Loading {}...'.format(inputdir))
    with h5py.File(output_path, 'a') as dset:
        dset.attrs['doc'] = 'angles: Ax , modalities: T1 + T1_E + T2 + T2_Flair_Ax, registration: True'
        print('Doc: {}'.format(dset.attrs['doc']))
        dset.attrs['angles'] = 'Ax'
        dset.attrs['modalities'] = 'T1_Ax+T1_E_Ax+T2_Ax+T2_Flair_Ax'
        dset.attrs['registration'] = True
        num_tumors = -1
        for tumor_type in os.listdir(inputdir):
            num_tumors += 1
            tumor_dir = osp.join(inputdir, tumor_type)
            tumor_grp = dset.create_group(tumor_type)
            num_patients = -1
            for patient in os.listdir(tumor_dir):
                ''' Use the after-registration nifti file, usually have more 180 channels'''
                num_patients += 1
                print('\nProcessing #{} tumor `{}`, #{} patient `{}`...'.format(
                    num_tumors, tumor_type, num_patients, patient))
                patient_dir = osp.join(tumor_dir, patient)
                patient_grp = tumor_grp.create_group(patient)
                for mod in os.listdir(patient_dir):
                    mod = os.path.join(patient_dir, mod)
                    if not mod.endswith('nii.gz'):
                        continue
                    mod_name = mod.split('/')[-1]
                    if mod_name.startswith('T'):
                        if 'Ax' in mod_name:
                            nii_file_path = mod
                            array = get_array_from_nifti(nii_file_path)
                            array = process_array(array)
                            array_to_write = array.astype(np.float16)   # 要保存的数据类型
                            patient_grp.create_dataset(mod_name[:-7], data=array_to_write)
                            num_arrays += 1
                            num_arrays_per_modality[mod_name[:-7]] += 1
                            mean_per_modality[mod_name[:-7]] += array_to_write.mean(dtype=np.float64).astype(np.float16)
                            std_per_modality[mod_name[:-7]] += array_to_write.std(dtype=np.float64).astype(np.float16)
                            print('num_arrays: ', num_arrays)
                            print('num_arrays of {}: {}'.format(mod_name[:-7], num_arrays_per_modality[mod_name[:-7]]))
                            print('array shape: {}'.format(array_to_write.shape))
                            print('mean of {}: {}'.format(mod_name[:-7], array_to_write.mean(dtype=np.float64).astype(np.float16)))
                            print('std of {}: {}'.format(mod_name[:-7], array_to_write.std(dtype=np.float64).astype(np.float16)))
                    
                    elif mod_name.startswith('Seg'):
                        mask_file_path = mod
                        mask_array = get_array_from_nifti(mask_file_path)
                        mask_array_to_write = mask_array.astype(np.int16)   # 要保存的数据类型
                        patient_grp.create_dataset(mod_name[:-7], data=mask_array_to_write)
                        num_arrays += 1
                        num_arrays_per_modality[mod_name[:-7]] += 1
                        mean_per_modality[mod_name[:-7]] += mask_array_to_write.mean(dtype=np.float64).astype(np.float16)
                        std_per_modality[mod_name[:-7]] += mask_array_to_write.std(dtype=np.float64).astype(np.float16)
                        print('num_arrays: ', num_arrays)
                        print('num_arrays of {}: {}'.format(mod_name[:-7], num_arrays_per_modality[mod_name[:-7]]))
                        print('array shape: {}'.format(mask_array_to_write.shape))
                            
        # write doc string in the attributes
        dset.attrs['num_arrays'] = num_arrays
        for mod in num_arrays_per_modality:
            dset.attrs['num_arrays_{}'.format(mod)] = num_arrays_per_modality[mod]
            dset.attrs['mean_{}'.format(mod)] = mean_per_modality[mod] / num_arrays_per_modality[mod]
            dset.attrs['std_{}'.format(mod)] = std_per_modality[mod] / num_arrays_per_modality[mod]
            print('Pixel mean of {}: {}'.format(mod, dset.attrs['mean_{}'.format(mod)]))
            print('Pixel std of {}: {}'.format(mod, dset.attrs['std_{}'.format(mod)]))


def process_array(array: np.ndarray) -> np.ndarray:
    """Process the image array, make sure the intensities is within [0, 255]

    Parameters
    ----------
    array : np.ndarray

    Returns
    -------
    np.ndarray
    """
    # make the pixels values is above 0
    array = array - array.min()
    values = array.reshape(-1).copy()
    values.sort()
    # clip off the top 1% pixel values
    top = values[int(len(values) * 0.99)]
    # make sure the pixel is in [0, 255]
    array = np.clip(array, 0, top) / top * 255.
    return array[:, ::-1, :]

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
    if not input_nii_path.endswith('nii.gz'):
        raise ValueError('{} does not ends with nii.gz'.format(input_nii_path))
    nii_image = sitk.ReadImage(input_nii_path)
    array = sitk.GetArrayFromImage(nii_image)
    return array


if __name__ == '__main__':
    create_hdf5_file(
        '/data/sd0809/TianTanData/data_align_4mod',
        '/data/sd0809/TianTanData/data_align_4mod.hdf5',
    )