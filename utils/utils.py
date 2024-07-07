from matplotlib import pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import math
import json
import os

def visualize_data_3mod(batch):  # 可视化 实例化之后的dataset中的数值
    ID = batch['ID']
    img = batch['image']  # shape: (3, 155, 240, 240)
    mask = batch['mask']
    
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.subplot(3,3,1)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :], cmap='gray')
    plt.title('t1')
    
    plt.subplot(3,3,2)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :], cmap='gray')
    plt.title('t1e')
    
    plt.subplot(3,3,3)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :], cmap='gray')
    plt.title('t2')
    
    plt.subplot(3,3,4)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :])
    plt.imshow(mask[0, :, :], cmap='jet', alpha=0.3)

    plt.subplot(3,3,5)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :])
    plt.imshow(mask[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(3,3,6)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :])
    plt.imshow(mask[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(3,3,7)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(mask[0, :, :], cmap='gray')
    
    plt.savefig(f'z_{ID}.jpg')
    plt.close()
    return 


def visualize_data_4mod(batch):  # 可视化 实例化之后的dataset中的数值
    ID = batch['ID']
    img = batch['image']  # shape: (3, 155, 240, 240)
    mask = batch['mask']
    
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.subplot(3,4,1)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :], cmap='gray')
    plt.title('t1')
    
    plt.subplot(3,4,2)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :], cmap='gray')
    plt.title('t1e')
    
    plt.subplot(3,4,3)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :], cmap='gray')
    plt.title('t2')
    
    plt.subplot(3,4,4)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[3, :, :], cmap='gray')
    plt.title('t2 flair')
    
    plt.subplot(3,4,5)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :])
    plt.imshow(mask[0, :, :], cmap='jet', alpha=0.3)

    plt.subplot(3,4,6)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :])
    plt.imshow(mask[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(3,4,7)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :])
    plt.imshow(mask[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(3,4,8)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[3, :, :])
    plt.imshow(mask[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(3,4,9)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(mask[0, :, :], cmap='gray')

    plt.savefig(f'z_{ID}.jpg')
    # plt.close()
    return 

def visualize_pred_mask_3mod(ID, img, mask_gt, mask_pred, save_path):  # 可视化 实例化之后的dataset中的数值
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.subplot(4,3,1)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :], cmap='gray')
    plt.title('T1')
    
    plt.subplot(4,3,2)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :], cmap='gray')
    plt.title('T1E')
    
    plt.subplot(4,3,3)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :], cmap='gray')
    plt.title('T2')
    
    plt.subplot(4,3,4)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :])
    plt.imshow(mask_gt[0, :, :], cmap='jet', alpha=0.3)
    plt.title('真实标签')

    plt.subplot(4,3,5)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :])
    plt.imshow(mask_gt[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,3,6)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :])
    plt.imshow(mask_gt[0, :, :], cmap='jet', alpha=0.3)

    plt.subplot(4,3,7)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :])
    plt.imshow(mask_pred[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,3,8)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :])
    plt.imshow(mask_pred[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,3,9)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :])
    plt.imshow(mask_pred[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,3,10)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(mask_gt[0, :, :], cmap='gray')
    plt.title('真实标签')

    plt.subplot(4,3,11)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(mask_pred[0, :, :], cmap='gray')
    plt.title('预测标签')

    vis_dir = os.path.join(save_path, 'visualize_predict_error')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    plt.savefig(f'{vis_dir}/{ID}.jpg')
    return 


def visualize_pred_mask_4mod(ID, img, mask_gt, mask_pred, save_path):  # 可视化 实例化之后的dataset中的数值
    plt.figure(figsize=(10,8), dpi=100)
    
    plt.subplot(4,4,1)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :], cmap='gray')
    plt.title('T1')
    
    plt.subplot(4,4,2)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :], cmap='gray')
    plt.title('T1E')
    
    plt.subplot(4,4,3)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :], cmap='gray')
    plt.title('T2')
    
    plt.subplot(4,4,4)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[3, :, :], cmap='gray')
    plt.title('FLAIR')
    
    plt.subplot(4,4,5)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :])
    plt.imshow(mask_gt[0, :, :], cmap='jet', alpha=0.3)
    plt.title('真实标签')

    plt.subplot(4,4,6)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :])
    plt.imshow(mask_gt[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,4,7)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :])
    plt.imshow(mask_gt[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,4,8)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[3, :, :])
    plt.imshow(mask_gt[0, :, :], cmap='jet', alpha=0.3)

    plt.subplot(4,4,9)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[0, :, :])
    plt.imshow(mask_pred[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,4,10)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[1, :, :])
    plt.imshow(mask_pred[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,4,11)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[2, :, :])
    plt.imshow(mask_pred[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,4,12)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(img[3, :, :])
    plt.imshow(mask_pred[0, :, :], cmap='jet', alpha=0.3)
    
    plt.subplot(4,4,13)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(mask_gt[0, :, :], cmap='gray')
    plt.title('真实标签')

    plt.subplot(4,4,14)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(mask_pred[0, :, :], cmap='gray')
    plt.title('预测标签')

    vis_dir = os.path.join(save_path, 'visualize_predict_error')
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    plt.savefig(f'{vis_dir}/{ID}.jpg')
    return 

# 3d
# def visualize_data_3mod(batch):  # 可视化 实例化之后的dataset中的数值
#     ID = batch['ID']
#     img = batch['image']  # shape: (3, 155, 240, 240)
#     mask = batch['mask']
#     n_slice = 5
    
#     plt.figure(figsize=(10,8), dpi=100)
    
#     plt.subplot(3,3,1)
#     plt.imshow(img[0, :, :, n_slice], cmap='gray')
#     plt.title('t1')
    
#     plt.subplot(3,3,2)
#     plt.imshow(img[1, :, :, n_slice], cmap='gray')
#     plt.title('t1e')
    
#     plt.subplot(3,3,3)
#     plt.imshow(img[2, :, :, n_slice], cmap='gray')
#     plt.title('t2')
    
#     plt.subplot(3,3,4)
#     plt.imshow(img[0, :, :, n_slice])
#     plt.imshow(mask[0, :, :, n_slice], cmap='jet', alpha=0.3)

#     plt.subplot(3,3,5)
#     plt.imshow(img[1, :, :, n_slice])
#     plt.imshow(mask[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(3,3,6)
#     plt.imshow(img[2, :, :, n_slice])
#     plt.imshow(mask[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(3,3,7)
#     plt.imshow(mask[0, :, :, n_slice], cmap='gray')

#     plt.savefig(f'z_{ID}.jpg')
#     return 

# def visualize_data_4mod(batch):  # 可视化 实例化之后的dataset中的数值
#     ID = batch['ID']
#     img = batch['image']  # shape: (3, 155, 240, 240)
#     mask = batch['mask']
#     n_slice = 5
    
#     plt.figure(figsize=(10,8), dpi=100)
    
#     plt.subplot(3,4,1)
#     plt.imshow(img[0, :, :, n_slice], cmap='gray')
#     plt.title('t1')
    
#     plt.subplot(3,4,2)
#     plt.imshow(img[1, :, :, n_slice], cmap='gray')
#     plt.title('t1e')
    
#     plt.subplot(3,4,3)
#     plt.imshow(img[2, :, :, n_slice], cmap='gray')
#     plt.title('t2')
    
#     plt.subplot(3,4,4)
#     plt.imshow(img[3, :, :, n_slice], cmap='gray')
#     plt.title('t2 flair')
    
#     plt.subplot(3,4,5)
#     plt.imshow(img[0, :, :, n_slice])
#     plt.imshow(mask[0, :, :, n_slice], cmap='jet', alpha=0.3)

#     plt.subplot(3,4,6)
#     plt.imshow(img[1, :, :, n_slice])
#     plt.imshow(mask[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(3,4,7)
#     plt.imshow(img[2, :, :, n_slice])
#     plt.imshow(mask[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(3,4,8)
#     plt.imshow(img[3, :, :, n_slice])
#     plt.imshow(mask[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(3,4,9)
#     plt.imshow(mask[0, :, :, n_slice], cmap='gray')

#     plt.savefig(f'z_{ID}.jpg')
#     return 


# def visualize_pred_mask_3mod(ID, img, mask_gt, mask_pred, save_path):  # 可视化 实例化之后的dataset中的数值
#     n_slice = 5
#     plt.figure(figsize=(10,8), dpi=100)
    
#     plt.subplot(4,3,1)
#     plt.imshow(img[0, :, :, n_slice], cmap='gray')
#     plt.title('t1')
    
#     plt.subplot(4,3,2)
#     plt.imshow(img[1, :, :, n_slice], cmap='gray')
#     plt.title('t1e')
    
#     plt.subplot(4,3,3)
#     plt.imshow(img[2, :, :, n_slice], cmap='gray')
#     plt.title('t2')
    
#     plt.subplot(4,3,4)
#     plt.imshow(img[0, :, :, n_slice])
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='jet', alpha=0.3)
#     plt.title('gt')

#     plt.subplot(4,3,5)
#     plt.imshow(img[1, :, :, n_slice])
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,3,6)
#     plt.imshow(img[2, :, :, n_slice])
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='jet', alpha=0.3)

#     plt.subplot(4,3,7)
#     plt.imshow(img[0, :, :, n_slice])
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,3,8)
#     plt.imshow(img[1, :, :, n_slice])
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,3,9)
#     plt.imshow(img[2, :, :, n_slice])
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,3,10)
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='gray')
#     plt.title('mask_gt')

#     plt.subplot(4,3,11)
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='gray')
#     plt.title('mask_pred')

#     vis_dir = os.path.join(save_path, 'visualize_predict_error')
#     if not os.path.exists(vis_dir):
#         os.makedirs(vis_dir)
#     plt.savefig(f'{vis_dir}/{ID}.jpg')
#     return 


# def visualize_pred_mask_4mod(ID, img, mask_gt, mask_pred, save_path):  # 可视化 实例化之后的dataset中的数值
#     n_slice = 5
#     plt.figure(figsize=(10,8), dpi=100)
    
#     plt.subplot(4,4,1)
#     plt.imshow(img[0, :, :, n_slice], cmap='gray')
#     plt.title('t1')
    
#     plt.subplot(4,4,2)
#     plt.imshow(img[1, :, :, n_slice], cmap='gray')
#     plt.title('t1e')
    
#     plt.subplot(4,4,3)
#     plt.imshow(img[2, :, :, n_slice], cmap='gray')
#     plt.title('t2')
    
#     plt.subplot(4,4,4)
#     plt.imshow(img[3, :, :, n_slice], cmap='gray')
#     plt.title('t2 flair')
    
#     plt.subplot(4,4,5)
#     plt.imshow(img[0, :, :, n_slice])
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='jet', alpha=0.3)
#     plt.title('gt')

#     plt.subplot(4,4,6)
#     plt.imshow(img[1, :, :, n_slice])
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,4,7)
#     plt.imshow(img[2, :, :, n_slice])
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,4,8)
#     plt.imshow(img[3, :, :, n_slice])
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='jet', alpha=0.3)

#     plt.subplot(4,4,9)
#     plt.imshow(img[0, :, :, n_slice])
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,4,10)
#     plt.imshow(img[1, :, :, n_slice])
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,4,11)
#     plt.imshow(img[2, :, :, n_slice])
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,4,12)
#     plt.imshow(img[3, :, :, n_slice])
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='jet', alpha=0.3)
    
#     plt.subplot(4,4,13)
#     plt.imshow(mask_gt[0, :, :, n_slice], cmap='gray')
#     plt.title('mask_gt')

#     plt.subplot(4,4,14)
#     plt.imshow(mask_pred[0, :, :, n_slice], cmap='gray')
#     plt.title('mask_pred')

#     vis_dir = os.path.join(save_path, 'visualize_predict_error')
#     if not os.path.exists(vis_dir):
#         os.makedirs(vis_dir)
#     plt.savefig(f'{vis_dir}/{ID}.jpg')
#     return 


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
 
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(
            len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas
 
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank *
                          self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)
 
    def __len__(self):
        return self.num_samples

    
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

def calculate_metric(y_pred=None, y=None, eps=1e-9):
    
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")
    
    batch_size, n_class = y_pred.shape[:2]
    
    dsc = np.empty((batch_size, n_class))
    hd = np.empty((batch_size, n_class))
    cnt = np.zeros((n_class))
    for b, c in np.ndindex(batch_size, n_class):
        edges_pred, edges_gt = y_pred[b, c], y[b, c]
        if not np.any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan distance.")
        if not np.any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan distance.")
        
        if  (edges_pred.sum()>0 and edges_gt.sum()>0): # pred和gt均不为0，正常计算dice和hausdorff
            dice = binary.dc(edges_pred, edges_gt)
            distance = binary.hd95(edges_pred, edges_gt)
            dsc[b, c] = dice
            hd[b, c] = distance
            cnt[c] += 1
        elif  (edges_pred.sum()==0 and edges_pred.sum() == 0):  # pred和gt均为0，dice=1 hausdorff 0
            dsc[b, c] = 1
            hd[b, c] = 0
            cnt[c] += 1
        # elif  ( (edges_pred.sum()>0 and edges_gt.sum()==0) or (edges_pred.sum()==0 and edges_gt.sum()>0) ): # pred和gt中有一个为0，dice=0 hausdorff 373.128664
        else:
            dsc[b, c] = 0
            hd[b, c] = 0
            cnt[c] += eps
    dsc = np.sum(dsc, axis=0)
    hd = np.sum(hd, axis=0)
    dsc = dsc / cnt
    hd = hd / cnt
    
    return torch.from_numpy(dsc), torch.from_numpy(hd)
