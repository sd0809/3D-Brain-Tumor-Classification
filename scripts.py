from calendar import c
import os
from matplotlib.pyplot import axis 
import numpy as np
import random
import shutil
import pandas as pd
# import xlrd
import openpyxl
import torch

# root = '/data1/sd0809/TianTanData/data/mb'
# print(len(os.listdir(root)))
# out = '/data1/sd0809/MINI-TianTanData/data/mb'

# print(len(mv_lst))
# for sample in mv_lst:
#     sample = str(sample)
#     sample_path = os.path.join(root, sample)
#     mv_path = os.path.join(out, sample)
#     print(mv_path)
#     shutil.copytree(sample_path, mv_path)


# root = '/data1/sd0809/TianTanData/data/mb'
# out =  '/data1/sd0809/TianTanData/mb_remove'

# for sample in lst:
#     sample_path = os.path.join(root, sample)
#     if os.path.exists(sample_path):
#         move_path = os.path.join(out, sample)
#         shutil.move(sample_path, move_path)
#         print(sample_path)


# root = '/data1/sd0809/TianTanData/data/mb'
# cur = os.listdir(root)
# count = 0
# for sample in lst:  
#     sample_id = sample.split('_')[0]
#     if sample_id not in cur:
#         count+=1
#         print(sample_id)
# print(count)


# # 05.31 输出重定向
# with open(f"log.txt", "a") as f:
#     f.write('Training start!\n')
#     print("Hello World")
# print('111')

# f.close()


# 06.05 测试gaussion noise

# noise = np.random.normal(0.0, 0.3, size=1)
# print(noise)
 

# labels = [0,1,1,2,3]
# preds = [1,1,1,2,0]
# ids  = [1,2,3,4,5]
# bool_lst = np.array(labels) == np.array(preds)
# # for i in range(len(bool_lst)):
# #     if bool_lst[i]:
# #         print(ids[i])
# index = np.arange(0,5)
# print(index[np.array(labels) == np.array(preds)])

# # tensor 拼接
# a = torch.Tensor([[1,2,3],
#                   [4,5,6]])
# b = torch.Tensor([[7,8,9],
#                   [10,11,12]])
# c = torch.cat((a,b),axis=1)
# print(c)



# print(df.values)
# print(df.index)
# print(df.columns)

# 读execl
# df = pd.read_excel('./molecular.xlsx',sheet_name='Sheet1')
# dict_={}
# for i in range(0, len(df.index)):
#     if ( df.values[i, 1] == 'SHH' or df.values[i, 1] == 'WNT' or df.values[i, 1] == 'G3' or df.values[i, 1] == 'G4'):
#         dict_[str(df.values[i, 0])] = df.values[i, 1]
# # print(dict_)

# data_dir = '/data/sd0809/TianTanData/data_align/mb'
# subject_lst = sorted(os.listdir(data_dir))
# dict_lst = sorted(list(dict_.keys()))
# for sub in dict_lst:
#     if sub not in subject_lst:
#         print(sub)

# 读txt
# with open ('./slice_info.txt', 'r') as f:
#     slice_info = f.read()
#     slice_info_dict = eval(slice_info)


from calendar import c
import os
from matplotlib.pyplot import axis 
import numpy as np
import random
import shutil
import pandas as pd
# import openpyxl

# # 05.18 将MINI-DATA中的mb 扩充到300例
# random.seed(42)


# root = '/data1/sd0809/TianTanData/data/mb'
# print(len(os.listdir(root)))
# out = '/data1/sd0809/MINI-TianTanData/data/mb'

# print(len(mv_lst))
# for sample in mv_lst:
#     sample = str(sample)
#     sample_path = os.path.join(root, sample)
#     mv_path = os.path.join(out, sample)
#     print(mv_path)
#     shutil.copytree(sample_path, mv_path)


# 05.19 移除mb中的错误病例

# lst1 = ['671062', '687286', '480256', '420824', '605138', '455006', '332234', '304699', '682836', '631537', '586892', '502745', '426460', '430732', '334486', '533195', '686780', '669071', '560714', '697092', '471404', '407609', '336258']
# lst2 = ['402624', '419976', '479061', '628396', '563729', '453065', '570143', '569744', '561101', '558285', '316841', '352763', '355759', '492743', '534491', '540259', '558055', '601504', '611231', '635252', '650284', '652577', '659711', '661522', '650716', '686340']

# lst = lst1 + lst2
# print(len(lst))
# exit(0)
# root = '/data1/sd0809/TianTanData/data/mb'
# out =  '/data1/sd0809/TianTanData/mb_remove'

# for sample in lst:
#     sample_path = os.path.join(root, sample)
#     if os.path.exists(sample_path):
#         move_path = os.path.join(out, sample)
#         shutil.move(sample_path, move_path)
#         print(sample_path)

# 05.19 添加新一批mb
# csv = openpyxl.load_workbook('list1.xlsx')
# sh = csv.worksheets[0]
# clo_lst =[]
# for col in sh['A']:
#     clo_lst.append(col.value)
# print(clo_lst)


# 统计data_reg中同时包含T1 T1E T2 T2F四模态的subjects并复制
root = '/data/sd0809/TianTanData/data_align'
out = '/data1/sd0809/TianTanData/data_align_4mod'
count_mb = 0
count_choroid = 0
count_ependymoma = 0
count_glioma = 0
for tumor_type in sorted(os.listdir(root)):
    tumor_dir = os.path.join(root, tumor_type)
    for subject in sorted(os.listdir(tumor_dir)):
        subject_dir = os.path.join(tumor_dir, subject)
        mod_lst = os.listdir(subject_dir)
        if ( 'T1_Ax.nii.gz' in mod_lst and 'T1_E_Ax.nii.gz' in mod_lst and 'T2_Ax.nii.gz' in mod_lst and 'T2_Flair_Ax.nii.gz' in mod_lst):
            mv_dir = out + '/' + tumor_type + '/' + subject
            # if not os.path.exists(mv_dir):
            #     os.makedirs(mv_dir)
            print(f'{subject_dir} --> {mv_dir}' )
            shutil.copytree(subject_dir, mv_dir)
            # exit(0)
            if tumor_type == 'mb':
                count_mb += 1
            elif tumor_type == 'choroid':
                count_choroid += 1
            elif tumor_type == 'ependymoma':
                count_ependymoma += 1
            elif tumor_type == 'glioma':
                count_glioma += 1

print(count_choroid)
print(count_ependymoma)
print(count_glioma)
print(count_mb)




# print(len(mv_lst))
# for sample in mv_lst:
#     sample = str(sample)
#     sample_path = os.path.join(root, sample)
#     mv_path = os.path.join(out, sample)
#     print(mv_path)
#     shutil.copytree(sample_path, mv_path)


# # 判断两个模型初始化权重是否完全一致
# pretrain_weight1 = torch.load('./init1.pth', map_location=torch.device('cuda'))
# pretrain_weight2 = torch.load('./init2.pth', map_location=torch.device('cuda'))

# for name in pretrain_weight1:
    
#     print( (pretrain_weight1[name].cpu().numpy() - pretrain_weight2[name].cpu().numpy()).sum())

# # 判断两个numpy矩阵是否完全一致
# arr1 = np.load('./train1.npy')
# arr2 = np.load('./train2.npy')

# print((arr1 == arr2).all())

# arr3 = np.load('./valid1.npy')
# arr4 = np.load('./valid2.npy')

# print((arr3 == arr4).all())


# 精简dir
# root = '/data1/sd0809/TianTanData_v2/data_preprocess/data_ax_'
# tumor_lst = sorted(os.listdir(root))
# for tumor_type in tumor_lst:
#     tumor_dir = os.path.join(root, tumor_type)
#     sub_lst = sorted(os.listdir(tumor_dir))
#     for sub in sub_lst:
#         sub_dir = os.path.join(tumor_dir, sub)
#         sub_dir_reg = sub_dir + '/Registration'
#         sub_dir_n4 = sub_dir + '/N4BiasFieldCorrection'
#         # mod_lst = sorted(os.listdir(sub_dir_reg))
#         # for mod in mod_lst:
#         #     mod_dir = os.path.join(sub_dir_reg, mod)
#         #     mod_file_path = mod_dir + f'/{mod}.nii.gz'
#         #     # mod_file_path = sub_dir_reg + f'/{mod}'
#         #     mv_path = sub_dir + f'/{mod}.nii.gz'
#         #     shutil.copy(mod_file_path, mv_path)
#         #     print(f'{mod_file_path} --> {mv_path}')
#         shutil.rmtree(sub_dir_reg)
#         shutil.rmtree(sub_dir_n4)


# # 把存在seg的ID的数据复制一份到 data_with_mask目录下
# data_dir = '/data1/sd0809/TianTanData_v2/data_preprocess/data_align_/mb'
# mask_dir = '/data1/sd0809/TianTanData_v2/mask_student'
# out_dir = '/data1/sd0809/TianTanData_v2/data_preprocess/data_with_mask'
# sub_lst = []
# for mask in sorted(os.listdir(mask_dir)):
#     ID = mask.split('_')[1]
#     sub_lst.append(ID)
# sub_lst = list(set(sub_lst))

# for sub in sub_lst:
#     sub_dir = os.path.join(data_dir, sub)
#     out_path = os.path.join(out_dir, sub)
#     shutil.copytree(sub_dir, out_path)


# # 把mask复制到 data_with_mask对应的目录下
# mask_dir = '/data1/sd0809/TianTanData_v2/mask_student_mb'
# data_dir = '/data1/sd0809/TianTanData_v2/data_preprocess/data_align_4mod/mb'
# sub_lst = []
# # for mask in sorted(os.listdir(mask_dir)):
# #     ID = mask.split('_')[1]
# #     sub_lst.append(ID)
# # sub_lst = list(set(sub_lst))

# sub_lst = list(set(os.listdir(data_dir)))
# # print(sub_lst)
# # exit(0)
# for sub in sub_lst:
#     mask_path = [i for i in os.listdir(mask_dir) if sub in i]
#     for i in mask_path:
#         path_i = os.path.join(mask_dir, i)
#         out_path = os.path.join(data_dir, sub)
#         shutil.copy(path_i, out_path)
#         print(f'{path_i} --> {out_path}')
#         # exit(0)

# seg重命名
# data_dir = '/data1/sd0809/TianTanData_v2/data_preprocess/data_align_4mod/mb'

# for sub in sorted(os.listdir(data_dir)):
#     sub_dir = os.path.join(data_dir, sub)
#     mask_path = [i for i in os.listdir(sub_dir) if sub in i]
#     for i in mask_path:
#         path_i = os.path.join(sub_dir, i)
#         if 'wt' in i:
#             rename = 'Seg_wt.nii.gz'
#             rename_path = os.path.join(sub_dir, rename)
#             os.rename(path_i, rename_path)
#         elif 'et' in i:
#             rename = 'Seg_et.nii.gz'
#             rename_path = os.path.join(sub_dir, rename)
#             os.rename(path_i, rename_path)

# # 统计data_with_mask中同时包含T1 T1E T2 T2F四模态的subjects数量
# root = '/data/sd0809/TianTanData/data_align/glioma'
# count_mb=0
# for subject in sorted(os.listdir(root)):
#     subject_dir = os.path.join(root, subject)
#     mod_lst = os.listdir(subject_dir)
#     if ( 'T1_Ax.nii.gz' in mod_lst and 'T1_E_Ax.nii.gz' in mod_lst and 'T2_Ax.nii.gz' and 'T2_Ax.nii.gz'  and 'T2_Flair_Ax.nii.gz' in mod_lst ):      
#         count_mb += 1
#     else:
#         print(subject)
#         continue
# print('num:', count_mb)

# # seg重命名
# data_dir = '/data1/sd0809/TianTanData_v2/data_preprocess/data_align_4mod/mb'

# for sub in sorted(os.listdir(data_dir)):
#     sub_dir = os.path.join(data_dir, sub)
#     file_lst = os.listdir(sub_dir)
#     if 'Seg_wt.nii.gz' not in file_lst:
#         shutil.rmtree(sub_dir)
