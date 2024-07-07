import os 
from cv2 import log
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from scipy import interp
from tqdm import tqdm
from typing import Iterator, List, Optional, Union

from sklearn.preprocessing import label_binarize 
from torch.autograd import Variable
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix,roc_curve, auc,roc_auc_score    # 生成混淆矩阵的函数
from operator import itemgetter
from torch.cuda.amp import autocast

from monai.transforms import Activations
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction

from utils.utils import visualize_pred_mask_3mod, visualize_pred_mask_4mod, calculate_metric

post_sigmoid = Activations(sigmoid=True)
cal_mean_dice= DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True) # 返回一批数据的TC WT ET dice平均值, 若nan 返回0

def accuracy(predictions, targets):
    predictions = torch.FloatTensor(predictions)
    targets = torch.FloatTensor(targets)
    return ((predictions == targets).sum().float() / targets.size(0)).item()


def train(model, optimizer, loss_fn_cla, loss_fn_seg, data_loader, device):
    model.train()
    loss_ctr = 0
    n_loss = 0
    n_loss_cla = 0
    n_loss_seg = 0
    n_acc = 0
    n_f1 = 0
    
    target_all = []
    pred_all = []
    
    for step, batch_data in enumerate(data_loader):
        # 梯度清零
        # if step == 1:
        #     break
        optimizer.zero_grad()
        IDs, images, labels, masks = batch_data['ID'], batch_data['image'], batch_data['label'], batch_data['mask']
        images = images.to(device, torch.float32)
        labels = labels.to(device)
        masks = masks.to(device, dtype=torch.float32)
        output_cla, output_seg = model(images)

        loss_cla = loss_fn_cla(output_cla, labels) #分类损失函数
        loss_seg = loss_fn_seg(output_seg, masks) # 分割损失函数
        loss = loss_cla + loss_seg
        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(output_cla,1)
        target_all.extend(labels.cpu().numpy())
        pred_all.extend(pred.cpu().numpy())
        
        loss_ctr += 1
        n_loss_cla += loss_cla.item()
        n_loss_seg += loss_seg.item()
        n_loss += loss.item()
    
    f1 = f1_score(target_all, pred_all, average = 'macro')
    acc = accuracy(pred_all, target_all)
    return n_loss/loss_ctr, n_loss_cla/loss_ctr, n_loss_seg/loss_ctr, acc, f1


def evaluate(model, loss_fn_cla, loss_fn_seg, data_loader, labels_name, device='cuda'):
    model.eval()
    tumor_index = torch.arange(0,len(labels_name)).cpu().numpy()
    target_cla_all = []
    target_seg_all = []
    pred_cla_all = []
    pred_seg_all = []
    with torch.no_grad():
        loss_ctr = 0
        n_loss = 0
        n_loss_cla = 0
        n_loss_seg = 0

        for step, batch_data in enumerate(data_loader):

            IDs, images, labels, masks = batch_data['ID'], batch_data['image'], batch_data['label'], batch_data['mask']
            images = images.to(device, torch.float32)
            labels = labels.to(device)
            masks = masks.to(device, dtype=torch.uint8)
            
            output_cla, output_seg = model(images)
            loss_cla = loss_fn_cla(output_cla, labels) #分类损失函数
            loss_seg = loss_fn_seg(output_seg, masks) #分类损失函数
            loss = loss_cla + loss_seg
            # 处理分类结果
            pred = torch.argmax(output_cla, 1)
            # 处理分割结果
            probs_sigmoid = post_sigmoid(output_seg)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.uint8)
            # 计算auc所需
            mm = torch.nn.Softmax(dim=1)
            output_cla_softmax = mm(output_cla)

            if not step:
                output_cla_softmax_all = output_cla_softmax
            else:
                output_cla_softmax_all = torch.cat((output_cla_softmax_all, output_cla_softmax), 0)
            
            target_cla_all.extend(labels.cpu().numpy())
            pred_cla_all.extend(pred.cpu().numpy())
            target_seg_all.extend(masks)
            pred_seg_all.extend(pred_masks)
            
            loss_ctr += 1
            n_loss_cla += loss_cla.item()
            n_loss_seg += loss_seg.item()
            n_loss += loss.item()
    
    acc = accuracy(pred_cla_all, target_cla_all)
    f1 = f1_score(target_cla_all, pred_cla_all, average = 'macro')
    dice = cal_mean_dice(pred_seg_all, target_seg_all)
    # dice, dice_not_nans = cal_mean_dice.aggregate() # 去除掉结果为nan的项
    dice = torch.nan_to_num(dice, nan=0.0)
    dice = dice.mean().item()

    #----draw ROC
    if len(labels_name) != 2:
        target_all_binarize = label_binarize(target_cla_all, classes=tumor_index)
    else:
        target_all_binarize = []
        for i in target_cla_all:
            if i==0:
                target_all_binarize.append([1,0])
            else:
                target_all_binarize.append([0,1])
        target_all_binarize = np.array(target_all_binarize)
    auroc_macro = roc_auc_score(target_all_binarize,output_cla_softmax_all.cpu(),average="macro")
    return n_loss/loss_ctr, n_loss_cla/loss_ctr, n_loss_seg/loss_ctr, acc, auroc_macro, f1, dice


def evaluate_and_plot(model, data_loader, labels_name, device='cuda', phase='valid', mod_num=3, title="confusion_matrix", save_path=None, normalize=False):
    
    model.eval()
    tumor_index = torch.arange(0,len(labels_name)).cpu().numpy()
    ID_all = []
    img_all = []
    target_cla_all = []
    target_seg_all = []
    pred_cla_all = []
    pred_seg_all = []
    with torch.no_grad():

        for step, batch_data in enumerate(data_loader):

            IDs, images, labels, masks = batch_data['ID'], batch_data['image'], batch_data['label'], batch_data['mask']
            images = images.to(device, torch.float32)
            labels = labels.to(device)
            masks = masks.to(device, dtype=torch.uint8)
            
            output_cla, output_seg = model(images)
            # 处理分类结果
            pred = torch.argmax(output_cla, 1)
            # 处理分割结果
            probs_sigmoid = post_sigmoid(output_seg)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.uint8)
            # 计算auc所需
            mm = torch.nn.Softmax(dim=1)
            output_cla_softmax = mm(output_cla)

            if not step:
                output_cla_softmax_all = output_cla_softmax
            else:
                output_cla_softmax_all = torch.cat((output_cla_softmax_all, output_cla_softmax), 0)
            ID_all.extend(IDs)
            img_all.extend(images)
            target_cla_all.extend(labels.cpu().numpy())
            pred_cla_all.extend(pred.cpu().numpy())
            target_seg_all.extend(masks)
            pred_seg_all.extend(pred_masks)
    
    if phase == 'test':
        index = np.arange(len(target_cla_all))
        pred_wrong_index = index[np.array(pred_cla_all) != np.array(target_cla_all)]
        # for i in pred_wrong_index:
        for i in len(ID_all):
            if mod_num == 3:
                visualize_pred_mask_3mod(ID_all[i], img_all[i].cpu().numpy(), target_seg_all[i].cpu().numpy(), pred_seg_all[i].cpu().numpy(), save_path)
            elif mod_num == 4:
                visualize_pred_mask_4mod(ID_all[i], img_all[i].cpu().numpy(), target_seg_all[i].cpu().numpy(), pred_seg_all[i].cpu().numpy(), save_path)
            else:
                print('mod num error!')
                break
    
    acc = accuracy(pred_cla_all, target_cla_all)
    f1 = f1_score(target_cla_all, pred_cla_all, average = 'macro')
    dice = cal_mean_dice(pred_seg_all, target_seg_all)
    # dice, dice_not_nans = cal_mean_dice.aggregate() # 去除掉结果为nan的项
    dice = torch.nan_to_num(dice, nan=0.0)
    dice = dice.mean().item()
    cm = confusion_matrix(target_cla_all,pred_cla_all)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #----draw confusion matrix
    cm_n = cm
    np.set_printoptions(precision=2)
    plt.figure(dpi=144)
    plt.imshow(cm_n, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title("Confusion_Matrix_"+title, fontsize=8)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, fontsize=8)    # rotation=90 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name, fontsize=8)    # 将标签印在y轴坐标上
    # add numbers to the picture
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i,cm[i, j] , fontsize=20, #format(cm[i, j], fmt)
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black")
    # show confusion matrix
    plt.ylabel('True label',fontsize=8)    
    plt.xlabel('Predicted label',fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        save_path_confusion = os.path.join(save_path,'confusion_matrix')
        if not os.path.exists(save_path_confusion):
            os.makedirs(save_path_confusion)
        plt.savefig(save_path_confusion +"/confusion_matrix_"+title+'.png', format='png') 
    else:
        plt.savefig('./fig/'+"confusion_matrix_"+title+'.png', format='png') 

    #----draw ROC
    if len(labels_name) != 2:
        target_all_binarize = label_binarize(target_cla_all, classes=tumor_index)
    else:
        target_all_binarize = []
        for i in target_cla_all:
            if i==0:
                target_all_binarize.append([1,0])
            else:
                target_all_binarize.append([0,1])
    target_all_binarize = np.array(target_all_binarize)
    n_classes = len(labels_name)

    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(target_all_binarize[:, i], output_cla_softmax_all[:,i].cpu().numpy())
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # macro（方法一）
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure(dpi = 144)
    
    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:.3f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','mediumspringgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:.3f})'
                ''.format(labels_name[i], roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    
    plt.legend(loc="lower right",fontsize=7, markerscale=2.)
    
    if save_path is not None:
        save_path_roc = os.path.join(save_path,'roc')
        if not os.path.exists(save_path_roc):
            os.makedirs(save_path_roc)
        plt.savefig(save_path_roc +"/roc_"+title+'.png', format='png') 
    else:
        plt.savefig('./fig/'+"roc_"+title+'.png', format='png') 
    plt.close('all')
    auroc_macro = roc_auc_score(target_all_binarize,output_cla_softmax_all.cpu(),average="macro")
    
    return acc, auroc_macro, f1, dice