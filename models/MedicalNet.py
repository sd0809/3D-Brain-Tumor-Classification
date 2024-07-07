import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.cuda.amp import autocast

class MedicalNet(nn.Module):
    def __init__(self, back_bone, path_to_weights, num_classes,  model_depth, pool_size=(1,1,1), device = 'cuda'):
        super(MedicalNet, self).__init__()
        self.back_bone = back_bone
        self.model_depth = model_depth
        self.num_classes = num_classes
        self.pool_cla = nn.Sequential(
            # nn.AdaptiveMaxPool3d(output_size = pool_size), # [bs, 512, 1, 7, 7] --> [bs, 512, 1, 1, 1]
            nn.AdaptiveAvgPool3d(output_size = pool_size), # [bs, 512, 1, 28, 28] --> [bs, 512, 1, 1, 1]
            nn.Flatten(start_dim=1),
            nn.Dropout(0.1),
        )

        if path_to_weights is not None:
            net_dict = self.back_bone.state_dict()
            pretrained_weights = torch.load(path_to_weights, map_location=torch.device(device))
            pretrain_dict = {k.replace('module.', ''): v for k, v in pretrained_weights['state_dict'].items() if k.replace('module.', '') in net_dict.keys()}
            print('pretrain weigth load successfully:\n', pretrain_dict.keys())
            net_dict.update(pretrain_dict)
            try:
                self.back_bone.load_state_dict(net_dict)
            except RuntimeError as e:
                print('Ignoring "' + str(e) + '"')
        else:
            print('No pretrained weight')
        # resnet 34
        if self.model_depth < 50:
            self.dense = nn.Sequential(
                                nn.Linear(512  * pool_size[0] * pool_size[1] * pool_size[2],  128),
                                nn.ReLU(),
                                nn.Dropout(p=0.1),
                                nn.Linear(128, self.num_classes)
                                )

        init.kaiming_normal_(self.dense[0].weight)
        init.kaiming_normal_(self.dense[3].weight)
    
    @autocast()
    def forward(self, x):
        
        features = self.back_bone(x)
        # print('features shape:', features.shape)
        features = self.pool_cla(features)
        features = self.dense(features)

        return features

# pretrain_path 里面的权重名称： module.
# 模型生成的权重名称： module.model.
# 加载预训练模型时模型需要的权重名称： model.