# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import numpy as np
import torch
from medpy import metric

def load_dicefunc(output, target):
    smooth = 1e-4
    if torch.is_tensor(output):
        output = torch.nn.Softmax(dim=1)(output)


        output = output.data.cpu().numpy()

    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    dice = [[], [], []]
    for i in range(output.shape[0]):
        class0 = (output[i][0]>output[i][1]) & (output[i][0]>output[i][2]) 
        class1 = (output[i][1]>output[i][0]) & (output[i][1]>output[i][2]) 
        class2 = (output[i][2]>output[i][0]) & (output[i][2]>output[i][1])
        intersection0 = (class0 & (target[i]==0)).sum()
        intersection1 = (class1 & (target[i]==1)).sum()
        intersection2 = (class2 & (target[i]==2)).sum()

        union0 = (class0 | (target[i]==0)).sum()
        union1 = (class1 | (target[i]==1)).sum()
        union2 = (class2 | (target[i]==2)).sum()
        cof0 = (intersection0 + smooth) / (union0 + smooth)
        cof1 = (intersection1 + smooth) / (union1 + smooth)
        cof2 = (intersection2 + smooth) / (union2 + smooth)
        dice[0].append(2*cof0/(1+cof0))
        dice[1].append(2*cof1/(1+cof1))
        dice[2].append(2*cof2/(1+cof2))
    return np.asarray(dice)

def load_hdfunc(outputs, targets):
    output = 0 * outputs[:,0] + 1 * outputs[:,1] + 2 * outputs[:,2]
    output = output.unsqueeze(1)
    hd95 = metric.binary.hd95(output.detach().cpu().numpy(), targets.detach().cpu().numpy())
    return hd95