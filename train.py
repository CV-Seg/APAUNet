# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import random
import torch
import monai
from utils import WarmupMultiStepLR
from build_model import build_model
from utils import is_npz
from dataset import mp_get_datas
from trainer import Trainer
from config import train_config 
from torch.utils.tensorboard import SummaryWriter

def main():
    writer      = SummaryWriter(os.path.join('./runs/',train_config['log']))
    
    # -------- load data --------
    assert train_config['dataset'] in ['Task007_Pancreas', 'Task004_Liver'], 'Other dataset is not implemented'
    data_dir    = os.path.join(train_config['data_path'], 'nnUNet_preprocessed', train_config['dataset'], 'nnUNetData_plans_v2.1_stage0')
    ids         = os.listdir(data_dir)
    ids         = list(filter(is_npz, ids))
    val_ids     = random.sample(ids, train_config["val_num"])
    train_ids   = []
    for idx in ids:
        if idx in val_ids:
            continue
        train_ids.append(idx)
    print('Val', val_ids)
    print('Train', train_ids)

    val_data    = mp_get_datas(data_dir, val_ids, train_config["dataset"])
    train_data  = mp_get_datas(data_dir, train_ids, train_config["dataset"])
    train_list  = list(range(len(train_ids)))
    val_list    = list(range(len(val_ids)))
    print(f'Get datas finished. Train data: {len(train_list)}, Validation data: {len(val_list)}')

    # -------- load model --------
    model       = build_model(train_config["model_name"], train_config["in_ch"], train_config['class_num'])

    # -------- Loss functions --------
    if train_config["criterion"] == 'DiceCE':
        criterion = monai.losses.DiceCELoss(softmax=True, to_onehot_y=True)
        print('---------Using DiceCE Loss')
    elif train_config["criterion"] == 'DiceFocal':
        criterion = monai.losses.DiceFocalLoss(softmax=True, to_onehot_y=True)
        print('---------Using DiceFocal Loss')
    else:
        raise NotImplementedError

    # -------- Optimizers --------
    if train_config["optimizer"] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=train_config["lr"], momentum=0.9, weight_decay=0.0001)
        print('---------Using SGD Optimizer')
    elif train_config["optimizer"] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.99))
        print('---------Using Adam Optimizer')
    elif train_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    else:
        raise NotImplementedError

    # -------- Learning rate schedulers & Warm up tricks --------
    if train_config["scheduler"] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=0.0001)
        print('---------Using CosineAnnealingLR Warmup')
    elif train_config["scheduler"] == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 332], gamma=0.1)
        print('---------Using MultiStepLR Warmup')
    elif train_config["scheduler"] == 'WarmupMultiStepLR':
        lr_scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=[250, 500], gamma=0.1, warmup_factor=0.1, warmup_iters=100, warmup_method="linear", last_epoch=-1)
        print('---------Using WarmupMultiStepLR Warmup')
    else:
        raise NotImplementedError

    # -------- Checkpoint resume ----------
    if train_config["resume"] is not None:
        print("loading saved Model...")
        checkpoint  = torch.load(train_config["resume"])
        model.load_state_dict(checkpoint['state_dict'])
        model       = model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch       = checkpoint['epoch']
        print("Model successfully loaded! Current step is: ", epoch)   

    # -------- Training ----------
    trainer = Trainer(model, criterion, optimizer, lr_scheduler, writer, train_list, val_list, train_data, val_data, train_config)
    trainer.run()

if __name__ == '__main__':
    main()
