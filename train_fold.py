# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import torch
import monai
from utils import WarmupMultiStepLR
from build_model import build_model
from utils import is_npz
from dataset import mp_get_datas
from trainer import Trainer
from config import kf_config 
from torch.utils.tensorboard import SummaryWriter

def get_kfold_data(k, i, ids):
    assert k > 1
    fold_size = len(ids)//k    
    idx = slice(i * fold_size, (i+1) * fold_size)
    val_ids = ids[idx]
    return val_ids

def main():
    writer      = SummaryWriter(os.path.join('./runs/',kf_config['log']))
    
    # -------- load data --------
    data_path   = kf_config['data_path'] 
    assert kf_config['dataset'] in ['Task007_Pancreas', 'Task004_Liver'], 'Other dataset is not implemented'
    data_dir    = os.path.join(data_path, 'nnFormer_preprocessed', kf_config['dataset'], 'nnFormerData_plans_v2.1_stage0')
    ids         = os.listdir(data_dir)
    ids         = list(filter(is_npz, ids))
    for i in range(kf_config["folds"]):
        print('fold:', i)
        val_ids     = get_kfold_data(kf_config["folds"], i, ids)
        
        train_ids   = []
        for idx in ids:
            if idx in val_ids:
                continue
            train_ids.append(idx)
        print('Val', val_ids)
        print('Train', train_ids)

        train_data  = mp_get_datas(data_dir, train_ids, kf_config["dataset"])
        val_data    = mp_get_datas(data_dir, val_ids, kf_config["dataset"])
        train_list  = list(range(len(train_ids)))
        val_list    = list(range(len(val_ids)))
        
        print('** Get datas finished. **')

        # -------- load model --------
        model = build_model(kf_config["model_name"], kf_config["in_ch"], kf_config['class_num'])

        # -------- Loss functions ----------
        if kf_config["criterion"] == 'DiceCE':
            criterion = monai.losses.DiceCELoss(softmax=True, to_onehot_y=True)
            print('---------Using DiceCE Loss')
        elif kf_config["criterion"] == 'DiceFocal':
            criterion = monai.losses.DiceFocalLoss(softmax=True, to_onehot_y=True)
            print('---------Using DiceFocal Loss')
        else:
            raise NotImplementedError

        # -------- Optimizers ----------
        if kf_config["optimizer"] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=kf_config["lr"], momentum=0.9, weight_decay=0.0001)
            print('---------Using SGD Optimizer')
        elif kf_config["optimizer"] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.99))
            print('---------Using Adam Optimizer')
        else:
            raise NotImplementedError

        # -------- Learning rate schedulers & Warm up tricks ----------
        if kf_config["scheduler"]   == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800, eta_min=0.0001)
            print('---------Using CosineAnnealingLR Warmup')
        elif kf_config["scheduler"] == 'MultiStepLR':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 332], gamma=0.1)
            print('---------Using MultiStepLR Warmup')
        elif kf_config["scheduler"] == 'WarmupMultiStepLR':
            lr_scheduler = WarmupMultiStepLR(optimizer=optimizer, milestones=[250, 500], gamma=0.1, warmup_factor=0.1, warmup_iters=100, warmup_method="linear", last_epoch=-1)
            print('---------Using WarmupMultiStepLR Warmup')
        else:
            raise NotImplementedError

        # -------- Checkpoint resume ----------
        if kf_config["resume"] is not None:
            print("loading saved Model...")
            checkpoint = torch.load(kf_config["resume"])
            model.load_state_dict(checkpoint['state_dict'])
            model = model.cuda()
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            print("Model successfully loaded! Current step is: ", epoch)   

        # -------- Training ----------
        trainer = Trainer(model, criterion, optimizer, lr_scheduler, writer, train_list, val_list, train_data, val_data, kf_config)
        trainer.run()

if __name__ == '__main__':
    main()
