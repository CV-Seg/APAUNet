# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import random
import time
import torch
import numpy as np
import time
from torch.cuda.amp import autocast, GradScaler
from metrics import load_dicefunc
from utils import ModelEma
from dataset import mp_get_batch
from datetime import datetime

class Trainer(object):

    def __init__(self, model, criterion, optimizer, scheduler, writer, train_list, val_list, train_data, val_data, config, use_cuda = True, use_ema = True):
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.writer     = writer
        self.use_cuda   = use_cuda
        self.use_ema    = use_ema
        self.train_list = train_list
        self.val_list   = val_list
        self.train_data = train_data
        self.val_data   = val_data
        self.config     = config
        self.epochs     = self.config["epochs"]
        self.best_dice  = 0 
        self.best_epoch = 0
        if self.use_cuda and config['resume'] is None:
            self.model  = self.model.cuda()        
        if self.use_ema:
            self.ema    = ModelEma(self.model, decay=0.9998)

    def run(self):
        scaler = GradScaler()
        for epoch in range(self.epochs):
            self.train(epoch, scaler)
            if self.config["model_name"] == "APAUNet":
                if (epoch + 1) % 10 == 0:
                    print(self.model.encoder1.beta, self.model.encoder2.beta, self.model.encoder3.beta, self.model.encoder4.beta, self.model.encoder5.beta)
                    print(self.model.decoder1.beta, self.model.decoder2.beta, self.model.decoder3.beta, self.model.decoder4.beta)
            if epoch % self.config["val_interval"] == 0:
                dice_mean = self.evaluation(epoch)
                if epoch >= 10 and self.best_dice < dice_mean:
                    self.save(dice_mean, epoch)

    def train(self, epoch, scaler):
        self.model.train()
        Time        = []
        loss_list   = []
        random.shuffle(self.train_list)

        for i in range(0, len(self.train_list), 2):
            if i + self.config["batch_size"] > len(self.train_list):
                break
            sample_a, target_a = mp_get_batch(self.train_data, self.train_list[i:i+self.config["batch_size"]//2], self.config["input_shape"], aug='random')
            sample_b, target_b = mp_get_batch(self.train_data, self.train_list[i+self.config["batch_size"]//2:i+self.config["batch_size"]], self.config["input_shape"], aug='bounding')
            inputs  = torch.cat((sample_a, sample_b), 0)
            targets = torch.cat((target_a, target_b), 0)
            inputs  = inputs.cuda()
            targets = targets.cuda()
            with autocast():
                times   = time.time()
                outputs = self.model(inputs)
                time_   = time.time() - times
                Time.append(time_)
                loss    = self.criterion(outputs, targets)
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()    
            self.ema.update(self.model)        
            loss_list.append(loss.item())
        self.scheduler.step()
        self.writer.add_scalar("training loss", np.mean(loss_list), epoch)
        print("-"*20)
        print(f"{datetime.now()} Training--epoch: {epoch+1}\t"f"lr: {self.scheduler.get_last_lr()[0]:.3f}\t"f"loss: {np.mean(loss_list):.3f}\t")
        print(f'Average inference Time:{sum(Time) / len(Time):.1f}', )

    def evaluation(self, epoch):
        self.model.eval()
        dice_mean_list  = []
        dice_organ_list = []
        dice_tumor_list = []
        with torch.no_grad():
            for i in range(0, len(self.val_list), 2):
                inputs, targets = mp_get_batch(self.val_data, self.val_list[i:i+2], self.config["input_shape"], aug='bounding')
                inputs  = inputs.cuda()
                targets = targets.cuda()
                outputs = self.model(inputs)
                dice        = load_dicefunc(outputs, targets)
                dice_organ  = dice[1]
                dice_tumor  = dice[2]
                dice_mean   = (dice_organ+dice_tumor)/2
                dice_mean_list.append(dice_mean)
                dice_organ_list.append(dice_organ)
                dice_tumor_list.append(dice_tumor)
        self.writer.add_scalar("validation dice average", np.mean(dice_mean_list), epoch)
        self.writer.add_scalar("validation dice organ", np.mean(dice_organ_list), epoch)
        self.writer.add_scalar("validation dice cancer", np.mean(dice_tumor_list), epoch)
        print("-"*20)
        
        dice_mean = np.mean(dice_mean_list)
        print(f"dice_average: {dice_mean:0.4f}")
        print(f"dice_organ: {np.mean(dice_organ_list):0.4f}\t"f"dice_cancer: {np.mean(dice_tumor_list):0.3f}\t")
        return dice_mean
    
    def save(self, dice_mean, epoch):
        self.best_dice = dice_mean
        self.best_epoch = epoch
        checkpoint = {
            'epoch': epoch, 
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
                    }
        torch.save(checkpoint, os.path.join(self.config["save_path"], 'best_'+self.config["model_name"]))
        print(f"best epoch: {self.best_epoch}, best dice: {self.best_dice:.4f}")


