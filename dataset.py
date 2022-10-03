# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import torch
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

def padding(sample, target, input_shape):
    HWD = np.array(input_shape)
    hwd = np.array(target.shape)
    tmp = np.clip((HWD - hwd) / 2, 0, None)
    rh, rw, rd = np.floor(tmp).astype(int)
    lh, lw, ld = np.ceil(tmp).astype(int)

    if sample.ndim == 3:
        sample = np.pad(sample, ((rh, lh), (rw, lw), (rd, ld)), 'constant', constant_values=-3)
        target = np.pad(target, ((rh, lh), (rw, lw), (rd, ld)), 'constant')
    else:
        sample = np.pad(sample, ((0,0), (rh, lh), (rw, lw), (rd, ld)), 'constant', constant_values=-3)
        target = np.pad(target, ((rh, lh), (rw, lw), (rd, ld)), 'constant')
    return sample, target

def random_crop(sample, target, input_shape):
    H, W, D = input_shape
    h, w, d = target.shape
    
    x = np.random.randint(0, h - H + 1)
    y = np.random.randint(0, w - W + 1)
    z = np.random.randint(0, d - D + 1)

    if sample.ndim == 3:
        return sample[x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]
    else:
        return sample[:, x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]

def bounding_crop(sample, target, input_shape):
    H, W, D = input_shape
    source_shape = list(target.shape)
    xyz = list(np.where(target > 0))

    lower_bound = []
    upper_bound = []
    for A, a, b in zip(xyz, source_shape, input_shape):
        lb = max(np.min(A), 0)
        ub = min(np.max(A)-b+1, a-b+1)

        if ub <= lb:
            lb = max(np.max(A) - b, 0)
            ub = min(np.min(A), a-b+1)
        
        lower_bound.append(lb)
        upper_bound.append(ub)
    x, y, z = np.random.randint(lower_bound, upper_bound)

    if sample.ndim == 3:
        return sample[x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]
    else:
        return sample[:, x:x+H, y:y+W, z:z+D], target[x:x+H, y:y+W, z:z+D]

def random_mirror(sample, target, prob=0.5):
    p = np.random.uniform(size=3)
    axis = tuple(np.where(p < prob)[0])
    sample = np.flip(sample, axis)
    target = np.flip(target, axis)
    return sample, target

def brightness(sample, target):    
    sample_new = np.zeros(sample.shape)
    for c in range(sample.shape[-1]):
        im = sample[:,:,:,c]        
        gain, gamma = (1.2 - 0.8) * np.random.random_sample(2,) + 0.8
        im_new = np.sign(im)*gain*(np.abs(im)**gamma)
        sample_new[:,:,:,c] = im_new 
    
    return sample_new, target

def totensor(data):
    return torch.from_numpy(np.ascontiguousarray(data))

def mp_get_datas(data_dir, data_ids, dataset, num_worker=6):

    def get_item(idx):
        sample, target = np.load(os.path.join(data_dir, idx))['data']
        target[target < 0] = 0
        return (sample, target)
    
    pool = ThreadPool(num_worker)
    datas = pool.map(get_item, data_ids)
    pool.close()
    pool.join()
    return datas

def mp_get_batch(data, idxs, input_shape, aug='random', num_worker=6):
    crop = random_crop if aug == 'random' else bounding_crop

    def batch_and_aug(idx):
        sample, target = data[idx]
        sample, target = padding(sample, target, input_shape)
        sample, target = crop(sample, target, input_shape)
        sample, target = random_mirror(sample, target)
        sample = totensor(sample)
        target = totensor(target)
        if sample.dim() == 3:
            sample.unsqueeze_(0)
        if target.dim() == 3:
            target.unsqueeze_(0)
        sample.unsqueeze_(0)
        target.unsqueeze_(0)
        return sample, target
    
    pool = ThreadPool(num_worker)
    batch = pool.map(batch_and_aug, idxs)
    pool.close()
    pool.join()
    samples, targets = zip(*batch)
    samples = torch.cat(samples)
    targets = torch.cat(targets)
    return samples, targets
