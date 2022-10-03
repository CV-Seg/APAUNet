# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

import os
import torch
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
from monai.inferers import sliding_window_inference
from config import test_config
from build_model import build_model

def main():
    # ------ load model checkpoint ------
    model = build_model(test_config["model_name"], test_config["in_ch"], test_config['class_num'])
    model.load_state_dict(torch.load(test_config['resume'])['state_dict'])
    model.cuda()
    model.eval()

    # ------ test data config ------
    data_dir    = os.path.join(test_config['data_path'], test_config['dataset'], 'imagesTS')
    input_size  = (96,96,96)

    # ------ inference ------
    print('inference start!')
    for index in tqdm(os.listdir(data_dir)):
        img = np.array(nib.load(os.path.join(data_dir, index)).get_fdata())
        img = torch.from_numpy(img)
        img = img.to(torch.float32)
        img = torch.unsqueeze(img, 0).unsqueeze(0)
        img = img.cuda()
        output = sliding_window_inference(img, roi_size=input_size, sw_batch_size=1, predictor=model)
        class2 = ((output[0][1]>output[0][0])&(output[0][1]>output[0][2])).float().cpu().numpy()
        class3 = ((output[0][2]>output[0][0])&(output[0][2]>output[0][1])).float().cpu().numpy()
        result = class2 + class3*2
        sitk.WriteImage(sitk.GetImageFromArray(result), os.path.join(test_config["output"], index))
    print('inference over!')

if __name__ == '__main__':
    main()
