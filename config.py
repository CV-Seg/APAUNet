# -*- coding:utf-8 -*-
# Author: Yuncheng Jiang, Zixun Zhang

train_config = {
        "log":          "log_name",
        "model_name":   "model_name",
        "dataset":      "Task004_Liver",
        "data_path":    "dataset_path",
        "save_path":    'save_model_path',
        "scheduler":    "CosineAnnealingLR",
        "criterion":    "DiceCE",
        "optimizer":    "SGD",
        "lr":           0.1,
        "warmup":       100,
        "epochs":       800,
        "val_interval": 10,
        "batch_size":   2,
        "in_ch":        1,
        "class_num":    3,
        "val_num":      26,
        "input_shape": (96,96,96),
        "resume":       None
}

test_config = {
        "model_name":   "model_name",
        "dataset":      "Task004_Liver",
        "data_path":    "dataset_path",
        "output":       "output_path",
        "in_ch":        1,
        "class_num":    3,
        "input_shape":  (96,96,96),
        "resume":       "checkpoint_path"
}

kf_config = {
        "log":          "log_name",
        "folds":        5,
        "model_name":   "model_name",
        "dataset":      "Task004_Liver",
        "data_path":    "dataset_path",
        "save_path":    'save_model_path',
        "scheduler":    "CosineAnnealingLR",
        "criterion":    "DiceCE",
        "optimizer":    "SGD",
        "lr":           0.1,
        "warmup":       100,
        "epochs":       800,
        "val_interval": 10,
        "batch_size":   2,
        "in_ch":        1,
        "class_num":    3,
        "val_num":      26,
        "input_shape": (96,96,96),
        "resume":       None
}