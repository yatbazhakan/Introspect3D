import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from datasets.activation_dataset import ActivationDataset
from definitions import ROOT_DIR
from modules.custom_networks import SwinIntrospection

dataset_info = {
    "root_dir": "/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_aggregated_raw/",
    "label_file": "kitti_pointpillars_activations_filtered_labels.csv",
    "label_field": "is_missed",
    "layer": 0,
    "is_multi_feature": False,
    "classes": ["No Error", "Error"],
    "name": "kitti"
}
if __name__ == "__main__":
    model_config = os.path.join("configs",'networks', "swin3d_t_fcn.yaml")
    dataset = ActivationDataset(dataset_info)
    sample = dataset[0][0]
    print(sample.shape)
    swin3d_t_model = SwinIntrospection({'layer_config':model_config})
    # print(swin3d_t_model)
    res = swin3d_t_model(sample[None,None,:,:,:])
    print(res)