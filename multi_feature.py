#%%
import os
import torch
import numpy as np
import pandas as pd
from glob import glob
import pickle
from utils.utils import generate_model_from_config
import yaml
#%%
pth = "/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_aggregated_raw/features/000000"
model_config = "/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn.yaml"
data = pickle.load(open(pth, "rb"))
#%%
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)
#%%
#load_yaml 
# from modules.custom_networks import CustomModel
# conf = {'layer_config': model_config}
# model = CustomModel(conf)
# out= model(data)
# print(out)
from mmdet.apis import DetInferencer

# Initialize the DetInferencer
inferencer = DetInferencer('/mnt/ssd2/Introspect3D/configs/mmdet/kitti_detr.py', '/mnt/ssd2/Introspect3D/work_dirs/kitti_detr/epoch_10.pth')

# Perform inference
inferencer('/mnt/ssd2/kitti/training/image_2/000011.png', show=True)