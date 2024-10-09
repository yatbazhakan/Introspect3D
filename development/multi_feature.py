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
pth = "/mnt/ssd2/custom_dataset/nus_centerpoint_activations_aggregated_raw/features/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915244548143"
model_config = "/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn_mult.yaml"
data = pickle.load(open(pth, "rb"))
#%%
print(data[0].shape)
print(data[1].shape)
print(data[2].shape)
first = torch.from_numpy(data[0][None,...])
second = torch.from_numpy(data[1][None,...])
third = torch.from_numpy(data[2][None,...])
print(first.shape,second.shape,third.shape)
tensor_feature = [first,second,third]       
#%%
#load_yaml 
from modules.custom_networks import CustomModel
conf = {'layer_config': model_config}
model = CustomModel(conf)
model = model.to('cuda:1')
out= model(tensor_feature)
# print(out)
# from mmdet.apis import DetInferencer

# # Initialize the DetInferencer
# inferencer = DetInferencer('/mnt/ssd2/Introspect3D/configs/mmdet/kitti_detr.py', '/mnt/ssd2/Introspect3D/work_dirs/kitti_detr/epoch_10.pth')

# # Perform inference
# inferencer('/mnt/ssd2/kitti/training/image_2/000011.png', show=True)