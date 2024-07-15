from datasets.nuscenes import NuScenesDataset
from utils.filter import *
from glob import glob
import os
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from modules.custom_networks import GenericInjection
import torchmetrics
from utils.filter import EllipseFilter
from glob import glob
from utils.utils import generate_model_from_config
import os
import cv2
from mmdet3d.apis import init_model, inference_detector
def register_activation_output(module, input, output):
    # print(output[0].shape,output[1].shape)
    # print(len(output))
    last_output = output.detach().cpu().numpy() #TODO: generalize this
    # print("Last output shape",last_output.shape)
    # print(last_output.shape)
    # print("-------------------")
    last_output = np.squeeze(last_output)
    activation_list.append(last_output)

def register_activation_input(module, input, output):
    # print(output[0].shape,output[1].shape)
    last_output = input[0].detach().cpu().numpy() #TODO: generalize this
    # print("Last output shape",last_output.shape)
    last_output = np.squeeze(last_output)
    activation_list.append(last_output)
def calculate_sparsity():
    early = activation_list[0]
    mid = activation_list[1]
    late = activation_list[2]
    #Calculate number of zeros 
    early_zeros = np.sum(early==0)
    mid_zeros = np.sum(mid==0)
    late_zeros = np.sum(late==0)
    #number of units in layers 
    early_units = np.prod(early.shape)
    mid_units = np.prod(mid.shape)
    late_units = np.prod(late.shape)
    #calculate sparsity
    early_sparsity = early_zeros/early_units
    mid_sparsity = mid_zeros/mid_units
    late_sparsity = late_zeros/late_units
    e_sp.append(early_sparsity)
    m_sp.append(mid_sparsity)
    l_sp.append(late_sparsity)   
activation_list = []
# det_model_checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
det_model_checkpoint = r'/home/wmg-5gcat/Desktop/Sajjad/DistEstIntrospection/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
# det_model_config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
det_model_config= r'/home/wmg-5gcat/Desktop/Sajjad/DistEstIntrospection/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
det_model = init_model(det_model_config, det_model_checkpoint, device='cuda:0')
dataset_config = {
    'name': 'NuScenesDataset',
    'root_dir': '/media/ssd_reza/nuscenes/',
    'version': 'v1.0-trainval',
    'process': False,
    'filter_labels_only': False,
    'save_path': '/media/wmg-5gcat/Co-op Autonomy 2/Hakan',
    'save_filename': 'nuscenes_train_fix.pkl',
    'filtering_style': 'FilterType.ELLIPSE',
    'filter_params': {
      'a': 15,
      'b': 25,
      'offset': -10,
      'axis': 1}}
nus = NuScenesDataset(**dataset_config) 
h1 = det_model.pts_backbone.blocks._modules['0'].register_forward_hook(register_activation_input)
h2 = det_model.pts_backbone.blocks._modules['1'].register_forward_hook(register_activation_input)
h3 = det_model.pts_backbone.blocks._modules['1'].register_forward_hook(register_activation_output)

    
from tqdm.auto import tqdm
e_sp = []
m_sp = []
l_sp = []

length = len(nus)
with tqdm(total=length) as pbar:
    for i in range(length):
        data = nus[i]
        cloud = data.get('pointcloud')
        labels = data.get('labels')
        with torch.no_grad():
            result = inference_detector(det_model,cloud.points)
        calculate_sparsity()
        pbar.update(1)
        
        activation_list = []
import pickle
with open('early_sparisty.pkl','wb') as f:
    pickle.dump(e_sp,f)
with open('mid_sparisty.pkl','wb') as f:
    pickle.dump(m_sp,f)
with open('late_sparisty.pkl','wb') as f:
    pickle.dump(l_sp,f)