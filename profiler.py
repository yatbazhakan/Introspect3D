#%%
from glob import glob
import os
os.chdir('/mnt/ssd2/Introspect3D')
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pickle
from utils.boundingbox import BoundingBox
from tqdm.auto import tqdm
from datasets.kitti import Kitti3D
from datasets.activation_dataset import ActivationDataset
from scipy.ndimage import zoom
import open3d as o3d
from mmdet3d.apis import init_model, inference_detector
from utils.utils import generate_model_from_config
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
#%%
#INTROSPECTION model
config_int = '/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn_vis.yaml'
model_dir = "/home/yatbaz_h@WMGDS.WMG.WARWICK.AC.UK"
model_pth = "kitti_mid_single.pth"

# det_root_dir = "/mnt/ssd2/mmdetection3d/"
# model_name = 'centerpoint'
# config = 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
# checkpoint = 'centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
model_name = 'pointpillars'
config = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
checkpoint = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'#%%
# kitti_path = r"/mnt/ssd2/kitti/training/"
file_path = r"/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_aggregated_raw/"
# file_path = r"/mnt/ssd2/custom_dataset/nus_centerpoint_activations_aggregated_raw/"
file_names = sorted(glob(os.path.join(file_path,'features','*')))
# files = [pickle.load(open(file_name,'rb')) for file_name in file_names[:10]]
# labels = pd.read_csv(os.path.join(file_path,'kitti_point_pillars_labels_aggregated_raw.csv'))
kitti_classes = ['Car', 'Pedestrian', 'Cyclist']
# kitti_dataset = Kitti3D(kitti_path, kitti_classes, 'FilterType.NONE', filter_params={})
#%%
import time
activation_dataset = ActivationDataset({'root_dir':file_path,
                                        'classes':["No Error","Error"],
                                        'label_file': 'kitti_point_pillars_labels_aggregated_raw.csv', #'nus_centerpoint_labels_aggregated_raw.csv', #'kitti_point_pillars_labels_aggregated_raw.csv', #
                                        'label_field':'is_missed',
                                        'layer':0,
                                        'is_multi_feature':False,
                                        'name':'kitti'})


introspection_model = generate_model_from_config({'layer_config': config_int})
# introspection_model.load_state_dict(torch.load(os.path.join(model_dir,model_pth)))
# introspection_model.to('cuda:1')
iters = 1000
bound = 1
ms= []
for i in range(bound):
    data = activation_dataset[i]
    image,label,_ = data
    image = image.unsqueeze(0)
    # image = image.to('cuda:1')
    # print(image.shape)
    with torch.no_grad():
        output = introspection_model(image)
    for j in range(iters):
        with torch.no_grad():
            start = time.perf_counter()
            output = introspection_model(image)
            ms.append((time.perf_counter()-start)*1000)
print(f"Inference time: {np.mean(ms):.4f} ms ({np.std(ms):.2f} ms)")

    
# %%
