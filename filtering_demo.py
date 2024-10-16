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
from utils.filter import FilterType
from datasets.nuscenes import NuScenesDataset
from datasets.activation_dataset import ActivationDataset
from scipy.ndimage import zoom
import open3d as o3d
from mmdet3d.apis import init_model, inference_detector
from utils.utils import generate_model_from_config
from operators.introspector import IntrospectionOperator
introspection_activations = []
def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    print(activation_batch.shape)
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        del reshaped_activations
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return projections[0]
def hook_func(module, input, output):
    introspection_activations.append(output)
    print(len(introspection_activations))
#%%
#INTROSPECTION model
config_int = '/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn2.yaml'
model_dir = "/home/yatbaz_h@WMGDS.WMG.WARWICK.AC.UK/" #'/mnt/ssd2/Introspect3D/'#
model_pth = 'nuscenes_filtered_labels_only.pth'
model_spatial = "nuscenes_filtered.pth"
#%%
det_root_dir = "/mnt/ssd2/mmdetection3d/"
model_name = 'centerpoint'
config = 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
checkpoint = 'centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
# model_name = 'pointpillars'
# config = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
# checkpoint = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'#%%
# kitti_path = r"/mnt/ssd2/kitti/training/"
# file_path = r"/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_filtered/"
file_path = r"/mnt/ssd2/custom_dataset/nus_centerpoint_activations_filtered/"
file_path2 = r"/mnt/ssd2/custom_dataset/nus_centerpoint_activations_aggregated_raw/"
#%%
activation_dataset_filtered = ActivationDataset({'root_dir':file_path,
                                        'classes':["No Error","Error"],
                                        'label_file': 'nus_centerpoint_labels_filtered.csv',#'nus_centerpoint_labels_filtered.csv', #'kitti_point_pillars_labels_filtered.csv',#'nus_centerpoint_labels_aggregated_raw.csv', ##'kitti_point_pillars_labels_aggregated_raw.csv',#
                                        'label_field':'is_missed',
                                        'layer':None,
                                        'is_multi_feature':False,
                                        'name':'nuscenes',
                                        'extension':'.npy'})
activation_dataset_raw = ActivationDataset({'root_dir':file_path2,
                                        'classes':["No Error","Error"],
                                        'label_file': 'nus_centerpoint_labels_aggregated_raw_filtered.csv',#'nus_centerpoint_labels_filtered.csv', #'kitti_point_pillars_labels_filtered.csv',#'nus_centerpoint_labels_aggregated_raw.csv', ##'kitti_point_pillars_labels_aggregated_raw.csv',#
                                        'label_field':'is_missed',
                                        'layer':2,
                                        'is_multi_feature':False,
                                        'name':'nuscenes',
                                        'extension':''})
#%%
from utils.process import MultiFeatureActivationEarlyFused
import torch.nn.functional as TF

multi = False
int_filtered= generate_model_from_config({'layer_config': config_int})
int_filtered.load_state_dict(torch.load(os.path.join(model_dir,model_spatial),map_location='cuda:0'))
int_raw = generate_model_from_config({'layer_config': config_int})
int_raw.load_state_dict(torch.load(os.path.join(model_dir,model_pth),map_location='cuda:0'))
#%%
from tqdm.auto import tqdm
results_dict = {'spatial_filtering' : {}, 'label_only': {}}
int_filtered.to('cuda:0')
int_filtered.eval()
int_raw.to('cuda:0')
int_raw.eval()
with tqdm(activation_dataset_filtered)as pbar:
    for samp,samp2 in zip(activation_dataset_filtered,activation_dataset_raw):
        
        tensor , label, file_name = samp
        tensor2 , label2, file_name2 = samp2
        if(label.item() != label2.item()):
            continue
        if not multi:
            tensor = tensor.to('cuda:0')
            res = int_filtered(tensor.unsqueeze(0))
            tensor2 = tensor.to('cuda:0')
            res2 = int_raw(tensor.unsqueeze(0))
        else:
            tensor = [t.unsqueeze(0) for t in tensor]
            # tensor = processor.process(activation=tensor,stack=True)
            tensor = tensor.to('cuda:0')
            print(tensor.shape)
            res = int_filtered(tensor)
            tensor= tensor.squeeze(0)
        res_sm, label = TF.softmax(res,dim=1), label
        predicted_label = torch.argmax(res_sm).item()
        res_sm2, label2 = TF.softmax(res2,dim=1), label2
        predicted_label2 = torch.argmax(res_sm2).item()
        # print(predicted_label, label)
        if predicted_label2 != label2.item() and predicted_label == label.item():
            results_dict['spatial_filtering'][file_name] = {'sf_prediction': predicted_label, 'label': label.item()}
            results_dict['label_only'][file_name2] = {'lo_prediction': predicted_label2, 'label': label2.item()}

        pbar.update(1)
    pd.DataFrame.from_dict(results_dict,orient='index').to_csv('nus_all.csv')

# #%%
# # %%
# int_raw.to('cuda:0')
# int_raw.eval()
# with tqdm(activation_dataset_raw)as pbar:
#     for samp in activation_dataset_raw:
        
#         tensor , label, file_name = samp
#         if not multi:
#             tensor = tensor.to('cuda:0')
#             res = int_raw(tensor.unsqueeze(0))
#         else:
#             tensor = [t.unsqueeze(0) for t in tensor]
#             # tensor = processor.process(activation=tensor,stack=True)
#             tensor = tensor.to('cuda:0')
#             print(tensor.shape)
#             res = int_raw(tensor)
#             tensor= tensor.squeeze(0)
#         res_sm, label = TF.softmax(res,dim=1), label
#         predicted_label = torch.argmax(res_sm).item()
#         # print(predicted_label, label)
#         results_dict['label_only'][file_name] = {'activations': predicted_label, 'label': label.item()}
#         pbar.update(1)

# # %%
# import pandas as pd
# pd.DataFrame.from_dict(results_dict['spatial_filtering'],orient='index').to_csv('nus_centerpoint_filtered_results.csv')
# pd.DataFrame.from_dict(results_dict['label_only'],orient='index').to_csv('nus_centerpoint_raw_results.csv')
# # %%
