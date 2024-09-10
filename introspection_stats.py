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
import random
#%%
def train_test_splitc(dataset):
    indices = list(range(len(dataset)))
    all_labels= dataset.get_all_labels()
    random_state = 1024
    train_indices, test_indices = train_test_split(indices, test_size=0.2,stratify=all_labels,random_state=random_state)
    after_val_train_indices, val_indices = train_test_split(train_indices, test_size=0.15,stratify=all_labels[train_indices],random_state=random_state)
    values,counts = np.unique(all_labels,return_counts=True)
    class_dist=  dict(zip(values,counts))
    c = 1e-3
    class_weights = [len(all_labels)/float(count) for cls, count in class_dist.items()]
    # if validation['balanced']:
    #     train_labels_after_val = [all_labels[i] for i in after_val_train_indices]
    #     class_counts = np.bincount(train_labels_after_val)
    #     min_class_count = np.min(class_counts)
    #     print("Min class count:",min_class_count,"Class counts:",class_counts)
    #     balanced_train_indices = []
    #     for cls in np.unique(train_labels_after_val):
    #         cls_indices = [i for i, label in zip(after_val_train_indices, train_labels_after_val) if label == cls]
    #         balanced_cls_indices = np.random.choice(cls_indices, min_class_count, replace=False)
    #         balanced_train_indices.extend(balanced_cls_indices)
    #     after_val_train_indices = balanced_train_indices
    #     balanced_train_labels = [all_labels[i] for i in balanced_train_indices]
    #     values, counts = np.unique(balanced_train_labels, return_counts=True)
    #     new_class_dist = dict(zip(values, counts))
    #     total_samples = len(balanced_train_indices)
    #     class_weights = [ total_samples / (len(values) * count) for cls, count in new_class_dist.items()]
    #     class_dist = new_class_dist
    
    
    
    train_dataset = Subset(dataset, after_val_train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    print("Class distribution:",class_dist)
    print("Class weights:",class_weights)
    return test_dataset
#%%
#INTROSPECTION model
config_int = '/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn_mult_c.yaml'
model_dir = "/home/yatbaz_h@WMGDS.WMG.WARWICK.AC.UK"
model_pth = "nuscenes_proposed.pth"

det_root_dir = "/mnt/ssd2/mmdetection3d/"
# model_name = 'centerpoint'
# config = 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
# checkpoint = 'centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
# model_name = 'pointpillars'
# config = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
# checkpoint = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'#%%
# kitti_path = r"/mnt/ssd2/kitti/training/"
# file_path = r"/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_aggregated_raw/"
file_path = r"/mnt/ssd2/custom_dataset/nus_centerpoint_activations_aggregated_raw/"
file_names = sorted(glob(os.path.join(file_path,'features','*')))
# files = [pickle.load(open(file_name,'rb')) for file_name in file_names[:10]]
# labels = pd.read_csv(os.path.join(file_path,'kitti_point_pillars_labels_aggregated_raw.csv'))
kitti_classes = ['Car', 'Pedestrian', 'Cyclist']
# kitti_dataset = Kitti3D(kitti_path, kitti_classes, 'FilterType.NONE', filter_params={})
#%%
activation_dataset = ActivationDataset({'root_dir':file_path,
                                        'classes':["No Error","Error"],
                                        'label_file':'nus_centerpoint_labels_aggregated_raw.csv', #'kitti_point_pillars_labels_aggregated_raw.csv',#
                                        'label_field':'is_missed',
                                        'layer':2,
                                        'is_multi_feature':True,
                                        'name':'nuscenes'})
activation_dataset = train_test_splitc(activation_dataset)
# %%
from modules.custom_networks import CustomModel
from utils.process import *
multi = True

introspection_model = generate_model_from_config({'layer_config': config_int})
# introspection_model = CustomModel({'layer_config': config_int},device='cuda:1')
introspection_model.load_state_dict(torch.load(os.path.join(model_dir,model_pth)))
processor = MultiFeatureActivationEarlyFused({})
import torch.nn.functional as TF
idx = 2
introspection_model.to('cuda:1')
introspection_model.eval()
tp, tn, fp, fn = [],[],[],[]
with tqdm(total=len(activation_dataset)) as pbar:
    for i in range(len(activation_dataset)):
        tensor, label, _ = activation_dataset[i]
        if not multi:
            tensor = tensor.to('cuda:1')
            tensor = tensor.unsqueeze(0)
        else:
            tensor = [t.unsqueeze(0) for t in tensor]
            # print(tensor[0].shape,tensor[1].shape,tensor[2].shape)
            tensor = processor.process(activation = tensor, stack = True)
            tensor = tensor.to('cuda:1')
            # tensor = tensor.unsqueeze(0)
        res = introspection_model(tensor)
        res_sm, label = TF.softmax(res,dim=1), label
        max_index= res_sm.argmax()
        val_in_max_index = res_sm[0][max_index].item()
        if label.item() == max_index.item():
            if label.item() == 1:
                tp.append(val_in_max_index)
            else:
                tn.append(val_in_max_index)
        else:
            if label.item() == 1:
                fn.append(val_in_max_index)
            else:
                fp.append(val_in_max_index)
        pbar.update(1)

    
#%%

import pickle 
with open('tp_proposed_nus.pkl','wb') as f:
    pickle.dump(tp,f)
with open('tn_proposed_nus.pkl','wb') as f:
    pickle.dump(tn,f)
with open('fp_proposed_nus.pkl','wb') as f:
    pickle.dump(fp,f)
with open('fn_proposed_nus.pkl','wb') as f:
    pickle.dump(fn,f)
# %%
# import pickle
# with open('tp.pkl','rb') as f:
#     tp = pickle.load(f)
# with open('tn.pkl','rb') as f:
#     tn = pickle.load(f)
# with open('fp.pkl','rb') as f:
#     fp = pickle.load(f)
# with open('fn.pkl','rb') as f:
#     fn = pickle.load(f)
