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
# model_pth = "kitti_mid_single.pth"

det_root_dir = "/mnt/ssd2/mmdetection3d/"
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
# kitti_classes = ['Car', 'Pedestrian', 'Cyclist']
# kitti_dataset = Kitti3D(kitti_path, kitti_classes, 'FilterType.NONE', filter_params={})
#%%
import time
activation_dataset = ActivationDataset({'root_dir':file_path,
                                        'classes':["No Error","Error"],
                                        'label_file': 'kitti_point_pillars_labels_aggregated_raw.csv', #'nus_centerpoint_labels_aggregated_raw.csv', #, #
                                        'label_field':'is_missed',
                                        'layer':2,
                                        'is_multi_feature':False,
                                        'name':'kitti'})

#%%
from thop import profile
detector = init_model(os.path.join(det_root_dir,"configs",model_name,config),
                      os.path.join(det_root_dir,"ckpts",checkpoint),
                      device='cuda:0')
pc_process = detector.data_preprocessor
#%
introspection_model = generate_model_from_config({'layer_config': config_int})
# introspection_model.load_state_dict(torch.load(os.path.join(model_dir,model_pth)))
# introspection_model.to('cuda:1')
from utils.process import MultiFeatureActivationEarlyFused
processor = MultiFeatureActivationEarlyFused(config={})
data ,_,_= activation_dataset[0]
# data = [t.unsqueeze(0) for t in data]


# data = processor.process(activation=data,stack=True)
input_shape = tuple(data.unsqueeze(0).shape)

input = torch.randn(input_shape)
macs, params = profile(introspection_model, inputs=(input, ))
from thop import clever_format
macs, params = clever_format([macs, params], "%.3f")
print(f"MACs: {macs}B, Parameters: {params}M")
# with get_accelerator().device(1):
#     flops, macs, params = get_model_profile(model=introspection_model, # model
#                                     input_shape=input_shape, # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
#                                     args=None, # list of positional arguments to the model.
#                                     kwargs=None, # dictionary of keyword arguments to the model.
#                                     print_profile=True, # prints the model graph with the measured profile attached to each module
#                                     detailed=True, # print the detailed profile
#                                     module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
#                                     top_modules=1, # the number of top modules to print aggregated profile
#                                     warm_up=10, # the number of warm-ups before measuring the time of each module
#                                     as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
#                                     output_file=None, # path to the output file. If None, the profiler prints to stdout.
#                                     ignore_modules=None, # the list of modules to ignore in the profiling
#                                     ) 
# total_params = sum(p.numel() for p in introspection_model.parameters())
# print(f"Number of parameters: {total_params/1e6}M")
# iters = 1000
# bound = 1
# ms= []
# for i in range(bound):
#     data = activation_dataset[i]
#     image,label,_ = data
#     image = image.unsqueeze(0)
#     # image = image.to('cuda:1')
#     # print(image.shape)
#     with torch.no_grad():
#         output = introspection_model(image)
#     for j in range(iters):
#         with torch.no_grad():
#             start = time.perf_counter()
#             output = introspection_model(image)
#             ms.append((time.perf_counter()-start)*1000)
# print(f"Inference time: {np.mean(ms):.4f} ms ({np.std(ms):.2f} ms)")

    
# %%
import numpy as np
from copy import deepcopy
from mmengine.dataset import Compose, pseudo_collate

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.structures import Box3DMode, Det3DDataSample, get_box_type
from mmdet3d.structures.det3d_data_sample import SampleList
def prepare_data(model,pcd):
    pass
cfg = detector.cfg
pcds = ["/mnt/ssd2/kitti/training/velodyne/000001.bin"]

# build the data pipeline
test_pipeline = deepcopy(cfg.test_dataloader.dataset.pipeline)
test_pipeline = Compose(test_pipeline)
box_type_3d, box_mode_3d = \
    get_box_type(cfg.test_dataloader.dataset.box_type_3d)

data = []
for pcd in pcds:
    # prepare data
    if isinstance(pcd, str):
        # load from point cloud file
        data_ = dict(
            lidar_points=dict(lidar_path=pcd),
            timestamp=1,
            # for ScanNet demo we need axis_align_matrix
            axis_align_matrix=np.eye(4),
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
    else:
        # directly use loaded point cloud
        data_ = dict(
            points=pcd,
            timestamp=1,
            # for ScanNet demo we need axis_align_matrix
            axis_align_matrix=np.eye(4),
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d)
    data_ = test_pipeline(data_)
    data.append(data_)
collate_data = pseudo_collate(data)

ppc = detector.data_preprocessor(collate_data)
features, num_points, coordinates = ppc['inputs']['points'], ppc['inputs']['voxels']['num_points'], ppc['inputs']['voxels']['voxel_centers']
test=  detector.voxel_encoder(features, num_points, coordinates)
# feat =detector.backbone(ppc)
# %%
times = []
for i in range(1000):
    start = time.perf_counter()
    res = inference_detector(detector, "/mnt/ssd2/kitti/training/velodyne/000001.bin")
    times.append(time.perf_counter()-start)
print(f"Average inference time: {np.mean(times):.4f} s")
#%%
import mmcv
from mmdet3d.apis import inference_detector, init_model

# Load the configuration
path = './test_model_config.py'

# Build the model from the configuration
model = init_model(path)
# %%
import mmcv
from mmengine import Config, build_from_cfg
from mmdet3d.models.detectors import Base3DDetector,VoxelNet

conf = Config.fromfile(path)
model = build_from_cfg(conf.model,VoxelNet)
# %%
