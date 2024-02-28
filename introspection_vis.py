#%%
from glob import glob
import os
os.chdir('/mnt/ssd2/Introspect3D')
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pickle
from tqdm.auto import tqdm
from datasets.kitti import Kitti3D
from datasets.activation_dataset import ActivationDataset
import open3d as o3d
from mmdet3d.apis import init_model, inference_detector
from utils.utils import generate_model_from_config
#%%
#INTROSPECTION model
config_int = '/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn_vis.yaml'
model_dir = "/home/yatbaz_h@WMGDS.WMG.WARWICK.AC.UK"
model_pth = "kitti_early_single.pth"
#%%
det_root_dir = "/mnt/ssd2/mmdetection3d/"
# model_name = 'centerpoint'
# config = 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
# checkpoint = 'centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
model_name = 'pointpillars'
config = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
checkpoint = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'#%%
kitti_path = r"/mnt/ssd2/kitti/training/"
file_path = r"/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_aggregated_raw/"
file_names = sorted(glob(os.path.join(file_path,'features','*')))
files = [pickle.load(open(file_name,'rb')) for file_name in file_names[:10]]
labels = pd.read_csv(os.path.join(file_path,'kitti_point_pillars_labels_aggregated_raw.csv'))
kitti_classes = ['Car', 'Pedestrian', 'Cyclist']
kitti_dataset = Kitti3D(kitti_path, kitti_classes, 'FilterType.NONE', filter_params={})
#%%
activation_dataset = ActivationDataset({'root_dir':file_path,
                                        'classes':["No Error","Error"],
                                        'label_file':'kitti_point_pillars_labels_aggregated_raw.csv',
                                        'label_field':'is_missed',
                                        'layer':0,
                                        'is_multi_feature':False,
                                        'name':'kitti'})
#%%

object_det_config = os.path.join(det_root_dir,"configs",model_name,config)
object_det_checkpoint = os.path.join(det_root_dir,"ckpts",checkpoint)
model = init_model(object_det_config, object_det_checkpoint, device='cuda:1')

# %%
introspection_model = generate_model_from_config({'layer_config': config_int})
introspection_model.load_state_dict(torch.load(os.path.join(model_dir,model_pth)))
# %%
import torch.nn.functional as TF
idx = 2
introspection_model.to('cuda:1')
introspection_model.eval()
    
tensor , label, file_name = activation_dataset[idx]
res = introspection_model(tensor.unsqueeze(0).cuda())
res_sm, label = TF.softmax(res,dim=1), label
res_sm,label
# %%
#RUn detection and visualize with open3d
kitti_dataset[idx]['pointcloud'].convert_to_kitti_points()
detections = inference_detector(model,kitti_dataset[idx]['pointcloud'].points)
detections[0].pred_instances_3d
#%%
dets= detections[0].pred_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
dets
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def get_rotated_corners(x, y, w, l, yaw):
    # Calculate the corners of the rectangle before rotation
    corners = np.array([
        [-l / 2, -w / 2], [l / 2, -w / 2],
        [l / 2, w / 2], [-l / 2, w / 2]
    ])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    # Rotate the corners
    rotated_corners = np.dot(corners, rotation_matrix)

    # Translate the corners to the position of the object
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y

    return rotated_corners
points = kitti_dataset[idx]['pointcloud'].points
# Assuming 'points' is your N,3 point cloud data
x = points[:, 0]  # X coordinates
y = points[:, 1]  # Y coordinates

plt.scatter(x, y, s=1)  # s is the size of each point
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D Top-Down View of Point Cloud')
plt.axis('equal')  # To maintain aspect ratio

for detection in dets:
    x, y, _, _, w, l, yaw = detection
    corners = get_rotated_corners(x, y, w, l, yaw)

    # Create a polygon patch
    polygon = patches.Polygon(corners, closed=True, linewidth=1, edgecolor='r', facecolor='none')

    # Add the polygon to the Axes
    plt.gca().add_patch(polygon)
for label in kitti_dataset[idx]['labels']:
    
    # x, y, _, _, w, l, yaw = label.center[0],label.center[1],label.center[2],label.dimensions[0],label.dimensions[1],label.dimensions[2],label.rotation[1]
    print(label.corners.shape)
    corners = label.corners[:,:2]
    polygon = patches.Polygon(corners, closed=True, linewidth=1, edgecolor='g', facecolor='none')
    plt.gca().add_patch(polygon)
plt.show()
# %%
import cv2
im = cv2.imread(kitti_dataset.image_paths[idx])
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
# %%
