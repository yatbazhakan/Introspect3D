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
config_int = '/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn_vis.yaml'
model_dir = "/home/yatbaz_h@WMGDS.WMG.WARWICK.AC.UK"
model_pth = "kitti_filtered_both.pth"
#%%
det_root_dir = "/mnt/ssd2/mmdetection3d/"
model_name = 'centerpoint'
config = 'centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
checkpoint = 'centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
# model_name = 'pointpillars'
# config = 'pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py' #pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py
# checkpoint = 'hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'#%%
kitti_path = r"/mnt/ssd2/kitti/training/"
# file_path = r"/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_aggregated_raw/"
file_path = r"/mnt/ssd2/custom_dataset/nus_centerpoint_activations_aggregated_raw/"
file_names = sorted(glob(os.path.join(file_path,'features','*')))
files = [pickle.load(open(file_name,'rb')) for file_name in file_names[:10]]
# labels = pd.read_csv(os.path.join(file_path,'kitti_point_pillars_labels_aggregated_raw.csv'))
kitti_classes = ['Car', 'Pedestrian', 'Cyclist']
kitti_dataset = Kitti3D(kitti_path, kitti_classes, 'FilterType.ELLIPSE', filter_params=dict(a = 25,
                                                                                            b = 15,
                                                                                            offset= -10,
                                                                                            axis= 0) )
# nuscenes_dataset = NuScenesDataset(root_dir='/mnt/ssd2/nuscenes/', 
#                                    version='v1.0-trainval', 
#                                    filtering_style='FilterType.NONE', 
#                                    filter_params={},
#                                    process=False,
#                                    save_path='/mnt/ssd2/nuscenes',
#                                    save_filename='nuscenes_train.pkl')
#%%
activation_dataset = ActivationDataset({'root_dir':file_path,
                                        'classes':["No Error","Error"],
                                        'label_file':'nus_centerpoint_labels_aggregated_raw.csv', ##'kitti_point_pillars_labels_aggregated_raw.csv',#
                                        'label_field':'is_missed',
                                        'layer':[0,1,2],
                                        'is_multi_feature':True,
                                        'name':'nuscenes'})
#%%

object_det_config = os.path.join(det_root_dir,"configs",model_name,config)
object_det_checkpoint = os.path.join(det_root_dir,"ckpts",checkpoint)
model = init_model(object_det_config, object_det_checkpoint, device='cuda:1')

# %%#
from utils.process import MultiFeatureActivationEarlyFused
multi = False
introspection_model = generate_model_from_config({'layer_config': config_int})
introspection_model.load_state_dict(torch.load(os.path.join(model_dir,model_pth)))
processor = MultiFeatureActivationEarlyFused(config={})

introspection_activations = None
introspection_activations = []


# my_hook  = introspection_model[0].layer4.register_forward_hook(hook_func)
# my_hook2 = introspection_model[0].layer3.register_forward_hook(hook_func)
# my_hook3 = introspection_model[0].layer2.register_forward_hook(hook_func)
my_hook4 = introspection_model[0].layer1.register_forward_hook(hook_func)

import torch.nn.functional as TF
idx = 2
introspection_model.to('cuda:1')
introspection_model.eval()
    
tensor , label, file_name = activation_dataset[idx]
if not multi:
    tensor = tensor.to('cuda:1')
    res = introspection_model(tensor.unsqueeze(0))
else:
    tensor = [t.unsqueeze(0) for t in tensor]
    tensor = processor.process(activation=tensor,stack=True)
    tensor = tensor.to('cuda:1')
    res = introspection_model(tensor)
    tensor= tensor.squeeze(0)
res_sm, label = TF.softmax(res,dim=1), label
# my_hook.remove()
# my_hook2.remove()
# my_hook3.remove()
my_hook4.remove()
# del my_hook, my_hook2, my_hook3, my_hook4
res_sm,label
# %%
# idx = 2
# #RUn detection and visualize with open3d
# print(nuscenes_dataset[idx]['pointcloud'].points.shape)
# print(type(nuscenes_dataset[idx]['pointcloud']))
# data = nuscenes_dataset[idx]['pointcloud']
# data.validate_and_update_descriptors(extend_or_reduce=5)
# # nuscenes_dataset[idx]['pointcloud'].points = nuscenes_dataset[idx]['pointcloud'].point
# print(nuscenes_dataset[idx]['pointcloud'].raw_points.shape)
# # nuscenes_dataset[idx]['pointcloud'].po
# detections = inference_detector(model,data.points)
# detections[0].pred_instances_3d
#%%
# dets= detections[0].pred_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
# scores = detections[0].pred_instances_3d.scores_3d.detach().cpu().numpy()
# filtered_indices = np.where(scores >= 0.5)[0]
# dets = dets[filtered_indices]
# dets
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
def get_rotated_corners(x, y, z, w, l, yaw, pitch=0, roll=0):
    # Define the corners of the rectangle before rotation
    corners = np.array([
        [-l / 2, -w / 2, z], [l / 2, -w / 2, z],
        [l / 2, w / 2, z], [-l / 2, w / 2, z]
    ])

    # Rotation matrices for yaw (Z-axis), pitch (Y-axis), and roll (X-axis)
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # Combined rotation matrix
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    # Rotate the corners
    rotated_corners = np.dot(corners, R.T)

    # Translate the corners to the position of the object
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y

    # For 2D visualization, return only the x and y coordinates
    return rotated_corners[:, :2]


dets = []
labels = []
# points = nuscenes_dataset[idx]['pointcloud'].points
points = kitti_dataset[idx]['pointcloud'].points
# Assuming 'points' is your N,3 point cloud data
x = points[:, 0]  # X coordinates
y = points[:, 1]  # Y coordinates
# indices = x[:] >=0 
# x = x[indices]
# y = y[indices]
fig = plt.figure()
fig.set_facecolor('black')

plt.scatter(x, y, s=0.01,c="white")  # s is the size of each point
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('2D Top-Down View of Point Cloud')
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.axis('off')
# plt.axis('equal')  # To maintain aspect ratio
# print("Labels", len(nuscenes_dataset[idx]['labels']))
for detection in dets:
    x, y, z, w, l, h, yaw,_,_= detection
    
    # box.rotation = R
    corners = get_rotated_corners(x, y,z, w, l, yaw=yaw)
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
plt.savefig('nus_original.png',bbox_inches='tight',pad_inches=0,dpi=300)
#%%
numpy_tens  = tensor.detach().cpu().numpy()

eigen_tens = get_2d_projection(numpy_tens[np.newaxis,:])
#%%
print(eigen_tens.shape)
#%%
import cv2
norm_tens = (numpy_tens - numpy_tens.min()) / (numpy_tens.max() - numpy_tens.min())
gray_tens = (norm_tens * 255).astype(np.uint8)
grayscale_tens_Cv = cv2.cvtColor(gray_tens[:,:,np.newaxis], cv2.COLOR_GRAY2RGB)
# cmap_rgb_Cv = cv2.applyColorMap(grayscale_tens_Cv, cv2.COLORMAP_JET)
plt.imshow(eigen_tens)
plt.xticks([])
plt.yticks([])
plt.show()
#%%
import cv2
import matplotlib.cm as cm
import matplotlib
max_acti = tensor.detach().cpu().numpy().max(axis=0)
print(max_acti.shape)
norm_max_acti = (max_acti - max_acti.min()) / (max_acti.max() - max_acti.min())
grayscale_org = (norm_max_acti * 255).astype(np.uint8)
# grayscale = cm.jet(grayscale)
#black to white and white to coloer 
# cmap = cm.get_cmap('viridis')
# testh = cmap(max_acti)
plt.imshow(grayscale_org,cmap='gray')
plt.xticks([])
plt.yticks([])
# plt.axis('off')
plt.savefig('nus_acti_proposed.png',bbox_inches='tight',pad_inches=0,dpi=600)
# %%
import cv2
im = cv2.imread(kitti_dataset.image_paths[idx])
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.show()
# %%

for i in range(4):
    print(introspection_activations[i].shape)
# introspection_activations[0].shape
# int_acti = introspection_activations[0].detach().cpu().numpy()



eigen_cams = []
for i in range(4):
    int_acti = introspection_activations[i].detach().cpu().numpy()
    eigen_cam = get_2d_projection(int_acti)
    eigen_cams.append(eigen_cam)

# eigen_cam = get_2d_projection(int_acti)

scaled_cams = []
for cam in eigen_cams:

    # max_int_acti = int_acti.squeeze(0).max(axis=0)
# mean_int_acti = int_acti.squeeze(0).mean(axis=0)
# std_int_acti = int_acti.squeeze(0).std(axis=0)  
#resize this to max_acti.shape
    target_shape = max_acti.shape
    zoom_factor = [t/m for t,m in zip(target_shape,cam.shape)]
    print(zoom_factor)
    int_acti_n = zoom(cam,  zoom_factor, order=1)
    scaled_cams.append(int_acti_n)
    # max_int_acti_n = zoom(max_int_acti,  zoom_factor, order=1)
    # mean_int_acti_n = zoom(mean_int_acti,  zoom_factor, order=1)
    # std_int_acti_n = zoom(std_int_acti,  zoom_factor, order=1)
    # print(int_acti_n.shape)
    # print(max_int_acti.shape)
#Resize to the shape of max acti
#%%
import cv2
import matplotlib.cm as cm
def inv(image):
    return abs(255-image)
# fig, ax = plt.subplots(1,6,figsize=(10,10))
# cmap = cm.jet
# r,c = 1,5
alpha, beta = 0.55, 0.45
# plt.subplot(r,c,1)
# plt.imshow(grayscale_org,cmap='gray')
plt.xticks([])
plt.yticks([])
# plt.axis('off')
for i,cam in enumerate(scaled_cams):
    # if i%2 == 0:
    #     cam = 1-cam
    #Subplot row,col =
    # row,col = (i+1)//5,(i+1)%5
    if i != 3:
        continue
    # plt.subplot(r,c,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    cam = 1-cam
    norm_cam = (cam - cam.min()) / (cam.max() - cam.min())
    grayscale = (norm_cam * 255).astype(np.uint8)
    # cmap_map = cmap(grayscale)
    grayscale_Cv = cv2.cvtColor(grayscale[:,:,np.newaxis], cv2.COLOR_GRAY2RGB)
    cmap_rgb = cv2.applyColorMap(grayscale_Cv, cv2.COLORMAP_JET)
    rgb_grayscale_org = cv2.cvtColor(grayscale_org, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(rgb_grayscale_org, alpha,cmap_rgb, beta,0.0)
    # blended = (blended - blended.min()) / (blended.max() - blended.min())
    # colored = (blended * 255).astype(np.uint8)
    # colored = cmap(colored)
    plt.imshow(blended,cmap='jet')
    plt.savefig('nus_eigen_proposed_last.png',bbox_inches='tight',pad_inches=0,dpi=600)
        

# cbar = plt.colorbar(fraction=0.1, pad=0.04)
# cbar.solids.set_edgecolor("face")
# # plt.tight_layout()
# print(cbar.cmap.name)
# plt.savefig('kitti_eigen_cam_layers_proposed_jet.png',bbox_inches='tight',pad_inches=0,dpi=600)
# %%
#subplots
fig, ax = plt.subplots(1,5)
plt.subplot(1,5,1)
plt.imshow(max_acti,cmap='gray')
plt.subplot(1,5,2)
plt.imshow(1-int_acti_n,cmap='gray')
plt.subplot(1,5,3)
plt.imshow(max_int_acti_n,cmap='gray')
plt.subplot(1,5,4)
plt.imshow(mean_int_acti_n,cmap='gray')
plt.subplot(1,5,5)
plt.imshow(std_int_acti_n,cmap='gray')
plt.show()
# %%
