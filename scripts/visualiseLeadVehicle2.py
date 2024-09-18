import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('.')
import pandas as pd 
import open3d as o3d
from nuscenes.nuscenes import NuScenes
import os
from enum import Enum
from nuscenes.utils.data_classes import LidarPointCloud,Box
from typing import Union
from utils.boundingbox import BoundingBox
from pyquaternion import Quaternion
import time
import cv2
from tqdm import tqdm
import pdb
from PIL import Image


class Colors(Enum):
    DARK_RED = (0.5,0.,0.)
    DARK_GREEN = (0.,0.5,0.)
    DARK_BLUE = (0.,0.,0.5)
    DARK_YELLOW = (0.5,0.5,0.)
    DARK_CYAN = (0.,0.5,0.5)
    DARK_MAGENTA = (0.5,0.,0.5)
    DARK_WHITE = (0.5,0.5,0.5)
    BLACK = (0.,0.,0.)
    LIGHT_GRAY = (0.75,0.75,0.75)

def isTrusted(res_dict):
    label = res_dict['label']
    pred = res_dict['pred']
    pred = np.argmax(pred)
    print(f'Pred: {pred}, Label: {label}')
    return pred<1, label<1

def create_line_set_bounding_box(box: BoundingBox,
                                offset:Union[float, np.ndarray],
                                axis:Union[int, np.ndarray],
                                color:Union[tuple, np.ndarray] = Colors.BLACK) -> o3d.geometry.LineSet:

    if box.corners.shape[0] != 8:
        temp_corners = box.corners.T
    else:
        temp_corners = box.corners
    o3d_box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(temp_corners))
    o3d_box.color = color
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_box)
    return line_set

with(open('outputs/lv_scene_dicts.pkl','rb')) as f:
        scene_list = pickle.load(f)

with(open('outputs/results/results_byfile.pkl','rb')) as f:
    introspection_results = pickle.load(f)

scene_number = 849
scene_data = scene_list[scene_number]
tokens = scene_data['sample_token']
gt_bbox = scene_data['gt_lead']
pred_bbox = scene_data['pred_lead']
ego_pose = scene_data['ego_pose']
lidar_path = scene_data['lidar_path']
#print(introspection_results.keys())
#print(lidar_path[0].split('/')[-1])
#exit()
sensor_calib = scene_data['sensor_calib']
s_translation = sensor_calib['translation']
s_rotation = Quaternion(sensor_calib['rotation']).rotation_matrix
resolution = (1600, 1600)
#progress = tqdm(total=len(tokens))
for itr, token in enumerate(tokens):
    print(f'Processing {itr}th frame')
    fig, ax = plt.subplots(figsize = (resolution[0]//100, resolution[1]//100))
    # write a text in top left corner
    if lidar_path[itr].split('/')[-1] in introspection_results.keys():
        valid_key = lidar_path[itr].split('/')[-1]
    else:
        print('Key not found')
    pred_trusted, gt_trusted = isTrusted(introspection_results[valid_key])
    #pred_trusted, gt_trusted = isTrusted(introspection_results['n008-2018-08-01-15-52-19-0400__LIDAR_TOP__1533153686798455'])
    if gt_trusted:
        ax.text(-60, 60, 'Distance Trusted! (GT)'.format(scene_number), color='Green', backgroundcolor='white', fontsize=25)
    else:
        ax.text(-60, 60, 'Distance Not Trusted! (GT)'.format(scene_number), color='Red', backgroundcolor='white', fontsize=25)
    if pred_trusted:
        ax.text(-60, 55, 'Distance Trusted! (Pred)'.format(scene_number), color='Green', backgroundcolor='white', fontsize=25)
    else:
        ax.text(-60, 55, 'Distance Not Trusted! (Pred)'.format(scene_number), color='Red', backgroundcolor='white', fontsize=25)
    # create a green border arund figure
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    # set the spines color to green
    ax.spines['top'].set_color('green')
    ax.spines['right'].set_color('green')
    ax.spines['bottom'].set_color('green')
    ax.spines['left'].set_color('green')

    current_ego_pose = [ego_pose[i] for i in range(itr, len(ego_pose))]
    
    ego_xyz = [pose['translation'] for pose in current_ego_pose]
    ego_xyz = np.array(ego_xyz)
    # we dont want to change the ego_pose values
    ego_xyz = np.copy(ego_xyz)
    ego_rotation = Quaternion(current_ego_pose[0]['rotation']).rotation_matrix
    ego_translation = current_ego_pose[0]['translation']
    for ego_i in range(0, len(ego_xyz)):
        ego_xyz[ego_i] = np.dot(ego_rotation.T,ego_xyz[ego_i] - ego_translation)
        ego_xyz[ego_i] = np.dot(s_rotation.T, ego_xyz[ego_i] - s_translation)
    # ego_xyz = np.dot(ego_xyz-ego_xyz[0], ego_rotation.T)
    # ego_xyz = np.dot(ego_xyz-s_translation, s_rotation.T)
    #plot ego xyz trajectory
    ego_xyz = ego_xyz[:,:2]
    if len(ego_xyz)<2:
        break
    # extend ego_xyz to 50m
    ego_dif = ego_xyz[-1,:] - ego_xyz[-2,:]
    ego_ext = np.ones((200,2))
    ego_ext[:,0] *= ego_dif[0]
    ego_ext[:,1] *= ego_dif[1]
    ego_ext = np.cumsum(ego_ext, axis=0)
    ego_ext += ego_xyz[-1,:]
    ego_xyz = np.concatenate((ego_xyz, ego_ext), axis=0)
    # calculate distance along curve
    distance = np.cumsum(np.sqrt(np.sum(np.diff(ego_xyz, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    if np.all(distance<50):
        break
    lidar_data = LidarPointCloud.from_file(lidar_path[itr]+ '.pcd.bin')
    lidar_points = lidar_data.points.T
    # transform lidar points to ego coordinate
    #lidar_points = np.dot(lidar_points[:,:3], s_rotation) + s_translation
    points_distances = np.sqrt(lidar_points[:, 0]**2 + lidar_points[:, 1]**2)

    ax.scatter(lidar_points[:,0], lidar_points[:,1], c=points_distances, cmap='viridis', s = 1)
    ax.set_xlim(-60,60)
    ax.set_ylim(-60,60)
    ax.axis('off')
    ax.set_aspect('equal')
    # GT BBOX
    if len(gt_bbox[itr])>0:
        # transform box to ego coordinate
        box = gt_bbox[itr][0]
        corners = box.corners
        #corners = np.dot(corners, s_rotation) + s_translation
        box.corners = corners
        #gt_bbox_vis = create_line_set_bounding_box(box,0,0,Colors.DARK_GREEN.value)
        #visualizer.add_geometry(gt_bbox_vis)
        #plot the bounding box in 2d
        x = corners[:,0]
        y = corners[:,1]
        ax.plot(x[[0,1,2,3,0]], y[[0,1,2,3,0]], c = 'g')
        ax.plot(x[[4,5,6,7,4]], y[[4,5,6,7,4]], c = 'g')
        for i in range(4):
            ax.plot(x[[i,i+4]], y[[i,i+4]], c = 'g')
            
    if len(pred_bbox[itr])>0:
        # transform box to ego coordinate
        box = pred_bbox[itr][0]
        corners = box.corners
        #corners = np.dot(corners, ego_rotation) + ego_xyz[0]
        #corners = np.dot(corners, s_rotation) + s_translation
        box.corners = corners
        x = corners[:,0]
        y = corners[:,1]
        ax.plot(x[[0,1,2,3,0]], y[[0,1,2,3,0]], c = 'r')
        ax.plot(x[[4,5,6,7,4]], y[[4,5,6,7,4]], c = 'r')
        for i in range(4):
            ax.plot(x[[i,i+4]], y[[i,i+4]], c = 'r')
    
    
    # cut ego_xyz up to 50m
    ego_xyz = ego_xyz[distance<50]
    ax.plot(ego_xyz[:,0], ego_xyz[:,1], c = 'black')
    x = ego_xyz[:,0]
    y = ego_xyz[:,1]
    dx = np.diff(x)
    dy = np.diff(y)
    length = np.sqrt(dx**2 + dy**2)
    nx = -dy/length
    ny=dx/length
    nx = np.concatenate(([nx[0]], (nx[:-1] + nx[1:]) / 2, [nx[-1]]))
    ny = np.concatenate(([ny[0]], (ny[:-1] + ny[1:]) / 2, [ny[-1]]))

    # Normalize the normals (just to be sure)
    norm = np.sqrt(nx**2 + ny**2)
    nx /= norm
    ny /= norm

    # Shift the points by 1.5m in the direction of the normal
    distance = 1.5
    x_1 = x + distance * nx
    y_1 = y + distance * ny

    x_2 = x - distance * nx
    y_2 = y - distance * ny


    ax.plot(x_1,y_1, 'k--')
    ax.plot(x_2,y_2, 'k--')

    # At the start
    plt.plot([x[0] - distance * nx[0], x[0] + distance * nx[0]],
            [y[0] - distance * ny[0], y[0] + distance * ny[0]], 'k--')

    # At the end
    plt.plot([x[-1] - distance * nx[-1], x[-1] + distance * nx[-1]],
            [y[-1] - distance * ny[-1], y[-1] + distance * ny[-1]], 'k--')



    fig.savefig('outputs/visualisation/{}_{}.png'.format(str(scene_number).zfill(3),str(itr).zfill(2)))
    plt.close(fig)
    #progress.update(1)
if True:
        # List to store the images
        images = []
        fps_divisor = 5
        output_folder = '/home/wmg-5gcat/Desktop/Sajjad/DistEstIntrospection/Introspect3D/outputs/visualisation'
        # Loop through the folder and collect all images
        counter = 0
        for file_name in sorted(os.listdir(output_folder)):
            counter+=1
            if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                file_path = os.path.join(output_folder, file_name)
                images.append(Image.open(file_path))

        # Save images as a GIF
        output_gif_path = os.path.join(output_folder, 'video.gif')
        if images:
            images[0].save(
                output_gif_path,
                save_all=True,
                append_images=images[1:],
                duration=int(100*fps_divisor),  # Duration between frames in milliseconds
                loop=0  # 0 means loop forever
            )
            print(f"GIF created successfully at {output_gif_path}")
        else:
            print("No images found in the folder.")
    

    
   