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

def set_custom_view(vis):
    ctr = vis.get_view_control()
    
    # Define the desired camera location and orientation
    camera_position = np.array([0, 0, 0.5], dtype=np.float64)  # Ensure dtype is float for calculations
    look_at_point = np.array([0, 0, 0], dtype=np.float64)
    up_vector = np.array([0, 0, 1], dtype=np.float64)
    
    # Calculate the new front vector (z-axis of the camera coordinate system)
    front_vector = look_at_point - camera_position
    front_vector /= np.linalg.norm(front_vector)  # Normalize to create a unit vector
    
    # Calculate the right vector (x-axis of the camera coordinate system)
    right_vector = np.cross(up_vector, front_vector)
    right_vector /= np.linalg.norm(right_vector)  # Normalize to create a unit vector
    
    # Re-calculate the up vector to ensure orthogonality (y-axis of the camera coordinate system)
    up_vector = np.cross(front_vector, right_vector)
    up_vector /= np.linalg.norm(up_vector)  # Normalize to create a unit vector
    
    # # Create the extrinsic matrix
    # extrinsic = np.eye(4, dtype=np.float64)
    # extrinsic[0:3, 0] = right_vector
    # extrinsic[0:3, 1] = up_vector
    # extrinsic[0:3, 2] = front_vector
    # extrinsic[0:3, 3] = camera_position

    extrinsic = np.eye(4)
    extrinsic[0:3, 3] = [100, 0, 30]  # Set camera position (x, y, z)
    
    # Create a rotation matrix for 30-degree downward view
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(-160)), -np.sin(np.radians(-160))],
        [0, np.sin(np.radians(-160)), np.cos(np.radians(-160))]
    ])
    
    # Apply rotation to the extrinsic matrix
    extrinsic[0:3, 0:3] = rotation
    # Set the extrinsic matrix to the camera parameters
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params)
    

    # Adjust rendering options if needed
    opt = vis.get_render_option()
    opt.background_color = np.asarray(Colors.LIGHT_GRAY.value)
    opt.point_size = 2.0  # Increase point size for better visibility at closer range

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

i = 849
scene_data = scene_list[i]
tokens = scene_data['sample_token']
gt_bbox = scene_data['gt_lead']
pred_bbox = scene_data['pred_lead']
ego_pose = scene_data['ego_pose']
lidar_path = scene_data['lidar_path']
sensor_calib = scene_data['sensor_calib']
s_translation = sensor_calib['translation']
s_rotation = Quaternion(sensor_calib['rotation']).rotation_matrix
resolution = (1920, 1080)
visualizer = o3d.visualization.Visualizer()
visualizer.create_window(width=resolution[0], height=resolution[1], visible=True)
print(s_translation)
print(s_rotation)
set_custom_view(visualizer)
for itr, token in enumerate(tokens):

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

    lidar_data = LidarPointCloud.from_file(lidar_path[itr]+ '.pcd.bin')
    lidar_points = lidar_data.points.T
    # transform lidar points to ego coordinate
    #lidar_points = np.dot(lidar_points[:,:3], s_rotation) + s_translation

    lidar_cloud = o3d.t.geometry.PointCloud()
    lidar_cloud.point.positions = o3d.core.Tensor(lidar_points[:,:3],dtype=o3d.core.Dtype.Float32)
    lidar_cloud.point.colors = o3d.core.Tensor(np.ones((len(lidar_points), 3))*np.array(Colors.BLACK.value),dtype=o3d.core.Dtype.Float32)
    # GT BBOX
    if len(gt_bbox[itr])>0:
        # transform box to ego coordinate
        box = gt_bbox[itr][0]
        corners = box.corners
        #corners = np.dot(corners, s_rotation) + s_translation
        box.corners = corners
        gt_bbox_vis = create_line_set_bounding_box(box,0,0,Colors.DARK_GREEN.value)
        visualizer.add_geometry(gt_bbox_vis)
    # PRED BBOX    
    if len(pred_bbox[itr])>0:
        # transform box to ego coordinate
        box = pred_bbox[itr][0]
        corners = box.corners
        #corners = np.dot(corners, ego_rotation) + ego_xyz[0]
        #corners = np.dot(corners, s_rotation) + s_translation
        box.corners = corners
        pred_bbox_vis = create_line_set_bounding_box(box,0,0,Colors.DARK_RED.value)
        visualizer.add_geometry(pred_bbox_vis)
    
    #draw ego xyz trajectory
    ego_line = o3d.t.geometry.LineSet()
    ego_line.point.positions = o3d.core.Tensor(ego_xyz,dtype=o3d.core.Dtype.Float32)
    ego_line.point.colors = o3d.core.Tensor(np.ones((ego_xyz.shape[0], 3))*np.array(Colors.DARK_BLUE.value),dtype=o3d.core.Dtype.Float32)
    lines = []
    for i in range(ego_xyz.shape[0]-1):
        lines.append([i,i+1])
    ego_line.line.indices = o3d.core.Tensor(lines,dtype=o3d.core.Dtype.Int32)
    ego_line = ego_line.to_legacy()
    lidar_cloud = lidar_cloud.to_legacy()
    
    visualizer.add_geometry(ego_line)
    visualizer.add_geometry(lidar_cloud)
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image('outputs/visualisation/{}.png'.format(itr))
    time.sleep(0.5)
    visualizer.clear_geometries()
visualizer.destroy_window()