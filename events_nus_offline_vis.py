import open3d as o3d
import os
from glob import glob
import numpy as np
from base_classes.base import DrivingDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud,Box
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
import numpy as np
from utils.boundingbox import BoundingBox
from utils.pointcloud import PointCloud
from utils.filter import *
import pickle
import os
import copy
from nuscenes.nuscenes import NuScenes
import numpy as np
import os
os.chdir('/mnt/ssd2/Introspect3D')
import open3d as o3d
from pprint import pprint
from glob import glob
from mmdet3d.apis import inference_detector, init_model
from math import cos,sin
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud,Box
import math
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from pyquaternion import Quaternion
from open3d import geometry
import cv2
from utils.boundingbox import BoundingBox

from utils.utils import create_bounding_boxes_from_predictions
from utils.utils import check_detection_matches
checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
object_label_to_index = {
    "human.pedestrian.adult": 6,
    "human.pedestrian.child": 6,
    "human.pedestrian.wheelchair": 6,
    "human.pedestrian.stroller": 6,
    "human.pedestrian.personal_mobility": 6,
    "human.pedestrian.police_officer": 6,
    "human.pedestrian.construction_worker": 6,
    "animal": 9,
    "vehicle.car": 3,
    "vehicle.motorcycle": 4,
    "vehicle.bicycle": 0,
    "vehicle.bus.bendy": 1,
    "vehicle.bus.rigid": 2,
    "vehicle.truck": 5,
    "vehicle.construction": 5,
    "vehicle.emergency.ambulance": 3,
    "vehicle.emergency.police": 3,
    "vehicle.trailer": 5,
    "movable_object.barrier": 13,
    "movable_object.trafficcone": 10,
    "movable_object.pushable_pullable": 8,
    "movable_object.debris": 12,
    "static_object.bicycle_rack": 20
}
def set_custom_view(vis):
    
    ctr = vis.get_view_control()
    print(ctr)
    # Create an extrinsic matrix for camera placement
    extrinsic = np.eye(4)
    extrinsic[0:3, 3] = [-10, 0, 30]  # Set camera position (x, y, z)
    
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
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 1.0
def make_point_cloud(path):
    cloud = np.load(path)
    points = cloud[:, :3]            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3))*[34,139,34])
    return pcd
def load_point_clouds(folder_path):
    """Load all point cloud files from the seeeeepecified folder."""
    files = glob(os.path.join(folder_path, 'lidar','*.npy'))

    point_clouds = [make_point_cloud(file) for file in files]
    return point_clouds
def read_labels(self, **kwargs):
    boxes = []
    id = kwargs['id']
    lidar_token = kwargs['lidar_token']
    sample_record = kwargs['sample_record']
    _,  boxes, _  = self.nusc.get_sample_data(lidar_token)
    for i in range(len(boxes)):
        annotation = self.nusc.get('sample_annotation', sample_record['anns'][i])
        box = boxes[i]
        if self.filter_boxes_with_category(annotation['category_name']):
            box.label = object_label_to_index[annotation['category_name']]
            custom_box = BoundingBox()
            custom_box.from_nuscenes_box(box)
            boxes.append(custom_box)
    return boxes
def create_oriented_bounding_box(box_params,rot_axis=2,calib=True,color=(1, 1, 0)):


    center = np.array([box_params[0], box_params[1], box_params[2]])
    extent = np.array([box_params[3], box_params[4], box_params[5]])
    if(len(box_params) > 9):
      print("Too many parameters for box")
      quat = Quaternion(box_params[6:10])
      rot_mat = quat.rotation_matrix
    elif(len(box_params) == 9):
      yaw = np.zeros(3)
      yaw[2] = box_params[6]

      rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

    center[2] += extent[2] / 2
    box3d = geometry.OrientedBoundingBox(center, rot_mat, extent)
    print(center)
    line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
    #yellow color
    line_set.paint_uniform_color(color)

    #  Move box to sensor coord system
    ctr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=center)
    # obb.color = (1, 0, 0)  # Red color
    return line_set#box3d#line_set #, ctr
            
def visualize_point_clouds(nuscenes_data, tokens, lidar = False, radar = False):
    """Visualize point clouds and switch between them using the spacebar."""
    model = init_model(config, checkpoint, device='cuda:0')

    is_nuscenes = True
    
    for token in tokens:
        scene = nuscenes_data.get('scene', token)
        print(scene['description'])

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        is_finished = False
        set_custom_view(vis)
        first_sample_token = scene['first_sample_token']
        sample_record = nuscenes_data.get('sample', first_sample_token)
        if lidar:
            lidar_token = sample_record['data']['LIDAR_TOP']
            lidar_data = nuscenes_data.get('sample_data', lidar_token)
            lidar_filepath = nuscenes_data.get_sample_data_path(lidar_token)
            point_cloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
            res, data = inference_detector(model, point_cloud)
            predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later

            filtered_predicted_boxes = predicted_boxes[score_mask]
            print(len(filtered_predicted_boxes))
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
            o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*[34,139,34])
        if radar:
            "Loading"
            radar_token = sample_record['data']['RADAR_FRONT']
            radar_data = nuscenes_data.get('sample_data', radar_token)
            radar_filepath = nuscenes_data.get_sample_data_path(radar_token)
            radar_point_cloud = RadarPointCloud.from_file(radar_filepath)
            point_cloud = radar_point_cloud.points
            o3d_cloud2 = o3d.geometry.PointCloud()
            o3d_cloud2.points = o3d.utility.Vector3dVector(radar_point_cloud.points[:, :3])
            #Create np array with yellow rgb code with point length
            o3d_cloud2.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*np.array([1,1,0]))
        #Combine if both lidar and radar are selected

        def load_next_point_cloud(vis):
            nonlocal sample_record,lidar, radar  # Declare nonlocal to modify the outer scope variable
            first_sample_token = sample_record['next']
            if first_sample_token == '':
                return
            sample_record = nuscenes_data.get('sample', first_sample_token)
            vis.clear_geometries()
            if lidar:
                lidar_token = sample_record['data']['LIDAR_TOP']
                lidar_data = nuscenes_data.get('sample_data', lidar_token)
                lidar_filepath = nuscenes_data.get_sample_data_path(lidar_token)
                point_cloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
                res, data = inference_detector(model, point_cloud)
                predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
                predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
                score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later
                filtered_predicted_boxes = predicted_boxes[score_mask]
                 
                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*[34,139,34])
                vis.add_geometry(o3d_cloud, reset_bounding_box=True)
                for box in filtered_predicted_boxes:
                    vis.add_geometry(create_oriented_bounding_box(box,calib=False))
            if radar:
                radar_token = sample_record['data']['RADAR_FRONT']

                radar_data = nuscenes_data.get('sample_data', radar_token)
                radar_filepath = nuscenes_data.get_sample_data_path(radar_token)
                # point_cloud = np.fromfile(radar_filepath, dtype=np.float32, count=-1).reshape([-1, 18])
                radar_point_cloud = RadarPointCloud.from_file(radar_filepath)
                
                point_cloud = radar_point_cloud.points

                o3d_cloud2 = o3d.geometry.PointCloud()
                o3d_cloud2.points = o3d.utility.Vector3dVector(radar_point_cloud.points[:, :3])
                #Create np array with yellow rgb code with point length
                o3d_cloud2.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*[255,255,0])
                vis.add_geometry(o3d_cloud2, reset_bounding_box=True)
            
            vis.add_geometry(o3d_cloud, reset_bounding_box=True)
            for box in filtered_predicted_boxes:
                vis.add_geometry(create_oriented_bounding_box(box,calib=False))
        vis.register_key_callback(ord(' '), load_next_point_cloud)  # Bind spacebar to switch point clouds


        if lidar and radar:
            vis.add_geometry(o3d_cloud, reset_bounding_box=True)
            vis.add_geometry(o3d_cloud2, reset_bounding_box=True)
        elif lidar:
            vis.add_geometry(o3d_cloud, reset_bounding_box=True)
        elif radar:
            vis.add_geometry(o3d_cloud2, reset_bounding_box=True)
        vis.run()  # Run the visualizer
        vis.destroy_window()  # Clean up after closing the window

import argparse
import pickle
# Usage
#Need to adapt to NuScenes dataset#
def arg_parse():
    parser = argparse.ArgumentParser(description='Visualize point clouds')
    parser.add_argument('-n', "--names",nargs='+', default=[], help='Names of the scenes to visualize')
    parser.add_argument('-l', "--lidar", default=True, help='Visualize lidar data')
    parser.add_argument('-r', "--radar", default=False, help='Visualize radar data')
    return parser.parse_args()
if __name__ == '__main__':
    folder_path = '/media/yatbaz_h/Jet/HYY/'
    root_dir = '/mnt/ssd2/nuscenes'
    version = 'v1.0-trainval'
    args = arg_parse()
    # if "nuscenes_data.pkl" in os.listdir():
    #     with open('nuscenes_data.pkl','rb') as f:
    #         nuscenes_data = pickle.load(f)
    # else:
    nuscenes_data= NuScenes(version=version, dataroot=root_dir, verbose=True)
        # with open('nuscenes_data.pkl','wb') as f:
        #     pickle.dump(nuscenes_data,f)
    # nuscenes_data= NuScenes(version=version, dataroot=root_dir, verbose=True)
    # with open('nuscenes_data.pkl','wb') as f:
    #     pickle.dump(nuscenes_data,f)
    import pandas as pd 
    nuscenes_scene_tokens = pd.DataFrame(columns=['scene_token','name','description'])
    scenes = nuscenes_data.scene
    for scene in scenes:
        scene_token = scene['token']
        name = scene['name']
        description = scene['description']
        print(scene_token,name,description)

        visualize_point_clouds(nuscenes_data,tokens = [scene_token], lidar = args.lidar, radar = args.radar)
    # nuscenes_data= NuScenes(version=versio                  n, dataroot=root_dir, verbose=True)
    # import pandas as pd
    # from tqdm.auto import tqdm
    # nuscenes_scene_tokens = pd.DataFrame(columns=['scene_token','name','description'])
    # scenes = 
    # with tqdm(total=len(scenes)) as pbar:
    #     for scene in scenes:
    #         scene_token = scene['token']
    #         name = scene['name']
    #         description = scene['description']
    #         print(scene_token,name,description)
    #         temp_df = pd.DataFrame([[scene_token,name,description]],columns=['scene_token','name','description'],index=[0])
    #         nuscenes_scene_tokens = pd.concat([nuscenes_scene_tokens,temp_df],ignore_index=True)
    #         pbar.update(1)
    # nuscenes_scene_tokens.to_csv('nuscenes_scene_tokens.csv')
    # point_clouds = load_point_clouds(folder_path)
#    visualize_point_clouds(nuscenes_data)






# import open3d as o3d
# import os
# from glob import glob
# import numpy as np

# from base_classes.base import DrivingDataset
# from nuscenes.nuscenes import NuScenes
# from nuscenes.utils.data_classes import LidarPointCloud,Box

# from pyquaternion import Quaternion
# import numpy as np
# from utils.boundingbox import BoundingBox
# from utils.pointcloud import PointCloud
# from utils.filter import *
# import pickle
# import os
# import copy

# def set_custom_view(vis):
    
#     ctr = vis.get_view_control()
#     print(ctr)
#     # Create an extrinsic matrix for camera placement
#     extrinsic = np.eye(4)
#     extrinsic[0:3, 3] = [-10, 0, 60]  # Set camera position (x, y, z)
    
#     # Create a rotation matrix for 30-degree downward view
#     rotation = np.array([
#         [1, 0, 0],
#         [0, np.cos(np.radians(-160)), -np.sin(np.radians(-160))],
#         [0, np.sin(np.radians(-160)), np.cos(np.radians(-160))]
#     ])
    
#     # Apply rotation to the extrinsic matrix
#     extrinsic[0:3, 0:3] = rotation
    
#     # Set the extrinsic matrix to the camera parameters
#     cam_params = ctr.convert_to_pinhole_camera_parameters()
#     cam_params.extrinsic = extrinsic
#     ctr.convert_from_pinhole_camera_parameters(cam_params)
#     opt = vis.get_render_option()
#     opt.background_color = np.asarray([0.0, 0.0, 0.0])
#     opt.point_size = 1.0
# def make_point_cloud(path):
#     cloud = np.load(path)
#     points = cloud[:, :3]            
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3))*[34,139,34])
#     return pcd
# def load_point_clouds(folder_path):
#     """Load all point cloud files from the seeeeepecified folder."""
#     files = glob(os.path.join(folder_path, 'lidar','*.npy'))

#     point_clouds = [make_point_cloud(file) for file in files]
#     return point_clouds

# def visualize_point_clouds(point_clouds):
#     """Visualize point clouds and switch between them using the spacebar."""
#     if not point_clouds:  # Check if list is empty
#         print("No point clouds found in the directory.")
#         return

#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window()
#     set_custom_view(vis)
#     current_index = [0]  # Using a list to hold the index as a mutable object

#     def load_next_point_cloud(vis):
#         """Callback function to load the next point cloud."""
#         current_index[0] = (current_index[0] + 1) % len(point_clouds)
#         vis.clear_geometries()
#         vis.add_geometry(point_clouds[current_index[0]], reset_bounding_box=True)

#     vis.register_key_callback(ord(' '), load_next_point_cloud)  # Bind spacebar to switch point clouds
#     vis.add_geometry(point_clouds[current_index[0]], reset_bounding_box=True)

#     vis.run()  # Run the visualizer
#     vis.destroy_window()  # Clean up after closing the window

# # Usage
# #Need to adapt to NuScenes dataset
# if __name__ == '__main__':

#     folder_path = '/media/yatbaz_h/Jet/HYY/'
#     root_dir = '/mnt/ssd2/nuscenes'
#     version = 'v1.0-trainval'

#     nuscenes_data= NuScenes(version=version, dataroot=root_dir, verbose=True)
#     # point_clouds = load_point_clouds(folder_path)
#     # visualize_point_clouds(point_clouds)
