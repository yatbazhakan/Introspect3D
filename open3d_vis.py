import open3d as o3d
import os
from glob import glob
import numpy as np
from base_classes.base import DrivingDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud,Box

from pyquaternion import Quaternion
import numpy as np
from utils.boundingbox import BoundingBox
from utils.pointcloud import PointCloud
from utils.filter import *
import pickle
import os
import copy
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

            
def visualize_point_clouds(nuscenes_data, tokens):
    """Visualize point clouds and switch between them using the spacebar."""

    for token in tokens:
        scene = nuscenes_data.get('scene', token)
        print(scene['description'])

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        is_finished = False
        set_custom_view(vis)
        first_sample_token = scene['first_sample_token']
        sample_record = nuscenes_data.get('sample', first_sample_token)
        lidar_token = sample_record['data']['LIDAR_TOP']
        lidar_data = nuscenes_data.get('sample_data', lidar_token)
        lidar_filepath = nuscenes_data.get_sample_data_path(lidar_token)
        point_cloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*[34,139,34])

        def load_next_point_cloud(vis):
            nonlocal sample_record  # Declare nonlocal to modify the outer scope variable
            first_sample_token = sample_record['next']
            if first_sample_token == '':
                return
            sample_record = nuscenes_data.get('sample', first_sample_token)
            lidar_token = sample_record['data']['LIDAR_TOP']
            lidar_data = nuscenes_data.get('sample_data', lidar_token)
            if lidar_data['is_key_frame']:
                vis.clear_geometries()
                lidar_filepath = nuscenes_data.get_sample_data_path(lidar_token)
                point_cloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*[34,139,34])
                vis.add_geometry(o3d_cloud, reset_bounding_box=True)

        vis.register_key_callback(ord(' '), load_next_point_cloud)  # Bind spacebar to switch point clouds


        vis.add_geometry(o3d_cloud, reset_bounding_box=True)
        vis.run()  # Run the visualizer
        vis.destroy_window()  # Clean up after closing the window

import argparse
import pickle
# Usage
#Need to adapt to NuScenes dataset#
def arg_parse():
    parser = argparse.ArgumentParser(description='Visualize point clouds')
    parser.add_argument('-n', "--names",nargs='+', default=[], help='Names of the scenes to visualize')
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
    visualize_point_clouds(nuscenes_data,tokens = args.names)
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
