# from nuscenes.nuscenes import NuScenes

# nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/ssd1/introspectionBase/datasets/nuscenes', verbose=True)
# print(nusc)
#@title Initial setup
from typing import Optional
import warnings
# Disable annoying warnings from PyArrow using under the hood.
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2
from glob import glob
import open3d as o3d
from matplotlib import pyplot as plt
def compute_colors_from_distance(points,max_distance):
    #If no disatance given return all black
    if max_distance==None:
        return np.zeros((points.shape[0],3))
    distances = np.linalg.norm(points[:, :3], axis=1)
    normalized_distances = distances / max_distance
    return plt.cm.jet(normalized_distances)[:,:3]
def create_point_cloud(points,distance=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = compute_colors_from_distance(points,distance)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
# Path to the directory with all components
dataset_dir = '/mnt/ssd2/waymo/training'

context_name = '10023947602400723454_1120_000_1140_000'

def read(tag: str) -> dd.DataFrame:
  """Creates a Dask DataFrame for the component specified by its tag."""
#   print('Reading',f'{dataset_dir}/{tag}/{context_name}.parquet')
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/*.parquet')#{context_name}
#   print(paths)
  return dd.read_parquet(paths)

lidar_df = read('lidar')
lidar_pose_df = read('lidar_pose')
lidar_calibration_df = read('lidar_calibration')
lidar_segmentation_df = read('lidar_segmentation')
stats_df = read('stats')
vehicle_pose_df = read('vehicle_pose')


_, lidar_row = next(iter(lidar_df.iterrows()))
_, lidar_pose_row = next(iter(lidar_pose_df.iterrows()))
_, lidar_calibration_row = next(iter(lidar_calibration_df.iterrows()))
_, lidar_segmentation_row = next(iter(lidar_segmentation_df.iterrows()))
_, vehicle_pose_row = next(iter(vehicle_pose_df.iterrows()))

lidar = v2.LiDARComponent.from_dict(lidar_row)
lidar_pose = v2.LiDARPoseComponent.from_dict(lidar_pose_row)
lidar_calibration = v2.LiDARCalibrationComponent.from_dict(lidar_calibration_row)
lidar_segmentation = v2.LiDARSegmentationLabelComponent.from_dict(lidar_segmentation_row)
vehicle_pose = v2.VehiclePoseComponent.from_dict(vehicle_pose_row)



from waymo_open_dataset.v2.perception.utils import lidar_utils
points = lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1,
                                                        lidar_calibration,
                                                        lidar_pose.range_image_return1,
                                                        vehicle_pose,
                                                        False)
max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))


pcd = create_point_cloud(points,max_distance)
o3d.visualization.draw_geometries([pcd])