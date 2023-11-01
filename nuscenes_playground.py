from nuscenes.nuscenes import NuScenes
import numpy as np
import os
import open3d as o3d
from pprint import pprint
from glob import glob
from nuscenes.utils.data_classes import LidarPointCloud
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
nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/ssd2/nuscenes_mini/v1.0-mini', verbose=True)
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
while not first_sample_token == '':
    sample_record = nusc.get('sample', first_sample_token)

    lidar_token = sample_record['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    if lidar_data['is_key_frame']:
      lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])
      pc_file = os.path.join(nusc.dataroot,lidar_filepath)
      pc = LidarPointCloud.from_file(pc_file)
      points = pc.points.T
      for i in range(1,len(sample_record['anns'])):
        annotations = nusc.get('sample_annotation', sample_record['anns'][i])
      print(points.shape)
    #   pprint(annotations)
      # print(points.shape)
    
    # print(lidar_filepath)
    first_sample_token = sample_record['next']

# my_sample = nusc.get('sample', first_sample_token)
# lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
# #get point cloud data from nuscenes
# pc_file = nusc.get('sample_data', lidar_data['token'])['filename']
# pc_file = os.path.join(nusc.dataroot,pc_file)
# points = np.fromfile(pc_file, dtype=np.float32, count=-1).reshape([-1, 5])

# max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))


# pcd = create_point_cloud(points,None)
# o3d.visualization.draw_geometries([pcd])