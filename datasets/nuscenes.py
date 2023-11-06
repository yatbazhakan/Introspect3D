from base_classes.base import DrivingDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud,Box

from pyquaternion import Quaternion
import numpy as np
from utils.pointcloud import PointCloud
from utils.filter import *
class NuScenesDataset(DrivingDataset):
    def __init__(self, dataroot, version='v1.0-mini', split='train', transform=None,
                 filtering_style: FilterType = FilterType.ELLIPSE,**kwargs):
        self.nuscenes = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.split = split
        self.transform = transform
        # Extract only the first sample token of each scene.
        self.sample_tokens = [s['first_sample_token'] for s in self.nuscenes.scene]
        self.filtering_style = eval(filtering_style)
        self.filter_params = kwargs['filter_params']
        self.filter = self.filtering_style.value(**self.filter_params)
        self.frame_pcd_locations = []
    def prepare_data(self, **kwargs):
        for first_sample_token in self.sample_tokens:
            while not first_sample_token == '':
                sample_record = nusc.get('sample', first_sample_token)
                frame_count += 1
                lidar_token = sample_record['data']['LIDAR_TOP']
                cs_record = nusc.get('calibrated_sensor', nusc.get('sample_data', lidar_token)['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', nusc.get('sample_data', lidar_token)['ego_pose_token'])
    
    # print(calibrated_lidar)
    lidar_data = nusc.get('sample_data', lidar_token)
    if lidar_data['is_key_frame']:
    def read_labels(self, **kwargs):
        return super().read_labels(**kwargs)
    def process_data(self, **kwargs):
        return super().process_data(**kwargs)
    
    
    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx):
        # Fetch the sample token, get sample data
        sample_token = self.sample_tokens[idx]
        sample = self.nuscenes.get('sample', sample_token)
        lidar_data = self.nuscenes.get('sample_data', sample['data']['LIDAR_TOP'])

        # Load the point cloud data
        lidar_filepath = self.nuscenes.get_sample_data_path(sample['data']['LIDAR_TOP'])
        pointcloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
        pc= PointCloud(pointcloud)
        
        
        # Transform point cloud to the global coordinate frame if required
        if self.transform is not None:
            cs_record = self.nuscenes.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
            pose_record = self.nuscenes.get('ego_pose', lidar_data['ego_pose_token'])
            pointcloud.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
            pointcloud.translate(np.array(cs_record['translation']))
            pointcloud.rotate(Quaternion(pose_record['rotation']).rotation_matrix)
            pointcloud.translate(np.array(pose_record['translation']))


        # Return the transformed pointcloud along with the original sample token.
        return {'pointcloud': pointcloud, 'sample_token': sample_token}