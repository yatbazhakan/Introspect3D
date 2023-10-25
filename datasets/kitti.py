from base_classes.base import DrivingDataset
from typing import Union, List
import os 
from glob import glob
import numpy as np
from utils.boundingbox import BoundingBox
from utils.pointcloud import PointCloud
import open3d as o3d
from math import cos,sin
import cv2 
from utils.filter import *
import open3d as o3d
class Kitti(DrivingDataset):

    def __init__(self,
                 root_dir: str,
                 class_names: Union[None, List],
                 filtering_style: FilterType = FilterType.ELLIPSE,**kwargs) -> None:
        self.root_dir = root_dir
        self.classes = class_names
        self.image_paths = self.get_image_paths()
        self.lidar_paths = self.get_lidar_paths()
        self.calibration_paths = self.get_calibration_paths()
        self.label_paths = self.get_label_paths()
        self.filtering_style = eval(filtering_style)
        self.filter_params = kwargs['filter_params']
        self.filter = self.filtering_style.value(**self.filter_params)
    def __getitem__(self, idx):       
        #Read data
        file_name = self.image_paths[idx].split('/')[-1]
        #image = self.read_image(idx) #Can be used later
        points = self.read_points(idx=idx)
        labels = self.read_labels(idx=idx)
        
        #Process data
        point_cloud = PointCloud(points=points)
        point_cloud.points = self.filter.filter_pointcloud(point_cloud.points)
        # point_cloud.convert_to_kitti_points()
        
        labels = self.filter.filter_bounding_boxes(labels)
        return point_cloud, labels, file_name #might look for a way to extend this to images
    
    def get_image_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'image_2', '*.png')))
    def get_calibration_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'calib', '*.txt')))
    def get_lidar_paths(self):
        return sorted(glob(os.path.join(self.root_dir, 'velodyne',  '*.bin')))
    def get_label_paths(self):
        return sorted(glob(os.path.join(self.root_dir, 'label_2', '*.txt')))
    
    def read_image(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    def read_points(self, **kwargs):
        idx = kwargs['idx']
        lidar_file = self.lidar_paths[idx]
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return points
    
    def read_labels(self, **kwargs):
        idx = kwargs['idx']
        objects = []
        from math import cos,sin
        label_file =  self.label_paths[idx]
        calibration_data = self.read_calibration(idx)
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                box = self.process_data(line=line, calibration_data=calibration_data)
                if box is not None:
                    objects.append(box)
        return objects
    
    def read_calibration(self, idx):
        calib_file = self.calibration_paths[idx]
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            calibration = {}

            for line in lines:
                # print(line)
                try:
                    key, value = line.split(':')
                except ValueError:
                    continue
                calibration[key] = np.array([float(x) for x in value.strip().split()])
            return calibration
    def __len__(self):
        return len(self.image_paths)
    def process_data(self,**kwargs):
        line = kwargs['line']
        calib_data = kwargs['calibration_data']
        tokens = line.strip().split(' ')
        obj_type = tokens[0]
        if obj_type not in self.classes:
            return None
        # print(line)
        dimensions_height, dimensions_width, dimensions_length = map(float, tokens[8:11])
        location_x, location_y, location_z = map(float, tokens[11:14])
        center_tr = np.array([location_x, location_y, location_z])
        rotation_y = float(tokens[14])
        l_div_2 = dimensions_length / 2
        x_corners = [l_div_2, l_div_2, -l_div_2, -l_div_2, l_div_2, l_div_2, -l_div_2, -l_div_2]
        w_div_2 = dimensions_width / 2
        y_corners = [0, 0, 0, 0, -dimensions_height, -dimensions_height, -dimensions_height, -dimensions_height]
        z_corners = [w_div_2, -w_div_2, -w_div_2, w_div_2, w_div_2, -w_div_2, -w_div_2, w_div_2]
        corner_matrix = np.array([x_corners, y_corners, z_corners])
        R = o3d.geometry.get_rotation_matrix_from_xyz((0,rotation_y,0))
        # R = np.array([[cos(rotation_y),0,sin(rotation_y)],[0,1,0],[-sin(rotation_y),0,cos(rotation_y)]])
        rotated_corners = np.matmul(R,corner_matrix)
        translated_corners = rotated_corners + center_tr.reshape(3,1)
        
        Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        Tr_velo_to_cam_extended = np.eye(4)  # Create a 4x4 identity matrix
        Tr_velo_to_cam_extended[:3, :] = Tr_velo_to_cam  # Replace the top-left 3x4 block
        T_inv = np.linalg.inv(Tr_velo_to_cam_extended)
        Homogeneous_corners = np.ones((4,8))
        Homogeneous_corners[:3,:] = translated_corners
        translated_corners = np.matmul(T_inv,Homogeneous_corners)[:3,:]
        real_center = np.mean(translated_corners,axis=1)
        box = BoundingBox(center=real_center, 
                            dimensions=(dimensions_height, dimensions_width, dimensions_length),
                            rotation=o3d.geometry.get_rotation_matrix_from_xyz((0,rotation_y,0)), 
                            label=obj_type)
        box.corners = translated_corners.T
        return box