from base_classes.base import DrivingDataset
from typing import Union, List
import os 
from glob import glob
import numpy as np
from utils.boundingbox import BoundingBox, BoundingBox2D
from utils.pointcloud import PointCloud
import open3d as o3d
from math import cos,sin
import cv2 
from utils.filter import *
import copy
import open3d as o3d
from definitions import KITTI_CLASSES
from registries.dataset_registry import dataset_registry
@dataset_registry.register('kitti3d')
class Kitti3D(DrivingDataset):

    def __init__(self,
                 root_dir: str,
                 class_names: Union[None, List],
                 filtering_style: FilterType = FilterType.NONE,**kwargs) -> None:
        self.root_dir = root_dir
        self.classes = class_names
        print("Getting image paths")
        self.image_paths = self.get_image_paths()
        print("Getting lidar paths")
        self.lidar_paths = self.get_lidar_paths()
        self.calibration_paths = self.get_calibration_paths()
        self.label_paths = self.get_label_paths()
        self.filtering_style = eval(filtering_style)
        self.filter_params = kwargs.get('filter_params',{})
        self.is_e2e = kwargs.get('is_e2e',False)
        self.filter = self.filtering_style.value(**self.filter_params)
    def __getitem__(self, idx):       
        #Read data
        file_name = self.lidar_paths[idx].split('/')[-1]
        #image = self.read_image(idx) #Can be used later
        points = self.read_points(idx=idx)
        labels = self.read_labels(idx=idx)
        #Process data
        point_cloud = PointCloud(points=points)
        if self.is_e2e:
            raw_point_cloud = copy.deepcopy(point_cloud)
            raw_labels = copy.deepcopy(labels)
        point_cloud.points = self.filter.filter_pointcloud(point_cloud.points)
        # point_cloud.convert_to_kitti_points()
        
        labels = self.filter.filter_bounding_boxes(labels)
        if self.is_e2e:
            item_dict = {'pointcloud': {'filtered':point_cloud,'raw':raw_point_cloud},
                        'labels': {'filtered':labels,'raw':raw_labels},
                        'file_name':file_name}
        else:
            item_dict = {'pointcloud': point_cloud, 'labels': labels, 'file_name':file_name}

        return item_dict
    
    def get_image_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'image_2', '*.png')))
    def get_calibration_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'calib', '*.txt')))
    def get_lidar_paths(self):
        return sorted(glob(os.path.join(self.root_dir, 'velodyne',  '*.bin')))
    def get_label_paths(self):
        return sorted(glob(os.path.join(self.root_dir, 'label_2', '*.txt')))
    
    def read_image(self, idx):
        image_path = self.lidar_paths[idx]
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
        return len(self.lidar_paths)
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


@dataset_registry.register('kitti2d')   
class Kitti2D(DrivingDataset):
    def __init__(self, root_dir: str,
                 class_names: Union[None, List],
                 load_image:bool =True,
                 image_size=(1242,375)) -> None:
        self.root_dir = root_dir
        self.classes = class_names
        self.class_name_to_idx = {class_name:KITTI_CLASSES[class_name] for idx,class_name in enumerate(self.classes)}
        self.image_size = image_size
        self.image_paths = self.get_image_paths()
        self.label_paths = self.get_label_paths()
        self.dataset_dict = { os.basename(image_path.split('/')[-1]):{'image':image_path,'label':label_path}
                                for image_path,label_path in zip(self.image_paths,self.label_paths)}
        self.load_image = load_image
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        kitti_string_format = list(self.dataset_dict.keys())[idx]
        image_path = self.dataset_dict[kitti_string_format]['image']
        label_path = self.dataset_dict[kitti_string_format]['label']
        image = image_path
        label = self.read_labels(label_path,classes=self.classes)
        if self.load_image:
            image = self.read_image(image_path,resolution=(1242,375))

            item_dict = {'image': image, 'labels': label,'file_name':image_path}
        else:
            item_dict = {'image': image_path, 'labels': label,'file_name':image_path}

        return item_dict 

    def read_labels(self, **kwargs):
        path = kwargs['path']
        classes = kwargs['classes']
        with open(path, 'r') as f:
            content = f.readlines()

        objects = []
        for line in content:
            line = line.replace(","," ")
            data = line.strip().split()
            class_name = data[0]
    # 
            # print(class_name in classes)
            if(str(class_name) in str(classes)):
                x1, y1, x2, y2 = float(data[4]), float(data[5]), float(data[6]), float(data[7])
                bbox = (x1, y1, x2, y2)
                objects.append({'bbox':bbox, 'label':class_name})
                # print(objects)

        
    def get_image_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'image_2', '*.png')))
    def get_calibration_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'calib', '*.txt')))
    def get_lidar_paths(self):
        return sorted(glob(os.path.join(self.root_dir, 'velodyne',  '*.bin')))
    def get_label_paths(self):
        return sorted(glob(os.path.join(self.root_dir, 'label_2', '*.txt')))
    