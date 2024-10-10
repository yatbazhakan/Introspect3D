
from base_classes.base import DrivingDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion
import numpy as np
from utils.boundingbox import BoundingBox
from utils.pointcloud import PointCloud
from utils.filter import *
import pickle
import os
import copy
import cv2
from registries.dataset_registry import dataset_registry
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

object_index_to_label = {v: k for k, v in object_label_to_index.items()}


reg
class NuScenesDataset(DrivingDataset):
    def __init__(self, root_dir, version='v1.0-mini', split='train', transform=None,
                 filtering_style: FilterType = FilterType.ELLIPSE, **kwargs):
        process = kwargs['process']
        self.save_path, self.save_filename = kwargs['save_path'], kwargs['save_filename']
        self.projection = kwargs.get('projection', False)
        print("Filtering style", filtering_style)
        self.filtering_style = eval(filtering_style)
        self.filter_params = kwargs.get('filter_params', {})
        self.is_e2e = kwargs.get('is_e2e', False)
        self.labels_only = kwargs.get('filter_labels_only', False)
        self.filter = self.filtering_style.value(**self.filter_params)
        self.dataset_flattened = {}
        self.dataset_flattened['train'] = {}
        self.dataset_flattened['val'] = {}
        self.dataset_flattened['test'] = {}
        if process:
            self.nusc = NuScenes(version=version, dataroot=root_dir, verbose=True)

            self.split = split
            self.transform = transform
            # Extract only the first sample token of each scene.
            self.sample_tokens = [s['first_sample_token'] for s in self.nusc.scene]
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)

            # Split sample tokens into train, test, val sets
            np.random.shuffle(self.sample_tokens)
            split_ratio = [0.7, 0.15, 0.15]
            train_end = int(len(self.sample_tokens) * split_ratio[0])
            val_end = train_end + int(len(self.sample_tokens) * split_ratio[1])
            self.sample_tokens = {'train': self.sample_tokens[:train_end],
                                  'val': self.sample_tokens[train_end:val_end],
                                  'test': self.sample_tokens[val_end:]}
            self.process_data()
        else:
            print("Loading dataset from file")
            self.dataset_flattened = pickle.load(open(os.path.join(self.save_path, self.save_filename), 'rb'))
            if split == 'all':
                self.dataset_flattened = list(self.dataset_flattened['train'].values()) + list(self.dataset_flattened['val'].values()) + list(self.dataset_flattened['test'].values())
            else:
                self.dataset_flattened = list(self.dataset_flattened[split].values())
            print("Loaded dataset with {} samples, {}".format(len(self.dataset_flattened), self.dataset_flattened[0].keys()))

    def read_labels(self, **kwargs):
        id = kwargs['id']
        split = kwargs['split']
        lidar_token = kwargs['lidar_token']
        sample_record = kwargs['sample_record']
        _, boxes, _ = self.nusc.get_sample_data(lidar_token)
        for i in range(len(boxes)):
            annotation = self.nusc.get('sample_annotation', sample_record['anns'][i])
            box = boxes[i]
            if self.filter_boxes_with_category(annotation['category_name']):
                box.label = object_label_to_index[annotation['category_name']]
                custom_box = BoundingBox()
                custom_box.from_nuscenes_box(box)
                self.dataset_flattened[split][id]['label'].append(custom_box)

    def process_data(self, **kwargs):
        for split, tokens in self.sample_tokens.items():
            id = 0
            for first_sample_token in tokens:
                while not first_sample_token == '':
                    sample_record = self.nusc.get('sample', first_sample_token)
                    lidar_token = sample_record['data']['LIDAR_TOP']
                    lidar_data = self.nusc.get('sample_data', lidar_token)
                    if lidar_data['is_key_frame']:
                        lidar_filepath = self.nusc.get_sample_data_path(lidar_token)
                        self.dataset_flattened[split][id] = {}
                        self.dataset_flattened[split][id]['path'] = lidar_filepath
                        self.dataset_flattened[split][id]['label'] = []
                        self.dataset_flattened[split][id]['sample_token'] = first_sample_token  # Store sample_token for later use
                        self.read_labels(id=id, lidar_token=lidar_token, sample_record=sample_record, split=split)
                        id += 1
                    first_sample_token = sample_record['next']
        pickle.dump(self.dataset_flattened, open(os.path.join(self.save_path, self.save_filename), 'wb'))

    def __len__(self):
        return len(self.dataset_flattened)

    def get_with_name(self, name):
        for i in range(len(self.dataset_flattened)):
            if name in self.dataset_flattened[i]['path']:
                return self.__getitem__(i)
        return None

    def __getitem__(self, idx):
        data = self.dataset_flattened[idx]
        lidar_filepath = data['path']
        labels = data['label']
        points = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
        point_cloud = PointCloud(points)

        if self.is_e2e:
            raw_point_cloud = copy.deepcopy(point_cloud)
            raw_labels = copy.deepcopy(labels)
        if not self.labels_only:
            point_cloud.points = self.filter.filter_pointcloud(point_cloud.points)
        point_cloud.raw_points = point_cloud.points.copy()
        
        labels = self.filter.filter_bounding_boxes(labels)
        out_labels = self.filter.filter_bounding_boxes(labels, FilteringArea.OUTSIDE)
        # Initialize the item dictionary
        item_dict = {'pointcloud': point_cloud, 'labels': labels, 'file_name': lidar_filepath, 'full_labels':out_labels}

        if self.projection:
            # Get camera image and calibration data
            sample_token = data['sample_token']
            camera_image, calibration_data = self.get_camera_data(sample_token)

            # Project lidar points onto image
            projected_points = self.project_lidar_to_image(point_cloud.points, calibration_data)

            # Add to item dictionary
            item_dict['camera_image'] = camera_image
            item_dict['projected_points'] = projected_points

        return item_dict

    def filter_boxes_with_category(self, box_label, accepted_categories=['vehicle.', 'human', 'cyclist']):
        for cat in accepted_categories:
            if box_label.startswith(cat):
                return True
        return False

    def get_camera_data(self, sample_token, camera_name='CAM_FRONT'):
        sample_record = self.nusc.get('sample', sample_token)
        camera_token = sample_record['data'][camera_name]
        camera_data = self.nusc.get('sample_data', camera_token)
        camera_filepath = self.nusc.get_sample_data_path(camera_token)
        camera_image = cv2.imread(camera_filepath)

        calibration_token = camera_data['calibrated_sensor_token']
        calibration_data = self.nusc.get('calibrated_sensor', calibration_token)

        return camera_image, calibration_data

    def project_lidar_to_image(self, points, calibration_data):
        # Transform from lidar to camera
        lidar_points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1)))).T
        lidar_to_camera = transform_matrix(calibration_data['translation'], Quaternion(calibration_data['rotation']), inverse=False)
        camera_points = lidar_to_camera @ lidar_points_homogeneous

        # Project to 2D
        intrinsic = np.array(calibration_data['camera_intrinsic'])
        camera_points = camera_points[:3, :]
        projected_points = intrinsic @ camera_points
        projected_points = projected_points[:2, :] / projected_points[2, :]

        return projected_points.T


#My implementation of the nuscenes dataset class
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
# object_label_to_index = {
#     "human.pedestrian.adult": 6,
#     "human.pedestrian.child": 6,
#     "human.pedestrian.wheelchair": 6,
#     "human.pedestrian.stroller": 6,
#     "human.pedestrian.personal_mobility": 6,
#     "human.pedestrian.police_officer": 6,
#     "human.pedestrian.construction_worker": 6,
#     "animal": 9,
#     "vehicle.car": 3,
#     "vehicle.motorcycle": 4,
#     "vehicle.bicycle": 0,
#     "vehicle.bus.bendy": 1,
#     "vehicle.bus.rigid": 2,
#     "vehicle.truck": 5,
#     "vehicle.construction": 5,
#     "vehicle.emergency.ambulance": 3,
#     "vehicle.emergency.police": 3,
#     "vehicle.trailer": 5,
#     "movable_object.barrier": 13,
#     "movable_object.trafficcone": 10,
#     "movable_object.pushable_pullable": 8,
#     "movable_object.debris": 12,
#     "static_object.bicycle_rack": 20
# }

# # object_label_to_index = {
# #     "vehicle.emergency.police car": 0,
# #     "vehicle.emergency.ambulance": 1,
# #     "vehicle.emergency.fire truck": 2,
# #     "vehicle.construction": 3,
# #     "vehicle.truck": 4,
# #     "vehicle.bus": 5,
# #     "vehicle.van": 6,
# #     "vehicle.motorcycle": 7,
# #     "vehicle.car": 8,
# #     "pedestrian": 9,
# #     "cyclist": 10,
# #     "traffic_cone": 11,
# #     "pole": 12,
# #     "fire_hydrant": 13,
# #     "stop_sign": 14,
# #     "traffic_light": 15,
# #     "parking_sign": 16,
# #     "speed_bump": 17,
# #     "sidewalk": 18,
# #     "parking_slot": 19,
# #     "road_marking": 20,
# #     "vegetation": 21,
# #     "terrain": 22,
# #     "other": 23,
# # }
# object_index_to_label = {v: k for k, v in object_label_to_index.items()}

# class NuScenesDataset(DrivingDataset):
#     def __init__(self, root_dir, version='v1.0-mini', split='train', transform=None,
#                  filtering_style: FilterType = FilterType.ELLIPSE,**kwargs):
#         process = kwargs['process']
#         self.save_path,self.save_filename = kwargs['save_path'], kwargs['save_filename']
#         print("Filtering style",filtering_style)
#         self.filtering_style = eval(filtering_style)
#         self.filter_params = kwargs.get('filter_params',{})
#         self.is_e2e = kwargs.get('is_e2e',False)
#         self.labels_only = kwargs.get('filter_labels_only',False)
#         self.filter = self.filtering_style.value(**self.filter_params)
#         self.dataset_flattened = {}
#         self.dataset_flattened['train'] = {}
#         self.dataset_flattened['val'] = {}
#         self.dataset_flattened['test'] = {}
#         if process:
#             self.nusc = NuScenes(version=version, dataroot=root_dir, verbose=True)

#             self.split = split
#             self.transform = transform
#             # Extract only the first sample token of each scene.
#             self.sample_tokens = [s['first_sample_token'] for s in self.nusc.scene]
#             if not os.path.exists(self.save_path):
#                 os.makedirs(self.save_path)

#             #SPlit sample tokens into train test val sets
#             np.random.shuffle(self.sample_tokens)
#             split_ratio = [0.7,0.15,0.15]
#             train_end = int(len(self.sample_tokens)*split_ratio[0])
#             val_end = train_end + int(len(self.sample_tokens)*split_ratio[1])
#             self.sample_tokens = {'train':self.sample_tokens[:train_end],
#                                   'val':self.sample_tokens[train_end:val_end],
#                                   'test':self.sample_tokens[val_end:]}
#             self.process_data()
#         else:
#             print("Loading dataset from file")
#             self.dataset_flattened = pickle.load(open(os.path.join(self.save_path,self.save_filename),'rb'))
#             # self.dataset_flattened = list(self.dataset_flattened.values()) #Test
#             if split == 'all':
#                 self.dataset_flattened = list(self.dataset_flattened['train'].values()) + list(self.dataset_flattened['val'].values())+ list(self.dataset_flattened['test'].values())
#             else:
#                 self.dataset_flattened = list(self.dataset_flattened[split].values())
#             print("Loaded dataset with {} samples, {}".format(len(self.dataset_flattened),self.dataset_flattened[0].keys()))


#     def read_labels(self, **kwargs):
#         id = kwargs['id']
#         split = kwargs['split']
#         lidar_token = kwargs['lidar_token']
#         sample_record = kwargs['sample_record']
#         _,  boxes, _  = self.nusc.get_sample_data(lidar_token)
#         for i in range(len(boxes)):
#             annotation = self.nusc.get('sample_annotation', sample_record['anns'][i])
#             box = boxes[i]
#             if self.filter_boxes_with_category(annotation['category_name']):
#                 box.label = object_label_to_index[annotation['category_name']]
#                 custom_box = BoundingBox()
#                 custom_box.from_nuscenes_box(box)

#                 self.dataset_flattened[split][id]['label'].append(custom_box)


#     def process_data(self, **kwargs):
#         for split,tokens in self.sample_tokens.items():
#             id = 0
#             for first_sample_token in tokens:
#                 while not first_sample_token == '':
#                     sample_record = self.nusc.get('sample', first_sample_token)
#                     lidar_token = sample_record['data']['LIDAR_TOP']
#                     # print(calibrated_lidar)
#                     lidar_data = self.nusc.get('sample_data', lidar_token)
#                     if lidar_data['is_key_frame']:
#                         lidar_filepath = self.nusc.get_sample_data_path(lidar_token)
#                         self.dataset_flattened[split][id] = {}
#                         self.dataset_flattened[split][id]['path'] = lidar_filepath
#                         self.dataset_flattened[split][id]['label'] = []
#                         self.read_labels(id=id, lidar_token=lidar_token, sample_record=sample_record,split=split)
#                         id += 1
#                     first_sample_token = sample_record['next']
#         pickle.dump(self.dataset_flattened,open(os.path.join(self.save_path,self.save_filename),'wb'))
    
    
#     def __len__(self):
#         return len(self.dataset_flattened)
#     def get_with_name(self,name):
#         for i in range(len(self.dataset_flattened)):
#             if name in self.dataset_flattened[i]['path']:
#                 return self.__getitem__(i)
#         return None
#     def __getitem__(self, idx):
#         # Fetch the sample token, get sample data
#         # if self.split != False:
#         #     data = self.dataset_flattened[self.split][idx]
#         # else:
#         #     data= self.dataset_flattened[idx]
#         data= self.dataset_flattened[idx]
#         lidar_filepath = data['path']
#         labels = data['label']
#         # Load the point cloud data
#         points = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
#         # print("Points shape",points.shape)
#         point_cloud = PointCloud(points)
#         # print("Before filtering",point_cloud.points.shape)
#         if self.is_e2e:
#             raw_point_cloud = copy.deepcopy(point_cloud)
#             raw_labels = copy.deepcopy(labels)
#         if not self.labels_only:
#             # print(point_cloud.points.shape)
#             point_cloud.points = self.filter.filter_pointcloud(point_cloud.points)
#         # print("After filtering",point_cloud.points.shape)
#         point_cloud.raw_points = point_cloud.points.copy()
#         # print("After filtering",point_cloud.points.shape)
#         # point_cloud.convert_to_kitti_points()
#         # print(type(labels),len(labels))
#         # print("------------------")
#         labels = self.filter.filter_bounding_boxes(labels)
#         if self.is_e2e:
#             item_dict = {'pointcloud': {'filtered':point_cloud,'raw':raw_point_cloud},
#                         'labels': {'filtered':labels,'raw':raw_labels},
#                         'file_name':lidar_filepath}
#         else:
#             item_dict = {'pointcloud': point_cloud, 'labels': labels, 'file_name': lidar_filepath}
#         return item_dict
    
#     def filter_boxes_with_category(self,box_label,accepted_categories=['vehicle.','human','cyclist']):
#         for cat in accepted_categories:
#             if box_label.startswith(cat):
#                 return True
#         return False
    
