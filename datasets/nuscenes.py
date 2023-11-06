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

# object_label_to_index = {
#     "vehicle.emergency.police car": 0,
#     "vehicle.emergency.ambulance": 1,
#     "vehicle.emergency.fire truck": 2,
#     "vehicle.construction": 3,
#     "vehicle.truck": 4,
#     "vehicle.bus": 5,
#     "vehicle.van": 6,
#     "vehicle.motorcycle": 7,
#     "vehicle.car": 8,
#     "pedestrian": 9,
#     "cyclist": 10,
#     "traffic_cone": 11,
#     "pole": 12,
#     "fire_hydrant": 13,
#     "stop_sign": 14,
#     "traffic_light": 15,
#     "parking_sign": 16,
#     "speed_bump": 17,
#     "sidewalk": 18,
#     "parking_slot": 19,
#     "road_marking": 20,
#     "vegetation": 21,
#     "terrain": 22,
#     "other": 23,
# }
object_index_to_label = {v: k for k, v in object_label_to_index.items()}

class NuScenesDataset(DrivingDataset):
    def __init__(self, dataroot, version='v1.0-mini', split='train', transform=None,
                 filtering_style: FilterType = FilterType.ELLIPSE,**kwargs):
        process = kwargs['process']
        self.save_path,self.save_filename = kwargs['save_path'], kwargs['save_filename']

        if process:
            self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
            self.split = split
            self.transform = transform
            # Extract only the first sample token of each scene.
            self.sample_tokens = [s['first_sample_token'] for s in self.nusc.scene]
            self.filtering_style = eval(filtering_style)
            self.filter_params = kwargs['filter_params']
            self.filter = self.filtering_style.value(**self.filter_params)
            self.dataset_flattened = {}
            self.process_data()
        else:
            self.dataset_flattened = pickle.load(open(os.path.join(self.save_path,self.save_filename),'rb'))

        

    def read_labels(self, **kwargs):
        id = kwargs['id']
        lidar_token = kwargs['lidar_token']
        sample_record = kwargs['sample_record']
        _,  boxes, _  = self.nusc.get_sample_data(lidar_token)
        filtered_boxes = []
        for i in range(len(boxes)):
            annotation = self.nusc.get('sample_annotation', sample_record['anns'][i])
            box = boxes[i]
            if self.filter_boxes_with_category(annotation['category_name']):
                box.label = object_label_to_index[annotation['category_name']]
                custom_box = BoundingBox()
                custom_box.from_nuscenes_box(box)
                filtered_boxes.append(custom_box)
        self.dataset_flattened[id]['label'].append(filtered_boxes)

        return super().read_labels(**kwargs)
    def process_data(self, **kwargs):
        id = 0
        for first_sample_token in self.sample_tokens:
            while not first_sample_token == '':
                sample_record = self.nusc.get('sample', first_sample_token)
                lidar_token = sample_record['data']['LIDAR_TOP']
                # print(calibrated_lidar)
                lidar_data = self.nusc.get('sample_data', lidar_token)
                if lidar_data['is_key_frame']:
                    lidar_filepath = self.nusc.get_sample_data_path(lidar_token)
                    self.dataset_flattened[id] = {}
                    self.dataset_flattened[id]['path'] = lidar_filepath
                    self.dataset_flattened[id]['label'] = []
                    self.read_labels(id=id, lidar_token=lidar_token, sample_record=sample_record)
                    id += 1
                first_sample_token = sample_record['next']
        pickle.dump(self.dataset_flattened,open(os.path.join(self.save_path,self.save_filename),'wb'))
    
    
    def __len__(self):
        return len(self.dataset_flattened)

    def __getitem__(self, idx):
        # Fetch the sample token, get sample data
        data= self.dataset_flattened[idx]
        lidar_filepath = data['path']
        labels = data['label']

        # Load the point cloud data
        pointcloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
        pc = PointCloud(pointcloud)
        
        return {'pointcloud': pc, 'sample_token': labels}
    
    def filter_boxes_with_category(self,box_label,accepted_categories=['vehicle.','human','cyclist']):
        for cat in accepted_categories:
            if box_label.startswith(cat):
                return True
        return False