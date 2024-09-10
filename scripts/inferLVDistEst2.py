from nuscenes.nuscenes import NuScenes
import numpy as np
import os
import logging
#os.chdir('/mnt/ssd2/Introspect3D')
import sys
sys.path.append('.')
import open3d as o3d
from pprint import pprint
from glob import glob
from mmdet3d.apis import inference_detector, init_model
from math import cos,sin
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud,Box
from nuscenes.nuscenes import NuScenes
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
import pickle
from utils.visualizer import Visualizer
from datasets.nuscenes import NuScenesDataset
import imageio
import os
from utils.pointcloud import PointCloud
from utils.utils import create_bounding_boxes_from_predictions
from utils.boundingbox import BoundingBox
from utils.utils import cart2frenet
from utils.logger import setup_logging
from tqdm import tqdm
import pandas as pd

def filter_boxes_with_category(box_label,accepted_categories=['vehicle.','human','cyclist']):
        for cat in accepted_categories:
            if box_label.startswith(cat):
                return True
        return False

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

def infer(mini=False):
    if mini:
        dataroot = '/media/ssd_reza/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
    else:
        dataroot = '/media/ssd_reza/nuscenes'
        version = 'v1.0-trainval'


    # Load Dataset and Model
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    checkpoint = r'/home/wmg-5gcat/Desktop/Sajjad/DistEstIntrospection/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
    config= r'/home/wmg-5gcat/Desktop/Sajjad/DistEstIntrospection/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'   
    model = init_model(config, checkpoint, device='cuda:0')


    scene_dicts = []
    print('Collecting Scene Data')
    progress_bar = tqdm(total=len(nusc.scene))

    for scene in nusc.scene:
        sampletoken = scene['first_sample_token']
        # Collect Scene data
        scene_dict = {
            'scene_token': scene['token'],
            'sample_token': [],
            'sensor_calib': None,
            'ego_pose': [],
            'ego_rot':[],
            'lidar_files':[],
            'gt_boxes': [],
            'pred_boxes': [],
        }
        sample = nusc.get('sample', sampletoken)
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        calib = nusc.get('calibrated_sensor',lidar['calibrated_sensor_token'])
        scene_dict['sensor_calib'] = calib
        while True:
            scene_dict['sample_token'].append(sampletoken)
            sample = nusc.get('sample', sampletoken)
            lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar['ego_pose_token'])
            quat = Quaternion(ego_pose['rotation'])
            scene_dict['ego_pose'].append(ego_pose)
            scene_dict['ego_rot'].append(quat.rotation_matrix)
            scene_dict['lidar_files'].append(lidar['filename'])
            
            _, boxes, _ = nusc.get_sample_data(sample['data']['LIDAR_TOP'])
            custom_boxes = []
            for i in range(len(boxes)):
                annotation = nusc.get('sample_annotation', sample['anns'][i])
                box = boxes[i]
                if filter_boxes_with_category(annotation['category_name']):
                    box.label = object_label_to_index[annotation['category_name']]
                    custom_box = BoundingBox()
                    custom_box.from_nuscenes_box(box)
                    custom_boxes.append(custom_box)
            scene_dict['gt_boxes'].append(custom_boxes)
            sampletoken = sample['next']
            if sampletoken == '':
                break
        progress_bar.update(1)
        scene_dicts.append(scene_dict)
    progress_bar.close()

    print('Inferencing')
    progress_bar = tqdm(total=len(scene_dicts))
    for i, scene_dict in enumerate(scene_dicts):
        # Infer
        lidar_files= scene_dict['lidar_files']
        for lidar_file in lidar_files:
            points = np.fromfile(os.path.join(dataroot,lidar_file),dtype=np.float32).reshape(-1,5)
            point_cloud = PointCloud(points)
            point_cloud.validate_and_update_descriptors(extend_or_reduce=5)
            res, data = inference_detector(model, point_cloud.points)
            predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later
            filtered_predicted_boxes = predicted_boxes[score_mask]
            prediction_bounding_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
            scene_dicts[i]['pred_boxes'].append(prediction_bounding_boxes)
        progress_bar.update(1)
    # save the scene_dicts
    pickle.dump(scene_dicts,open('outputs/lv_scene_dicts.pkl','wb'))
    
def postprocess(x_limit=50, y_limit=1.5):
    scene_dicts = pickle.load(open('outputs/lv_scene_dicts.pkl','rb'))
    logger.info('Postprocessing')
    result_list = []
    for scene_itr, scene in enumerate(scene_dicts):
        scene_token = scene['scene_token']
        gt_boxes = scene['gt_boxes']
        pred_boxes = scene['pred_boxes']
        lidar_files = scene['lidar_files']
        ego_poses = scene['ego_pose']
        sensor_calib = scene['sensor_calib']
        sample_token = scene['sample_token']
        # convert ego poses to frenet frame
        ego_xy = np.array([pose['translation'][0:2] for pose in ego_poses])
        ego_sd = cart2frenet(ego_xy, ego_xy)
        # extend in s direction 51 meters
        ego_sd_ext = np.concatenate((np.arange(1,52,1).reshape(-1,1),np.zeros((51,1))),axis=1)
        ego_sd = np.concatenate((ego_sd,ego_sd_ext),axis=0)
        # convert bbox to global frame
        sensor_translation = np.array(sensor_calib['translation'])
        sensor_rotation = np.array(sensor_calib['rotation'])
        sensor_rotation = Quaternion(sensor_rotation).rotation_matrix
        for i in range(len(gt_boxes)):
            ego_translation = ego_poses[i]['translation']
            ego_rotation = ego_poses[i]['rotation']
            ego_rotation = Quaternion(ego_rotation).rotation_matrix
            for j in range(len(gt_boxes[i])):
                box_center = np.array(gt_boxes[i][j].center)
                box_center = np.dot(sensor_rotation,box_center) + sensor_translation
                box_center = np.dot(ego_rotation,box_center) + ego_translation
                gt_boxes[i][j].global_center = box_center

            for j in range(len(pred_boxes[i])):
                box_center = np.array(pred_boxes[i][j].center)
                box_center = np.dot(sensor_rotation,box_center) + sensor_translation
                box_center = np.dot(ego_rotation,box_center) + ego_translation
                pred_boxes[i][j].global_center = box_center
        
        # convert to frenet frame
        for i in range(len(gt_boxes)):
            for j in range(len(gt_boxes[i])):
                gt_boxes[i][j].frenet_center = cart2frenet(np.array(gt_boxes[i][j].global_center[:2]).reshape(1,2), ego_xy)
            for j in range(len(pred_boxes[i])):
                pred_boxes[i][j].frenet_center = cart2frenet(np.array(pred_boxes[i][j].global_center[:2]).reshape(1,2), ego_xy)
    
        def get_lead_vehicle(boxes, ego_x, y_lim, x_lim):
            lead_vehicle = []
            for box in boxes:
                # Check if any corner point is inside the ellipse
                x = box.frenet_center[0,0]
                y = box.frenet_center[0,1]
                if np.abs(y)<y_lim and x>ego_x and x<(ego_x+x_lim):
                    if len(lead_vehicle) == 0:
                        lead_vehicle.append(box)
                    elif box.frenet_center[0,0] < lead_vehicle[0].frenet_center[0,0]:
                        lead_vehicle[0] = box
            if len(lead_vehicle)>0:
                lead_vehicle[0].frenet_dist2ego = lead_vehicle[0].frenet_center[0,0] - ego_x
            return lead_vehicle        
        # get lead vehicle
        scene_dicts[scene_itr]['gt_lead'] = []
        scene_dicts[scene_itr]['pred_lead'] = []
        for i in range(len(gt_boxes)):
            gt_lead_vehicle = get_lead_vehicle(gt_boxes[i],ego_sd[i,0],y_limit,x_limit)
            pred_lead_vehicle = get_lead_vehicle(pred_boxes[i],ego_sd[i,0],y_limit,x_limit)
            scene_dicts[scene_itr]['gt_lead'].append(gt_lead_vehicle)
            scene_dicts[scene_itr]['pred_lead'].append(pred_lead_vehicle)
            result_dict = {
                'token': sample_token[i],
                'pred_lead':pred_lead_vehicle,
                'gt_lead': gt_lead_vehicle,
                'ego_xy': ego_xy,
                'iter':i,
                }
            result_list.append(result_dict)

    with open('notebooks/result_list.pkl','wb') as f:
        pickle.dump(result_list,f)
    with open('outputs/lv_scene_dicts.pkl','wb') as f:
        pickle.dump(scene_dicts,f)

def generate_csv(mini):
    if mini:
        dataroot = '/media/ssd_reza/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
    else:
        dataroot = '/media/ssd_reza/nuscenes'
        version = 'v1.0-trainval'
    with(open('notebooks/result_list.pkl','rb')) as f:
        result_list = pickle.load(f)
    result_df = pd.DataFrame(columns=['name','token','gt_lead','pred_lead', 'ego_xy'])
    gt_lead = []
    pred_lead = []
    name = []
    ego_xy = []
    tokens = []
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    

    for result in result_list:
        if len(result['gt_lead'])>0:
            gt_lead.append(np.array(result['gt_lead'][0].frenet_center))
        else:
            gt_lead.append([])
        if len(result['pred_lead'])>0:
            pred_lead.append(np.array(result['pred_lead'][0].frenet_center))
        else:
            pred_lead.append([])
        token = result['token']
        sample_data = nusc.get('sample', token)
        lidar_token = sample_data['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)
        lidar_path = os.path.join('/media/ssd_reza/nuscenes/',lidar_data['filename'].split('.pcd.bin')[0])
        name.append(lidar_path)
        ego_xy.append(result['ego_xy'])
        tokens.append(token)
    result_df['name'] = name
    result_df['gt_lead'] = gt_lead
    result_df['pred_lead'] = pred_lead
    result_df['ego_xy'] = ego_xy
    result_df['token'] = tokens
    logger.debug(result_df.head())
    result_df.to_csv('/media/ssd_reza/sajjad/distest/custom_dataset/nus_centerpoint_early_activations/lv_dist_est2.csv', index=False)

def add_lidar_path_to_dict(mini):
    if mini:
        dataroot = '/media/ssd_reza/nuscenes/v1.0-mini'
        version = 'v1.0-mini'
    else:
        dataroot = '/media/ssd_reza/nuscenes'
        version = 'v1.0-trainval'
    with(open('outputs/lv_scene_dicts.pkl','rb')) as f:
        data_list = pickle.load(f)
    
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    

    for i,data in enumerate(data_list):
        tokens = data['sample_token']
        lidar_paths = []
        for token in tokens:
            sample_data = nusc.get('sample', token)
            lidar_token = sample_data['data']['LIDAR_TOP']
            lidar_data = nusc.get('sample_data', lidar_token)
            lidar_path = os.path.join('/media/ssd_reza/nuscenes/',lidar_data['filename'].split('.pcd.bin')[0])
            lidar_paths.append(lidar_path)
        data_list[i]['lidar_path'] = lidar_paths
    with open('outputs/lv_scene_dicts.pkl','wb') as f:
        pickle.dump(data_list,f)
if __name__ == '__main__':
    logger = setup_logging('inferLVDistEst2', level=logging.DEBUG)
    mini = False
    #infer(mini) 
    postprocess()
    #generate_csv(mini)
    #add_lidar_path_to_dict(mini)