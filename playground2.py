
from nuscenes.nuscenes import NuScenes
import numpy as np
import os
#os.chdir('/mnt/ssd2/Introspect3D')
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
import imageio.v2 as imageio
import os
from torch.utils.data import DataLoader
dataroot = '/media/ssd_reza/nuscenes'
verbose = True
from enum import Enum
from tqdm import tqdm
import pdb
class Colors(Enum):
    RED = (1,0,0)
    GREEN = (0,1,0)
    BLUE = (0,0,1)
    YELLOW = (1,1,0)
    CYAN = (0,1,1)
    MAGENTA = (1,0,1)
    WHITE = (1,1,1)
    BLACK = (0,0,0)
    ORANGE = (1,0.5,0)
    PURPLE = (0.5,0,1)
    PINK = (1,0,0.5)
lv_length = 100
lv_width = 4

def get_lead_vehicle(data, length, width):
        lead_vehicle = []
        for box in data:
            # Check if any corner point is inside the ellipse
            # print(box)
            center = box.center.copy()
            x,y,_ = center
            if y > 0 and y <length and x > -width/2 and x < width/2:
                if len(lead_vehicle) == 0:
                    lead_vehicle.append(box)
                elif box.center[1] < lead_vehicle[0].center[1]:
                    lead_vehicle[0] = box
            
        return lead_vehicle




def custom_collate(batch):
    point_clouds = []
    labels = []
    file_names = []
    tokens = []
    for i, item in enumerate(batch):
        batch[i]['pointcloud'].validate_and_update_descriptors(extend_or_reduce=5)
        point_clouds.append(batch[i]['pointcloud'].points)
        labels.append(batch[i]['labels'])
        file_names.append(batch[i]['file_name'])
        tokens.append(batch[i]['sample_record']['token'])
    return point_clouds, labels, file_names, tokens
dataset = NuScenesDataset(root_dir='/media/ssd_reza/nuscenes',
                          version='v1.0-trainval',
                          split='train',
                          transform=None,
                          filtering_style = "FilterType.LEAD_VEHICLE",
                          filter_params = {'width':lv_width,'length':lv_length}, # offset -5
                          filter_labels_only = True,
                          save_path='/media/ssd_reza/nuscenes',
                          save_filename='nuscenes_train_lv.pkl',
                          process=False,)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn = custom_collate)

checkpoint = r'/home/wmg-5gcat/Desktop/Sajjad/DistEstIntrospection/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
config= r'/home/wmg-5gcat/Desktop/Sajjad/DistEstIntrospection/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
    
model = init_model(config, checkpoint, device='cuda:0')
if verbose:
    progress_bar = tqdm(total=len(dataloader))
result_list = []
for i,item in enumerate(dataloader):
    point_clouds, labels, file_names, tokens = item
    
    #visualizer = Visualizer()
    # config =r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py'
    # checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'
    # config =r'/mnt/ssd2/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py'
    # checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth'
    # config =r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-3class.py'
    # checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144-d1a706b1.pth'
    res_batch, data = inference_detector(model, point_clouds)
    for batch_itr, res in enumerate(res_batch):
        predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
        predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
        score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later
        filtered_predicted_boxes = predicted_boxes[score_mask]
        is_nuscenes = True
        from utils.utils import create_bounding_boxes_from_predictions
        prediction_bounding_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
        pred_lead_vehicle = get_lead_vehicle(prediction_bounding_boxes,lv_length,lv_width)
        result_dict = {
            'token':tokens[batch_itr],
            'pred_lead':pred_lead_vehicle,
            'gt_lead':labels[batch_itr],
        }
        result_list.append(result_dict)
        
    if verbose:
        progress_bar.update(1)
    # print("Number of matches: {}".format(len(matches)))
    # print("Number of unmatched ground truths: {}".format(len(unmatched_ground_truths)))
    # print("Number of unmatched predictions: {}".format(len(unmatched_predictions)))
    # print(type(prediction_bounding_boxes[0]),type(item['labels'][0]))
    #gt_box_color = []
    # list_colors = list(Colors)
    # for gt_itr, box in enumerate(item['labels']):
    #     #gt_box_color.append(list_colors[gt_itr % len(list_colors)])
    #     print(box.center,box.dimensions)#, gt_box_color[-1].name)
    # visualizer.visualize_save(cloud= item['pointcloud'].points[:,:3],
    #                           gt_boxes = item['labels'],
    #                           pred_boxes = prediction_bounding_boxes,
    #                           #gt_box_color= [color.value for color in gt_box_color],
    #                           save_path="./nuscenes_pointcloud3.png")  # item['labels']
    # print("saved")
# export result_list
import pickle
with open('result_list.pkl','wb') as f:
    pickle.dump(result_list,f)


