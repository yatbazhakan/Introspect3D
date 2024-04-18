from nuscenes.nuscenes import NuScenes
import numpy as np
import os
os.chdir('/mnt/ssd2/Introspect3D')
import open3d as o3d
from pprint import pprint
from glob import glob
from mmdet3d.apis import inference_detector, init_model
from math import cos,sin
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud,Box
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
def rotate_points(points, R):
    return np.dot(points, R.T)

def match_detections_3d(ground_truths, predictions, iou_threshold=0.5):
    matches = []
    unmatched_ground_truths = []
    unmatched_predictions = list(predictions)

    for gt in ground_truths:
        max_iou = -1
        max_iou_idx = -1
        for idx, pred in enumerate(unmatched_predictions):
            current_iou = calculate_iou_3d(gt, pred)
            if current_iou > max_iou:
                max_iou = current_iou
                max_iou_idx = idx
        print("max iou is",max_iou)
        if max_iou >= iou_threshold:
            matches.append((gt, unmatched_predictions[max_iou_idx]))
            del unmatched_predictions[max_iou_idx]
        else:
            unmatched_ground_truths.append(gt)

    return matches, unmatched_ground_truths, unmatched_predictions
def calculate_iou_3d(box1,box2):
    # Create corner points for both boxes
    if type(box1) == o3d.geometry.OrientedBoundingBox:
        center1 = np.array(box1.center)
        dimensions1 = np.array(box1.extent)
        half_dims1 = dimensions1 / 2
        corners1 = np.array([np.array([x, y, z]) for x in [-half_dims1[0], half_dims1[0]] for y in [-half_dims1[1], half_dims1[1]] for z in [-half_dims1[2], half_dims1[2]]])
        corners1 = rotate_points(corners1, box1.R) + center1

    else:
        #get corners from lineset
        # print("Getting corners from lineset")
        corners1 = np.array(box1.corners)
 
        dimensions1 = box1.dimensions
        
    if type(box2)== o3d.geometry.OrientedBoundingBox:
        center2 = np.array(box2.center)
        dimensions2 = np.array(box2.extent)
        half_dims2 = dimensions2 / 2
        corners2 = np.array([np.array([x, y, z]) for x in [-half_dims2[0], half_dims2[0]] for y in [-half_dims2[1], half_dims2[1]] for z in [-half_dims2[2], half_dims2[2]]])
        # Rotate and translate corners
        corners2 = rotate_points(corners2, box2.R) + center2
    else:
        #get corners from lineset
        # print("Getting corners from lineset")
    
        corners2 = np.array(box2.corners)
        
        dimensions2 = box2.dimensions
    # Calculate axis-aligned bounding boxes for intersection
    print(corners1.shape,corners2.shape)
    pprint(corners1)

    pprint(corners2)
    min_bound1 = np.min(corners1, axis=0)
    max_bound1 = np.max(corners1, axis=0)
    min_bound2 = np.min(corners2, axis=0)
    max_bound2 = np.max(corners2, axis=0)
    
    # Calculate intersection
    min_intersect = np.maximum(min_bound1, min_bound2)
    max_intersect = np.minimum(max_bound1, max_bound2)
    intersection_dims = np.maximum(0, max_intersect - min_intersect)
    intersection_volume = np.prod(intersection_dims)
    
    # Calculate volumes of the individual boxes
    vol1 = np.prod(dimensions1)
    vol2 = np.prod(dimensions2)
    
    # Calculate IoU
    iou = intersection_volume / (vol1 + vol2 - intersection_volume)
    return iou
def is_inside_ellipse(x, y, a, b):
    return (x**2 / a**2) + (y**2 / b**2) <= 1
def is_outside_ellipse(x, y, a, b):
    return (x**2 / a**2) + (y**2 / b**2) > 1
def filter_objects_outside_ellipse(objects, a, b,offset=5,axis=0):
    """
    Filters ground truth objects to only include those outside a specified ellipse.

    Parameters:
        objects (list): The input objects, each as a dictionary.
        a (float): Semi-major axis length of the ellipse.
        b (float): Semi-minor axis length of the ellipse.

    Returns:
        list: The filtered objects.
    """
    filtered_objects = []
    
    for obj in objects:
        corners = obj.copy()
        if(len(corners) != 8):
          corners = corners.Tq
        corners[:, axis] = corners[:, axis] + offset
        inside_ellipse = is_inside_ellipse(corners[:, 0], corners[:, 1], a, b)
        if np.any(inside_ellipse):
            filtered_objects.append(obj)
    # print(filtered_objects)
    return filtered_objects
def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.
  
    Output
      :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  
    return [qx, qy, qz, qw]
def plot_bounding_box_from_corners(corners,offset =0, calib=None, color=[0, 1, 0]):
    # Define the lines connecting the corners based on their indices
    if corners.shape[0] != 8:
       corners = corners.T
    # # pprint(corners)
    # lines = [[0, 1], [1, 2], [2, 3], [3, 0], # Lower face
    #          [4, 5], [5, 6], [6, 7], [7, 4], # Upper face
    #           [0, 4], [1, 5], [2, 6], [3, 7]] # Connect the faces

    # # Create a LineSet object
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(corners)
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
    line_set = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))
    line_set.color = color
    return line_set
def create_oriented_bounding_box(box_params,rot_axis=2,calib=True,color=(1, 0, 0)):


    center = np.array([box_params[0], box_params[1], box_params[2]])
    extent = np.array([box_params[3], box_params[4], box_params[5]])
    if(len(box_params) > 9):
      print("Too many parameters for box")
      quat = Quaternion(box_params[6:10])
      rot_mat = quat.rotation_matrix
    elif(len(box_params) == 9):
      yaw = np.zeros(3)
      yaw[2] = box_params[6]

      rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

    center[2] += extent[2] / 2
    print(rot_mat)
    box3d = geometry.OrientedBoundingBox(center, rot_mat, extent)
    
    line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
    line_set.paint_uniform_color(color)

    #  Move box to sensor coord system
    ctr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=center)
    # obb.color = (1, 0, 0)  # Red color
    return box3d#line_set #, ctr
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

def filter_points_inside_ellipse(points, a, b,offset=5,axis=0):
    xyz  = points.copy()
    xyz[:,axis] += offset
    x,y,z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    inside = is_inside_ellipse(x, y, a, b)
    return points[inside]
def filter_points_outside_ellipse(points, a, b,offset=5,axis=0):
    xyz  = points.copy()
    xyz[:,axis] += offset
    x,y,z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    outside = is_outside_ellipse(x, y, a, b)
    return points[outside]
def filter_boxes_with_category(box_label,accepted_categories=['vehicle.','human','cyclist']):
    for cat in accepted_categories:
        if box_label.startswith(cat):
            return True
    return False
from datasets.nuscenes import NuScenesDataset
from utils.filter import FilterType
from definitions import ROOT_DIR
def get_lat_long_str(dataset,box):
    lat,lng ="",""
    if dataset == "nuscenes":
        # print(box.center)
        if box.center[0] > 0:
            lat = "R"
        else:
            lat = "L"
        if box.center[1] > 0:
            lng = "F"
        else:
            lng = "B"
    # print("-",lat,lng)
    return lat,lng




def captionize_missed(missed_objects,dataset="nuscenes",mode="closest"):
    closest_distance = np.inf
    closest_location = ""
    find_locations = []
    each_location = []
    for box in missed_objects:
        euc_distance = np.linalg.norm(box.center)
        lateral_location = ""
        long_location = ""
        lateral_location,long_location = get_lat_long_str(dataset,box)
        each_location.append((f"{lateral_location}{long_location}",euc_distance))
    each_location = sorted(each_location,key=lambda x:x[1])
    if len(each_location) > 0:
        locs = [loc[0] for loc in each_location]
        text = f"{len(missed_objects)},{','.join(locs)}"
    else:
        text = f"{len(missed_objects)},NN"
    return text
                

      
from tqdm.auto import tqdm


dataset = NuScenesDataset(root_dir='/mnt/ssd2/nuscenes/',
                          version='v1.0-trainval',
                          split='train',
                          transform=None,
                          filtering_style = "FilterType.NONE",
                          filter_params = {'a':15,'b':25,'offset':-5,'axis':1},
                          save_path='/mnt/ssd2/nuscenes/',
                          save_filename='nuscenes_train.pkl',
                          process=False,)
# print(len(dataset))
# exit()  
sindex = 0
checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
model = init_model(config, checkpoint, device='cuda:0')
with open('/mnt/ssd2/test_captions2.txt','w') as f:
    with tqdm(total=len(dataset)) as pbar:
        for i,item in enumerate(dataset):
            # if i != sindex:
            #     continue
            # if i>sindex:
            #     break
            # visualizer = Visualizer()
            # config =r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py'
            # checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'
            checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
            config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
            # config =r'/mnt/ssd2/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py'
            # checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth'
            # config =r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-3class.py'
            # checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144-d1a706b1.pth'
            
            name = item['file_name']
            item['pointcloud'].validate_and_update_descriptors(extend_or_reduce=5)
            res, data = inference_detector(model, item['pointcloud'].points)
            predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later
            filtered_predicted_boxes = predicted_boxes[score_mask]
            is_nuscenes = True
            from utils.utils import create_bounding_boxes_from_predictions
            prediction_bounding_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
            from utils.utils import check_detection_matches
            matches , unmatched_ground_truths, unmatched_predictions = check_detection_matches(item['labels'],prediction_bounding_boxes)
            missed_info = captionize_missed(unmatched_ground_truths)
            f.write(f"{name},{missed_info}\n")
            pbar.update(1)
            # print(missed_info)
            # print("Number of matches: {}".format(len(matches)))
            # print("Number of unmatched ground truths: {}".format(len(unmatched_ground_truths)))
            # print("Number of unmatched predictions: {}".format(len(unmatched_predictions)))