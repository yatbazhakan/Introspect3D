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
from utils.pointcloud import PointCloud
import pickle
from utils.visualizer import Visualizer
from datasets.nuscenes import NuScenesDataset
from utils.filter import FilterType, EllipseFilter,FilteringArea
from definitions import ROOT_DIR
import pandas as pd
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
def make_point_cloud(path):
    cloud = np.load(path)
    points = cloud[:, :3]            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(points), 3)))
    return pcd
def load_point_clouds(folder_path):
    """Load all point cloud files from the seeeeepecified folder."""
    files = sorted(glob(os.path.join(folder_path, 'lidar','*.npy')))

    point_clouds = [np.load(file) for file in files]
    return point_clouds

if __name__ == '__main__':
        # folder_path = '/media/yatbaz_h/Extreme SSD/HYY/Urban/2024-04-30-12-28-22/'
        folder_path = '/mnt/ssd2/HYY/Motorway/Rainy/run1_2024-05-16-13-36-20/'#/run7_2024-05-16-14-33-23/'
        # folder_path = '/mnt/ssd2/HYY/Urban/2024-04-30-12-28-22/'

        files = sorted(glob(os.path.join(folder_path, 'lidar','*.npy')))
        point_clouds = load_point_clouds(folder_path)   
        checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
        config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
        # config = r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_pillar02_second_secfpn_head-dcn_8xb4-cyclic-20e_nus-3d.py'
        # checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20220811_045458-808e69ad.pth' #r"D:\mmdetection3d\ckpts\hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth" #
        
        model = init_model(config, checkpoint, device='cuda:0')
        # label_df =pd.DataFrame(columns=['sample_path','is_error'])
        filt = EllipseFilter(15,25,-5,1)
        t_i = 1000
        for i,cloud in enumerate(point_clouds):
            print("---------NEW SAMPLE--------")
            if i < t_i:
                file_wo_extension = os.path.splitext(os.path.basename(files[i]))[0]
                if str(file_wo_extension) == '20240516_133942':
                    t_i = i
                else:
                    continue

            file_wo_extension = os.path.splitext(os.path.basename(files[i]))[0]
            print(file_wo_extension)
            visualizer = Visualizer()
            item = PointCloud(cloud)
            item.set_points_as_raw()
            print(item.points.shape)
            # print(item.points.shape)
            inside_points = filt.filter_pointcloud(cloud,FilteringArea.INSIDE)
            outside_points = filt.filter_pointcloud(cloud,FilteringArea.OUTSIDE)
            print(inside_points.shape,outside_points.shape) 
            # print(item.points.shape)
            item.validate_and_update_descriptors(extend_or_reduce=5)
            res, data = inference_detector(model, item.points)
            predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later
            filtered_predicted_boxes = predicted_boxes[score_mask]
            is_nuscenes = True
            from utils.utils import create_bounding_boxes_from_predictions
            prediction_bounding_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
            visualizer.visualize(cloud=inside_points,
                                outside_cloud=outside_points,
                                pred_boxes=prediction_bounding_boxes,
                                colors={'inside':np.full((inside_points.shape[0],3),[0.56470588235, 0.93333333333, 0.56470588235]),
                                        'outside':np.full((outside_points.shape[0],3),[0.56470588235, 0.93333333333, 0.56470588235])})#[0.0,0.5,1.0])})
            print("-----------------")

            # while error not in ['Y','y','N','n']:
            #     error = input("Please enter a valid input")

            # error = input("Press enter to continue")

            # print(file_wo_extension)
            # if error=='Y' or error=='y':
            #     temp_df = pd.DataFrame({'sample_path':file_wo_extension,'is_error':1},index=[0])
            #     label_df = pd.concat([label_df,temp_df])
            # elif error=='N' or error=='n':
            #     temp_df = pd.DataFrame({'sample_path':file_wo_extension,'is_error':0},index=[0])
            #     label_df = pd.concat([label_df,temp_df])

    #     label_df.to_csv('label_df_run1_urban.csv')
    # except KeyboardInterrupt as e:
    #     label_df.to_csv('label_df_run1_urban.csv')
    # except Exception as e:
    #     label_df.to_csv('label_df_run1_urban.csv')
    #     print(e)










# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 44.846057891845703, 131.50129699707031, 12.466060638427734 ],
# 			"boundingbox_min" : [ -43.393062591552734, -131.52217102050781, -5.5637688636779785 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ 0.018693435956732646, -0.82589250425491179, 0.56351763669621036 ],
# 			"lookat" : [ 0.72649765014648438, -0.01043701171875, 3.4511458873748779 ],
# 			"up" : [ 0.011721879862023592, 0.56375842025419354, 0.82585654996796909 ],
# 			"zoom" : 0.13999999999999962
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }




    
    # visualizer.visualize_save(cloud= item['pointcloud'].points[:,:3],
    #                           gt_boxes = [], #item['labels'],
    #                           pred_boxes = [],#prediction_bounding_boxes,
    #                           save_path="./nuscenes_pointcloud3.png")  # item['labels']
   
    # vis.run()
    # vis.destroy_window()
