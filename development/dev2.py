import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from glob import glob
import open3d as o3d
import mmdet3d
import numpy as np
import pandas as pd
import cv2
from plyfile import PlyData
from mmcv.transforms.base import BaseTransform
from mmengine.registry import TRANSFORMS, VISUALIZERS
from mmengine.structures import InstanceData
from mmdet3d.utils import register_all_modules
from mmdet3d.apis import init_model as init_3ddetector
from mmdet3d.apis import inference_detector
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmseg.apis import init_model as init_segmentor
CITYSCAPES_COLORS={
    "road": (128, 64, 128), #0
    "sidewalk": (244, 35, 232),#1
    "building": (70, 70, 70),#2
    "wall": (102, 102, 156),#3
    "fence": (190, 153, 153),#4
    "pole": (153, 153, 153),#5
    "traffic_light": (250, 170, 30),#6
    "traffic_sign": (220, 220, 0),#7
    "vegetation": (107, 142, 35),#8
    "terrain": (152, 251, 152),#9
    "sky": (70, 130, 180),#10
    "person": (220, 20, 60),#11
    "rider": (255, 0, 0),#12
    "car": (0, 0, 142),#13
    "truck": (0, 0, 70),#14
    "bus": (0, 60, 100),#15
    "train": (0, 80, 100),#16
    "motorcycle": (0, 0, 230),#17
    "bicycle": (119, 11, 32)#18
}

CITYSCAPES_INDEX_TO_COLOR= {i:color for i,(name,color) in enumerate(CITYSCAPES_COLORS.items())}

from mmseg.apis import inference_model
print("Open3D Version: " + o3d.__version__)
# o3d.visualization.webrtc_server.enable_webrtc()

save_name = ""

save_dir = "/mnt/ssd1/Introspect3D/custom_dataset/pointpillars_kitti_class3"
def get_rotation_matrix_from_corners(corners):
    """
    Get the rotation matrix from 8 corners of a 3D bounding box.
    
    Parameters:
        corners (numpy.ndarray): 3x8 array containing the coordinates of the 8 corners.
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix.
    """
    
    # Assume the corners are ordered such that:
    # - corners[:, 0] and corners[:, 1] are opposite corners on the "bottom" face of the box
    # - corners[:, 0] and corners[:, 4] are opposite corners on the "front" face of the box
    # - corners[:, 0] and corners[:, 3] are opposite corners on the "left" face of the box
    
    # Compute the vectors representing the edges
    edge1 = corners[:, 1] - corners[:, 0]  # Vector from corner 0 to corner 1
    edge2 = corners[:, 4] - corners[:, 0]  # Vector from corner 0 to corner 4
    edge3 = corners[:, 3] - corners[:, 0]  # Vector from corner 0 to corner 3
    
    # Normalize the vectors
    edge1 /= np.linalg.norm(edge1)
    edge2 /= np.linalg.norm(edge2)
    edge3 /= np.linalg.norm(edge3)
    
    # Construct the rotation matrix
    R = np.column_stack((edge1, edge2, edge3))
    
    return R
def is_inside_ellipse(x, y, a, b):
    return (x**2 / a**2) + (y**2 / b**2) <= 1
def is_outside_ellipse(x, y, a, b):
    return (x**2 / a**2) + (y**2 / b**2) > 1
def set_custom_view(vis):
    ctr = vis.get_view_control()
    
    # Create an extrinsic matrix for camera placement
    extrinsic = np.eye(4)
    extrinsic[0:3, 3] = [-10, 0, 0]  # Set camera position (x, y, z)
    
    # Create a rotation matrix for 30-degree downward view
    rotation = np.array([
        [1, 0, 0],
        [0, np.cos(np.radians(-160)), -np.sin(np.radians(-160))],
        [0, np.sin(np.radians(-160)), np.cos(np.radians(-160))]
    ])
    
    # Apply rotation to the extrinsic matrix
    extrinsic[0:3, 0:3] = rotation
    
    # Set the extrinsic matrix to the camera parameters
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])

def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points
def read_calib_file(calib_file):
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
def plot_bounding_box_from_corners(corners,offset =0, calib=None, color=[0, 1, 0]):
    # Define the lines connecting the corners based on their indices

    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Create a LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])

    return line_set
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
        # print(obj)
        if 'corners' not in obj.keys():
            x, y = obj['x'], obj['y']
            
            # Generate corner points for the box
            dx, dy = obj['dx'], obj['dy']

        else:
            x,y,_ = obj['real_center']
        dx, dy = obj['dx'], obj['dy']
    
        # Check if any corner point is inside the ellipse
        corners = np.array([
                [x - dx/2, y - dy/2],
                [x - dx/2, y + dy/2],
                [x + dx/2, y - dy/2],
                [x + dx/2, y + dy/2]
            ])
        adjusted_corners_x = corners[:, axis] + offset
        inside_ellipse = is_inside_ellipse(adjusted_corners_x, corners[:, 1], a, b)
        if np.any(inside_ellipse):
            filtered_objects.append(obj)
    # print(filtered_objects)
    return filtered_objects

def create_oriented_bounding_box_gt(box_params, color=(0, 1, 0),calib=None, offset=0,axis=0):
    center = np.array([box_params['x'], box_params['y'], box_params['z']/2])
    extent = np.array([box_params['dx'], box_params['dy'], box_params['dz']])
    R = box_params['yaw']

    # Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    if("corners" in box_params.keys()):
    # # Convert center to homogeneous coordinates and transform to LiDAR coordinates
    # point_homogeneous = np.hstack((center, np.ones(1)))
    # center_tr = transform_camera_to_lidar(point_homogeneous, Tr_velo_to_cam)
        center = box_params['real_center']
        coners = box_params['corners']
        # coners[:,2] /=2
        coners[:,axis] += offset
        return plot_bounding_box_from_corners(coners,offset=offset, calib=calib, color=color)
        # R_empty = np.eye(3)
        # obb = o3d.geometry.OrientedBoundingBox(center=real_center, R=R, extent=extent)
        # cntr = o3d.geometry.OrientedBoundingBox(center=real_center, R=R, extent=[0.1,0.1,0.1])
        # cntr.color = (1,1,1)
        # obb.color = color
        # return obb, cntr
    # Rotate around the Y-axis for KITTI's rotation_y
    # print("Plotting from center")
    # Create OrientedBoundingBox
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    cntr = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=[0.1,0.1,0.1])
    cntr.color = (1,1,1)
    obb.color = color 
    
    return obb
def filter_points_inside_ellipse(points, a, b,offset=5):
    x, y, z = points[:, 0]+offset, points[:, 1], points[:, 2]
    inside = is_inside_ellipse(x, y, a, b)
    return points[inside]
def filter_points_outside_ellipse(points, a, b,offset=5):
    x, y, z = points[:, 0]+offset, points[:, 1], points[:, 2]
    outside = is_outside_ellipse(x, y, a, b)
    return points[outside]
def read_kitti_label_file(bin_file_path, filter_classes=['Car', 'Pedestrian', 'Cyclist']):
    """
    Reads a KITTI label .txt file and returns a list of dictionaries containing
    object type and bounding box parameters, filtered by specified classes.

    Parameters:
        txt_file_path (str): Path to the KITTI label .txt file.
        filter_classes (List[str]): List of classes to include.

    Returns:
        List[Dict]: A list of dictionaries containing 'type', 'x', 'y', 'z', 'dx', 'dy', 'dz', and 'yaw'.
    """
    txt_file_path = bin_file_path.replace('.bin', '.txt').replace("velodyne","label_2")
    calib_file_path = bin_file_path.replace('.bin', '.txt').replace("velodyne","calib")
    calib = read_calib_file(calib_file_path)
    objects = []
    from math import cos,sin

    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.strip().split(' ')
            obj_type = tokens[0]
            if obj_type not in filter_classes:
                continue
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
            R = np.array([[cos(rotation_y),0,sin(rotation_y)],[0,1,0],[-sin(rotation_y),0,cos(rotation_y)]])
            rotated_corners = np.matmul(R,corner_matrix)
            translated_corners = rotated_corners + center_tr.reshape(3,1)
            
            Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
            Tr_velo_to_cam_extended = np.eye(4)  # Create a 4x4 identity matrix
            Tr_velo_to_cam_extended[:3, :] = Tr_velo_to_cam  # Replace the top-left 3x4 block
            
            # rotation_component = np.eye(4)
            # rotation_component[:3,:3] = Tr_velo_to_cam_extended[:3,:3]
            
            # translation_component = np.eye(4)
            # translation_component[:3,3] = Tr_velo_to_cam_extended[:3,3]
            
            # inverse_rotation = np.linalg.inv(rotation_component)
            # inverse_translation =  -translation_component
            
            # new_transform = np.eye(4)
            # new_transform[:3,:3] = inverse_rotation[:3,:3]
            # new_transform[:3,3] = inverse_translation[:3,3]

            # print(new_transform.shape,translated_corners.shape)
            T_inv = np.linalg.inv(Tr_velo_to_cam_extended)
            Homogeneous_corners = np.ones((4,8))
            Homogeneous_corners[:3,:] = translated_corners
            translated_corners = np.matmul(T_inv,Homogeneous_corners)[:3,:]
            real_center = np.mean(translated_corners,axis=1)
            # print(translated_corners.shape)
            # real_center = np.mean(translated_corners,axis=1)
            # print(real_center)
            objects.append({
                'type': obj_type,
                'x': real_center[0],
                'y': real_center[1],
                'z': real_center[2],
                'dx': dimensions_length,
                'dy': dimensions_height,
                'dz': dimensions_width,
                'yaw': o3d.geometry.get_rotation_matrix_from_xyz((0,rotation_y,0)),
                'corners': translated_corners.T,
                'real_center': real_center
            })
    return objects
def create_oriented_bounding_box(box_params,offset=100,axis=0,calib=None,color=(1, 0, 0)):
    offset_array = np.zeros(3)
    offset_array[axis] = offset
    center = np.array([box_params[0], box_params[1], box_params[2]])
    extent = np.array([box_params[3], box_params[4], box_params[5]])
    center[2] += extent[2]/2
    R = o3d.geometry.get_rotation_matrix_from_xyz((0,0,box_params[6]))

    # Transform_lidar_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    # extended_transform = np.eye(4)
    # extended_transform[:3, :] = Transform_lidar_to_cam
    # corners = np.array([center + np.dot(R, np.array([x, y, z])) for x in [-extent[0], extent[0]] for y in [-extent[1], extent[1]] for z in [-extent[2], extent[2]]])
    # center_from_corners = np.mean(corners, axis=0)
    # center  = center_from_corners
    center += offset_array
    R_empty = np.eye(3)
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    obb.color = (1, 0, 0)  # Red color
    return obb
def create_oriented_bounding_box_nuscenes(detection, offset=100, axis=0, color=(1, 0, 0),calib=None):
    offset_array = np.zeros(3)
    offset_array[axis] = offset

    # Extract parameters from the detection array
    x, y, z, dx, dy, dz, r, _, _ = detection

    # Define the center and extent of the box
    center = np.array([x, y, z/2])
    extent = np.array([dx, dy, dz])

    # Create the rotation matrix from the heading angle
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0 ))

    # Apply the offset to the center
    center += offset_array

    # Create the Oriented Bounding Box
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    obb.color = color  # Set the color
    obb_for_center = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=[0.1,0.1,0.1])
    obb_for_center.color = (1,1,1)
    return obb , obb_for_center
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
        # print("For gt",gt,"max iou is",max_iou)
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
        corners1 = np.array(box1.points)
        dimensions_from_corners = np.max(corners1,axis=0) - np.min(corners1,axis=0)
        dimensions1 = dimensions_from_corners
        
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

        corners2 = np.array(box2.points)
        dimensions_from_corners = np.max(corners2,axis=0) - np.min(corners2,axis=0)
        dimensions2 = dimensions_from_corners
    # Calculate axis-aligned bounding boxes for intersection
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


def is_point_inside_obb(obb, point):
    """
    Check if a point is inside an oriented bounding box (OBB).
    
    Parameters:
    - obb: Open3D OrientedBoundingBox object.
    - point: NumPy array of shape (3,) representing the point.
    
    Returns:
    - bool: True if the point is inside the OBB, False otherwise.
    """
    # Transform the point to the OBB's local coordinate system
    point_local = np.linalg.inv(obb.R).dot((point[:3] - obb.center))
    
    # Check if the transformed point is inside the axis-aligned box
    return np.all(np.abs(point_local) <= (obb.extent / 2))

def remove_points_inside_obbs(point_cloud, obbs):
    """
    Remove points inside oriented bounding boxes from a point cloud.
    
    Parameters:
    - point_cloud: NumPy array of shape (N, 3), where N is the number of points.
    - obbs: List of Open3D OrientedBoundingBox objects.
    
    Returns:
    - filtered_point_cloud: NumPy array containing points outside the bounding boxes.
    """
    mask = np.ones(point_cloud.shape[0], dtype=bool)
    
    for obb in obbs:
        for i, point in enumerate(point_cloud):
            if is_point_inside_obb(obb, point):
                mask[i] = False
    
    filtered_point_cloud = point_cloud[mask]
    
    return filtered_point_cloud
def backbone_extraction_hook(ins,inp,out):
    global image_name, save_dir,flag
    last_output = out[2].detach().cpu().numpy()
    last_output = np.squeeze(last_output)
    if(flag):
        np.save(os.path.join(save_dir,"features" ,save_name + '.npy'), last_output)
    else:
        np.save(os.path.join(save_dir,"removed_features" ,save_name + '.npy'), last_output)


if __name__ == '__main__':
    import random
    from tqdm.auto import tqdm
    #Define paths
    used_model = "pointpillars"
    training_set = "kitti"
    num_classes = 3
    # kitti_velodyne_path= r"D:\introspectionBase\datasets\Kitti\raw\training\velodyne"
    # img_path = r"D:\introspectionBase\datasets\Kitti\raw\training\image_2"
    # config_file = r'D:\mmdetection3d\configs\pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'#r'D:\mmdetection3d\configs\pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py' # 
    # checkpoint = r'D:\mmdetection3d/ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth' #r"D:\mmdetection3d\ckpts\hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth" #
    kitti_velodyne_path= r"/mnt/ssd2/kitti/training/velodyne"
    img_path = r"/mnt/ssd2/kitti/training/image_2/"
    config_file = r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'#r'D:\mmdetection3d\configs\pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py' # 
    checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth' #r"D:\mmdetection3d\ckpts\hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth" #
    # sem_config_file = r'/mnt/ssd2/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r18b-d8_4xb2-80k_cityscapes-512x1024.py'
    # sem_checkpoint = r'/mnt/ssd2/mmsegmentation/ckpts/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth'
    sem_config_file = r'/mnt/ssd2/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py'

    sem_checkpoint=r"/mnt/ssd2/mmsegmentation/ckpts/mask2former_r50_8xb2-90k_cityscapes-512x1024_20221202_140802-ffd9d750.pth"
    save_path = f"./custom_dataset/{used_model}_{training_set}_class{str(num_classes)}/labels/"
    model = init_3ddetector(config_file, checkpoint, device='cuda:0')
    seg_model = init_segmentor(sem_config_file, sem_checkpoint, device='cuda:0')
    # seg_model = 
    os.makedirs(save_path,exist_ok=True)
    new_dataset = pd.DataFrame(columns=['image_path', 'is_missed','missed_objects','total_objects'])
    # hook = model.backbone.register_forward_hook(backbone_extraction_hook)
    t = 11
    with tqdm(total=7481) as pbar:
        for i in range(7481):
            print("Processing image",i)
            test_num= i #random.randint(0,7480)# 11 #
            if i < t:
                continue
            if i>t:
                break
            filename = os.path.join(kitti_velodyne_path, f'{test_num:06}.bin')
            save_name = f'{test_num:06}'
            print(filename)
            #Read Calibration file, load point cloud and transfrom to image coordinates
            path = os.path.join(img_path, f'{test_num:06}.png')
            print(path)
            image = cv2.imread(path)
            print(image.shape)
            rgb_image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            segmentation_result = inference_model(seg_model, rgb_image)
            # print(segmentation_result)
            # best_class = np.argmax(segmentation_result[0],axis=0)
            #Overlay with the original image
            pred_seg_mask = segmentation_result.pred_sem_seg.data.cpu().numpy()
            print(np.unique(pred_seg_mask), pred_seg_mask.shape)

            # Reshape pred_seg_mask to add a singleton third dimension
            pred_seg_mask = pred_seg_mask.reshape(pred_seg_mask.shape[1], pred_seg_mask.shape[2], 1)
            print(pred_seg_mask.shape)

            # Create colorified_mask with three channels
            colorified_mask = np.zeros((pred_seg_mask.shape[0], pred_seg_mask.shape[1], 3), dtype=np.uint8)
            print(colorified_mask.shape)

            # Iterate over the CITYSCAPES_INDEX_TO_COLOR items
            for idx, color in CITYSCAPES_INDEX_TO_COLOR.items():
                mask = (pred_seg_mask == idx).squeeze()  # Squeeze to make it a 2D mask
                colorified_mask[mask] = color  # Assign the color


            
            calib_data = read_calib_file(filename.replace('.bin', '.txt').replace("velodyne","calib"))
            points = load_velodyne_points(filename)
            project_points = points.copy()
            # Transformation matrix is in (3x4) shape, so we extend it to (4x4) by adding a row of [0,0,0,1] for homogeneous coordinates
            Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
            Tr_velo_to_cam_extended = np.eye(4)  # Create a 4x4 identity matrix
            Tr_velo_to_cam_extended[:3, :] = Tr_velo_to_cam  # Replace the top-left 3x4 block

            rect = calib_data['R0_rect'].reshape(3, 3)
            extend_rect = np.eye(4)
            extend_rect[:3,:3] = rect

            P0 = calib_data['P0'].reshape(3, 4)
            P0_extended = np.zeros((4, 4))
            P0_extended[:3, :] = P0
            P0_extended[3, 3] = 1

            points_in_camera = np.matmul(Tr_velo_to_cam_extended, points.T)
            points_in_camera = np.matmul(extend_rect, points_in_camera)
            points_in_camera = np.matmul(P0, points_in_camera).T

            rescale = points_in_camera[:, 2].copy()
            points_in_camera[:, 0] /= points_in_camera[:, 2]
            points_in_camera[:, 1] /= points_in_camera[:, 2]

            image_width = image.shape[1]
            image_height = image.shape[0]
            # Filter out points that are behind the camera or outside the image dimensions
            valid_indices = (points_in_camera[:, 2] > 0) & \
                            (points_in_camera[:, 0] >= 0) & (points_in_camera[:, 0] < image_width) & \
                            (points_in_camera[:, 1] >= 0) & (points_in_camera[:, 1] < image_height)
            print(valid_indices.shape)
            colorified_mask2 = colorified_mask.copy()
            projected_points = points_in_camera[valid_indices, :2]
            org_depths = points[valid_indices, 2]
            for point in projected_points:
                cv2.circle(colorified_mask, (int(point[0]), int(point[1])), radius=1, color=(0, 255, 255), thickness=-1)
            overlayed_image = cv2.addWeighted(image,0.5,colorified_mask,0.5,0)

            # # Display the image with the points
            bol= cv2.imwrite('/mnt/ssd2/Introspect3D/debug/points_projected.png', overlayed_image)
            print("Saved",bol)

            unwanted_points = np.zeros(len(points), dtype=bool)
            filtered_projected_points = []
            index_to_colors = [CITYSCAPES_INDEX_TO_COLOR[idx] for idx in range(9) ]
            
            for i,point in enumerate(projected_points):
                original_idx = np.where(valid_indices)[0][i]  # Index in the original 'points' array
                mask_class = tuple(colorified_mask2[int(point[1]), int(point[0])])

                if mask_class not in index_to_colors:
                    unwanted_points[original_idx] = True
                    filtered_projected_points.append(point)
                    cv2.circle(colorified_mask2, (int(point[0]), int(point[1])), radius=1, color=(0, 255, 255), thickness=-1)
            
            # Now filtered_projected_points contains only the points you want to keep
            # Overlay the modified colorified_mask on the original image
            overlayed_image2 = cv2.addWeighted(image, 0.5, colorified_mask2, 0.5, 0)
            bol = cv2.imwrite('/mnt/ssd2/Introspect3D/debug/points_projected_filtered.png', overlayed_image2)
            print("Saved",bol)
            # sliced_depts=  np.array(filtered_depths[unwanted_points])
            # print("Sliced",sliced_depts.shape)
            # rescaled_filtered_points = np.zeros((len(filtered_projected_points),4))
            # rescaled_filtered_points[:,0] *= np.array(sliced_depts)
            # rescaled_filtered_points[:,1] *= np.array(sliced_depts)
            # rescaled_filtered_points[:,2] = np.array(sliced_depts)
            # rescaled_filtered_points[:,3] = 1
            # # Assuming 'depths' is a 1D array with the depth for each point in 'filtered_projected_points'
            # # And P0_inv is the inverse of the P0 matrix
            # P0_inv = np.linalg.inv(P0_extended)
            # Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam_extended)
            # extend_rect_inv = np.linalg.inv(extend_rect)
            # # Convert 2D points to homogeneous coordinates
            # print("Homogenous Points",rescaled_filtered_points.shape)
            # points_in_camera_3d_fov = rescaled_filtered_points @ P0_inv @ extend_rect_inv @ Tr_velo_to_cam_inv
            # points_in_camera_3d_fov[:,2] = org_depths[valid_indices][unwanted_points]
            # print(points_in_camera_3d_fov.shape)
            points_in_camera_3d_fov = points[unwanted_points]
            points_in_camera_3d_non_fov =points[~valid_indices]
            total_pc = np.vstack([points_in_camera_3d_fov, points_in_camera_3d_non_fov])
            print(total_pc.shape)
            print(points_in_camera_3d_non_fov.shape,points_in_camera_3d_fov.shape)

            # Apply inverse transformations to points in FOV
            
            # Combine both sets of points
            # points_in_original_coords = np.vstack([points_in_camera_3d_fov, points_in_camera_3d_non_fov])
        
            # print(points_in_camera_3d_fov.shape)
            # points_in_camera_3d_non_fov = points[~valid_indices]
            # print(points_in_camera_3d_non_fov.shape)
            # print("---")
            # # Combine both sets of points
            # points_in_camera_3d = np.vstack([points_in_camera_3d_fov, points[~unwanted_points]])
            # # Inverse transformations
            # Tr_velo_to_cam_inv = np.linalg.inv(Tr_velo_to_cam_extended)
            # extend_rect_inv = np.linalg.inv(extend_rect)

            # # Apply inverse transformations to get points in original coordinates
            # points_in_original_coords = np.matmul(Tr_velo_to_cam_inv, np.matmul(extend_rect_inv, points_in_camera_3d.T)).T
            # print(points_in_original_coords.shape)
            max_distance = np.max(np.linalg.norm(total_pc[:, :3], axis=1))
            transated_original = points + np.array([0,-50,0,0])
            # original_pcd = create_point_cloud(points,distance=max_distance)
            filtered_pcd = create_point_cloud(total_pc,distance=max_distance)
            translated_pcd = create_point_cloud(points,distance=max_distance)
            # # filtered_pcd.translate([0,0, 100], relative=False)
            # radian_from_degree = lambda x: x * np.pi / 180
            # R = o3d.geometry.get_rotation_matrix_from_xyz((radian_from_degree(0),radian_from_degree(90),radian_from_degree(0)))
            # filtered_pcd.rotate(R, center=(0,0,0))
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(window_name=str(test_num)) 

            # # vis.add_geometry(original_pcd)
            # vis.add_geometry(filtered_pcd)
            # set_custom_view(vis)
            # # vis.add_geometry(coordinate_frame)
            # vis.run()
            # break
            #Simple transformation to image coordinates
            # y = Tr_velo_to_cam_extended @ points.T
            #y_below_ground_removed = y.T[y.T[:,1] > -1] -> Not sure if this is needed or the axis is correct
            

            
            # #Create a Open3D point cloud from the points for visualization
            # max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))
            # original_pcd = create_point_cloud(points,distance=max_distance)    
            
            
            # nuscenes_compatible_points = np.ones((points.shape[0],5))
            # nuscenes_compatible_points[:,:3] = points[:,:3]
            
            # print("Filtering points")
            # #Define the ellipse parameters and filter the points inside the ellipse, then create a point cloud from the filtered points
            # a, b = 25, 15  # Semi-major and semi-minor axes
            # print("Before filtering",points.shape[0],"points")
            # filtered_points = filter_points_inside_ellipse(points, a, b,offset=-10) #filter_points_inside_pyramid(points, min_x, max_x, min_y,max_y ) #
            # print("After filtering",filtered_points.shape[0],"points")
            # filtered_pcd = create_point_cloud(filtered_points,distance=max_distance)
            # outside_points = filter_points_outside_ellipse(filtered_points, a, b,offset=-10)
            # outside_pcd = create_point_cloud(outside_points)
            
            # #Retrieve the ground truth objects from the label file, then create oriented bounding boxes for visualization
            labels = read_kitti_label_file(filename)
            # filtered_gt_boxes = filter_objects_outside_ellipse(labels, a, b,offset=-10,axis=0)
            # print(len(filtered_gt_boxes),len(labels))
            # gt_oriented_boxes_orj = [create_oriented_bounding_box_gt(label,axis=1,offset=100) for label in labels]
            gt_oriented_boxes = [create_oriented_bounding_box_gt(label,offset=0) for label in labels]
            # # Initialize the object detector
            score_threshold = 0.5 #Threshold for filtering detections with low confidence
            
            # Run inference on the raw point cloud, and create oriented bounding boxes for visualization
            # res,data = inference_detector(model, points)
            # pred_boxes_box = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            # pred_boxes_score = res.pred_instances_3d.scores_3d.cpu().numpy()
            # filtered_indices = np.where(pred_boxes_score >= score_threshold)[0]
            # filtered_boxes = pred_boxes_box[filtered_indicesdev2.py]
            # obb_list = [create_oriented_bounding_box(box,offset=100,axis=1,calib=calib_data) for box in filtered_boxes]
            print("Running inference")
            flag = 1
            
            # Run inference on the filtered point cloud, and create oriented bounding boxes for visualization, filter detections with low confidence
            # print(filtered_points.shape)
            print(total_pc.shape)
            register_all_modules()
            res_f,data_f = inference_detector(model, total_pc)
            pred_boxes_box_f = res_f.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            pred_boxes_score_f = res_f.pred_instances_3d.scores_3d.cpu().numpy()
            print(pred_boxes_score_f)
            filtered_indices = np.where(pred_boxes_score_f >= score_threshold)[0]
            filtered_boxes = pred_boxes_box_f[filtered_indices]
            obb_list_f = [create_oriented_bounding_box(box,offset=0,calib=calib_data) for box in filtered_boxes]
            # # print("Object shapes",len(obb_list_f),"Prediction shape", filtered_boxes.shape,filtered_boxes[0])
            # #Remove points from the filtered point cloud that are inside the predicted bounding boxes
            # removed_object_points = remove_points_inside_obbs(filtered_points, obb_list_f)
            # # removed_pcd = create_point_cloud(removed_object_points,distance=max_distance)
            print("Running inference on the removed points")

            # flag = 0
            one_more_chance_res, one_more_chance_data = inference_detector(model, points)
            one_more_chance_pred_boxes_box = one_more_chance_res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            one_more_chance_pred_boxes_score = one_more_chance_res.pred_instances_3d.scores_3d.cpu().numpy()
            filtered_indices = np.where(one_more_chance_pred_boxes_score >= score_threshold)[0]
            one_more_chance_filtered_boxes = one_more_chance_pred_boxes_box[filtered_indices]
            one_more_chance_obb_list_f = [create_oriented_bounding_box(box,offset=-50,axis=1,calib=calib_data,color=(1,0,0)) for box in one_more_chance_filtered_boxes]
            
            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

            

            # #Visualize the results
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=str(test_num)) 
            # original_pcd.translate([0,50, 0], relative=False)
            translated_pcd.translate([0,-50, 0], relative=False)
            vis.add_geometry(filtered_pcd)
            # vis.add_geometry(original_pcd)
            vis.add_geometry(translated_pcd)
            # #
            # # original_pcd.translate([0,100, 0], relative=False)
            # vis.add_geometry(outside_pcd)
            # vis.add_geometry(filtered_pcd)
            # # vis.add_geometry(removed_pcd)
            
            # render_option = vis.get_render_option()
            # render_option.point_size = 2.0
            # # print("Difference between orj and regular gt boxes",len(gt_oriented_boxes_orj),len(gt_oriented_boxes))
            # for gt_obb in gt_oriented_boxes_orj:
            #     vis.add_geometry(gt_obb)
            for gt_obb in gt_oriented_boxes:
                vis.add_geometry(gt_obb)
            for obb in obb_list_f:
                vis.add_geometry(obb) 
            for obb in one_more_chance_obb_list_f:
                vis.add_geometry(obb)
            set_custom_view(vis)
            # vis.add_geometry(coordinate_frame)
            vis.run()

            # # Close the visualizer window
            # vis.destroy_window()
            # # only_gt_boxes = [obb for obb,cntr in gt_oriented_boxes]
            # detected_objects, missed_gt, not_matched_predictions = match_detections_3d(gt_oriented_boxes, obb_list_f)
            # if(len(gt_oriented_boxes) > 0):
            #     row = {'image_path':f"{test_num:06}.png",'is_missed':len(missed_gt) > 0,'missed_objects':len(missed_gt),'total_objects':len(gt_oriented_boxes)}
            #     new_dataset = pd.concat([new_dataset,pd.DataFrame([row])])
            
            # print("Detected objects",len(detected_objects),"Missed gt",len(missed_gt),"Not matched predictions",len(not_matched_predictions))
            # break
            # This is for the custom dataset creation
            # if(len(missed_gt) > 0 and len(detected_objects) > 0):
            #     custom_dataset_label_str = ""
            #     for gt in missed_gt:
            #         # custom_dataset_label_str += f"{gt['type']} "
            #         box_3d_corners = np.array(gt.points)
            #         for corner in box_3d_corners:
            #             custom_dataset_label_str += f"{corner[0]} {corner[1]} {corner[2]} "
            #         custom_dataset_label_str += "\n"
            #     with open(os.path.join(save_path,f"{test_num:06}.txt"), "w") as f:
            #         f.write(custom_dataset_label_str)
            # pbar.update(1)
            # break
    # hook.remove()
    # new_dataset.to_csv(f"./custom_dataset/{used_model}_{training_set}_class{str(num_classes)}_dataset.csv",index=False)


# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import os
# from glob import glob
# import pandas as pd
# from tqdm.auto import tqdm
# import torch
# from definitions import ROOT_DIR
# import open3d as o3d
# counterg = 0
# class ExtremelySimpleActivationShaping():
#     def __init__(self, config):
#         self.config = config
#     def process(self, **kwargs):
#         activation = kwargs.get('activation')
#         percentile = self.config['percentile']
#         method = self.config['method']
#         if isinstance(activation,np.ndarray):
#             activation = torch.from_numpy(activation)
#         result = eval(f"self.ash_{method}(activation,percentile)")
#         return result.detach().cpu().numpy()
    
#     def ash_b(self,x, percentile=65):
#         assert x.dim() == 4
#         assert 0 <= percentile <= 100
#         b, c, h, w = x.shape

#         # calculate the sum of the input per sample
#         s1 = x.sum(dim=[1, 2, 3])

#         n = x.shape[1:].numel()
#         k = n - int(np.round(n * percentile / 100.0))
#         t = x.view((b, c * h * w))
#         v, i = torch.topk(t, k, dim=1)
#         fill = s1 / k
#         fill = fill.unsqueeze(dim=1).expand(v.shape)
#         t.zero_().scatter_(dim=1, index=i, src=fill)
#         return x

#     def ash_p(self,x, percentile=65):

#         assert x.dim() == 4
#         assert 0 <= percentile <= 100
#         # print(percentile)
#         original_x = x.detach().clone()
#         # normalize original_x
#         # original_x = original_x / original_x.sum(dim=[1, 2, 3], keepdim=True)
#         #plot histogram of normlized
#         figure = plt.figure()
#         #subplot with 1 row and 2 columns
#         figure.add_subplot(1,2,1)
#         hist, edges = np.histogram(original_x.flatten(),bins=10)
#         print(hist,edges)
#         # plt.bar(range(len(hist)),hist)
#         # plt.title("Org")
#         #put into first plot
#         global counterg

#         b, c, h, w = x.shape
#         #Plot distribution of values of x 
#         percentile_value = np.percentile(original_x.flatten(), 90)  # Replace 65 with your desired percentile
#         # plt.axvline(percentile_value, color='r', linestyle='dashed', linewidth=2)
#         n = x.shape[1:].numel()
#         k = n - int(np.round(n * percentile / 100.0))
#         t = x.view((b, c * h * w))
#         v, i = torch.topk(t, k, dim=1)
#         t.zero_().scatter_(dim=1, index=i, src=v)
#         #normalize x to other variable
#         x_test = x / x.sum(dim=[1, 2, 3], keepdim=True)
#         #plot histogram of normlized
#         figure.add_subplot(1,2,2)
#         hist = np.histogram(x.flatten(),bins=25)[0]
#         # plt.bar(range(len(hist)),hist)
#         # plt.title("Ash")
#         #save the figure
#         # plt.savefig(os.path.join(ROOT_DIR,'debug','histograms',f"ash_{counterg}.png"))
#         # plt.clf()
#         counterg += 1
#         return x

#     def ash_s(self,x, percentile=65):
#         assert x.dim() == 4
#         assert 0 <= percentile <= 100
#         b, c, h, w = x.shape
#         original_x = x.clone()
#         # calculate the sum of the input per sample
#         s1 = x.sum(dim=[1, 2, 3])
#         n = x.shape[1:].numel()
#         k = n - int(np.round(n * percentile / 100.0))
#         t = x.view((b, c * h * w))
#         v, i = torch.topk(t, k, dim=1)
#         t.zero_().scatter_(dim=1, index=i, src=v)

#         # calculate new sum of the input per sample after pruning
#         s2 = x.sum(dim=[1, 2, 3])

#         # apply sharpening
#         scale = s1 / s2
#         x = x * torch.exp(scale[:, None, None, None])

#         return x
# class StatisticalFeatureExtraction():
#     def __init__(self, config):
#         self.config = config
#     def process(self, **kwargs):
#         processed_activation_list= []
#         activation = kwargs.get('activation')
#         functions = self.config['functions']
#         #if the activation is a tensor, convert it to numpy
#         if isinstance(activation,torch.Tensor):
#             activation = activation.detach().cpu().numpy()
#         #if the activation is a list, convert it to numpy, may be not needed
#         if isinstance(activation,list):
#             activation = np.array(activation)
#         #if the activation is a numpy array, process it
#         #Assumption is that channel is 0th index, which makes the operation global pooling
#         for function in functions:
#             processed_activation = eval(f"np.{function}(activation,axis=(2,3),keepdims=True)")
#             processed_activation_list.append(processed_activation.squeeze((2,3)))
#             # print(processed_activation.shape)
#         return np.concatenate(processed_activation_list,axis=1)
# class VisualDNA():
#     def __init__(self, config):
#         self.config = config
#     def process(self, **kwargs):
#         processed_activation_list= []
#         activation = kwargs.get('activation')
#         bins = self.config['bins']
#         #if the activation is a tensor, convert it to numpy
#         if isinstance(activation,torch.Tensor):
#             activation = activation.detach().cpu().numpy()
#         #if the activation is a list, convert it to numpy, may be not needed
#         if isinstance(activation,list):
#             activation = np.array(activation)
#         #if the activation is a numpy array, process it
#         #Assumption is that channel is 0th index, which makes the operation global pooling
#         base_histogram = np.zeros((activation.shape[0],bins))
#         for i in range(activation.shape[0]):
#             for j in range(activation.shape[1]):
#                 act_map = activation[i,j,:,:]
#                 histogram = np.histogram(act_map,bins=bins)[0]
#                 base_histogram[i,:] += histogram
#         # print(base_histogram.shape)
#         return base_histogram
# def mapper_for_labels(labels):
#     bins = [0, 0.25, 0.5, 0.75, 1]

#     # Digitize the values
#     # The right parameter is set to False to make the intervals open on the right, i.e., (0, 0.25], (0.25, 0.5], etc.
#     labels = np.digitize(labels, bins, right=False)

#     print(np.unique(labels,return_counts=True))

#     return labels
# from definitions import *
# file_path = r"/mnt/ssd2/custom_dataset/kitti_pointpillars_activations_raw/"
# file_names = glob(os.path.join(file_path,'features','*.npy'))
# files = []
# params=  {"percentile": 95, "method": "p"}
# # params = {"functions":['mean','std','max']}
# # params = {"bins":2000}
# processor = ExtremelySimpleActivationShaping(params)#VisualDNA(params)#StatisticalFeatureExtraction(config=params)#
# dist_diff = []
# # o3d.visualization.webrtc_server.enable_webrtc()
# with tqdm(range(len(file_names))) as pbar:
#     for i,file in enumerate(file_names):
#         if i <100:
#             act = np.load(file)
#             #get base name for the file
#             base_name = os.path.basename(file)
#             #remove extention
#             base_name = os.path.splitext(base_name)[0]
#             pcd_path = os.path.join(ROOT_DIR,"..","kitti",'training','velodyne',f"{base_name}.bin")
#             loaded_pointcloud = np.fromfile(pcd_path, dtype=np.float32, count=-1).reshape([-1, 4])
#             # pcd = o3d.geometry.PointCloud()
#             # pcd.points = o3d.utility.Vector3dVector(loaded_pointcloud[:, :3])  # X, Y, Z

#             # # Normalize intensity data and set it as colors if needed
#             # intensities = loaded_pointcloud[:, 3]  # Assuming the 4th column is intensity
#             # max_intensity = np.max(intensities)
#             # colors = plt.get_cmap("viridis")(intensities / max_intensity)[:, :3]  # Normalize and apply colormap
#             # pcd.colors = o3d.utility.Vector3dVector(colors)
#             # vis = o3d.visualization.Visualizer()
#             # vis.create_window(visible=False)  # We don't want to open a window hence visible=False
#             # vis.add_geometry(pcd)
#             # vis.update_geometry(pcd)
#             # vis.poll_events()
#             # vis.update_renderer()

#             # Capture the image and save
#             # image = vis.capture_screen_float_buffer(do_render=True)
#             # plt.imshow(np.asarray(image))
#             # plt.axis('off')
#             image_path = os.path.join(ROOT_DIR,'debug','pointclouds',f"output_image_{i}.png")
#             # plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
#             # o3d.io.write_image(, image)
#             # vis.destroy_window()
#             ash = processor.process(activation=act[None,:,:,:]).squeeze(0)
#             #Get channelwise mean of the activation
#             # print(act.shape)
#             cam  = np.mean(act,axis=(0))
#             # print(cam.shape)
#             #save this to debug/activations folder as a grayscale image
#             # plt.imsave(os.path.join(ROOT_DIR,"debug","activations",f"acti_{i}.png"),cam,cmap='gray')
#             # plt.clf()
#             pbar.update(1)
        #absolute difference between ash and raw val
#         print(ash.shape,act.shape)
#         diff = np.abs(act - ash)
#         print(np.sum(diff.flatten()))
#         exit()

#         dist_diff.append(np.sum(diff.flatten()))
#         # files.append(act)
#         pbar.update(1)
# #write the values to file
# with open(os.path.join(file_path,"dist_diff.txt"),'w') as f:
#     for item in dist_diff:
#         f.write("%s\n" % item)
# hist = np.histogram(dist_diff,bins=10)[0]
# #Plot histogram
# plt.bar(range(len(hist)),hist)
# plt.show()
# label_csv= pd.read_csv(os.path.join(file_path,"kitti_point_pillars_labels_raw.csv"))
# # label_csv['missed_ratio'] = label_csv['missed_objects']/label_csv['total_objects']
# # labels = list(map(int,list(label_csv['missed_ratio'] > 0.5)))
# labels = list(mapper_for_labels(label_csv['is_missed'].values))
# # Let's assume 'activations' is your array of activation maps with shape (num_samples, 256, 50, 50)
# # You need to reshape it into (num_samples, 256*50*50)
# activations = np.array(files)
# num_samples = len(activations)  # replace with your actual number of samples
# activations_flattened = activations.reshape(num_samples, -1)

# # Standardize the features (if needed)
# # from sklearn.preprocessing import StandardScaler
# # scaler = StandardScaler()
# # activations_standardized = scaler.fit_transform(activations_flattened)

# # Use t-SNE to reduce dimensionality
# tsne = TSNE(n_components=4, verbose=1, perplexity=40, n_iter_without_progress=300)
# tsne_results = tsne.fit_transform(activations_flattened)

# # Visualize the results
# plt.figure(figsize=(16,10))
# plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='viridis')
# plt.colorbar()  # if 'your_labels' is provided, otherwise omit this line
# plt.xlabel('t-SNE component 1')
# plt.ylabel('t-SNE component 2')
# plt.savefig('dna_multi_label.png')