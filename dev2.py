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
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
print("Open3D Version: " + o3d.__version__)
# o3d.visualization.webrtc_server.enable_webrtc()
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
        coners[:,axis] += offset
        return plot_bounding_box_from_corners(coners,offset=offset, calib=calib, color=color)
        # R_empty = np.eye(3)
        # obb = o3d.geometry.OrientedBoundingBox(center=real_center, R=R, extent=extent)
        # cntr = o3d.geometry.OrientedBoundingBox(center=real_center, R=R, extent=[0.1,0.1,0.1])
        # cntr.color = (1,1,1)
        # obb.color = color
        # return obb, cntr
    # Rotate around the Y-axis for KITTI's rotation_y
    print("Plotting from center")
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
            print(translated_corners.shape)
            # real_center = np.mean(translated_corners,axis=1)
            # print(real_center)
            objects.append({
                'type': obj_type,
                'x': center_tr[0],
                'y': center_tr[1],
                'z': center_tr[2],
                'dx': dimensions_length,
                'dy': dimensions_height,
                'dz': dimensions_width,
                'yaw': o3d.geometry.get_rotation_matrix_from_xyz((0,rotation_y,0)),
                'corners': translated_corners.T,
                'real_center': np.mean(translated_corners,axis=1)
            })
    return objects
def create_oriented_bounding_box(box_params,offset=100,axis=0,calib=None):
    offset_array = np.zeros(3)
    offset_array[axis] = offset
    center = np.array([box_params[0], box_params[1], box_params[2]/2])
    extent = np.array([box_params[3], box_params[4], box_params[5]])
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
        print("For gt",gt,"max iou is",max_iou)
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
        print("Getting corners from lineset")

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
if __name__ == '__main__':
    import random
    #Define paths
    kitti_velodyne_path= r"D:\introspectionBase\datasets\Kitti\raw\training\velodyne"
    img_path = r"D:\introspectionBase\datasets\Kitti\raw\training\image_2"
    config_file = r'D:\mmdetection3d\configs\pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
    checkpoint = r'D:\mmdetection3d/ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
    test_num  = random.randint(0,7480) #Selecting a random data point
    filename = os.path.join(kitti_velodyne_path, f'{test_num:06}.bin')
    
    #Read Calibration file, load point cloud and transfrom to image coordinates
    calib_data = read_calib_file(filename.replace('.bin', '.txt').replace("velodyne","calib"))
    points = load_velodyne_points(filename)
    
    # Transformation matrix is in (3x4) shape, so we extend it to (4x4) by adding a row of [0,0,0,1] for homogeneous coordinates
    Tr_velo_to_cam = calib_data['Tr_velo_to_cam'].reshape(3, 4)
    Tr_velo_to_cam_extended = np.eye(4)  # Create a 4x4 identity matrix
    Tr_velo_to_cam_extended[:3, :] = Tr_velo_to_cam  # Replace the top-left 3x4 block
    
    #Simple transformation to image coordinates
    # y = Tr_velo_to_cam_extended @ points.T
    #y_below_ground_removed = y.T[y.T[:,1] > -1] -> Not sure if this is needed or the axis is correct
    

    
    #Create a Open3D point cloud from the points for visualization
    max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))
    original_pcd = create_point_cloud(points,distance=max_distance)    
    
  
    
    
    #Define the ellipse parameters and filter the points inside the ellipse, then create a point cloud from the filtered points
    a, b = 25, 15  # Semi-major and semi-minor axes
    filtered_points = filter_points_inside_ellipse(points, a, b,offset=-10) #filter_points_inside_pyramid(points, min_x, max_x, min_y,max_y ) #
    filtered_pcd = create_point_cloud(filtered_points,distance=max_distance)
    outside_points = filter_points_outside_ellipse(points, a, b,offset=-10)
    outside_pcd = create_point_cloud(outside_points)
    
    #Retrieve the ground truth objects from the label file, then create oriented bounding boxes for visualization
    labels = read_kitti_label_file(filename)
    filtered_gt_boxes = filter_objects_outside_ellipse(labels, a, b,offset=-10,axis=0)
    print(len(filtered_gt_boxes),len(labels))
    # gt_oriented_boxes_orj = [create_oriented_bounding_box_gt(label,axis=1,offset=100) for label in labels]
    gt_oriented_boxes = [create_oriented_bounding_box_gt(label,offset=0) for label in filtered_gt_boxes]
    # Initialize the object detector
    model = init_model(config_file, checkpoint, device='cuda:0')
    score_threshold = 0.5 #Threshold for filtering detections with low confidence
    
    # Run inference on the raw point cloud, and create oriented bounding boxes for visualization
    # res,data = inference_detector(model, points)
    # pred_boxes_box = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    # pred_boxes_score = res.pred_instances_3d.scores_3d.cpu().numpy()
    # filtered_indices = np.where(pred_boxes_score >= score_threshold)[0]
    # filtered_boxes = pred_boxes_box[filtered_indicesdev2.py]
    # obb_list = [create_oriented_bounding_box(box,offset=100,axis=1,calib=calib_data) for box in filtered_boxes]
    
    # Run inference on the filtered point cloud, and create oriented bounding boxes for visualization, filter detections with low confidence
    res_f,data_f = inference_detector(model, filtered_points)
    pred_boxes_box_f = res_f.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    pred_boxes_score_f = res_f.pred_instances_3d.scores_3d.cpu().numpy()
    filtered_indices = np.where(pred_boxes_score_f >= score_threshold)[0]
    filtered_boxes = pred_boxes_box_f[filtered_indices]
    obb_list_f = [create_oriented_bounding_box(box,offset=0,calib=calib_data) for box in filtered_boxes]
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    # Visualize the results
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(test_num)) 
    #
    # original_pcd.translate([0,100, 0], relative=False)
    vis.add_geometry(outside_pcd)
    vis.add_geometry(filtered_pcd)
    
    render_option = vis.get_render_option()
    render_option.point_size = 1.5
    # print("Difference between orj and regular gt boxes",len(gt_oriented_boxes_orj),len(gt_oriented_boxes))
    # for gt_obb in gt_oriented_boxes_orj:
    #     vis.add_geometry(gt_obb)
    for gt_obb in gt_oriented_boxes:
        vis.add_geometry(gt_obb)
    for obb in obb_list_f:
        vis.add_geometry(obb)
    # for obb in obb_list:
    #     vis.add_geometry(obb)
    set_custom_view(vis)
    vis.add_geometry(coordinate_frame)
    vis.run()

    # Close the visualizer window
    vis.destroy_window()
    # only_gt_boxes = [obb for obb,cntr in gt_oriented_boxes]
    detected_objects, missed_gt, not_matched_predictions = match_detections_3d(gt_oriented_boxes, obb_list_f)
    print(len(detected_objects),len(missed_gt),len(not_matched_predictions))