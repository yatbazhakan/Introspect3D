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
kitti_velodyne_path= r"D:\introspectionBase\datasets\Kitti\raw\training\velodyne"
def set_custom_view(vis):
    ctr = vis.get_view_control()
    ctr.translate(0, -50, 0)
    ctr.set_constant_z_far(1000)
    ctr.set_constant_z_near(0.1)
    ctr.set_front([0, 0, 1])
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.5)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.5, 0.5, 0.5])

def rotate_point_around_y(point, theta):
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    rotated_point = np.dot(R, point)
    
    return rotated_point
def get_yaw_from_rotation_matrix(R):
    """
    Extract the yaw angle from a 3x3 rotation matrix.
    
    Parameters:
        R (numpy.ndarray): 3x3 rotation matrix.
        
    Returns:
        float: Yaw angle in radians.
    """
    # Convert rotation matrix to Euler angles
    yaw_angle = np.arctan2(R[1, 0], R[0, 0])
    
    return yaw_angle
def project_to_axis(points, axis):
    return np.dot(points, axis)

def overlap_on_axis(points1, points2, axis):
    min1 = np.min(project_to_axis(points1, axis))
    max1 = np.max(project_to_axis(points1, axis))
    min2 = np.min(project_to_axis(points2, axis))
    max2 = np.max(project_to_axis(points2, axis))
    overlap = max(0, min(max1, max2) - max(min1, min2))
    return overlap
def create_oriented_box(center, dimensions, yaw):
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=dimensions)

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
            rotation_y = float(tokens[14])
            l_div_2 = dimensions_length / 2
            x_corners = [l_div_2, l_div_2, -l_div_2, -l_div_2, l_div_2, l_div_2, -l_div_2, -l_div_2]
            w_div_2 = dimensions_width / 2
            y_corners = [0, 0, 0, 0, -dimensions_height, -dimensions_height, -dimensions_height, -dimensions_height]
            z_corners = [w_div_2, -w_div_2, -w_div_2, w_div_2, w_div_2, -w_div_2, -w_div_2, w_div_2]
            corner_matrix = np.array([x_corners, y_corners, z_corners])
            rotated_corners = rotate_point_around_y(corner_matrix, rotation_y)
            translated_corners = rotated_corners + np.array([location_x, location_y, location_z]).reshape((3, 1))
            translated_center = np.mean(translated_corners, axis=1)
            # Adjust location_y to get the center of the bounding box
            location_y += dimensions_height / 2.0

            Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
            # R = Tr_velo_to_cam[:3, :3]
            # R_inv = np.linalg.inv(R)
            # R_cam_bb = o3d.geometry.get_rotation_matrix_from_xyz((0, rotation_y, 0))
            # R_cam_vel = R_cam_bb @  R_inv
            
            # center = np.array([location_x, location_y, location_z])
            # Convert center to homogeneous coordinates and transform to LiDAR coordinates
            point_homogeneous = np.hstack((translated_center, np.ones(1)))
            center_tr = transform_camera_to_lidar(point_homogeneous, Tr_velo_to_cam)
            objects.append({
                'type': obj_type,
                'corners': translated_corners,
            })
            # objects.append({
            #     'type': obj_type,
            #     'x': center_tr[0],
            #     'y': center_tr[1],
            #     'z': center_tr[2],
            #     'dx': dimensions_length,
            #     'dy': dimensions_height,
            #     'dz': dimensions_width,
            #     'yaw': o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0)),
            # })
    return objects
def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    return points
def compute_colors_from_distance_bw(points):
    distances = np.linalg.norm(points[:, :3], axis=1)
    max_distance = np.max(distances)
    colors = distances / max_distance
    return np.c_[colors, colors, colors]
def compute_colors_from_distance(points,max_distance):
    distances = np.linalg.norm(points[:, :3], axis=1)
    normalized_distances = distances / max_distance
    return plt.cm.jet(normalized_distances)[:,:3]
def filter_points_inside_ellipse(points, a, b,offset=5):
    x, y, z = points[:, 0]+offset, points[:, 1], points[:, 2]
    inside = (x**2 / a**2) + (y**2 / b**2) <= 1
    return points[inside]
def create_point_cloud(points,distance=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    colors = compute_colors_from_distance(points,distance)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
def visualize(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
def create_oriented_bounding_box(box_params,offset=100):
    center = np.array([box_params[0], box_params[1]+offset, box_params[2]/2])
    extent = np.array([box_params[3], box_params[4], box_params[5]])
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, box_params[6]))
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    obb.color = (1, 0, 0)  # Red color
    return obb
def transform_camera_to_lidar(point, Tr_velo_to_cam):
    # Inverse the transformation matrix
    R = Tr_velo_to_cam[:3, :3]
    translation = Tr_velo_to_cam[:3, 3]
    R_inv = np.linalg.inv(R)
    translation_inv = -translation
    Tr_cam_to_velo = np.hstack((R_inv, translation_inv.reshape(-1, 1)))
    Tr_cam_to_velo = np.vstack((Tr_cam_to_velo, np.array([[0, 0, 0, 1]])))
    #Inverse the transformation matrix
    # Tr_velo_to_cam = Tr_velo_to_cam.reshape(3, 4)
    # Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, np.array([[0, 0, 0, 1]])))
    # Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    # Perform the transformation
    point_lidar = point @ Tr_cam_to_velo.T
    
    return point_lidar
#Some generic implementation needed this has become ellipse specific
def create_oriented_bounding_box_gt(box_params, color=(0, 1, 0), calib=None):
    center = np.array([box_params['x'], box_params['y'], box_params['z']/2])
    extent = np.array([box_params['dx'], box_params['dy'], box_params['dz']])
    # Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    # # Convert center to homogeneous coordinates and transform to LiDAR coordinates
    # point_homogeneous = np.hstack((center, np.ones(1)))
    # center_tr = transform_camera_to_lidar(point_homogeneous, Tr_velo_to_cam)
    
    # Rotate around the Y-axis for KITTI's rotation_y
    R = box_params['yaw']

    # Create OrientedBoundingBox
    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    obb.color = color 
    
    return obb
# def create_oriented_bounding_box_gt(box_params, color=(0, 1, 0),calib=None):
#     center = np.array([box_params['x'], box_params['y'], box_params['z']])
#     extent = np.array([box_params['dx'], box_params['dy'], box_params['dz']])
#     Tr_velo_to_cam = calib['Tr_velo_to_cam'].reshape(3, 4)
    
#     point_homogeneous = np.hstack((center, np.ones(1)))
#     print(point_homogeneous.shape,Tr_velo_to_cam.shape)
#     center_tr = transform_camera_to_lidar(point_homogeneous, Tr_velo_to_cam)
#     # Rotate around the Y-axis for KITTI's rotation_y
#     # Explicitly set the rotation matrix for Y-axis rotation
#     R = o3d.geometry.get_rotation_matrix_from_xyz((0,box_params['yaw'],  0))

#     obb = o3d.geometry.OrientedBoundingBox(center=center_tr, R=R, extent=extent)
#     obb.color = color  # Set color
#     return obb
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
def transform_lidar_to_camera(points, Tr_velo_to_cam):
    # Add a fourth dimension for homogeneous coordinate transformation
    points_cam = np.dot(points, np.transpose(Tr_velo_to_cam))
    return points_cam[:, :3]
def filter_cone(points, min_angle, max_angle, min_radius, max_radius):
    """
    Filters a point cloud to only include points within a specified cone.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        min_angle (float): The minimum azimuth angle in radians.
        max_angle (float): The maximum azimuth angle in radians.
        min_radius (float): The minimum radial distance from the origin.
        max_radius (float): The maximum radial distance from the origin.

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud.
    """
    # Calculate azimuth angle and radial distance
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    radius = np.linalg.norm(points[:, :2], axis=1)
    
    # Apply the cone filter
    mask = (azimuth >= min_angle) & (azimuth <= max_angle) & (radius >= min_radius) & (radius <= max_radius)
    filtered_points = points[mask]
    
    # Create a new point cloud for the filtered points
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    
    return filtered_cloud
def filter_points_inside_pyramid(points, z_min, z_max, x_max, y_max):
    """
    Filters points to only include those inside a specified pyramid.

    Parameters:
        points (numpy.ndarray): The input points (N, 3).
        z_min (float): The minimum z-coordinate for the pyramid base.
        z_max (float): The maximum z-coordinate for the pyramid tip.
        x_max (float): The maximum x-coordinate for the pyramid base.
        y_max (float): The maximum y-coordinate for the pyramid base.

    Returns:
        numpy.ndarray: The filtered points (N, 3).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Normalize x and y based on z and pyramid dimensions
    z_range = z_max - z_min
    norm_x = x / (z - z_min) * z_range / x_max
    norm_y = y / (z - z_min) * z_range / y_max
    
    # Check if points are inside the pyramid and within the z-range
    inside = (np.abs(norm_x) <= 1) & (np.abs(norm_y) <= 1) & (z >= z_min) & (z <= z_max)
    
    return points[inside]
def plot_bounding_box_from_corners(label, color=[0, 1, 0]):
    # Define the lines connecting the corners based on their indices
    corners = label['corners']
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Create a LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners.T)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])

    return line_set
# def filter_points_inside_pyramid(points, min_x, max_x, min_y, max_y):
#     """
#     Filters points to only include those inside a specified pyramid.

#     Parameters:
#         points (numpy.ndarray): The input points (N, 3).
#         min_x (float): The minimum x-coordinate for the pyramid base.
#         max_x (float): The maximum x-coordinate for the pyramid base.
#         min_y (float): The minimum y-coordinate for the pyramid base.
#         max_y (float): The maximum y-coordinate for the pyramid base.

#     Returns:
#         numpy.ndarray: The filtered points (N, 3).
#     """
#     x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
#     # Normalize x and y based on z and pyramid dimensions
#     norm_x = x / (z * (max_x - min_x) + min_x)
#     norm_y = y / (z * (max_y - min_y) + min_y)
    
#     # Check if points are inside the pyramid
#     inside = (np.abs(norm_x) <= 1) & (np.abs(norm_y) <= 1)
    
#     return points[inside]
def filter_objects_outside_ellipse(objects, a, b,offset=5):
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
        print(obj)
        if 'corners' not in obj.keys():
            x, y = obj['x'], obj['y']
            
            # Generate corner points for the box
            dx, dy = obj['dx'], obj['dy']
            corners = np.array([
                [x - dx/2, y - dy/2],
                [x - dx/2, y + dy/2],
                [x + dx/2, y - dy/2],
                [x + dx/2, y + dy/2]
            ])
            adjusted_corners_x = corners[:, 0] + offset
        else:
            print("corners",obj['corners'].shape)
            corners = obj['corners'].T
            adjusted_corners_x = corners[:, 0] + offset
        # Check if any corner point is inside the ellipse
        inside_ellipse = (adjusted_corners_x**2 / a**2) + (corners[:, 1]**2 / b**2) <= 1
        if np.any(inside_ellipse):
            filtered_objects.append(obj)
    
    return filtered_objects
def rotate_point(x, y, yaw):
    x_rot = x * np.cos(yaw) - y * np.sin(yaw)
    y_rot = x * np.sin(yaw) + y * np.cos(yaw)
    return x_rot, y_rot

def box_corners(center, dimensions, yaw):
    dx, dy, dz = dimensions
    x, y, z = center
    half_dx, half_dy = dx / 2, dy / 2
    
    # Define box corners relative to the center
    corners = np.array([
        [-half_dx, -half_dy],
        [+half_dx, -half_dy],
        [+half_dx, +half_dy],
        [-half_dx, +half_dy]
    ])
    
    # Rotate and translate corners
    corners_rotated = np.array([rotate_point(x, y, yaw) for x, y in corners])
    corners_rotated += [x, y]
    
    return corners_rotated

def iou_3d(box1, box2):
    center1, dimensions1, yaw1 = box1['center'], box1['dimensions'], box1['yaw']
    center2, dimensions2, yaw2 = box2['center'], box2['dimensions'], box2['yaw']
    
    corners1 = box_corners(center1, dimensions1, yaw1)
    corners2 = box_corners(center2, dimensions2, yaw2)
    
    # Compute intersection volume
    x_min_int = max(np.min(corners1[:, 0]), np.min(corners2[:, 0]))
    x_max_int = min(np.max(corners1[:, 0]), np.max(corners2[:, 0]))
    y_min_int = max(np.min(corners1[:, 1]), np.min(corners2[:, 1]))
    y_max_int = min(np.max(corners1[:, 1]), np.max(corners2[:, 1]))
    z_min_int = max(center1[2] - dimensions1[2] / 2, center2[2] - dimensions2[2] / 2)
    z_max_int = min(center1[2] + dimensions1[2] / 2, center2[2] + dimensions2[2] / 2)
    
    if x_max_int < x_min_int or y_max_int < y_min_int or z_max_int < z_min_int:
        intersection_volume = 0
    else:
        intersection_volume = (x_max_int - x_min_int) * (y_max_int - y_min_int) * (z_max_int - z_min_int)
    
    # Compute union volume
    volume1 = dimensions1[0] * dimensions1[1] * dimensions1[2]
    volume2 = dimensions2[0] * dimensions2[1] * dimensions2[2]
    union_volume = volume1 + volume2 - intersection_volume
    
    # Compute IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0
    
    return iou

def rotate_points(points, yaw):
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return np.dot(points, R.T)

def calculate_iou_3d(box1,box2):
    # Create corner points for both boxes
    center1 = np.array(box1.center)
    center2 = np.array(box2.center)
    dimensions1 = np.array(box1.extent)
    dimensions2 = np.array(box2.extent)
    yaw1 = get_yaw_from_rotation_matrix(box1.R)
    yaw2 = get_yaw_from_rotation_matrix(box2.R)
    half_dims1 = dimensions1 / 2
    half_dims2 = dimensions2 / 2
    corners1 = np.array([np.array([x, y, z]) for x in [-half_dims1[0], half_dims1[0]] for y in [-half_dims1[1], half_dims1[1]] for z in [-half_dims1[2], half_dims1[2]]])
    corners2 = np.array([np.array([x, y, z]) for x in [-half_dims2[0], half_dims2[0]] for y in [-half_dims2[1], half_dims2[1]] for z in [-half_dims2[2], half_dims2[2]]])
    
    # Rotate and translate corners
    corners1 = rotate_points(corners1, yaw1) + center1
    corners2 = rotate_points(corners2, yaw2) + center2
    
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

        if max_iou >= iou_threshold:
            matches.append((gt, unmatched_predictions[max_iou_idx]))
            del unmatched_predictions[max_iou_idx]
        else:
            unmatched_ground_truths.append(gt)

    return matches, unmatched_ground_truths, unmatched_predictions

if __name__ == '__main__':
    import random
    config_file = r'D:\mmdetection3d\configs\pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'
    # download the checkpoint from model zoo and put it in `checkpoints/`
    checkpoint = r'D:\mmdetection3d/ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
    test_num  = 1220#random.randint(0,7480)
    filename = os.path.join(kitti_velodyne_path, f'{test_num:06}.bin')
    calib_data = read_calib_file(filename.replace('.bin', '.txt').replace("velodyne","calib"))
    P_rect = calib_data['P2'].reshape(3, 4)
    points = load_velodyne_points(filename)
    max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))
    score_threshold = 0.5

    # Define cone parameters
    # min_angle = -np.pi / 6  # -30 degrees
    # max_angle = np.pi / 6   # 30 degrees
    # min_radius = 1.0
    # max_radius = 50.0
    min_x = 0.5
    max_x = 100.0
    min_y = -20.0
    max_y = 20.0
    a, b = 20, 20  # Semi-major and semi-minor axes
    filtered_points = filter_points_inside_ellipse(points, a, b,offset=-10) #filter_points_inside_pyramid(points, min_x, max_x, min_y,max_y ) #
    original_pcd = create_point_cloud(points,distance=max_distance)
    filtered_pcd = create_point_cloud(filtered_points,distance=max_distance)
    
    # Visualize the original point cloud
    model = init_model(config_file, checkpoint, device='cuda:0')
    #Visualize both point clouds
    # o3d.visualization.draw_geometries([original_pcd, filtered_pcd])
    # res,data = inference_detector(model, points)
    # pred_boxes_box = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    # pred_boxes_score = res.pred_instances_3d.scores_3d.cpu().numpy()
    # obb_list = [create_oriented_bounding_box(box) for box in pred_boxes_box]
    
    res_f,data_f = inference_detector(model, filtered_points)
    pred_boxes_box_f = res_f.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()

    pred_boxes_score_f = res_f.pred_instances_3d.scores_3d.cpu().numpy()
    filtered_indices = np.where(pred_boxes_score_f >= score_threshold)[0]
    filtered_boxes = pred_boxes_box_f[filtered_indices]
    obb_list_f = [create_oriented_bounding_box(box,offset=0) for box in filtered_boxes]
    
    
    labels = read_kitti_label_file(filename)
    filtered_labels = filter_objects_outside_ellipse(labels, a, b,offset=-10)
    gt_oriented_boxes = [plot_bounding_box_from_corners(label) for label in filtered_labels]
    original_pcd.translate([0, 100, 0], relative=False)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=str(test_num)) 
    vis.add_geometry(original_pcd)
    vis.add_geometry(filtered_pcd)
    for gt_obb in gt_oriented_boxes:
        vis.add_geometry(gt_obb)
    # for obb in obb_list:
    #     vis.add_geometry(obb)
    for obbf in obb_list_f:
        vis.add_geometry(obbf)
    set_custom_view(vis)

    render_option = vis.get_render_option()
    render_option.point_size = 1.5
    # o3d.visualization.draw_geometries([original_pcd] + obb_list)
    vis.run()

    # Close the visualizer window
    vis.destroy_window()
    
    # detected_objects, missed_gt, not_matched_predictions = match_detections_3d(gt_oriented_boxes, obb_list_f)
    # print(len(detected_objects),len(missed_gt),len(not_matched_predictions))
    
    
    """
    Some code snippets for later use if needed
    
    This is for projecting lidar points into an image
     # center_points = np.array([[label['x'],label['y'],label['z'],1] for label in labels])
    # # print(center_points.shape,P_rect.shape,R0_rect_extended.shape)
    # transformedd_centers = ((P_rect@ R0_rect_extended@ center_points.T).T)[:, :3]
    # transformedd_centers[:,0] /= transformedd_centers[:,2]
    # transformedd_centers[:,1] /= transformedd_centers[:,2]
    # plt.scatter(transformedd_centers[:,0],transformedd_centers[:,1],zorder=1,s=1,color="red")
    # plt.imshow(img,zorder=0)
    # plt.scatter(img_point_cloud.T[:,1],img_point_cloud.T[:,0],zorder=1,s=0.3,color="red")

    # plt.imshow(img,zorder=0)
    # print(P_rect.shape,R0_rect_extended.shape,Tr_velo_to_cam_extended.shape)
    # P_velo_to_img = R0_rect_extended @ Tr_velo_to_cam_extended
    # transformed_points = ((Tr_velo_to_cam_extended @ points_homogenous.T).T)[:, :3]
    # transformed_points = points_homogenous @ Tr_velo_to_cam_extended
    # print(transformed_points.shape)
    # plt.show()
    
    # This is needed for projecting lidar points into an image
    R0_rect = calib_data['R0_rect'].reshape(3, 3)
    R0_rect_extended = np.eye(4)  # Create a 4x4 identity matrix
    R0_rect_extended[:3, :3] = R0_rect  # Replace the top-left 3x3 block
    # print(R0_rect.shape,Tr_velo_to_cam.shape)
    P_rect = calib_data['P2'].reshape(3, 4)
    
    img_point_cloud = P_rect @ R0_rect_extended @ Tr_velo_to_cam_extended @ points.T
    # print(img_point_cloud.shape)
    # print(img_point_cloud)
    img_pth = os.path.join(img_path, f'{test_num:06}.png')
    # img = cv2.imread(img_pth)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    """