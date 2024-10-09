import open3d as o3d
import os
from glob import glob
import numpy as np
from base_classes.base import DrivingDataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud,Box
from nuscenes.utils.data_classes import RadarPointCloud
from pyquaternion import Quaternion
import numpy as np
from utils.boundingbox import BoundingBox
from utils.pointcloud import PointCloud
from utils.filter import *
import pickle
import os
import copy
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
from utils.filter import EllipseFilter,FilteringArea
from pyquaternion import Quaternion
from open3d import geometry
import cv2
from utils.boundingbox import BoundingBox
import torch
from utils.utils import create_bounding_boxes_from_predictions
from utils.utils import check_detection_matches
from utils.utils import generate_model_from_config
import time
int_model_config = "/mnt/ssd2/Introspect3D/configs/networks/resnet18_fcn.yaml"
model_load_path = "/mnt/ssd2/nus_late_single.pth"
checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
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
activation_list = []
def register_activation_output(module, input, output):
    # print(output[0].shape,output[1].shape)
    # print(len(output))
    last_output = output.detach().cpu().numpy() #TODO: generalize this
    # print("Last output shape",last_output.shape)
    # print(last_output.shape)
    # print("-------------------")
    last_output = np.squeeze(last_output)
    activation_list.append(last_output)
def set_custom_view(vis):
    ctr = vis.get_view_control()
    
    # Define the desired camera location and orientation
    camera_position = np.array([-5, 0, 10], dtype=np.float64)  # Ensure dtype is float for calculations
    look_at_point = np.array([0, 0, 0], dtype=np.float64)
    up_vector = np.array([0, 0, 1], dtype=np.float64)
    
    # Calculate the new front vector (z-axis of the camera coordinate system)
    front_vector = look_at_point - camera_position
    front_vector /= np.linalg.norm(front_vector)  # Normalize to create a unit vector
    
    # Calculate the right vector (x-axis of the camera coordinate system)
    right_vector = np.cross(up_vector, front_vector)
    right_vector /= np.linalg.norm(right_vector)  # Normalize to create a unit vector
    
    # Re-calculate the up vector to ensure orthogonality (y-axis of the camera coordinate system)
    up_vector = np.cross(front_vector, right_vector)
    up_vector /= np.linalg.norm(up_vector)  # Normalize to create a unit vector
    
    # Create the extrinsic matrix
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[0:3, 0] = right_vector
    extrinsic[0:3, 1] = up_vector
    extrinsic[0:3, 2] = front_vector
    extrinsic[0:3, 3] = camera_position

    # Set the extrinsic matrix to the camera parameters
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam_params)
    
    # Adjust rendering options if needed
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 1.0  # Increase point size for better visibility at closer range
# def set_custom_view(vis):
    
#     ctr = vis.get_view_control()
#     print(ctr)
#     # Create an extrinsic matrix for camera placement
#     extrinsic = np.eye(4)
#     extrinsic[0:3, 3] = [-10, 0, 30]  # Set camera position (x, y, z)
    
#     # Create a rotation matrix for 30-degree downward view
#     rotation = np.array([
#         [1, 0, 0],
#         [0, np.cos(np.radians(-160)), -np.sin(np.radians(-160))],
#         [0, np.sin(np.radians(-160)), np.cos(np.radians(-160))]
#     ])
    
#     # Apply rotation to the extrinsic matrix
#     extrinsic[0:3, 0:3] = rotation
    
#     # Set the extrinsic matrix to the camera parameters
#     cam_params = ctr.convert_to_pinhole_camera_parameters()
#     cam_params.extrinsic = extrinsic
#     ctr.convert_from_pinhole_camera_parameters(cam_params)
#     opt = vis.get_render_option()
#     opt.background_color = np.asarray([0.0, 0.0, 0.0])
#     opt.point_size = 1.0
def make_point_cloud(path):
    cloud = np.load(path)
    points = cloud[:, :3]            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3))*[34,139,34])
    return pcd
def load_point_clouds(folder_path):
    """Load all point cloud files from the seeeeepecified folder."""
    files = glob(os.path.join(folder_path, 'lidar','*.npy'))

    point_clouds = [make_point_cloud(file) for file in files]
    return point_clouds
def filter_boxes_with_category(box_label,accepted_categories=['vehicle.','human','cyclist']):
        for cat in accepted_categories:
            if box_label.startswith(cat):
                return True
        return False
def read_labels(**kwargs):
    c_boxes = []
    id = kwargs.get('id', None)
    lidar_token = kwargs['lidar_token']
    sample_record = kwargs['sample_record']
    nuscenes_data = kwargs.get('nuscenes_data', None)
    _,  boxes, _  = nuscenes_data.get_sample_data(lidar_token)
    for i in range(len(boxes)):
        annotation = nuscenes_data.get('sample_annotation', sample_record['anns'][i])
        box = boxes[i]
        if filter_boxes_with_category(annotation['category_name']):
            box.label = object_label_to_index[annotation['category_name']]
            custom_box = BoundingBox()
            custom_box.from_nuscenes_box(box)
            print(custom_box)
            c_boxes.append(custom_box)
    return c_boxes
def create_oriented_bounding_box(box_params,rot_axis=2,calib=True,color=(1, 1, 0)):

    if type(box_params) == BoundingBox:
        corners = box_params.corners.copy()
        box_params = box_params.to_list()
        print(box_params)
        center = np.array([box_params[0], box_params[1], box_params[2]])
        extent = np.array([box_params[3], box_params[4], box_params[5]])
        yaw = box_params[6]
        rot_mat = geometry.get_rotation_matrix_from_xyz([0,0,0])
        #Create LIneset with corners
        line_set = geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(
            [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
        #yellow color
        line_set.paint_uniform_color(color)
        return line_set
    else:
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
    box3d = geometry.OrientedBoundingBox(center, rot_mat, extent)
    line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
    #yellow color
    line_set.paint_uniform_color(color)

    #  Move box to sensor coord system
    ctr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=center)
    # obb.color = (1, 0, 0)  # Red color
    return line_set#box3d#line_set #, ctr
def visualize_point_clouds(nuscenes_data, tokens, lidar=False, radar=False):
    idx = 0
    model = init_model(config, checkpoint, device='cuda:0')
    int_model = generate_model_from_config({'layer_config': int_model_config})
    int_model.load_state_dict(torch.load(model_load_path))
    int_model.eval()
    int_model.to('cuda:1')
    hook = model.pts_backbone.blocks._modules[str(1)].register_forward_hook(register_activation_output)
    
    # Iterate through each token
    for token in tokens:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        set_custom_view(vis)  # Set up custom camera view if necessary
        
        scene = nuscenes_data.get('scene', token)
        sample_token = scene['first_sample_token']
        sample_record = nuscenes_data.get('sample', sample_token)

        # Process Lidar data
        while sample_record['next'] != '':
            lidar_token = sample_record['data']['LIDAR_TOP']
            gt_boxes = read_labels(nuscenes_data=nuscenes_data, lidar_token=lidar_token, sample_record=sample_record)
            pc_filter = EllipseFilter(a=15, b=25, offset=-10, axis=1)
            filtered_gt_boxes = pc_filter.filter_bounding_boxes(gt_boxes, mode=FilteringArea.INSIDE)
            
            lidar_filepath = nuscenes_data.get_sample_data_path(lidar_token)
            point_cloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
            inside_points = pc_filter.filter_pointcloud(point_cloud, mode=FilteringArea.INSIDE)
            
            res, data = inference_detector(model, inside_points)
            activations = activation_list.pop()
            t_act = torch.from_numpy(activations).unsqueeze(0).to('cuda:1')
            res_int = int_model(t_act)
            indice_max = torch.argmax(res_int)
            predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
            predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
            score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later
            filtered_predicted_boxes = predicted_boxes[score_mask]

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(inside_points[:, :3])
            o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(inside_points), 3)) * [1.0, 1.0, 1.0])  # Green color
            vis.add_geometry(o3d_cloud)
            outside_points = pc_filter.filter_pointcloud(point_cloud, mode=FilteringArea.OUTSIDE)
            out_o3d_cloud = o3d.geometry.PointCloud()
            out_o3d_cloud.points = o3d.utility.Vector3dVector(outside_points[:, :3])
            out_o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(outside_points), 3)) * [0.7, 0.7, 1])  # Blue color
            vis.add_geometry(out_o3d_cloud)
            obb_predicted = [create_oriented_bounding_box(box, calib=False) for box in filtered_predicted_boxes]
            obb_gt = [create_oriented_bounding_box(box, calib=False, color=(0, 1, 0)) for box in filtered_gt_boxes]
            for box in obb_predicted:
                vis.add_geometry(box)
            for box in obb_gt:
                vis.add_geometry(box)
            filtered_predicted_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)
            matched , unmatched_ground_truths, unmatched_predictions = check_detection_matches(filtered_gt_boxes,filtered_predicted_boxes)
            with open('errors.txt', 'a') as f:
                if len(unmatched_ground_truths):
                    f.write(f"{token}_{idx},GT: Error, Predicted {res_int}-{indice_max}\n")
                else:
                    f.write(f"{token}_{idx},GT: No Error, Predicted {res_int}-{indice_max}\n")
            sample_token = sample_record['next']
            sample_record = nuscenes_data.get('sample', sample_token)


            # Run visualizer to ensure all data is loaded
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f"outputs/demo/{token}_2frame_{str(idx).zfill(2)}.png")
            idx += 1
            vis.clear_geometries()
            time.sleep(0.5)
        # Close and reset for next token
    vis.destroy_window()


# def visualize_point_clouds(nuscenes_data, tokens, lidar = False, radar = False):
#     """Visualize point clouds and switch between them using the spacebar."""
#     idx =0
#     model = init_model(config, checkpoint, device='cuda:0')
#     int_model = generate_model_from_config({'layer_config':int_model_config})
#     int_model.load_state_dict(torch.load(model_load_path))
#     int_model.eval()
#     int_model.to('cuda:1')
#     is_nuscenes = True
#     hook = model.pts_backbone.blocks._modules[str(1)].register_forward_hook(register_activation_output)
#     for token in tokens:
#         scene = nuscenes_data.get('scene', token)

#         vis = o3d.visualization.VisualizerWithKeyCallback()
#         vis.create_window()
#         is_finished = False
#         set_custom_view(vis)
#         pc_filter = EllipseFilter(a=15,b=25,offset=-10,axis=1)
#         first_sample_token = scene['first_sample_token']
#         sample_record = nuscenes_data.get('sample', first_sample_token)
#         if lidar:
#             lidar_token = sample_record['data']['LIDAR_TOP']
#             gt_boxes = read_labels(nuscenes_data=nuscenes_data,lidar_token=lidar_token,sample_record=sample_record)

#             filtered_gt_boxes = pc_filter.filter_bounding_boxes(gt_boxes,mode=FilteringArea.INSIDE)
#             lidar_data = nuscenes_data.get('sample_data', lidar_token)
#             lidar_filepath = nuscenes_data.get_sample_data_path(lidar_token)
#             point_cloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
#             outside_points = pc_filter.filter_pointcloud(point_cloud,mode=FilteringArea.OUTSIDE)
#             inside_points = pc_filter.filter_pointcloud(point_cloud,mode=FilteringArea.INSIDE)
#             res, data = inference_detector(model, inside_points)
#             activations = activation_list.pop()
#             t_act = torch.from_numpy(activations).unsqueeze(0).to('cuda:1')
#             res_int = int_model(t_act)
#             softmaxed = torch.nn.functional.softmax(res_int,dim=1)

#             print(res_int)
#             predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
#             predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
#             score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later

#             filtered_predicted_boxes = predicted_boxes[score_mask]
#             print(len(filtered_predicted_boxes))
#             o3d_cloud = o3d.geometry.PointCloud()
            
#             o3d_cloud.points = o3d.utility.Vector3dVector(inside_points[:, :3])
#             o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(inside_points), 3))*[34,139,34])
#             out_o3d_cloud = o3d.geometry.PointCloud()
#             out_o3d_cloud.points = o3d.utility.Vector3dVector(outside_points[:, :3])
#             out_o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(outside_points), 3))*[0.4,0.44,1])
#         if radar:
#             # "Loading"
#             radar_token = sample_record['data']['RADAR_FRONT']
#             radar_data = nuscenes_data.get('sample_data', radar_token)
#             radar_filepath = nuscenes_data.get_sample_data_path(radar_token)
#             radar_point_cloud = RadarPointCloud.from_file(radar_filepath)
#             point_cloud = radar_point_cloud.points
#             o3d_cloud2 = o3d.geometry.PointCloud()
#             o3d_cloud2.points = o3d.utility.Vector3dVector(radar_point_cloud.points[:, :3])
#             #Create np array with yellow rgb code with point length
#             o3d_cloud2.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*np.array([1,1,0]))
#         #Combine if both lidar and radar are selected

#         def load_next_point_cloud(vis):
#             nonlocal sample_record,lidar, radar,idx,token # Declare nonlocal to modify the outer scope variable
#             first_sample_token = sample_record['next']
#             if first_sample_token == '':
#                 return
#             sample_record = nuscenes_data.get('sample', first_sample_token)
#             vis.clear_geometries()
#             if lidar:
#                 lidar_token = sample_record['data']['LIDAR_TOP']
#                 gt_boxes = read_labels(nuscenes_data=nuscenes_data,lidar_token=lidar_token,sample_record=sample_record)
#                 filtered_gt_boxes = pc_filter.filter_bounding_boxes(gt_boxes,mode=FilteringArea.INSIDE)
#                 lidar_data = nuscenes_data.get('sample_data', lidar_token)
#                 lidar_filepath = nuscenes_data.get_sample_data_path(lidar_token)
#                 point_cloud = np.fromfile(lidar_filepath, dtype=np.float32, count=-1).reshape([-1, 5])
#                 outside_points = pc_filter.filter_pointcloud(point_cloud,mode=FilteringArea.OUTSIDE)
#                 inside_points = pc_filter.filter_pointcloud(point_cloud,mode=FilteringArea.INSIDE)
#                 res, data = inference_detector(model, inside_points)
#                 activations = activation_list.pop()
#                 t_act = torch.from_numpy(activations).unsqueeze(0).to('cuda:1')
#                 res_int = int_model(t_act)
#                 softmaxed = torch.nn.functional.softmax(res_int,dim=1)
#                 # print(res_int)
#                 predicted_boxes = res.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
#                 predicted_scores = res.pred_instances_3d.scores_3d.cpu().numpy()
#                 score_mask = np.where(predicted_scores >= 0.5)[0] # May require edit later
#                 filtered_predicted_boxes = predicted_boxes[score_mask]
                
#                 o3d_cloud = o3d.geometry.PointCloud()
#                 o3d_cloud.points = o3d.utility.Vector3dVector(inside_points[:, :3])
#                 o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(inside_points), 3))*[34,139,34])
#                 out_o3d_cloud = o3d.geometry.PointCloud()
#                 out_o3d_cloud.points = o3d.utility.Vector3dVector(outside_points[:, :3])
#                 out_o3d_cloud.colors = o3d.utility.Vector3dVector(np.ones((len(outside_points), 3))*[0.4,0.4,1])
#                 vis.add_geometry(o3d_cloud, reset_bounding_box=True)
#                 vis.add_geometry(out_o3d_cloud, reset_bounding_box=True)
#                 obb_boxes = [create_oriented_bounding_box(box,calib=False) for box in filtered_predicted_boxes]
#                 obb_list_f = [create_oriented_bounding_box(box,color=(0,1,0)) for box  in filtered_gt_boxes]
#                 #convert detected boxes to BoundingBox objects
#                 detected_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)

#                 matches , unmatched_ground_truths, unmatched_predictions = check_detection_matches(filtered_gt_boxes,detected_boxes)
#                 for box in obb_boxes:
#                     vis.add_geometry(box)
#                 # for box in obb_list_f:
#                 #     vis.add_geometry(box)
#                 if len(unmatched_ground_truths):
#                     with open('errors.txt','a') as f:
#                         f.write(f"{token}_{idx},GT: Error, Predicted {softmaxed}\n")
#                 else:
#                     with open('errors.txt','a') as f:
#                         f.write(f"{token}_{idx},GT: No Error, Predicted {softmaxed}\n")
#                 print("idx",idx)
#                 idx += 1
#             if radar:
#                 radar_token = sample_record['data']['RADAR_FRONT']

#                 radar_data = nuscenes_data.get('sample_data', radar_token)
#                 radar_filepath = nuscenes_data.get_sample_data_path(radar_token)
#                 # point_cloud = np.fromfile(radar_filepath, dtype=np.float32, count=-1).reshape([-1, 18])
#                 radar_point_cloud = RadarPointCloud.from_file(radar_filepath)
                
#                 point_cloud = radar_point_cloud.points

#                 o3d_cloud2 = o3d.geometry.PointCloud()
#                 o3d_cloud2.points = o3d.utility.Vector3dVector(radar_point_cloud.points[:, :3])
#                 #Create np array with yellow rgb code with point length
#                 o3d_cloud2.colors = o3d.utility.Vector3dVector(np.ones((len(point_cloud), 3))*[255,255,0])
#                 vis.add_geometry(o3d_cloud2, reset_bounding_box=True)
#             vis.poll_events()  # Process any pending events
#             vis.update_renderer()  # Update the renderer
#             vis.capture_screen_image(f"outputs/demo/{token}_lidar_{idx}.png")

#             # vis.add_geometry(o3d_cloud, reset_bounding_box=True)
            
#         vis.register_key_callback(ord(' '), load_next_point_cloud)  # Bind spacebar to switch point clouds

        
#         if lidar and radar:
#             vis.add_geometry(o3d_cloud, reset_bounding_box=True)
#             vis.add_geometry(o3d_cloud2, reset_bounding_box=True)
#         elif lidar:
#             detected_boxes = create_bounding_boxes_from_predictions(filtered_predicted_boxes)

#             matches , unmatched_ground_truths, unmatched_predictions = check_detection_matches(filtered_gt_boxes,detected_boxes)
#             if len(unmatched_ground_truths):
#                 with open('errors.txt','a') as f:
#                     f.write(f"{token}_{idx},GT: Error, Predicted {softmaxed}\n")
#             else:
#                 with open('errors.txt','a') as f:
#                     f.write(f"{token}_{idx},GT: No Error, Predicted {softmaxed}\n")
#             obb_boxes = [create_oriented_bounding_box(box,calib=False) for box in filtered_predicted_boxes]
#             obb_list_f = [create_oriented_bounding_box(box,color=(0,1,0)) for box in filtered_gt_boxes]
#             vis.add_geometry(o3d_cloud, reset_bounding_box=True)
#             vis.add_geometry(out_o3d_cloud, reset_bounding_box=True)
#             for box in obb_boxes:
#                 vis.add_geometry(box)
#             print("idx",idx)
#             idx +=1
#         elif radar:
#             vis.add_geometry(o3d_cloud2, reset_bounding_box=True)
#         vis.poll_events()  # Process any pending events
#         vis.update_renderer()  # Update the renderer          
#         vis.capture_screen_image(f"outputs/demo/{token}_lidar_{idx}.png")
#         vis.run()  # Run the visualizer

#         vis.destroy_window()  # Clean up after closing the window

import argparse
import pickle
# Usage
#Need to adapt to NuScenes dataset#
def arg_parse():
    parser = argparse.ArgumentParser(description='Visualize point clouds')
    parser.add_argument('-n', "--names",nargs='+', default=[], help='Names of the scenes to visualize')
    parser.add_argument('-l', "--lidar", default=True, help='Visualize lidar data')
    parser.add_argument('-r', "--radar", default=False, help='Visualize radar data')
    return parser.parse_args()
if __name__ == '__main__':
    folder_path = '/media/yatbaz_h/Jet/HYY/'
    root_dir = '/mnt/ssd2/nuscenes'
    version = 'v1.0-trainval'
    args = arg_parse()
    # if "nuscenes_data.pkl" in os.listdir():
    #     with open('nuscenes_data.pkl','rb') as f:
    #         nuscenes_data = pickle.load(f)
    # else:
    nuscenes_data= NuScenes(version=version, dataroot=root_dir, verbose=True)
        # with open('nuscenes_data.pkl','wb') as f:
        #     pickle.dump(nuscenes_data,f)
    # nuscenes_data= NuScenes(version=version, dataroot=root_dir, verbose=True)
    # with open('nuscenes_data.pkl','wb') as f:
    #     pickle.dump(nuscenes_data,f)
    import pandas as pd 
    nuscenes_scene_tokens = pd.DataFrame(columns=['scene_token','name','description'])
    scenes = nuscenes_data.scene
    for scene in scenes:
        scene_token = scene['token']
        name = scene['name']
        description = scene['description']
        print(scene_token,name,description)

        visualize_point_clouds(nuscenes_data,tokens = [scene_token], lidar = args.lidar, radar = args.radar)
    # nuscenes_data= NuScenes(version=versio                  n, dataroot=root_dir, verbose=True)
    # import pandas as pd
    # from tqdm.auto import tqdm
    # nuscenes_scene_tokens = pd.DataFrame(columns=['scene_token','name','description'])
    # scenes = 
    # with tqdm(total=len(scenes)) as pbar:
    #     for scene in scenes:
    #         scene_token = scene['token']
    #         name = scene['name']
    #         description = scene['description']
    #         print(scene_token,name,description)
    #         temp_df = pd.DataFrame([[scene_token,name,description]],columns=['scene_token','name','description'],index=[0])
    #         nuscenes_scene_tokens = pd.concat([nuscenes_scene_tokens,temp_df],ignore_index=True)
    #         pbar.update(1)
    # nuscenes_scene_tokens.to_csv('nuscenes_scene_tokens.csv')
    # point_clouds = load_point_clouds(folder_path)
#    visualize_point_clouds(nuscenes_data)






# import open3d as o3d
# import os
# from glob import glob
# import numpy as np

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

# def set_custom_view(vis):
    
#     ctr = vis.get_view_control()
#     print(ctr)
#     # Create an extrinsic matrix for camera placement
#     extrinsic = np.eye(4)
#     extrinsic[0:3, 3] = [-10, 0, 60]  # Set camera position (x, y, z)
    
#     # Create a rotation matrix for 30-degree downward view
#     rotation = np.array([
#         [1, 0, 0],
#         [0, np.cos(np.radians(-160)), -np.sin(np.radians(-160))],
#         [0, np.sin(np.radians(-160)), np.cos(np.radians(-160))]
#     ])
    
#     # Apply rotation to the extrinsic matrix
#     extrinsic[0:3, 0:3] = rotation
    
#     # Set the extrinsic matrix to the camera parameters
#     cam_params = ctr.convert_to_pinhole_camera_parameters()
#     cam_params.extrinsic = extrinsic
#     ctr.convert_from_pinhole_camera_parameters(cam_params)
#     opt = vis.get_render_option()
#     opt.background_color = np.asarray([0.0, 0.0, 0.0])
#     opt.point_size = 1.0
# def make_point_cloud(path):
#     cloud = np.load(path)
#     points = cloud[:, :3]            
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(np.ones((len(points), 3))*[34,139,34])
#     return pcd
# def load_point_clouds(folder_path):
#     """Load all point cloud files from the seeeeepecified folder."""
#     files = glob(os.path.join(folder_path, 'lidar','*.npy'))

#     point_clouds = [make_point_cloud(file) for file in files]
#     return point_clouds

# def visualize_point_clouds(point_clouds):
#     """Visualize point clouds and switch between them using the spacebar."""
#     if not point_clouds:  # Check if list is empty
#         print("No point clouds found in the directory.")
#         return

#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window()
#     set_custom_view(vis)
#     current_index = [0]  # Using a list to hold the index as a mutable object

#     def load_next_point_cloud(vis):
#         """Callback function to load the next point cloud."""
#         current_index[0] = (current_index[0] + 1) % len(point_clouds)
#         vis.clear_geometries()
#         vis.add_geometry(point_clouds[current_index[0]], reset_bounding_box=True)

#     vis.register_key_callback(ord(' '), load_next_point_cloud)  # Bind spacebar to switch point clouds
#     vis.add_geometry(point_clouds[current_index[0]], reset_bounding_box=True)

#     vis.run()  # Run the visualizer
#     vis.destroy_window()  # Clean up after closing the window

# # Usage
# #Need to adapt to NuScenes dataset
# if __name__ == '__main__':

#     folder_path = '/media/yatbaz_h/Jet/HYY/'
#     root_dir = '/mnt/ssd2/nuscenes'
#     version = 'v1.0-trainval'

#     nuscenes_data= NuScenes(version=version, dataroot=root_dir, verbose=True)
#     # point_clouds = load_point_clouds(folder_path)
#     # visualize_point_clouds(point_clouds)
