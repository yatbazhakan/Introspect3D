from nuscenes.nuscenes import NuScenes
import numpy as np
import os
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
    # pprint(corners)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Create a LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])

    return line_set
def create_oriented_bounding_box(box_params,offset=0,axis=0,calib=None,color=(1, 0, 0)):
    offset_array = np.zeros(3)
    offset_array[axis] = offset
    center = np.array([box_params[0], box_params[1], box_params[2]/2+0.5])
    extent = np.array([box_params[3], box_params[4], box_params[5]])
    if(len(box_params) > 9):
      quat = Quaternion(box_params[6:10])
      R = quat.rotation_matrix
    else:
      rotation = box_params[6]
      rotataion = get_quaternion_from_euler(0,0,rotation)
      quat = Quaternion(rotataion)
      R = quat.rotation_matrix
    box = Box(center=center, size=extent, orientation=quat,label=0)

    obb = plot_bounding_box_from_corners(box.corners(),color=color)#o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    ctr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=center)
    # obb.color = (1, 0, 0)  # Red color
    return obb , ctr
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
def create_index_dict_for_categories(categories):
   dicti = {}
   for cat in categories:
      try:
        dicti[cat['name']] = cat['index']
        pprint(cat)
      except:
        print("Error with category {}".format(cat['name']))
   return dicti


nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/ssd2/nuscenes_mini/v1.0-mini', verbose=True)
# mapping= create_index_dict_for_categories(nusc.category)
print("Length of nuScenes database: {}".format(len(nusc.scene)))
key_frame_count = 0
frame_count = 0
my_scene = nusc.scene[0]
# print(nusc.calibrated_sensor)
first_sample_token = my_scene['first_sample_token']
while not first_sample_token == '':
    sample_record = nusc.get('sample', first_sample_token)
    frame_count += 1
    lidar_token = sample_record['data']['LIDAR_TOP']
    calibrated_lidar = nusc.get('calibrated_sensor', nusc.get('sample_data', lidar_token)['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', nusc.get('sample_data', lidar_token)['ego_pose_token'])
    # print(calibrated_lidar)
    lidar_data = nusc.get('sample_data', lidar_token)
    if lidar_data['is_key_frame']:
      key_frame_count += 1
      lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])
      pc_file = os.path.join(nusc.dataroot,lidar_filepath)
      pc = LidarPointCloud.from_file(pc_file)
      points2 = np.fromfile(pc_file, dtype=np.float32, count=-1).reshape([-1, 5])
      img_cam_front = nusc.get('sample_data', sample_record['data']['CAM_FRONT'])
      img_filepath = os.path.join(nusc.dataroot, img_cam_front['filename'])
      img = plt.imread(img_filepath)
      # transformation_matrix = np.eye(4)
      # transformation_matrix[:3,:3] = Quaternion(calibrated_lidar['rotation']).rotation_matrix
      # transformation_matrix[:3,3] = calibrated_lidar['translation']
      # pc.transform(transformation_matrix)

      # ego_to_global_transform = np.eye(4)
      # ego_to_global_transform[:3,:3] = Quaternion(ego_pose['rotation']).rotation_matrix
      # ego_to_global_transform[:3,3] = ego_pose['translation']
      # pc.transform(ego_to_global_transform)
      points = pc.points.T
      obb_boxes = []
      print(len(sample_record['anns']))
      _, boxes, _  = nusc.get_sample_data(lidar_token)
      for i in range(len(boxes)):

        box = boxes[i]
        # corners = np.array(box.corners())
        #box_params = np.concatenate((box_center,box_size,annotations['rotation']))
        # print(box_rotation.rotation_matrix)
        obb_boxes.append(plot_bounding_box_from_corners(box.corners())) #create_oriented_bounding_box(box_params,offset=0,axis=0,calib=None,color=(1, 0, 0))
      


      # kitti_velodyne_path= r"/mnt/ssd1/introspectionBase/datasets/KITTI/training/velodyne"
      # img_path = r"/mnt/ssd1/introspectionBase/datasets/KITTI/training/image_2"
      # config_file = r'/mnt/ssd1/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'#r'D:\mmdetection3d\configs\pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py' # 
      # checkpoint = r'/mnt/ssd1/mmdetection3d/ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth' #r"D:\mmdetection3d\ckpts\hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth" #
      config_file = r'/mnt/ssd1/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py'
      checkpoint = r'/mnt/ssd1/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'
      model = init_model(config_file, checkpoint, device='cuda:0')
      res = inference_detector(model, points2)
      res_f = res[0]
      data = res[1]
      
      # print(res_f)
      # print(res_f.keys())
      pred_boxes_box_f = res_f.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
      pred_boxes_score_f = res_f.pred_instances_3d.scores_3d.cpu().numpy()
      # print(pred_boxes_score_f)
      filtered_indices = np.where(pred_boxes_score_f >= 0.5)[0]
      filtered_boxes = pred_boxes_box_f[filtered_indices]
      obb_list_f = [create_oriented_bounding_box(box,offset=0,calib=None) for box in filtered_boxes]
      # print("Number of boxes: {}".format(len(obb_list_f)))
      
      max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))
      given_inputs_np= data['inputs']['points'][:, :3].cpu().numpy()
      given_inputs_pcd = create_point_cloud(given_inputs_np,max_distance)
    
      # print("Number of key frames: {}".format(key_frame_count))
      # print("Number of frames: {}".format(frame_count))
      pcd = create_point_cloud(points,max_distance)
      vis = o3d.visualization.Visualizer()
      vis.create_window()
      vis.add_geometry(given_inputs_pcd)
      #Plot axes
      x_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0,0,0])
      vis.add_geometry(x_axis)
      
      for box in obb_boxes:
        vis.add_geometry(box)
      for box,ctr in obb_list_f:
        vis.add_geometry(box)
        vis.add_geometry(ctr)
      # o3d.visualization.draw_geometries([pcd])
      vis.run()
      vis.destroy_window()
      first_sample_token = sample_record['next']
      break
# my_sample = nusc.get('sample', first_sample_token)
# lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
# #get point cloud data from nuscenes
# pc_file = nusc.get('sample_data', lidar_data['token'])['filename']
# pc_file = os.path.join(nusc.dataroot,pc_file)
# points = np.fromfile(pc_file, dtype=np.float32, count=-1).reshape([-1, 5])

# max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))
# # print("Number of key frames: {}".format(key_frame_count))
# # print("Number of frames: {}".format(frame_count))
# pcd = create_point_cloud(points,max_distance)
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# for box in boxes:
#   vis.add_geometry(box)
# # o3d.visualization.draw_geometries([pcd])
# vis.run()
# vis.destroy_window()














"""
  
        annotations = nusc.get('sample_annotation', sample_record['anns'][i])
        # pprint(annotations)
        #Get category index with name
        
        # index = mapping[annotations['category_name']]
        # category =  index
        box_center = annotations['translation']
      
        box_size = annotations['size']
        box_rotation = annotations['rotation']
        box_params = np.concatenate((box_center,box_size,annotations['rotation']))
        # print(type(category))
        #convert label category to integer for nuscenes
        rotation = Quaternion(box_rotation)
        box = Box(center=box_center,size=box_size,orientation=rotation,label = 0)
        box.translate(calibrated_lidar['translation'])
        box.rotate(Quaternion(calibrated_lidar['rotation']))
"""