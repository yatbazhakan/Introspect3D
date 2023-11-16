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
          corners = corners.T
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
dataset = NuScenesDataset(root_dir='/mnt/ssd2/nuscenes/',
                          version='v1.0-trainval',
                          split='train',
                          transform=None,
                          filtering_style = "FilterType.ELLIPSE",
                          filter_params = {'a':15,'b':25,'offset':-5,'axis':1},
                          save_path='/mnt/ssd2/nuscenes/',
                          save_filename='nuscenes_train_filtered.pkl',
                          process=False,)
print("Length of nuScenes database: {}".format(len(dataset)))

for item in dataset:
    visualizer = Visualizer()
    config =r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py'
    checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'
    # checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
    # config= r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
    # config =r'/mnt/ssd2/mmdetection3d/configs/pv_rcnn/pv_rcnn_8xb2-80e_kitti-3d-3class.py'
    # checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/pv_rcnn_8xb2-80e_kitti-3d-3class_20221117_234428-b384d22f.pth'
    config =r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_16xb2-2x_waymo-3d-3class.py'
    checkpoint=  r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn_2x16_2x_waymoD5-3d-3class_20200831_204144-d1a706b1.pth'
    
    model = init_model(config, checkpoint, device='cuda:0')
    item['pointcloud'].validate_and_update_descriptors(extend_or_reduce=4)
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
    print("Number of matches: {}".format(len(matches)))
    print("Number of unmatched ground truths: {}".format(len(unmatched_ground_truths)))
    print("Number of unmatched predictions: {}".format(len(unmatched_predictions)))
    # print(type(prediction_bounding_boxes[0]),type(item['labels'][0]))
    visualizer.visualize(cloud= item['pointcloud'].points[:,:3],gt_boxes = item['labels'],pred_boxes = prediction_bounding_boxes)  # item['labels']
    
    if input() == "q":
        break
    # nusc = NuScenes(version='v1.0-mini', dataroot='/mnt/ssd2/nuscenes_mini/v1.0-mini', verbose=True)
    # # kitti_velodyne_path= r"/mnt/ssd1/introspectionBase/datasets/KITTI/training/velodyne"
    # # img_path = r"/mnt/ssd1/introspectionBase/datasets/KITTI/training/image_2"
    # # config_file = r'/mnt/ssd1/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py'#r'D:\mmdetection3d\configs\pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py' # 
    # # checkpoint = r'/mnt/ssd1/mmdetection3d/ckpts/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth' #r"D:\mmdetection3d\ckpts\hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d_20201021_120719-269f9dd6.pth" #
    # # config_file = r'/mnt/ssd1/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-amp-2x_nus-3d.py'
    # # checkpoint = r'/mnt/ssd1/mmdetection3d/ckpts/hv_pointpillars_secfpn_sbn-all_fp16_2x8_2x_nus-3d_20201020_222626-c3f0483e.pth'
    # # config_file = r'/mnt/ssd2/mmdetection3d/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
    # # checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'
    # checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_20e_nus_20220810_025930-657f67e0.pth'
    # config_file = r'/mnt/ssd2/mmdetection3d/configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'
    # # checkpoint = r'/mnt/ssd2/mmdetection3d/ckpts/tpvformer_8xb1-2x_nus-seg_20230411_150639-bd3844e2.pth'
    # # config_file = r'/mnt/ssd2/mmdetection3d/configs/tpvformer/tpvformer_8xb1-2x_nus-seg.py'

    # model = init_model(config_file, checkpoint, device='cuda:0')
    # # nusc = NuScenes(version='v1.0-trainval', dataroot='/mnt/ssd2/nuscenes/', verbose=True)
    # # pickle.dump(nusc,open("nuscenes_trainval.pkl","wb"))
    # # exit()
    # # mapping= create_index_dict_for_categories(nusc.category)
    # # nusc = pickle.load(open("nuscenes_trainval.pkl","rb"))
    # print("Length of nuScenes database: {}".format(len(nusc.scene)))
    # key_frame_count = 0
    # frame_count = 0
    # my_scene = nusc.scene[0]
    # # print(nusc.calibrated_sensor)
    # first_sample_token = my_scene['first_sample_token']
    # frame_count = 0

    # while not first_sample_token == '':
    #     sample_record = nusc.get('sample', first_sample_token)
    #     frame_count += 1
    #     lidar_token = sample_record['data']['LIDAR_TOP']
    #     cs_record = nusc.get('calibrated_sensor', nusc.get('sample_data', lidar_token)['calibrated_sensor_token'])
    #     pose_record = nusc.get('ego_pose', nusc.get('sample_data', lidar_token)['ego_pose_token'])
        
    #     # print(calibrated_lidar)
    #     lidar_data = nusc.get('sample_data', lidar_token)
    #     if lidar_data['is_key_frame']:
    #       key_frame_count += 1
    #       lidar_filepath = os.path.join(nusc.dataroot, lidar_data['filename'])
    #       pc_file = os.path.join(nusc.dataroot,lidar_filepath)
    #       pc = LidarPointCloud.from_file(pc_file)
    #       points2 = np.fromfile(pc_file, dtype=np.float32, count=-1).reshape([-1, 5])
    #       img_cam_front = nusc.get('sample_data', sample_record['data']['CAM_FRONT'])
    #       img_filepath = os.path.join(nusc.dataroot, img_cam_front['filename'])
    #       max_distance =  None#np.max(np.linalg.norm(points2[:, :3], axis=1))

    #       a,b = 15, 25
    #       offset = -5
    #       axis = 1
    #       filtered_points = filter_points_inside_ellipse(points2,a,b,offset=offset,axis=axis)
    #       filtered_pcd = create_point_cloud(filtered_points,max_distance)
    #       # img = cv2.imread(img_filepath)
    #       # cv2.imshow("Image",img)
    #       # cv2.waitKey(0)
    #       # cv2.destroyAllWindows()
        
    #       # transformation_matrix = np.eye(4)
    #       # transformation_matrix[:3,:3] = Quaternion(calibrated_lidar['rotation']).rotation_matrix
    #       # transformation_matrix[:3,3] = calibrated_lidar['translation']
    #       # pc.transform(transformation_matrix)

    #       # ego_to_global_transform = np.eye(4)
    #       # ego_to_global_transform[:3,:3] = Quaternion(ego_pose['rotation']).rotation_matrix
    #       # ego_to_global_transform[:3,3] = ego_pose['translation']
    #       # pc.transform(ego_to_global_transform)
    #       points = pc.points.T
    #       obb_boxes = []
    #       print(len(sample_record['anns']))
    #       _, boxes, _  = nusc.get_sample_data(lidar_token)
    #       filtered_boxes = []
    #       for i in range(len(boxes)):
    #         annotation = nusc.get('sample_annotation', sample_record['anns'][i])
    #         box = boxes[i]
    #         if filter_boxes_with_category(annotation['category_name']):
    #           #May need to add some coloring
    #         #   temp_box = BoundingBox(center=box.center,size=box.wlh,orientation=Quaternion(box.orientation),label=annotation['category_name'])
    #         #   print("Adding box with category {}".format(annotation['category_name']))
    #           filtered_boxes.append(box.corners())
    #           # obb_boxes.append(plot_bounding_box_from_corners(box.corners())) #create_oriented_bounding_box(box_params,offset=0,axis=0,calib=None,color=(1, 0, 0))
    #           # for i in range(len(sample_record['anns'])):
    #       #   annotations = nusc.get('sample_annotation', sample_record['anns'][i])
    #       #   print(annotations['category_name'])
    #       #   box = Box(center=annotations['translation'],size=annotations['size'],orientation=Quaternion(annotations['rotation']),label=0)

    #       #   # Move box to ego vehicle coord system.
    #       #   box.translate(-np.array(pose_record['translation']))
    #       #   box.rotate(Quaternion(pose_record['rotation']).inverse)

    #       #   #  Move box to sensor coord system.
    #       #   box.translate(-np.array(cs_record['translation']))
    #       #   box.rotate(Quaternion(cs_record['rotation']).inverse)

    #         # box.translate(np.array([0, 0, 0.5]))
    #         # corners = np.array(box.corners())
    #         #box_params = np.concatenate((box_center,box_size,annotations['rotation']))
    #         # print(box_rotation.rotation_matrix)
            
        


    #       filtered_gt_boxes = filter_objects_outside_ellipse(filtered_boxes,a,b,offset=offset,axis=axis)
    #       obb_boxes = [plot_bounding_box_from_corners(box,offset=offset,color=(0,0,1)) for box in filtered_gt_boxes]
    #       res = inference_detector(model, filtered_points)
    #       res_f = res[0]
    #       data = res[1]
        
    #       # print(res_f)
    #       # print(res_f.keys())
    #       pred_boxes_box_f = res_f.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    #       pred_boxes_score_f = res_f.pred_instances_3d.scores_3d.cpu().numpy()
    #       pred_boxes_labels = res_f.pred_instances_3d.labels_3d.cpu().numpy()
    #       filtered_indices = np.where(pred_boxes_score_f >= 0.5)[0]
    #       filtered_boxes = pred_boxes_box_f[filtered_indices]
    #       filtered_labels = pred_boxes_labels[filtered_indices]
    #       obb_list_f = [create_oriented_bounding_box(box,color=(1,1,1)) for box in filtered_boxes]
    #       matches , unmatched_ground_truths, unmatched_predictions = match_detections_3d(obb_boxes,obb_list_f)
    #       print("Number of matches: {}".format(len(matches)))
    #       print("Number of unmatched ground truths: {}".format(len(unmatched_ground_truths)))
    #       print("Number of unmatched predictions: {}".format(len(unmatched_predictions)))
    #       # print("Number of key frames: {}".format(key_frame_count))
    #       # print("Number of frames: {}".format(frame_count))
    #     #   pcd = create_point_cloud(points,max_distance)
    #       vis = o3d.visualization.Visualizer()
    #       vis.create_window()
    #       vis.add_geometry(filtered_pcd)
    #       #Plot axes
    #       x_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1,origin=[0,0,0])
    #       vis.add_geometry(x_axis)
        
    #       for box in obb_boxes:
    #         vis.add_geometry(box)
    #       for box in obb_list_f:
    #         vis.add_geometry(box)
    #         # vis.add_geometry(ctr)
    #       # o3d.visualization.draw_geometries([pcd])
    #     #   vis.capture_screen_image(f"./outputs/frame{frame_count}.jpg", do_render=True)
    #       frame_count += 1
    #     #   opt = vis.get_render_option()
    #     #   opt.background_color = np.asarray([0.6, 0.6, 0.6])
        
    #       vis.run()
    #       vis.destroy_window()
    #       first_sample_token = sample_record['next']
    #       # q = input("Press enter to continue")
    #       # if q == 'q':
        
        
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