import os
import numpy as np
import cv2
import mmcv
from mmseg.apis import init_model as init_segmentor
from mmseg.apis import inference_model as inference_segmentor
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
config_file = r'/mnt/ssd2/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py'

checkpoint_file=r"/mnt/ssd2/mmsegmentation/ckpts/mask2former_r50_8xb2-90k_cityscapes-512x1024_20221202_140802-ffd9d750.pth"
nuscenes_path = '/mnt/ssd2/nuscenes'
version = 'v1.0-trainval'
scene_name = 'scene-0061'  # Example scene
cam_name = 'CAM_FRONT'
lidar_name = 'LIDAR_TOP'
class_ids_of_interest = [0,1,12,13,14,15,16,17,18]  # Example class IDs of interest

# Initialize MMSegmentation model
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Initialize NuScenes dataset
nusc = NuScenes(version=version, dataroot=nuscenes_path, verbose=True)
scene_token = nusc.scene[0]['token']
# Retrieve a sample from the scene
scene = nusc.get('scene', scene_token)
sample = nusc.get('sample', scene['first_sample_token'])
# Get camera and LiDAR data
cam_data = nusc.get('sample_data', sample['data'][cam_name])
lidar_data = nusc.get('sample_data', sample['data'][lidar_name])

# Load image
img_path = os.path.join(nusc.dataroot, cam_data['filename'])
img = mmcv.imread(img_path)
# from PIL import Image
# img = Image.open(img_path)
# Perform semantic segmentation
result = inference_segmentor(model, img)
segmented_img = result.pred_sem_seg.data.cpu().numpy()
unique_classes = np.unique(segmented_img,return_counts=True)
print(unique_classes)
# Load point cloud
pc_path = os.path.join(nusc.dataroot, lidar_data['filename'])
pc = LidarPointCloud.from_file(pc_path)
print("--",pc.points.shape)
cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
pc.translate(np.array(cs_record['translation']))

# Second step: transform from ego to the global frame.
poserecord = nusc.get('ego_pose', lidar_data['ego_pose_token'])
pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
pc.translate(np.array(poserecord['translation']))

# Third step: transform from global into the ego vehicle frame for the timestamp of the image.
poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
pc.translate(-np.array(poserecord['translation']))
pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
pc.translate(-np.array(cs_record['translation']))
pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

# Fifth step: actually take a "picture" of the point cloud.
# Grab the depths (camera frame z axis points away from the camera).
depths = pc.points[2, :]
coloring = depths
points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

# Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
# Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
# casing for non-keyframes which are slightly out of sync.
mask = np.ones(depths.shape[0], dtype=bool)
mask = np.logical_and(mask, depths > 1.0)
mask = np.logical_and(mask, points[0, :] > 1)
mask = np.logical_and(mask, points[0, :] < img.shape[0] - 1)
mask = np.logical_and(mask, points[1, :] > 1)
mask = np.logical_and(mask, points[1, :] < img.shape[1] - 1)
points = points[:, mask]
coloring = coloring[mask]
print(points.shape)
# #plot
from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(9, 16))
ax.imshow(img)
ax.scatter(points[0, :], points[1, :], c=coloring, s=5)
ax.axis('off')
plt.show()

# Project LiDAR points to camera image
# cam_intrinsic = np.array(nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['camera_intrinsic'])
# lidar2cam_translation = np.array(nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['translation'])
# lidar2cam_rotation = np.array(nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])['rotation'])
# quat = pyquaternion.Quaternion(lidar2cam_rotation)
# lidar2cam_rotation = quat.transformation_matrix
# print(lidar2cam_translation.shape, lidar2cam_rotation.shape, cam_intrinsic.shape)
# # Create the 4x4 transformation matrix from LIDAR to camera
# lidar2cam_translation = np.array(lidar2cam_translation).reshape(1, 3)
# lidar2cam = np.eye(4)
# lidar2cam = lidar2cam_rotation
# lidar2cam[:3, 3] = lidar2cam_translation

# # Add a row of ones to the point cloud to handle the homogeneous coordinates
# pc_homogeneous = np.vstack((pc.points[:3, :], np.ones(pc.points.shape[1])))
# print("pc_homogeneous",pc_homogeneous.shape)    
# # Transform point cloud to camera coordinate system
# pc_cam_frame = lidar2cam @ pc_homogeneous
# segmented_img = segmented_img.squeeze()
# points = view_points(pc_cam_frame[:3, :], cam_intrinsic, normalize=True)
# points = points[:2, :]
# # Filter points based on segmentation result
# filtered_points = []
# xys = []
# for i in range(points.shape[1]):
#     x, y = int(points[0, i]), int(points[1, i])
#     if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
#         if segmented_img[y, x] in class_ids_of_interest:
#             filtered_points.append(pc.points[:, i])
#             xys.append((y, x))

# filtered_points = np.array(filtered_points)
# print(filtered_points.shape)
# # Visualize filtered points on image
# for point,xy in zip(filtered_points,xys):
#     # x, y = int(point[0]), int(point[1])
#     x, y = xy
#     if x < 0 or y < 0:
#         print(x, y)
#     img = cv2.circle(img, (x, y), 3, (0, 255, 0), 3)

# # Save or display the resulting image
# cv2.imshow('Filtered Points', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # # Optionally, save the image
# output_img_path = 'path/to/save/output_image.jpg'
# cv2.imwrite(output_img_path, img)
