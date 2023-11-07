import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils.boundingbox import BoundingBox
from utils.pointcloud import PointCloud
from typing import Union, List
from enum import Enum
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
    
class Visualizer:
    def __init__(self) -> None:
        pass
    def set_custom_view(self,vis):
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

    def create_oriented_bounding_boxes(self,
                                       box: BoundingBox,
                                       offset:Union[float, np.ndarray],
                                       axis:Union[int, np.ndarray],
                                       color:Union[tuple, np.ndarray] = Colors.BLUE) -> o3d.geometry.OrientedBoundingBox:
            
            if isinstance(offset, float):
                offset_array = np.zeros(3)
                offset_array[axis] = offset
            elif isinstance(offset, np.ndarray):
                offset_array = offset
                
            center = box.center
            dims = box.dimensions
            R = box.orientation

            center += offset_array
            R_empty = np.eye(3)
            obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=dims)
            obb.color = (1, 0, 0)  # Red color
            return obb
        
    def create_line_set_bounding_box(self,
                                       box: BoundingBox,
                                       offset:Union[float, np.ndarray],
                                       axis:Union[int, np.ndarray],
                                       color:Union[tuple, np.ndarray] = Colors.BLUE) -> o3d.geometry.LineSet:

        if box.corners.shape[0] != 8:
            temp_corners = box.corners.T
        else:
            temp_corners = box.corners
        o3d_box = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(temp_corners))

            

        o3d_box.color = color
        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d_box)
        return line_set
    def create_pcd_from_points(self,points,max_distance=True):
        if max_distance:
            max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))
        else:
            max_distance = None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(self.compute_colors_from_distance(points,max_distance))
        return pcd
    def compute_colors_from_distance(self,points,max_distance):
        #If no disatance given return all black
        if max_distance==None:
            return np.zeros((points.shape[0],3))
        distances = np.linalg.norm(points[:, :3], axis=1)
        normalized_distances = distances / max_distance
        return plt.cm.jet(normalized_distances)[:,:3]
    def visualize(self,**kwargs):
        cloud = kwargs['cloud']
        gt_boxes = kwargs['gt_boxes']
        pred_boxes = kwargs['pred_boxes']
        #print(cloud.shape)
        cloud = self.create_pcd_from_points(cloud)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        self.set_custom_view(visualizer)
        visualizer.add_geometry(cloud)
        for box in gt_boxes:
            # print(type(box))
            print(box.type)
            visualizer.add_geometry(self.create_line_set_bounding_box(box,0,0,Colors.GREEN.value))
        for box in pred_boxes:
            # print(type(box))
            visualizer.add_geometry(self.create_line_set_bounding_box(box,0,0,Colors.RED.value))
        
        visualizer.run()
        visualizer.destroy_window()