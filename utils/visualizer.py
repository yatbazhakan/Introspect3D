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
camera_parameters = {
    "field_of_view": 60.0,
    "lookat": [2.0621228213350746, 13.9162883221339, 1.8207356271047817],
    "up": [-0.032179561539771549, 0.39573627678482143, 0.91780023700999092],
    "front": [-0.021788454380564679, -0.91833534235703385, 0.39520306455504062],
    "zoom": 0.079999999999999613
}

class Visualizer:
    def __init__(self) -> None:
        pass
    def set_custom(self,vis):
        vc = vis.get_view_control()
        vc.set_lookat([2.0621228213350746, 13.9162883221339, 1.8207356271047817])
        vc.set_up([-0.032179561539771549, 0.39573627678482143, 0.91780023700999092])
        vc.set_front([-0.021788454380564679, -0.91833534235703385, 0.39520306455504062])
        vc.set_zoom(0.08) 
    def set_custom_view(self,vis):
        
        ctr = vis.get_view_control()
        
        # Create an extrinsic matrix for camera placement
        extrinsic = np.eye(4)
        extrinsic[0:3, 3] = [-10, 0, 30]  # Set camera position (x, y, z)
        
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
        opt.background_color = np.asarray([0.0, 0.0, 0.0])
        opt.point_size = 2.0

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
    def create_pcd_from_points(self,points,max_distance=True,colors=None) -> o3d.geometry.PointCloud:
        if max_distance:
            max_distance = np.max(np.linalg.norm(points[:, :3], axis=1))
        else:
            max_distance = None
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if colors is not None:
            print("Colors are given")
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(self.compute_colors_from_distance(points,None))
        return pcd
    def compute_colors_from_distance(self,points,max_distance):
        #If no disatance given return all black
        if max_distance==None:
            return np.ones((points.shape[0],3)) * [0.56470588235, 0.93333333333, 0.56470588235]
        distances = np.linalg.norm(points[:, :3], axis=1)
        normalized_distances = distances / max_distance
        return plt.cm.jet(normalized_distances)[:,:3]
    def visualize(self,**kwargs):
        cloud = kwargs.get('cloud', [])
        outside_cloud = kwargs.get('outside_cloud', None)
        gt_boxes = kwargs.get('gt_boxes', [])
        pred_boxes = kwargs.get('pred_boxes', [])
        colors = kwargs.get('colors', {})
        #print(cloud.shape)
        cloud = self.create_pcd_from_points(cloud,colors= colors.get('inside',None))

        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        self.set_custom_view(visualizer)
        visualizer.add_geometry(cloud)
        if outside_cloud is not None:
            print("Outside cloud is given")
            colors = colors.get('outside',None)
            if colors is not None:
                outside_cloud = self.create_pcd_from_points(outside_cloud,colors=colors)
            else:
                cols =  np.full((outside_cloud.shape[0],3),[0,0,1])
                outside_cloud = self.create_pcd_from_points(outside_cloud,colors=cols )
            visualizer.add_geometry(outside_cloud)
        for box in gt_boxes:
            # print(type(box))
            print(box.type)
            visualizer.add_geometry(self.create_line_set_bounding_box(box,0,0,Colors.GREEN.value))
        for box in pred_boxes:
            # print(type(box))
            visualizer.add_geometry(self.create_line_set_bounding_box(box,0,0,Colors.RED.value))
        
        visualizer.run()
        visualizer.destroy_window()
    def visualize_save(self, cloud, gt_boxes, pred_boxes, save_path=None, resolution=(1920, 1080)):
        # This function now has parameters for the point cloud, ground truth boxes, predicted boxes,
        # an optional path to save the screenshot, and an optional resolution for the screenshot.
        cloud = self.create_pcd_from_points(cloud, max_distance=False)
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window(width=resolution[0], height=resolution[1])
        self.set_custom_view(visualizer)
        visualizer.add_geometry(cloud)

        for box in gt_boxes:
            visualizer.add_geometry(self.create_line_set_bounding_box(box,0,0, Colors.GREEN.value))
        for box in pred_boxes:
            visualizer.add_geometry(self.create_line_set_bounding_box(box,0,0, Colors.RED.value))

        if save_path:
            # Adjust the viewpoint before capturing the screenshot
            visualizer.poll_events()
            visualizer.update_renderer()
            # Capture the screenshot
            image_float = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
            plt.imsave(save_path, np.asarray(image_float), dpi=1)  # dpi=1 means save the image at the resolution given by width and height

        visualizer.run()
        visualizer.destroy_window()