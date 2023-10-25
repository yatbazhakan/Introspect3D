from __future__ import annotations
import numpy as np
from typing import Union, List
from math import cos, sin
import open3d as o3d
class BoundingBox:
    def __init__(self,
                 center: np.ndarray, 
                 dimensions: Union[np.ndarray,tuple], 
                 rotation:  Union[float, np.ndarray],
                 label: Union[str,int]) -> None:
        
        self.center = center
        self.dimensions = dimensions
        self.rotation = rotation
        self.type = label
        self.corners = None
        if isinstance(self.rotation, np.float32):
            self.calcukate_corners_from_prediction()
        else:
            self.calculate_corners()

    def rotate_points(self,points, R): # For now, migt move somewhere else
        return np.dot(points, R.T)
    
    def calculate_corners(self):
        dimensions_height, dimensions_width, dimensions_length = self.dimensions[0], self.dimensions[1], self.dimensions[2]
        R = self.rotation
        l_div_2 = dimensions_length / 2
        x_corners = [l_div_2, l_div_2, -l_div_2, -l_div_2, l_div_2, l_div_2, -l_div_2, -l_div_2]
        w_div_2 = dimensions_width / 2
        y_corners = [0, 0, 0, 0, -dimensions_height, -dimensions_height, -dimensions_height, -dimensions_height]
        z_corners = [w_div_2, -w_div_2, -w_div_2, w_div_2, w_div_2, -w_div_2, -w_div_2, w_div_2]

        corner_matrix = np.array([x_corners, y_corners, z_corners])
        self.corners = self.rotate_points(corner_matrix.T, R) + self.center

    def calcukate_corners_from_prediction(self):
        corners_list = []
        
        x, y, z = self.center
        dx, dy, dz = self.dimensions
        yaw = self.rotation
        # Step 1: Create local corner points
        local_corners = np.array([
            [-dx/2, -dy/2, -dz/2],  # corner 1
            [+dx/2, -dy/2, -dz/2],  # corner 2
            [+dx/2, +dy/2, -dz/2],  # corner 3
            [-dx/2, +dy/2, -dz/2],  # corner 4
            [-dx/2, -dy/2, +dz/2],  # corner 5
            [+dx/2, -dy/2, +dz/2],  # corner 6
            [+dx/2, +dy/2, +dz/2],  # corner 7
            [-dx/2, +dy/2, +dz/2],  # corner 8
        ])
        
        # Step 2: Rotate corners
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        rotated_corners = np.dot(local_corners, rotation_matrix.T)
        
        # Step 3: Translate corners
        translated_corners = rotated_corners + np.array([x, y, z])
        
        self.corners = np.array(translated_corners)

    def calculate_iou(self, box: BoundingBox) -> float:
        center1 = self.center
        dimensions1 = self.dimensions
        corners1 = self.corners
        # There might be an issue if the rotation is not a matrix
        #corners1 =  self.rotate_points(corners1, self.rotation) + center1

            
        center2 = box.center
        dimensions2 = box.dimensions
        corners2 = box.corners
        # Rotate and translate corners
        #corners2 = self.rotate_points(corners2, box.rotation) + center2

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
    
    def calculate_all_ious(self, boxes: List[BoundingBox]) -> List[float]:
        return [self.calculate_iou(box) for box in boxes]
    
    def find_max_iou_box(self, boxes: List[BoundingBox]) -> BoundingBox:
        #Function to return the box_idx with the highest iou, and iou value
        if len(boxes) == 0:
            return None, 0
        ious = self.calculate_all_ious(boxes)
        max_iou = max(ious)
        max_iou_idx = np.argmax(ious)
        return max_iou_idx, max_iou

    def transform_box(self,transform):
        dimensions_height, dimensions_width, dimensions_length = self.dimensions[0], self.dimensions[1], self.dimensions[2]
        if isinstance(self.rotation) == float:
            rotation_y = self.rotation
            R = np.array([[cos(rotation_y),0,sin(rotation_y)],[0,1,0],[-sin(rotation_y),0,cos(rotation_y)]])
        elif isinstance(self.rotation) == np.ndarray: #Might not be what I need
            R = self.rotation
        # TODO: This might be KITTI specific, moving this part to KITTI dataset for now, if same problem occurs in other datasets, I will move it to here.
        # l_div_2 = dimensions_length / 2
        # x_corners = [l_div_2, l_div_2, -l_div_2, -l_div_2, l_div_2, l_div_2, -l_div_2, -l_div_2]
        # w_div_2 = dimensions_width / 2
        # y_corners = [0, 0, 0, 0, -dimensions_height, -dimensions_height, -dimensions_height, -dimensions_height]
        # z_corners = [w_div_2, -w_div_2, -w_div_2, w_div_2, w_div_2, -w_div_2, -w_div_2, w_div_2]
        # corner_matrix = np.array([x_corners, y_corners, z_corners])
        # rotated_corners = np.matmul(R,corner_matrix)
        # translated_corners = rotated_corners + self.center.reshape(3,1)
        
        # I expect this will be done by the dataset classes and will be send as a parameter
        # transform = transform.reshape(3, 4)
        # transform = np.eye(4)  # Create a 4x4 identity matrix
        # transform[:3, :] = transform  # Replace the top-left 3x4 block
        # T_inv = np.linalg.inv(transform)
        
        # TODO: This part is only needed if the corners do not match with the transform matrix, need further guards.
        # Homogeneous_corners = np.ones((4,8))
        # Homogeneous_corners[:3,:] = translated_corners
        # translated_corners = np.matmul(transform,Homogeneous_corners)[:3,:]
        # real_center = np.mean(translated_corners,axis=1)
        # self.center = real_center
        # self.corners = translated_corners
        # self.rotation = o3d.geometry.get_rotation_matrix_from_xyz((0,rotation_y,0)) #Assuming rotation is only around y axis and given with float radian
        
    def is_point_inside(self,point):
        point_local = np.linalg.inv(self.rotation).dot((point[:3] - self.center))
        return np.all(np.abs(point_local) <= (self.dimensions / 2))
    
    

    def remove_points_inside_box(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        Remove points inside oriented bounding boxes from a point cloud.
        
        Parameters:
        - point_cloud: NumPy array of shape (N, 3), where N is the number of points.
        - obbs: List of Open3D OrientedBoundingBox objects.
        
        Returns:
        - filtered_point_cloud: NumPy array containing points outside the bounding boxes.
        """
        mask = np.ones(point_cloud.shape[0], dtype=bool)
        for i, point in enumerate(point_cloud):
            if self.is_point_inside_obb(point):
                mask[i] = False
        
        filtered_point_cloud = point_cloud[mask]  
        return filtered_point_cloud

    def get_open3d_oriented_boxes(self):
        obb = o3d.geometry.OrientedBoundingBox(center=self.center, R=self.rotation, extent=self.dimensions)
        return obb
    
    def __str__(self) -> str:
        st = ""
        st += "Center: " + str(self.center) + "\n"
        st += "Dimensions: " + str(self.dimensions) + "\n"
        st += "Rotation: " + str(self.rotation) + "\n"
        return st

