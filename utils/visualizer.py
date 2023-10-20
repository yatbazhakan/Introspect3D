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
        lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

        # Create a LineSet object
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box.corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])

        return line_set
