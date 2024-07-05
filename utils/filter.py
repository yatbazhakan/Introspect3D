from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import Union, List
from utils.boundingbox import BoundingBox
from utils.pointcloud import PointCloud
from base_classes.base import FilterStrategy
class FilteringArea(Enum):
    INSIDE = 0
    OUTSIDE = 1 
class ObjectFilter(FilterStrategy):
    def __init__(self) -> None:
        """Initialize ObjectFilter with a list of bounding boxes.
        
        Args:
            bounding_boxes: A list of bounding boxes to filter points inside.
        """
        super().__init__()

    def filter_pointcloud(self, data: PointCloud, bounding_boxes: List[BoundingBox], mode: FilteringArea = FilteringArea.INSIDE):
        points = data.points  # Assuming data is an instance of PointCloud
        mask = np.ones(len(points), dtype=bool)
        
        for box in bounding_boxes:
            inside_box = self.is_inside(points, box)
            if mode == FilteringArea.INSIDE:
                mask = np.logical_and(mask, ~inside_box)
            elif mode == FilteringArea.OUTSIDE:
                mask = np.logical_and(mask, inside_box)

        return data.points[mask]
    def filter_bounding_boxes(self, data, mode):
        pass
    def is_inside(self, points: np.ndarray, box: BoundingBox) -> np.ndarray:
        """Check if points are inside the given bounding box.
        
        Args:
            points: The point cloud data.
            box: The bounding box to check against.

        Returns:
            A boolean array indicating whether each point is inside the bounding box.
        """
        x_min, y_min, z_min = np.min(box.corners, axis=0)
        x_max, y_max, z_max = np.max(box.corners, axis=0)

        inside_x = np.logical_and(points[:, 0] >= x_min, points[:, 0] <= x_max)
        inside_y = np.logical_and(points[:, 1] >= y_min, points[:, 1] <= y_max)
        inside_z = np.logical_and(points[:, 2] >= z_min, points[:, 2] <= z_max)

        return np.logical_and(np.logical_and(inside_x, inside_y), inside_z)
    def is_outside(self, **kwargs):
        pass
class EllipseFilter(FilterStrategy):
    """Implements ellipse-based filtering."""

    def __init__(self,
                 a: float,
                 b: float,
                 offset: Union[float, np.ndarray],
                 axis: int) -> None:
        """Initialize EllipseFilter with major and minor axes, and an offset.
        Args:
            a: The length of the major axis.
            b: The length of the minor axis.
            offset: The offset for the ellipse position.
        """
        super().__init__()

        if isinstance(offset, float) or isinstance(offset, int):
            offset_array = np.zeros(3)
            offset_array[axis] = offset
        else:
            offset_array = offset
        self.a = a
        self.b = b   
        self.offset = offset_array
        
    def filter_pointcloud(self, data: PointCloud, mode: FilteringArea = FilteringArea.INSIDE):
        points = data
        temp_points = points[:,:3] + self.offset #Types of points might be problem such as (N,4) or (N,5)
        x,y,z = temp_points[:,0],temp_points[:,1],temp_points[:,2] #What if I need to send yz, xz combinations
        if mode == FilteringArea.INSIDE:
            # print("Before filtering",data.shape)
            inside_ellipse = self.is_inside(x=x,y=y)
            # print("After filtering",data[inside_ellipse].shape)
            return data[inside_ellipse]
        elif mode == FilteringArea.OUTSIDE:
            outside_ellipse = self.is_outside(x=x,y=y)
            return data[outside_ellipse]
        
    def filter_bounding_boxes(self, data, mode: FilteringArea = FilteringArea.INSIDE):
        filtered_objects = []
        for box in data:

            corners = box.corners.copy()
            if corners.shape[0] != 8:
                corners = corners.T
            adjusted_corners= corners + self.offset
            if mode == FilteringArea.INSIDE:
                inside_ellipse = self.is_inside(x = adjusted_corners[:,0], y=adjusted_corners[:,1])
            elif mode == FilteringArea.OUTSIDE:
                inside_ellipse = self.is_outside(x= adjusted_corners[:,0], y = adjusted_corners[:,1])
            # print("Corners inside ellipse",np.sum(inside_ellipse))
            if np.sum(inside_ellipse) >= 4 and FilteringArea.INSIDE == mode:
                filtered_objects.append(box)
            elif np.sum(inside_ellipse) == 0 and FilteringArea.OUTSIDE == mode:
                filtered_objects.append(box)
        return filtered_objects
    
    def is_inside(self,**kwargs):
        x,y = kwargs['x'],kwargs['y']
        return (x**2 / self.a**2) + (y**2 / self.b**2) <= 1
    
    def is_outside(self,**kwargs):
        x,y = kwargs['x'],kwargs['y']
        return (x**2 / self.a**2) + (y**2 / self.b**2) > 1
    
     
class RectangleFilter(FilterStrategy):
    def __init__(self,height,width,offset) -> None:
        """Initialize RectangleFilter with dimensions and offset.
        
        Args:
            height: The height of the rectangle.
            width: The width of the rectangle.
            offset: The offset for the rectangle position.
        """
        super().__init__()
        self.height = height
        self.width = width
        self.offset = offset
        
    def filter_pointcloud(self, data, mode=0):
        # Rectangle filtering logic here
        pass
    
    def filter_bounding_boxes(self, data, mode=0):
        pass
    
    def is_inside(self,**kwargs):
        pass
    
    def is_outside(self,**kwargs):
        pass   
class NoFilter(FilterStrategy):
    def __init__(self,**kwargs) -> None:
        super().__init__()
    def filter_pointcloud(self, data, mode=0):
        return data
    def filter_bounding_boxes(self, data, mode=0):
        return data
    def is_inside(self,**kwargs):
        return True
    def is_outside(self,**kwargs):
        return False
#TODO: Implement a grid filter to split pc into grids
class GridFilter(FilterStrategy):
    def __init__(self,**kwargs) -> None:
        super().__init__()   
    def filter_pointcloud(self, data, mode=0):
        # Rectangle filtering logic here
        pass
    def filter_bounding_boxes(self, data, mode=0):
        pass
    
    def is_inside(self,**kwargs):
        pass
    
    def is_outside(self,**kwargs):
        pass  
class FilterType(Enum):
    RECTANGLE = RectangleFilter
    ELLIPSE = EllipseFilter
    OBJECT = ObjectFilter
    NONE = NoFilter
