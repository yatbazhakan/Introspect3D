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
            # Check if any corner point is inside the ellipse
            # print(box)
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


class FilterType(Enum):
    RECTANGLE = RectangleFilter
    ELLIPSE = EllipseFilter 
    NONE = NoFilter
