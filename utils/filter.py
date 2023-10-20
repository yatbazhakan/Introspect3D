from abc import ABC, abstractmethod
from enum import Enum

    
class FilterStrategy(ABC):
    #Mode 0: Inside the shape
    #Mode 1: Outside the shape
    @abstractmethod
    def filter_pointcloud(self, data,mode):
        pass
    
    def filter_bounding_boxes(self, data,mode):
        pass
    
    
    
class RectangleFilter(FilterStrategy):
    def __init__(self,height,width,offset) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.offset = offset
        
    def filter_pointcloud(self, data, mode=0):
        # Rectangle filtering logic here
        pass
    
    def filter_bounding_boxes(self, data, mode=0):
        pass
    
class EllipseFilter(FilterStrategy):
    def __init__(self,a,b,offset) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.offset = offset
    def filter_pointcloud(self, data, mode=0):
        # Ellipse filtering logic here
        pass
    def filter_bounding_boxes(self, data, mode=0):
        pass
    
    

class FilterType(Enum):
    RECTANGLE = RectangleFilter
    ELLIPSE = EllipseFilter 
