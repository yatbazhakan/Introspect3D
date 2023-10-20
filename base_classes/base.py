from abc import ABC, abstractmethod
from utils.filter import FilteringArea
#TODO: If scales more, might be better to separate these into indivudal files
class DrivingDataset(ABC):
    #As placeholder for now
    @abstractmethod
    def read_label(self, **kwargs):
        """Load dataset into memory."""
        pass

    @abstractmethod
    def process_data(self, **kwargs):
        """Process the data, including any calibration steps."""
        pass



class FilterStrategy(ABC):
    """Abstract base class for different filtering strategies."""

    @abstractmethod
    def filter_pointcloud(self, data,mode: FilteringArea):
        """Apply filtering to a point cloud based on a specified mode.
        
        Args:
            data: The point cloud data to be filtered.
            mode: The filtering mode, either INSIDE or OUTSIDE.
        """
        pass
    
    @abstractmethod
    def filter_bounding_boxes(self, data,mode: FilteringArea):
        """Apply filtering to bounding boxes based on a specified mode.
        
        Args:
            data: The bounding box data to be filtered.
            mode: The filtering mode, either INSIDE or OUTSIDE.
        """
        pass
    
    @abstractmethod
    def is_inside(self,**kwargs):
        pass
        
    @abstractmethod
    def is_outside(self,**kwargs):
        pass
    