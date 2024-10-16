from abc import ABC, abstractmethod
#TODO: If scales more, might be better to separate these into indivudal files
class DrivingDataset(ABC):
    #As placeholder for now
    @abstractmethod
    def read_labels(self, **kwargs):
        """Load dataset into memory."""
        pass

    @abstractmethod
    def process_data(self, **kwargs):
        """Process the data, including any calibration steps."""
        pass

#Not sure if needed but for consistency will not use for now
class ErrorDataset(ABC):
    @abstractmethod
    def read_labels(self, **kwargs):
        """Load dataset into memory."""
        pass

    @abstractmethod
    def read_data(self, **kwargs):
        """Load activation paths."""
        pass

    @abstractmethod
    def process_data(self, **kwargs):
        """Process the data, including any calibration steps."""
        pass
class FilterStrategy(ABC):
    """Abstract base class for different filtering strategies."""

    @abstractmethod
    def filter_pointcloud(self, data,mode):
        """Apply filtering to a point cloud based on a specified mode.
        
        Args:
            data: The point cloud data to be filtered.
            mode: The filtering mode, either INSIDE or OUTSIDE.
        """
        pass
    
    @abstractmethod
    def filter_bounding_boxes(self, data,mode):
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
    
class Operator(ABC):
    @abstractmethod
    def execute(self, **kwargs):
        pass
    
class Factory(ABC):
    @abstractmethod
    def get(self,name, **kwargs):
        pass


class ActivationProcessor(ABC):
    @abstractmethod
    def process(self, **kwargs):
        pass