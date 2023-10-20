from abc import ABC, abstractmethod

class DrivingDataset(ABC):
    #As placeholder for now
    @abstractmethod
    def load_data(self):
        """Load dataset into memory."""
        pass

    @abstractmethod
    def process_data(self):
        """Process the data, including any calibration steps."""
        pass
