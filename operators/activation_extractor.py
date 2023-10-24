from base_classes.base import Operator
from utils.utils import load_detection_model
from utils.factories import DatasetFactory
from utils.activations import Activations
class ActivationExractionOperator(Operator):
    def __init__(self, config):
        self.model = load_detection_model(config.extraction)
        self.dataset = DatasetFactory().get(config.extraction)
        self.method = config.extraction['method']
        self.activation = Activations(self.model, self.method)
        
    def execute(self, **kwargs):
        verbose = kwargs.get('verbose', False)
        if verbose:
            print("Extracting activations")
        for i in range(len(self.dataset)):
            cloud, label, file_name = self.dataset[i]
            self.activation(cloud,file_name)
            