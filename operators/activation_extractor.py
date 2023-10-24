from base_classes.base import Operator
from utils.utils import load_detection_model
from utils.factories import DatasetFactory
from utils.activations import Activations
import os 
class ActivationExractionOperator(Operator):
    def __init__(self, config):
        self.config = config.extraction
        self.model = load_detection_model(self.config)
        self.dataset = DatasetFactory().get(**self.config['dataset'])
        self.method = self.config['method']
        self.activation = Activations(self.model, self.method)
        os.makedirs(self.config['method']['save_dir'], exist_ok=True)
    def execute(self, **kwargs):
        verbose = kwargs.get('verbose', False)
        if verbose:
            print("Extracting activations")
        for i in range(len(self.dataset)):
            # print(i)
            cloud, label, file_name = self.dataset[i]
            print(cloud.points.shape)
            self.activation(cloud.points,file_name)
            