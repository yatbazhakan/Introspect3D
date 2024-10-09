from datasets.kitti import Kitti2D, Kitti3D
from datasets.nuscenes import NuScenesDataset
from datasets.activation_dataset import ActivationDataset
from datasets.custom_dataset import CustomDataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from glob import glob
from base_classes.base import Factory
from enum import Enum
import yaml


#Is it worth moving model generation here?
class ModelFactory(Factory):
    def __init__(self) -> None:
        pass
    
class DatasetFactory(Factory):
    def load_defaults(self):
        path = 'configs/defaults.yaml'
        with open(path) as file:
           self.defaults = yaml.load(file, Loader=yaml.FullLoader)
        self.defaults = self.defaults['datasets']
    def __init__(self):
        self.load_defaults()
        self.datasets = {}
        for name in self.defaults:
            self.datasets[name] = name

    def get(self,name,**kwargs):
        return eval(self.datasets[name])(**kwargs)
    
