from datasets.kitti import Kitti2D, Kitti3D
from datasets.nuscenes import NuScenesDataset
from datasets.activation_dataset import ActivationDataset
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from glob import glob
import open3d as o3d

import numpy as np
import pandas as pd
import cv2
from base_classes.base import Factory
try:
    import mmdet3d
    from mmcv.transforms.base import BaseTransform
    from mmengine.registry import TRANSFORMS, VISUALIZERS
    from mmengine.structures import InstanceData
    from mmdet3d.utils import register_all_modules
    from mmdet3d.apis import inference_detector, init_model
    from mmdet3d.evaluation.metrics.nuscenes_metric import NuScenesMetric

    from mmdet3d.structures import Det3DDataSample, LiDARInstance3DBoxes
    from mmdet3d.visualization import Det3DLocalVisualizer
    from mmdet3d.structures import LiDARInstance3DBoxes
except:
    print("DIfferent environment, some packages will be missing")
    pass
from enum import Enum
import yaml



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
    
