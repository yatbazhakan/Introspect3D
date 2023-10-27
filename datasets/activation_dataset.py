import numpy as np
import torch
from torch import nn
import pandas as pd
from utils.process import *
from glob import glob
import os

class ActivationDataset:
    def __init__(self,config) -> None:
        self.config = config
        self.root_dir = config['root_dir']
        self.classes = config['classes']
        self.feature_paths = self.get_feature_paths()
        self.label_file = self.get_label_file()
        self.labels = pd.read_csv(self.label_file)

    def get_feature_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'features', '*.npy')))
    def get_label_file(self):
        return os.path.join(self.root_dir,self.config["label_file"])
    def get_label(self,idx):
        label = self.label_file['is_missed'][idx]
        return label
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = feature_path.split('/')[-1].replace('.npy','')
        feature = np.load(feature_path)
        label = self.get_label(feature_path)
        tensor_feature = torch.from_numpy(feature)
        tensor_label = torch.tensor(label)
        return tensor_feature, tensor_label, feature_name
    def __len__(self):
        return len(self.feature_paths)
    def get_all_labels(self):
        return self.labels['is_missed'].values