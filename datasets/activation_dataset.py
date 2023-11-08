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
        self.labels['name'] = self.labels['name'].astype(str)
        #remove if any leading path is there in self labels['name']
        self.labels['name'] = self.labels['name'].apply(lambda x: x.split('/')[-1].replace('.npy',''))
        #fill names with leading zeros to make them 6 digits
        if self.config['name'] == 'kitti':
            self.labels['name'] = self.labels['name'].apply(lambda x: x.zfill(6))
        if len(self.labels) != len(self.feature_paths):
            temp_paths = []
            for path in self.feature_paths:
                name = path.split('/')[-1].replace('.npy','')
            
                # print(type(name),type(self.labels['name'].values[-1]),name in self.labels['name'].values)
                if name in self.labels['name'].values:
                    
                    temp_paths.append(path)
            self.feature_paths = temp_paths
            print("Feature paths and labels are not equal, some features are missing")
            print(len(self.feature_paths),len(self.labels))

    def get_feature_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'features', '*.npy')))
    def get_label_file(self):
        return os.path.join(self.root_dir,self.config["label_file"])
    def get_label(self,idx):
        # name_from_idx = int(self.feature_paths[idx].split('/')[-1].replace('.npy',''))
        # print(idx,self.labels['name'].values)
        # print(idx in self.labels['name'].values)
        label = self.labels[self.labels['name']==idx]['is_missed'].values[0]
        label = 1 if label else 0
        return label
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = feature_path.split('/')[-1].replace('.npy','')
        feature = np.load(feature_path)
        label = self.get_label(feature_name)
        tensor_feature = torch.from_numpy(feature)
        tensor_label = torch.LongTensor([label])
        # print(tensor_label)
        return tensor_feature, tensor_label, feature_name
    def __len__(self):
        return len(self.feature_paths)
    def get_all_labels(self):
        return self.labels['is_missed'].values