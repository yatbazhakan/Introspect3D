#CUrrently implemented for custom numpy files for pointclouds, it is extremely custom
import numpy as np
import os
import pandas as pd
from glob import glob
import torch
from torch.utils.data import Dataset
from utils.pointcloud import PointCloud
from registries.dataset_registry import dataset_registry
@dataset_registry.register('custom')
class CustomDataset(Dataset):
    def __init__(self, **kwargs):
        self.data_dir = kwargs.get('root_dir',"")
        self.transform = kwargs.get('transform', None)
        self.label_file = kwargs.get('label_file', "labels.csv")
        self.data_paths = glob(os.path.join(self.data_dir,'lidar' ,"*.npy"))
        self.labels = pd.read_csv(os.path.join(self.data_dir, self.label_file))
        columns_to_convert = ['error_annotation', 'no_error_annotation', 'error_annotation_filtered', 'no_error_annotation_filtered']
        self.labels[columns_to_convert] = self.labels[columns_to_convert].astype(int)
        # print(self.labels.head())
    def __len__(self):
        return len(self.data_paths)
    def get_labels(self,file_path):
        base_name = os.path.basename(file_path)
        # print(base_name)
        label = self.labels[self.labels['file_name'] == base_name]['error_annotation_filtered']
        # print(label)
        return 1 if label.values[0]=="True" else 0
    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        label = self.labels.iloc[idx]
        if self.transform:
            data = self.transform(data)
        items = {}
        items['pointcloud'] = PointCloud(data)
        items['labels'] = self.labels['error_annotation_filtered'].iloc[idx].item()
    
        items['file_name'] = self.data_paths[idx]
        return items

    def get_all_labels(self):
        return self.labels['error_annotation_filtered'].values
    