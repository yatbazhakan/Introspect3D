#CUrrently implemented for custom numpy files for pointclouds, it is extremely custom
import numpy as np
import os
import pandas as pd
from glob import glob
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, config):
        self.data_dir = config.get('data_dir',"")
        self.transform = config.get('transform', None)
        self.label_file = config.get('label_file', "labels.csv")
        self.data_paths = glob(os.path.join(self.data_dir, "*.npy"))
        self.labels = pd.read_csv(os.path.join(self.data_dir, self.label_file))
    def __len__(self):
        return len(self.data_paths)
    def get_labels(self,file_path):
        label = self.labels[self.labels['file_path'] == file_path]['error_annotation_filtered']
        return 1 if label=="True" else 0
    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        label = self.labels.iloc[idx]
        if self.transform:
            data = self.transform(data)
        items = {}
        items['point_cloud'] = torch.tensor(data).float()
        items['label'] = torch.tensor(self.get_labels(self.data_paths[idx]),dtype=torch.long)
        items['file_path'] = self.data_paths[idx]
        return items

    