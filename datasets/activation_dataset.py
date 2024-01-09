import numpy as np
import torch
from torch import nn
import pandas as pd
from utils.process import *
from glob import glob
import os
import pickle
class ActivationDataset:
    def __init__(self,config,extension ="") -> None:
        self.config = config
        self.extension = extension  
        self.root_dir = config['root_dir']
        self.classes = config['classes']
        self.is_multi_feature = config.get('is_multi_feature',False)
        self.feature_paths = self.get_feature_paths()
        self.label_file = self.get_label_file()
        self.label_field = config['label_field']
        self.threshold = config.get('threshold',None)
        print("Threshold is ",self.threshold)
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
                name = path.split('/')[-1].replace(self.extension,'')
            
                # print(type(name),type(self.labels['name'].values[-1]),name in self.labels['name'].values)
                if name in self.labels['name'].values:
                    
                    temp_paths.append(path)
            self.feature_paths = temp_paths
            print("Feature paths and labels are not equal, some features are missing")
            print(len(self.feature_paths),len(self.labels))

    def get_feature_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'features', f'*{self.extension}')))
    def get_label_file(self):
        return os.path.join(self.root_dir,self.config["label_file"])
    def get_label(self,idx):
        # name_from_idx = int(self.feature_paths[idx].split('/')[-1].replace('.npy',''))
        # print(idx,self.labels['name'].values)
        # print(idx in self.labels['name'].values)
        if self.threshold == None:
            label = self.labels[self.labels['name']==idx]['is_missed'].values[0]
            label = 1 if label else 0
            return label
        label = self.labels[self.labels['name']==idx][self.label_field].values[0]
        label = 1 if label > self.threshold else 0
        return label
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = feature_path.split('/')[-1].replace(self.extension,'')
        # print(feature_name)
        if self.is_multi_feature:
            pickle_path = feature_path.replace('.npy','')
            with open(pickle_path,'rb') as f:
                feature = pickle.load(f)
            first = torch.from_numpy(feature[0])
            second = torch.from_numpy(feature[1])
            third = torch.from_numpy(feature[2])
            tensor_feature = [first,second,third]        

        else:
            feature = np.load(feature_path)
            tensor_feature = torch.from_numpy(feature)
        label = self.get_label(feature_name)
        
        tensor_label = torch.LongTensor([label])

        # print(tensor_label)
        return tensor_feature, tensor_label, feature_name
    def __len__(self):
        return len(self.feature_paths)
    def get_all_labels(self):
        if self.threshold == None:
            return self.labels['is_missed'].values
        map_values = self.labels[self.label_field].values
        bool_values = map_values > self.threshold
        return bool_values.astype(int)
    
class ActivationDatasetLegacy:
    def __init__(self) -> None:
        pass
    #Init to be used for legacy code if needed