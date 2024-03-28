import numpy as np
import torch
from torch import nn
import pandas as pd
from utils.process import *
from glob import glob
import os
import pickle
class ActivationDataset:
    def __init__(self,config) -> None:
        self.config = config
        self.extension = config.get('extension','')  
        self.root_dir = config['root_dir']
        self.classes = config['classes']
        self.is_multi_feature = config.get('is_multi_feature',False)
        self.feature_paths = self.get_feature_paths()
        print("Number of features found: ",len(self.feature_paths), " in ",self.feature_paths[:5])
        self.label_file = self.get_label_file()
        self.label_field = config['label_field']
        self.layer = config.get('layer',None)
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
        print(len(self.feature_paths),len(self.labels))

    def get_feature_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'features', f'*{self.extension}')))
    def get_label_file(self):
        return os.path.join(self.root_dir,self.config["label_file"])
    def get_label(self,idx):
        # name_from_idx = int(self.feature_paths[idx].split('/')[-1].replace('.npy',''))
        # print(idx,self.labels['name'].values)
        # print(idx in self.labels['name'].values)
        # idx = str(idx).replace('.npy','')
        # idx = str(idx).replace('.pkl','')
        if self.threshold == None:
            label = self.labels[self.labels['name']==idx]['is_missed'].values[0]
            label = 1 if label else 0
            return label
        else:
            if type(self.label_field) == str:
                label = self.labels[self.labels['name']==idx][self.label_field].values[0]
                label = 1 if label > self.threshold else 0
                return label
            elif type(self.label_field) == list: #Assuming this is to calculate missed ratio
                missed_objects = self.labels[self.labels['name']==idx][self.label_field[0]].values[0]
                total_objects = self.labels[self.labels['name']==idx][self.label_field[1]].values[0]
                missed_ratio = missed_objects/total_objects
                print(missed_ratio)
                label = 1 if missed_ratio < self.threshold else 0
                return label
            else:
                print("Label field is not recognized")
        # label = self.labels[self.labels['name']==idx][self.label_field].values[0]
        # label = 1 if label > self.threshold else 0
        # return label
    def __getitem__(self, idx):
        feature_path = self.feature_paths[idx]
        feature_name = feature_path.split('/')[-1].replace(self.extension,'')
        # print(feature_name)
        if self.is_multi_feature:
            pickle_path = feature_path.replace('.npy','')
            with open(pickle_path,'rb') as f:
                feature = pickle.load(f)
            #making the grouping agnostic from the number of layers selected
            # print(first.shape,second.shape,third.shape)
            if type(self.layer) == list:
                tensor_feature = []
                for i in range(len(self.layer)):
                    data = torch.from_numpy(feature[self.layer[i]])
                # Handled in the model, will be changed to sort out here as well
                #     if i == 0:
                #         data = torch.from_numpy(feature[self.layer[i]])
                #     elif i == 1:
                #         data = torch.from_numpy(feature[self.layer[i]])
                #     elif i == 2:
                #         data = torch.from_numpy(feature[self.layer[i]])
                    tensor_feature.append(data)
    
            # tensor_feature = [first,second,third]        
        else:
            if self.layer == None:
                feature = np.load(feature_path)
                tensor_feature = torch.from_numpy(feature)
            else:
                pickle_path = feature_path.replace('.npy','')
                # print("----",pickle_path)
                with open(pickle_path,'rb') as f:
                    feature = pickle.load(f)
                tensor_feature = torch.from_numpy(feature[int(self.layer)])
            # print(tensor_feature.shape)
        # print(feature_name)
        feature_name = feature_name.replace('.npy','')
        label = self.get_label(feature_name)
        
        tensor_label = torch.LongTensor([label])
        # print(tensor_feature.shape,tensor_label.shape,feature_name)
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