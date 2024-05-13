import numpy as np
import torch
from torch import nn
import pandas as pd
from utils.process import *
from glob import glob
import os
import pickle
import pdb
from numpy import array, float32
class ActivationDataset:
    def __init__(self,config) -> None:
        self.config = config
        self.extension = config.get('extension','')  
        self.root_dir = config['root_dir']
        #self.classes = config['classes']
        self.is_multi_feature = config.get('is_multi_feature',False)
        self.feature_paths = self.get_feature_paths()
        print("Number of features found: ",len(self.feature_paths), " in ",self.feature_paths[:5])
        self.label_file = self.get_label_file()
        self.labels = pd.read_csv(self.label_file)
        self.labels.head()
        self.gt_dist, self.pred_dist = self.get_de_error(self.labels['gt_lead'].values, self.labels['pred_lead'].values)
        self.layer = config.get('layer',None)
        self.threshold = config.get('threshold',None)
        print("Threshold is ",self.threshold)
        self.labels['name'] = self.labels['name'].astype(str)
        #remove if any leading path is there in self labels['name']
        # if self.is_multi_feature: #Need to fix this extension issue
        self.labels['name'] = self.labels['name'].apply(lambda x: x.split('/')[-1].replace('.npy',''))
        #fill names with leading zeros to make them 6 digits
        if self.config['name'] == 'kitti':
            self.labels['name'] = self.labels['name'].apply(lambda x: x.zfill(6))
        if len(self.labels) != len(self.feature_paths):
            temp_paths = []
            for path in self.feature_paths:
                name = path.split('/')[-1].replace(self.extension,'')
                # print(name)
                # print(type(name),type(self.labels['name'].values[-1]),name in self.labels['name'].values)
                if name in self.labels['name'].values:
                    
                    temp_paths.append(path)
            self.feature_paths = temp_paths
            print("Feature paths and labels are not equal, some features are missing")
            # print(len(self.feature_paths),len(self.labels))
        print(len(self.feature_paths),len(self.labels))

    def get_de_error(self, gt_bboxes, pred_bboxes, filter_boundry = 100):
        gt_dists = []
        pred_dists = []
        for gt_bbox, pred_bbox in zip(gt_bboxes, pred_bboxes):    
            gt_bbox = eval(gt_bbox)
            pred_bbox = eval(pred_bbox)
            if len(gt_bbox) > 1 or len(pred_bbox) > 1:
                print(gt_bbox, pred_bbox, len(gt_bbox), len(pred_bbox))
                raise ValueError("Only one bounding box is expected")
            gt_exists = len(gt_bbox) != 0
            pred_exists = len(pred_bbox) != 0
            if gt_exists and pred_exists: # True Positive
                gt_dist = gt_bbox[0][1]
                pred_dist = pred_bbox[0][1]
            elif not gt_exists and not pred_exists: # True Negative
                gt_dist = filter_boundry
                pred_dist = filter_boundry
            elif not gt_exists and pred_exists: # False Positive
                gt_dist = filter_boundry
                pred_dist = pred_bbox[0][1]
            elif gt_exists and not pred_exists: # False Negative
                gt_dist = gt_bbox[0][1]
                pred_dist = filter_boundry
            gt_dists.append(gt_dist)
            pred_dists.append(pred_dist)
        gt_dists = np.array(gt_dists)
        pred_dists = np.array(pred_dists)
        return gt_dists, pred_dists
    def get_feature_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'features', f'*{self.extension}')))
    def get_label_file(self):
        return os.path.join(self.root_dir,self.config["label_file"])
    def get_label(self,idx):
        itr = np.nonzero(self.labels['name'] == idx)[0][0]
        return np.abs(self.gt_dist[itr] - self.pred_dist[itr])/100
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
        
        tensor_label = torch.FloatTensor([label])

        # print(tensor_feature.shape,tensor_label.shape,feature_name)
        # print(tensor_label)
        return tensor_feature, tensor_label, feature_name
    def __len__(self):
        return len(self.feature_paths)
    def get_all_labels(self):
        return np.abs(self.gt_dist - self.pred_dist)/100
    
class ActivationDatasetLegacy:
    def __init__(self) -> None:
        pass
    #Init to be used for legacy code if needed