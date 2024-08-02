import numpy as np
import torch
from torch import nn
import pandas as pd
from utils.process import *
from glob import glob
import os
import torch.nn.functional as F

import pickle
class ActivationDataset:
    def __init__(self,config) -> None:
        self.config = config
        self.extension = config.get('extension','')  
        self.root_dir = config['root_dir']
        print("ROOT DIR: ",self.root_dir)
        self.classes = config['classes']
        self.is_multi_feature = config.get('is_multi_feature',False)
        self.is_caption = config.get('is_caption',False)
        self.is_sparse = config.get('is_sparse',False)
        if self.is_caption:
            self.caption_file = config.get('caption_file',None)
            with open(self.caption_file,'r') as f:
                self.captions = f.readlines()
            self.generic_captions = self.captions
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
        # if self.is_multi_feature: #Need to fix this extension issue
        self.labels['name'] = self.labels['name'].apply(lambda x: x.split('/')[-1].replace('.npy',''))
        # print(self.labels.head())

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
        if self.is_caption:
            self.generate_caption_dataset()
    def get_feature_paths(self):
        return sorted(glob(os.path.join(self.root_dir,'features', f'*{self.extension}')))
    def get_label_file(self):
        #*DEBUG
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
        feature_name = feature_path.split('/')[-1]
        # print(feature_name,feature_path,idx)
        # print(feature_name)
        if self.is_sparse:
            if self.is_multi_feature:
               tensor_feature  = torch.load(feature_path)
               tensor_feature = [tensor.to_dense() if tensor.is_sparse else tensor for tensor in tensor_feature]
            else:
                if self.layer == None:
                    tensor_feature = torch.load(feature_path)
                else:
                    # print("Loading sparse tensor")
                    # print(feature_path)
                    tensor_feature = torch.load(feature_path)
                    # print(tensor_feature)
                    tensor_feature =tensor_feature[int(self.layer)]
                if tensor_feature.is_sparse:
                    # print("Converting to dense")
                    tensor_feature = tensor_feature.to_dense()
        else:           
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
        # print(tensor_feature.shape)
        feature_name = feature_name.replace('.npy','')
        feature_name = feature_name.replace('.pkl','')
        feature_name = feature_name.replace('.pt','')
        label = self.get_label(feature_name)
        
        tensor_label = torch.LongTensor([label])
        # print(tensor_feature.shape,tensor_label.shape,feature_name)
        # print(tensor_label)
        # print(len(tensor_feature))
        # print(tensor_feature.shape,tensor_label,feature_name)
        return tensor_feature, tensor_label, feature_name
    def __len__(self):
        return len(self.feature_paths)
    def get_all_labels(self):
        if self.threshold == None:
            return self.labels['is_missed'].values
        map_values = self.labels[self.label_field].values
        bool_values = map_values > self.threshold
        return bool_values.astype(int)
    def quant_distance(self,distance):
        distance = float(distance)
        if distance < 5:
            return "<5m"
        elif distance < 10:
            return "5-10m"
        elif distance < 20:
            return "10-20m"
        elif distance < 40:
            return "20-40m"
        elif distance < 80:
            return "40-80m"
        else:
            return ">80m"
        
    def get_processed_caption(self,caption_details):
        mapper = {
            "LB":"left back",
            "LF":"left front",
            "RB":"right back",
            "RF":"right front"
        }
        file = caption_details[0]
        num_missed = int(caption_details[1])
        rest = caption_details[2:]
        for i in range(len(rest)):
            rest[i] = rest[i].replace('\n','')
            rest[i] = rest[i].replace(' ','')

        rest_set = set(rest)
 
        # # print(len(rest_set),num_missed,rest_set)
        # if num_missed > 0 and len(rest_set) < 4:
        #     processed_caption = str(num_missed) + " objects missed closest at "+ mapper[rest[0]]
        # elif num_missed > 0 and len(rest_set) == 4:
        #     processed_caption = str(num_missed) + " objects missed at the scene"
        if num_missed > 0:
            distance = rest[1]
            processed_caption = "objects missed closest at "+ mapper[rest[0]] +" " +self.quant_distance(distance)
        elif num_missed == 0 :
            processed_caption= "no object detections missed in the scene"
        return processed_caption
    def generate_caption_dataset(self):
        data_path = "/mnt/ssd1/test/"
        dataset_dict = {'x_features':[],'captions':[]}
        from tqdm.auto import tqdm
        print("Generating caption dataset")
        print("Length of feature paths: ",len(self.feature_paths))
        with tqdm(total=len(self.feature_paths)) as pbar:
            for i in range(len(self.feature_paths)):
                feature_path = self.feature_paths[i]
                feature_name = feature_path.split('/')[-1].replace(self.extension,'')
                print(feature_name)
                # if self.is_multi_feature:
                #     pickle_path = feature_path.replace('.npy','')
                #     with open(pickle_path,'rb') as f:
                #         feature = pickle.load(f)
                #     if type(self.layer) == list:
                #         tensor_feature = []
                #         for i in range(len(self.layer)):
                #             data = torch.from_numpy(feature[self.layer[i]])
                #             tensor_feature.append(data)
                # else:
                #     print("Not implemented for single feature")
                # ppc = tensor_feature[0]
                # lla = tensor_feature[2]
                # # print(ppc.shape,lla.shape)
                # reconstructed_map = ppc.max(dim=0).values.unsqueeze(0)
                # # print(reconstructed_map.shape)
                # #Single value decomposition on channels of ppc
                # # C,H,W = ppc.shape[0], ppc.shape[1],ppc.shape[2]
                # # ppc_flattened = ppc.view(C, -1)
                # # mean_centered = ppc_flattened - ppc_flattened.mean(dim=0)  # Subtract the mean of each column

                # # u, s,v = torch.linalg.svd(mean_centered, full_matrices=False)

                # # # Step 3: Reconstruct the map using the first singular value and vector
                # # reconstructed_map = s[0] * torch.ger(u[:, 0], v[:, 0])  # torch.ger produces the outer product
                # # reconstructed_map = reconstructed_map.unsqueeze(0)  
                # import matplotlib.pyplot as plt
                # # print(reconstructed_map.shape)
                # # plt.imshow(reconstructed_map.detach().cpu().numpy()[0, :, :])
                # # plt.show()
                # weight_ppc = 0.5  # Define the weight for ppc
                # weight_lla = 1 - weight_ppc  # Define the weight for lla
                # test_map = reconstructed_map.unsqueeze(0)
                # # print(test_map.shape,lla.shape)
                # ppc_resized = F.interpolate(test_map, size=(lla.shape[1], lla.shape[2]), mode='bilinear', align_corners=False)
                # blended = weight_ppc * ppc_resized + weight_lla * lla
                feature_name = feature_name.replace('.npy','')
                print("length generic captions: ",len(self.generic_captions))
                for i,cap in enumerate(self.generic_captions):
                    
                    caption_details = cap.split(',')
                    # print(caption_details)
                    file = caption_details[0].split('/')[-1].replace('.pcd.bin','')
                    if file in feature_name:
                        # print(i,cap)
                        processed_caption = self.get_processed_caption(caption_details)
                        print(processed_caption)
                        # print(processed_caption)
                        # torch.save(blended.detach().cpu(),os.path.join(data_path,'features',file+'.pt'))
                        dataset_dict['x_features'].append(os.path.join(data_path,'features',file+'.pt'))
                        dataset_dict['captions'].append(processed_caption)
                        
                pbar.update(1)
            name = 'caption_dataset_nus_v3.json'
            import json
            with open(os.path.join(data_path,name), "w") as outfile: 
                json.dump(dataset_dict, outfile)

            # Load the JSON data from the file
            with open(os.path.join(data_path,name), 'r') as json_file:
                data = json.load(json_file)

            # Open the corpus.txt file in write mode
            with open(os.path.join(data_path,'corpus.txt'), 'w') as text_file:
                # Iterate through each caption in the JSON data
                for caption in data.get('captions', []):
                    # Write each caption on a new line in the text file
                    text_file.write(caption + '\n')

            print("Captions have been written to corpus.txt")
                        # torch.save(dataset_dict,'caption_dataset_nus.pt')
class ActivationDatasetLegacy:
    def __init__(self) -> None:
        pass
    #Init to be used for legacy code if needed