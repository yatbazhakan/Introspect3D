from torch.utils.data import Dataset
from torchvision import transforms
from utils.data_utils import get_labels_from_json, get_labels_from_txt
import json
import torch
import pandas as pd
from ast import literal_eval
import os
import numpy as np
from tqdm.auto import tqdm
from glob import glob
import pickle

class BDD(Dataset):
    
    def __init__(self,root_dir,classes,mode='training',split=0):
        super().__init__()
        self.root_dir = root_dir
        self.classes = classes
        if mode == 'training': self.mode = "train"
        elif mode =='validation': self.mode = "val"
        else: self.mode="test"
        self.split = split
        self.image_directory_path = f"{self.root_dir}/images/100k/{self.mode}/"
        self.label_json_path = f"{self.root_dir}/labels/bdd100k_labels_images_{self.mode}.json"
        
        if self.split != -1:
            image_file = f"images_{split}.pkl"
            label_file = f"labels_{split}.pkl"
            self.path = os.path.join(root_dir,"split",mode)
            images = os.path.join(self.path,image_file)
            with open(images,"rb") as f:
                image_paths =pickle.load(f)

            labels = os.path.join(self.path,label_file)
            with open(labels,"rb") as f:
                label_paths =pickle.load(f)
        else:
            self.labels = get_labels_from_json(self.label_json_path,self.classes)
            self.image_paths = [self.image_directory_path+imgid for imgid in list(self.labels.keys())]
        self.dataset_dict = {image_id: {'image': self.image_directory_path+image_id,
                                        'label': label} for image_id,label in self.labels.items()}
    def get_labels_from_json(pth, classes):
        file = open(pth, 'r').read()
        json_file = json.loads(file)
        labels = {}
        for image in json_file:
            key = image['name']
            objs = []
            for obj in image['labels']:
                label = obj['category']
                if (label in classes):
                    box = tuple(obj['box2d'].values())
                    if (len(box) == 0):
                        print("WARNING",obj)
                    else:
                        objs.append({'bbox':tuple(box), 'label':label})
            if (len(objs) != 0):
                labels[key] = objs
        return labels
    def __len__(self):
        return len(self.dataset_dict.values())
    def __getitem__(self, index):
        image_id = list(self.dataset_dict.keys())[index]
        image_path = self.dataset_dict[image_id]['image']
        label = self.dataset_dict[image_id]['label']
        image = image_path
        return  image, label
    