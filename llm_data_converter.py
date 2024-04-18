from utils.factories import DatasetFactory
from utils.config import Config
from definitions import ROOT_DIR
import os
import random
import torch
captions = [
    "Missing objects on both sides",
    "Missing objects on the right side",
    "Missing objects on the left side",
    "Missing object on the right side",
    "Missing object on the left side",
    "No missing objects",
]
if __name__ == "__main__":
    # config_path = "configs/wmg-pc/yolov8_kitti.yaml"
    # config = Config(os.path.join(config_path)).introspection
    # dataset = DatasetFactory().get(**config['dataset'])
    # dataset_test = {'features':[], 'captions':[]}
    # for i in range(len(dataset)):
    #     tensor_data, tensor_label, feature_name = dataset[i]
    #     flattened = tensor_data.flatten(start_dim=1)
    #     num = random.randint(0, len(captions)-1)
    #     dataset_test['features'].append(flattened)
    #     dataset_test['captions'].append(captions[num])
    # torch.save(dataset_test, os.path.join(ROOT_DIR, 'test_data.pt'))
    config_path = "configs/wmg-pc/centerpoint_nus_caption_blended.yaml"
    config = Config(os.path.join(config_path)).introspection
    dataset = DatasetFactory().get(**config['dataset'])


