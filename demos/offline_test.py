from datasets.activation_dataset import ActivationDataset
import numpy as np
from modules import *
from utils.utils import generate_model_from_config
import torch
import torch.nn as nn
from modules.custom_networks import GenericInjection
import pandas as pd
config = {
      'root_dir': '/media/wmg-5gcat/Co-op Autonomy 2/Hakan/custom_dataset/vehicle_centerpoint_activations_aggregated_raw',
      'label_file': 'vehicle_centerpoint_labels_aggregated_raw_filtered.csv',
      'classes': ['No Error', 'Error'],
      'label_field': 'is_missed',
      'layer': 0,
      'is_multi_feature': False,
      'is_sparse': True,
      'extension': '.pt',
      'name': 'custom'}
# config={
#       'root_dir': '/mnt/ssd2/custom_dataset/nus_centerpoint_activations_filtered_all/',
#       'label_file': 'nus_centerpoint_labels_filtered_all.csv',
#       'classes': ['No Error', 'Error'],
#       'label_field': 'is_missed',
#       'layer': 0,
#       'is_multi_feature': False,
#       'name': 'nuscenes'
#       }
preds = pd.DataFrame(columns=['filename','pred','label'])
if __name__ == "__main__":

    model_config = {'layer_config': 'configs/networks/resnet18_fcn.yaml'}#
    model_load_from = "./outputs/ckpts/nuscenes_none_resnet18_fcn_rawjolly-terrain-490__best.pth"
    model = generate_model_from_config(model_config)
    # model = GenericInjection(model_config)
    model.load_state_dict(torch.load(model_load_from))
    dataset = ActivationDataset(config)
    model.to('cuda:0')
    model.eval()
    # print(len(dataset))
    with torch.no_grad():
        for i,data in enumerate(dataset):
            tensor,label, meta = data
            # print(tensor,label,meta)
            tensor = tensor.to('cuda:0').unsqueeze(0)
            # tensor = [t.unsqueeze(0).to('cuda:0') for t in tensor]
            res = model(tensor)
            #softmaxed:
            res = nn.functional.softmax(res, dim=1)
            print(res)
            pred = torch.argmax(res)
            temp_df = pd.DataFrame({'filename':meta,'pred':pred.item(),'label':label.item()},index=[0])
            preds = pd.concat([preds,temp_df])

    preds.to_csv('veh_late_preds.csv',index=False)