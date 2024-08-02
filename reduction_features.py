import torch
import torch.nn as nn
from utils.utils import generate_model_from_config
from utils.factories import DatasetFactory
from datasets.activation_dataset import ActivationDataset
import os
from tqdm.auto import tqdm
import pandas as pd
save_folder = "../custom_dataset/pca_features_from_late/"
current_folder = "train"
current_file_name= None
isTrue = True
def data_generator(data, batch_size):
    total_samples = data.shape[0]
    for i in range(0, total_samples, batch_size):
        yield data[i:i + batch_size]

if __name__ == "__main__":
    config = {
        'model_path' : "./outputs/ckpts/nuscenes_none_resnet18_fcn_rawnus_centerpoint_filtered_wobjf late0__best.pth",
        'model_architecture': {'layer_config':"./configs/networks/resnet18_fcn.yaml"},
        'dataset': 
            {
                'name': 'ActivationDataset',
                'config': {
                    'root_dir': "/media/wmg-5gcat/Co-op Autonomy 2/Hakan/custom_dataset/nus_centerpoint_activations_filtered_objects_lonly"  ,
                    'label_file': 'nus_centerpoint_labels_filtered_objects_lonly.csv',
                    'classes': ['No Error', 'Error'],
                    'label_field': 'is_missed',
                    'layer': 2,
                    'is_multi_feature': False,
                    'is_sparse': True,
                    'extension': '.pt',
                    'name': 'nuscenes'
                    }
            }
    }
    os.makedirs(save_folder, exist_ok=True)
    
    root = config.get('dataset',None).get('config',None).get('root_dir',None)
    config['dataset']['config']['root_dir'] = os.path.join(root,'train')
    train_dataset = DatasetFactory().get(**config['dataset'])
    config['dataset']['config']['root_dir'] = os.path.join(root,'test')
    test_dataset = DatasetFactory().get(**config['dataset'])
    config['dataset']['config']['root_dir'] = os.path.join(root,'val')
    val_dataset = DatasetFactory().get(**config['dataset'])
    # model = generate_model_from_config(config['model_architecture'])
    # model.load_state_dict(torch.load(config['model_path']))
    # model.eval()
    # model = model.to('cpu')
    # hook = model[0].register_forward_hook(activation_hook)
    #Create SVD
    
    # print(f"Extracting features from {current_folder} dataset")
    # if(current_folder == "train"):
    #     train_dataset_labels = pd.DataFrame(columns=['name', 'label'])
    #     os.makedirs(os.path.join(save_folder,current_folder), exist_ok=True)
    #     os.makedirs(os.path.join(save_folder,current_folder,"features"), exist_ok=True)
    #     with torch.no_grad():
    #         with tqdm(total=len(train_dataset)) as pbar:
    #             for i in range(len(train_dataset)):
    #                 input_tensor,label_tensor,file_name = train_dataset[i]
                    
    #                 # current_file_name= file_name
    #                 # cloud = torch.unsqueeze(input_tensor,0)
    #                 # cloud = cloud.to('cpu')
    #                 # output = model(cloud)
    #                 temp_df = pd.DataFrame({'name': [file_name], 'label': [label_tensor.item()]})
    #                 pbar.update(1)
    #         train_dataset_labels = pd.concat([train_dataset_labels,temp_df])
    #     train_dataset_labels.to_csv(os.path.join(save_folder,current_folder,"labels.csv"))
    # elif(current_folder == "test"):  
    #     test_dataset_labels = pd.DataFrame(columns=['name', 'label'])
    #     current_folder = "test"
    #     os.makedirs(os.path.join(save_folder,current_folder), exist_ok=True)
    #     os.makedirs(os.path.join(save_folder,current_folder,"features"), exist_ok=True)
    #     with torch.no_grad():
    #         with tqdm(total=len(test_dataset)) as pbar:
    #             for i in range(len(test_dataset)):
    #                 input_tensor,label_tensor,file_name = test_dataset[i]
    #                 current_file_name= file_name
    #                 cloud = torch.unsqueeze(input_tensor,0)
    #                 cloud = cloud.to('cpu')
    #                 output = model(cloud)
    #                 temp_df = pd.DataFrame({'name': [file_name], 'label': [label_tensor.item()]})
    #                 test_dataset_labels = pd.concat([test_dataset_labels,temp_df])
    #                 pbar.update(1)
    
    #     test_dataset_labels.to_csv(os.path.join(save_folder,"labels.csv"))
    # else:
    #     val_dataset_labels = pd.DataFrame(columns=['name', 'label'])
    #     current_folder = "val"
    #     os.makedirs(os.path.join(save_folder,current_folder), exist_ok=True)
    #     os.makedirs(os.path.join(save_folder,current_folder,"features"), exist_ok=True)
    #     with torch.no_grad():
    #         with tqdm(total=len(val_dataset)) as pbar:
    #             for i in range(len(val_dataset)):
    #                 input_tensor,label_tensor,file_name = val_dataset[i]
    #                 current_file_name= file_name
    #                 cloud = torch.unsqueeze(input_tensor,0)
    #                 cloud = cloud.to('cpu')
    #                 output = model(cloud)
    #                 temp_df = pd.DataFrame({'name': [file_name], 'label': [label_tensor.item()]})
    #                 val_dataset_labels = pd.concat([val_dataset_labels,temp_df])
    #                 pbar.update(1)
    #     val_dataset_labels.to_csv(os.path.join(save_folder,current_folder,"labels.csv"))