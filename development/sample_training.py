import os
import torch
import torch.nn as nn
import torchvision as vision
import torchvision.transforms as transforms
# import cv2
import numpy as np
import pandas as pd
import torchmetrics
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm  
class MiniErrorDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir=None,
                      feature_path=None,
                      label_path=None,
                      transform=None):
        convert_bool_to_int = lambda x: 1 if x else 0
        self.root_dir = root_dir
        self.feature_path = os.path.join(root_dir, feature_path)
        self.label_path = os.path.join(root_dir, label_path)
        
        self.input_paths = glob(os.path.join(self.feature_path, '*.npy'))
        self.labels = pd.read_csv(self.label_path)
        self.labels['is_missed'] = self.labels['is_missed'].apply(convert_bool_to_int)
        #Filter any sample that does not appear in the label csv
        self.input_paths = [path for path in self.input_paths if os.path.basename(path).replace('.npy','.png') in self.labels['image_path'].values]
    def override_dataset(self,indices):
        # print(indices)
        temp_labels = pd.DataFrame(columns=self.labels.columns)
        for i in indices:
            path = self.input_paths[i]
            temp_row = self.labels[self.labels['image_path'] == os.path.basename(path).replace('.npy','.png')]
            temp_labels = pd.concat([temp_labels,temp_row])
        self.labels = temp_labels
        self.input_paths = [self.input_paths[i] for i in indices]
    
    def __getitem__(self,idx):
        features = np.load(self.input_paths[idx])
        label = self.labels[self.labels['image_path'] == os.path.basename(self.input_paths[idx]).replace('.npy','.png')]['is_missed'].values[0]
        processed_features = pre_process_tensor(torch.from_numpy(features))
        label = torch.tensor(label)
        return processed_features, label

    def __len__(self):
        return len(self.input_paths)
    def get_y_true(self):
        return self.labels['is_missed'].values
    def get_x_indices(self):
        return list(range(len(self.input_paths)))
def auto_pad_to_multiple(tensor, multiple=14):
    # Get the shape of the tensor
    shape = tensor.shape
    height, width = shape[-2], shape[-1]

    # Calculate the required padding
    pad_height = ((height + multiple - 1) // multiple) * multiple - height
    pad_width = ((width + multiple - 1) // multiple) * multiple - width

    # Calculate individual padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Perform padding
    padded_tensor = nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return padded_tensor
def pre_process_tensor(tensor):
    groups = torch.chunk(tensor, 4, dim=0)

    # Initialize a list to store the output for each group
    output_list = []

    # Iterate through each group to stack 62x54 images to form a 496x432 image
    for group in groups:
        reshaped_group = group.reshape(8, 8, 62, 54).permute(0, 2, 1, 3).reshape(496, 432)
        output_list.append(reshaped_group)

    # Stack the outputs along a new dimension to get the final tensor
    output_tensor = torch.stack(output_list, dim=0)
    #Make divisible by 14 by padding
    # print(output_tensor.shape)
    output_tensor = auto_pad_to_multiple(output_tensor, multiple=14)
    # print(output_tensor.shape)
    return output_tensor



if __name__ == '__main__':
    print(os.getcwd())
    root_dir = './custom_dataset/pointpillars_kitti_class3/'
    features_path = 'features/'
    label_csv = 'pointpillars_kitti_class3_dataset.csv'

    dataset= MiniErrorDataset(root_dir=root_dir, 
                              feature_path=features_path,
                              label_path=label_csv) 
    x_train, x_test, y_train, y_test = train_test_split(dataset.get_x_indices(),dataset.get_y_true(),test_size=0.2, random_state=42,stratify=dataset.get_y_true())
    
    
    train_set =  MiniErrorDataset(root_dir=root_dir, 
                              feature_path=features_path,
                              label_path=label_csv) 
    train_set.override_dataset(x_train)
    test_set =  MiniErrorDataset(root_dir=root_dir, 
                              feature_path=features_path,
                              label_path=label_csv) 
    test_set.override_dataset(x_test)
    print(np.unique(train_set.get_y_true(),return_counts=True))
    print(np.unique(test_set.get_y_true(),return_counts=True))
    values, class_samples = np.unique(train_set.get_y_true(),return_counts=True)
    total_samples = sum(class_samples)

    # Calculate class weights
    class_weights = [total_samples / class_sample for class_sample in class_samples]

    # Convert to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    # Model preparation
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
    temp_dino_init= torch.nn.Conv2d(4, 384, kernel_size=(14,14), stride=(14,14))
    temp_dino_init.weight.data[:,:3,:,:] = dinov2.backbone.patch_embed.proj.weight.data[:,:3,:,:]
    #FIll remeaning weights with kaimler init
    nn.init.kaiming_normal_(temp_dino_init.weight.data[:,3:,:,:], mode='fan_out', nonlinearity='relu')
    
    temp_dino_init.bias.data = dinov2.backbone.patch_embed.proj.bias.data
    # print(temp_dino_init.weight.data.shape)
    dinov2.backbone.patch_embed.proj = temp_dino_init
    linear_head = torch.nn.Linear(1920, 2)
    #Changing 1920 -> 1000 for binary classification with weights
    linear_head.weight.data = dinov2.linear_head.weight.data[:2,:]
    linear_head.bias.data = dinov2.linear_head.bias.data[:2]
    dinov2.linear_head = linear_head
    # print(dinov2)
    # input("Enter to continue")
    # #Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2.to(device)
    # Initialize Cross-Entropy loss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    criterion= criterion.to(device)
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(dinov2.parameters(), lr=0.0001)
    num_epochs = 100
    predictions=[]
    # output_shape_hook = dinov2.backbone.patch_embed.register_forward_hook(lambda ins, input, output: print(output.shape))
    labelss = []
    with tqdm(total=num_epochs) as pbar:
        for epoch in range(1,num_epochs+1):
            dinov2.train()
            for i, (features, labels) in enumerate(train_loader):
                # input("Enter to continue")
                features = features.to(device)
                labels = labels.to(device)
                # print("Feature shape: ", features.shape)
                # print(dinov2.backbone)
                # print(dinov2.backbone.patch_embed.proj)
                outputs = dinov2(features)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if (i+1) % 100 == 0:
                #     break
                #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i+1, len(train_loader), loss.item()))
            pbar.update(1)
            predictions=[]
            # output_shape_hook = dinov2.backbone.patch_embed.register_forward_hook(lambda ins, input, output: print(output.shape))
            labelss = []
            #Test LOop
            dinov2.eval()
            # with tqdm(total=len(test_loader)) as pbar:
            print("Evaluation: Epoch [{}/{}]".format(epoch, num_epochs))
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                outputs = dinov2(features)
                _, predicted = torch.max(outputs.data, 1)
                # print("Before: ", predicted.shape)
                predictions.append([item.item() for item in predicted.cpu().numpy()])
                # print("After: ", predicted.shape)
                # print(len(predictions))
                labelss.append([item.item() for item in labels.cpu().numpy()])
                # pbar.update(1)
            #metric calculation
            flatten_predictions = [item for sublist in predictions for item in sublist]
            flatten_labels = [item for sublist in labelss for item in sublist]
            tensor_predictions = torch.tensor(flatten_predictions)
            tensor_labels = torch.tensor(flatten_labels)
            metric_collection = torchmetrics.MetricCollection(
                [torchmetrics.Accuracy(task="multiclass",num_classes=2),
                 torchmetrics.F1Score(task="multiclass",num_classes=2,average='macro'),
                 torchmetrics.ConfusionMatrix(task='multiclass',num_classes=2)])
            tensor_predictions = tensor_predictions.to(device)
            tensor_labels = tensor_labels.to(device)
            metric_collection.to(device)
            metric_collection.update(tensor_predictions, tensor_labels)
            print(metric_collection.compute())
