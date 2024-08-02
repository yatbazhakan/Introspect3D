try: 
    from mmdet3d.utils import register_all_modules
    from mmdet3d.apis import inference_detector, init_model
    from pyquaternion import Quaternion
    import open3d as o3d
    from nuscenes.utils.data_classes import LidarPointCloud,Box
except:
    print("DIfferent environment, some packages will be missing")
    pass

import os
from definitions import ROOT_DIR
from utils.config import Config
from utils.boundingbox import BoundingBox
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.nn import *
from torchvision.models import *
from modules.loss import *
from modules.attention import *
import torchvision
import torch
import yaml
from modules.other import Identity
from modules.conv_blocks import *
try:
    from torch_geometric.nn import *
    from torch_geometric.data import *
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import global_mean_pool

except:
    print("torch_geometric not installed")
    pass
import gc


def clear_memory():
    # Clear GPU memory if using PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Garbage collection to free up RAM
    gc.collect()
def load_detection_model(config: Config):
    """Loads the detection model."""
    root_dir = config['model']['root_dir']
    checkpoint_full_path = os.path.join(root_dir,config['model']['checkpoint_dir'], config['model']['checkpoint'])
    model_config_full_path = os.path.join(root_dir,config['model']['config_dir'],config['model']['name'] , config['model']['config'])
    model = init_model(model_config_full_path, checkpoint_full_path, config['device'])
    # print("Model Address",id(model))
    return model

def check_and_create_folder(path: str):
    """Checks if a folder exists and creates it if not."""
    if not os.path.exists(path):
        os.makedirs(path)

def create_bounding_boxes_from_predictions(boxes: np.ndarray):
    """Creates a list of BoundingBox objects from a numpy array of boxes."""
    #TODO: checks for the boxes in terms of shape (what if more values are given)
    bounding_boxes = []
    for box in boxes:
        if isinstance(box,BoundingBox):
            print("Box is already a BoundingBox object")
            bounding_boxes.append(box)
            continue
        center = box[:3]
        dimensions = box[3:6]
        rotation = box[6]

        center[2] += dimensions[2]/2
        bounding_box = BoundingBox(center, dimensions, rotation, 0)
    
        bounding_boxes.append(bounding_box)
    return bounding_boxes

def check_detection_matches(ground_truth_boxes, predicted_boxes, iou_threshold:float=0.5):
    """Checks if the predicted boxes match with the ground truth boxes."""
    matches = []
    unmatched_ground_truths = []
    unmatched_predictions = list(predicted_boxes)
    print("Unmatched Predictions",len(unmatched_predictions))
    for gt_box in ground_truth_boxes:

        max_iou_idx, max_iou = gt_box.find_max_iou_box(unmatched_predictions)
        print("Max IOU",max_iou)
        if max_iou != None and max_iou >= iou_threshold:
            matches.append((gt_box, unmatched_predictions[max_iou_idx]))
            del unmatched_predictions[max_iou_idx]
        else:
            unmatched_ground_truths.append(gt_box)

    return matches, unmatched_ground_truths, unmatched_predictions

def clone_weights(model,layer_type,layer_config):
    if "resnet" in layer_type:
        # Clone the first 3 channels from the original weights
        # Make this generalizable for other torchvision models like densenet
        original_weight = model.conv1.weight.clone()  # Shape: [64, 3, 7, 7]
        model.conv1 = nn.Conv2d(layer_config['in_channels'], 64,
                                        kernel_size=7, stride=2, padding=3, bias=False)
        # Calculate the mean across the channel dimension
        mean_weight = torch.mean(original_weight, dim=1, keepdim=True)  # Shape: [64, 1, 7, 7]

        # Initialize a new weight tensor filled with zeros
        new_weight = torch.zeros((64, layer_config['in_channels'], 7, 7), dtype=original_weight.dtype, device=original_weight.device)
        if layer_config['in_channels'] > 3:
            # Copy the original 3 channels into the new weight tensor
            new_weight[:, :3, :, :] = original_weight

            # Fill the remaining 253 channels with the mean channel
            new_weight[:, 3:, :, :] = mean_weight
        else:
            new_weight[:,:layer_config['in_channels'],:,:] = original_weight[:,:layer_config['in_channels'],:,:]

        # Assign the new weight tensor to the conv1 layer
        model.conv1.weight = nn.Parameter(new_weight)


        num_ftrs = model.fc.in_features 
        model.fc = Identity()

    return model,num_ftrs

def generate_model_from_config(config):
    print("Loading model yaml")
    path = os.path.join(ROOT_DIR,config['layer_config'])
    print(path)
    layer_data = yaml.load(open(path),Loader=yaml.FullLoader)
    print("Model yaml loaded")
    layers = []
    pretrained = False
    num_ftrs = None
    # print(layer_data)
    for layer_config in layer_data['layers']:
        # print(layer_config)
        layer_type = layer_config['type']
        if layer_type.startswith('torchvision'): #expectation is to use ResNets, may adapt later to other models
            pretrained=True
            model = eval(f"{layer_type}(**layer_config['params'])")
            if not "swin" in layer_type:   
                model,num_ftrs = clone_weights(model,layer_type,layer_config)
            # print(model)
            layers.append(model)
        else:
            layer_params = layer_config['params']
            if num_ftrs != None and 'in_features' in layer_params.keys():
                # print("in_features is set to",num_ftrs,"for layer",layer_type)
                layer_params['in_features'] = num_ftrs
                num_ftrs = None
            elif 'output_size' in layer_params.keys():
                ouput_size = eval(layer_params['output_size'])
                layer_params['output_size'] = ouput_size
            # print(layer_type,layer_params)
            layers.append(eval(f"{layer_type}(**layer_params)"))
    if pretrained:
        pt_model = layers[0]
        layers = layers[1:]
        layers = weight_init(Sequential(*layers))
        model = Sequential(pt_model,*layers)
    else:
        model = Sequential(*layers)
        model = weight_init(model)
    return model

def weight_init(model):
    #initialize weights for better convergence  
    for m in model.modules():
        
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    return model
def generate_optimizer_from_config(config,model):
    optimizer_type = config['type']
    optimizer_params = config['params']
    return eval(f"{optimizer_type}(model.parameters(),**optimizer_params)")

def generate_scheduler_from_config(config,optimizer):
    scheduler_type = config['type']
    scheduler_params = config['params']
    return eval(f"{scheduler_type}(optimizer,**scheduler_params)")

def generate_criterion_from_config(config,**kwargs):
    print(config)
    loss_type = config['type']
    loss_params = config['params']
    if "weight" in loss_params.keys():
        loss_params['weight'] = torch.tensor(loss_params['weight'],dtype=torch.float32)
    return eval(f"{loss_type}(**loss_params)")

def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
    
    Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]