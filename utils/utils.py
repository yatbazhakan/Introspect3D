from mmdet3d.utils import register_all_modules
from mmdet3d.apis import inference_detector, init_model
import os
from utils.config import Config
from utils.boundingbox import BoundingBox
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.nn import *
from torchvision.models import *
import torchvision
import torch
from modules.other import Identity
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
        center = box[:3]
        #Some fix I dont;know for sure
        center[2]= center[2]/2
        dimensions = box[3:6]
        rotation = box[6]
        #These rotations might be a problem later, but for now they are fine
        #Type is integer and not available with box, some mapping may be needed, but I dont care about classes for now
        # print("Rotation from prediction",rotation,type(rotation))
        bounding_box = BoundingBox(center, dimensions, rotation, 0)
        bounding_boxes.append(bounding_box)
    return bounding_boxes

def check_detection_matches(ground_truth_boxes:BoundingBox, predicted_boxes:BoundingBox, iou_threshold:float=0.5):
    """Checks if the predicted boxes match with the ground truth boxes."""
    matches = []
    unmatched_ground_truths = []
    unmatched_predictions = list(predicted_boxes)

    for gt_box in ground_truth_boxes:

        max_iou_idx, max_iou = gt_box.find_max_iou_box(unmatched_predictions)

        if max_iou != None and max_iou >= iou_threshold:
            matches.append((gt_box, unmatched_predictions[max_iou_idx]))
            del unmatched_predictions[max_iou_idx]
        else:
            unmatched_ground_truths.append(gt_box)

    return matches, unmatched_ground_truths, unmatched_predictions


def generate_model_from_config(config):

    
    layers = []
    num_ftrs = None
    for layer_config in config['layers']:
        layer_type = layer_config['type']
        if layer_type.startswith('torchvision'): #expectation is to use ResNets, may adapt later to other models
            model = eval(f"{layer_type}(**layer_config['params'])")
            # Clone the first 3 channels from the original weights
            original_weight = model.conv1.weight.clone()  # Shape: [64, 3, 7, 7]
            model.conv1 = nn.Conv2d(layer_config['in_channels'], 64,
                                            kernel_size=7, stride=2, padding=3, bias=False)
            # Calculate the mean across the channel dimension
            mean_weight = torch.mean(original_weight, dim=1, keepdim=True)  # Shape: [64, 1, 7, 7]

            # Initialize a new weight tensor filled with zeros
            new_weight = torch.zeros((64, layer_config['in_channels'], 7, 7), dtype=original_weight.dtype, device=original_weight.device)

            # Copy the original 3 channels into the new weight tensor
            new_weight[:, :3, :, :] = original_weight

            # Fill the remaining 253 channels with the mean channel
            new_weight[:, 3:, :, :] = mean_weight

            # Assign the new weight tensor to the conv1 layer
            model.conv1.weight = nn.Parameter(new_weight)


            num_ftrs = model.fc.in_features 
            model.fc = Identity()
            layers.append(model)
        else:
            layer_params = layer_config['params']
            if num_ftrs != None and 'in_features' in layer_params.keys():
                print("in_features is set to",num_ftrs,"for layer",layer_type)
                layer_params['in_features'] = num_ftrs
                num_ftrs = None
            layers.append(eval(f"{layer_type}(**layer_params)"))
    return nn.Sequential(*layers)

def generate_optimizer_from_config(config,model):
    optimizer_type = config['type']
    optimizer_params = config['params']
    return eval(f"{optimizer_type}(model.parameters(),**optimizer_params)")

def generate_scheduler_from_config(config,optimizer):
    scheduler_type = config['type']
    scheduler_params = config['params']
    return eval(f"{scheduler_type}(optimizer,**scheduler_params)")

def generate_criterion_from_config(config):
    loss_type = config['type']
    loss_params = config['params']
    if "weight" in loss_params.keys():
        loss_params['weight'] = torch.tensor(loss_params['weight'],dtype=torch.float32)
    return eval(f"{loss_type}(**loss_params)")