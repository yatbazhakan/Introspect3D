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

    for layer_config in config['layers']:
        layer_type = layer_config['type']
        layer_params = layer_config['params']
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
    return eval(f"{loss_type}(**loss_params)")