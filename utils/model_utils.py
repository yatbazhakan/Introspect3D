from torch import nn
import torch
from definitions import ROOT_DIR
import yaml
import os
import logging
try:
    from mmdet3d .apis import init_model
except:
    logging.log(logging.INFO,"mmdet3d not installed")
    exit()

def load_detection_model(config):
    """Loads the detection model."""
    root_dir = config['model']['root_dir']
    checkpoint_full_path = os.path.join(root_dir,config['model']['checkpoint_dir'], config['model']['checkpoint'])
    model_config_full_path = os.path.join(root_dir,config['model']['config_dir'],config['model']['name'] , config['model']['config'])
    model = init_model(model_config_full_path, checkpoint_full_path, config['device'])
    # print("Model Address",id(model))
    logging.log(logging.INFO,"Model loaded")
    return model
"""
TODO: Have a separate function for each clone weights? Support more functions if needed/possible
""" 
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
        model.fc = nn.Identity()
    logging.log(logging.INFO,"Weights cloned")
    return model,num_ftrs

def generate_model_from_config(config):
    path = os.path.join(ROOT_DIR,config['layer_config'])
    logging.log(logging.INFO,f"Loading model yaml from {path}")

    layer_data = yaml.load(open(path),Loader=yaml.FullLoader)
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
            # num_ftrs = model.fc.in_features
            # model.fc = Identity()
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
                if isinstance(layer_params['output_size'],str):
                    ouput_size = eval(layer_params['output_size'])
                elif isinstance(layer_params['output_size'],int):
                    ouput_size = layer_params['output_size']
                layer_params['output_size'] = ouput_size
            # print(layer_type,layer_params)
            layers.append(eval(f"{layer_type}(**layer_params)"))
    #TODO: Add support for other models, check if this is needed
    if pretrained:
        pt_model = layers[0]
        layers = layers[1:]
        layers = weight_init(nn.Sequential(*layers))
        model = nn.Sequential(pt_model,*layers)
    else:
        model = nn.Sequential(*layers)
        model = weight_init(model)
    logging.log(logging.INFO,"Model generated")
    return model

def weight_init(model):
    #initialize weights for better convergence
    logging.log(logging.INFO,"Initializing weights")
    for m in model.modules():
        
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m,nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    logging.log(logging.INFO,"Weights initialized")
    return model