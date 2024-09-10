import cv2
from ultralytics import YOLO
import os
# from utils.utils import generate_model_from_config
import yaml
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Flatten, Linear, ReLU, Dropout
from definitions import ROOT_DIR
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fcos_resnet50_fpn
activation_maps = []

def ash_p(x, percentile=75):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
def custom_hook(module, input, output):
    # print(output.keys())
    # for k,v in output.items():
    #     print(k,v.shape)
    # print(input[0].shape)
    # print(output[0].shape)
    
    activation_maps.append(output['2'])
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
def detect_objects_in_video(video_path,model_path,int_model):
    # Initialize the YOLOv8 model
    global activation_maps
    model = fcos_resnet50_fpn(pretrained=True)
    model = model.cuda()
    model.eval()
    print(model.backbone)
    hook = model.backbone.body.register_forward_hook(custom_hook)
    # model = YOLO(model_path)  # Replace with your model version if necessary
    # print(model.model.model[21])
    # hook = model.model.model[8].register_forward_hook(custom_hook)
    # print(int_model)
    softmax = nn.Softmax(dim=1)
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    prev_activation = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        im = frame.copy()
        # Perform detection
        # results = model(frame,imgsz=640, conf=0.5)
        im_tens = torchvision.transforms.ToTensor()(frame).cuda()
        results = model(im_tens[None, ...])
        activation = activation_maps[0]
        activation_maps = []
        print(activation.shape)
        activation = activation.detach().cpu().numpy()
        activation_tens = torch.from_numpy(activation).cuda()
        # activation_tens = ash_p(activation_tens, percentile=90)

        # activation_tens = ash_p(activation_tens[None, ...], percentile=90)
        # activation_tens = activation_tens[None, ...]
        int_model.eval()

        int_Res=  int_model(activation_tens)
        #softmaxed
        print(int_Res.shape)
        int_Res = softmax(int_Res)
        # int_Res = torch.nn.functional.softmax(int_Res.squeeze(),dim=0)
        print(int_Res)
        # im = results[0].plot()
        argmax = torch.argmax(int_Res)
        if argmax == 0:
            argmax = "Safe"
        else:
            argmax = "Unsafe"
        if argmax == 0:
            color = (0,255,0)
        else:
            color = (0,0,255)
        # Optionally: Show the frame with detections
        im = cv2.putText(im, f"Prediction: {argmax}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # break
    # Release the video capture object
    hook.remove()
    cap.release()
    cv2.destroyAllWindows()

def introspect(int_model):
        for activation in activation_maps:
            
            activation = activation.detach().cpu().numpy()
            activation_tens = torch.from_numpy(activation).cuda()
            print("inp",activation_tens.shape)
            # activation_tens = ash_p(activation_tens.squeeze(), percentile=90)
            # activation_tens = ash_p(activation_tens[None, ...], percentile=90)
            activation_tens = activation_tens[None, ...]
            int_Res=  int_model(activation_tens)
            #softmaxed
            print(int_Res.shape)
            int_Res = torch.nn.functional.softmax(int_Res.squeeze(),dim=0)
            print(int_Res)
# Example usage
introspector_config = 'configs/networks/resnet18_fcn.yaml'
introspector_model_path= '/mnt/ssd2/bddvideo'
introspector_model_name= 'bdd8.pth'
root_path = '/mnt/ssd2/bddvideo/bdd100k_videos_train_00/bdd100k/videos/train'
video_name= '025ed087-7cb7665c.mov' #'024dd592-94359ff1.mov' #
yolo_path =""#'/mnt/ssd2/introspection-object-detection/checkpoints'
model_name='yolov8n.pt'

model = generate_model_from_config({'layer_config':introspector_config})
model = model.cuda()
# model.load_state_dict(torch.load(os.path.join(introspector_model_path,introspector_model_name)))
# model = model.eval()
print("Model loaded",model)
detect_objects_in_video(os.path.join(root_path,video_name),
                        os.path.join(yolo_path,model_name),
                        model)
# introspect(model)
# for i in range(len(activation_maps)-2):
#     print(torch.equal(activation_maps[i],activation_maps[i+1]))
    