import torch
import torch.nn as nn
import yaml
import os
from collections import OrderedDict
from utils.utils import weight_init,generate_model_from_config
from torchvision.models.video import swin3d_t ,Swin3D_T_Weights
import torchvision
class SwinIntrospection(nn.Module):
    def __init__(self, model_config,hooks=None,device = 'cuda:1',in_channels=1) -> None:
        super(SwinIntrospection, self).__init__()
        self.swin = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
        self.layers = generate_model_from_config(model_config)
        self.swin.head = self.layers
        self.swin.patch_embed.proj = nn.Conv3d(in_channels, self.swin.patch_embed.proj.out_channels,kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.swin.patch_embed = weight_init(self.swin.patch_embed)
    
    def forward(self,x):
        return self.swin(x)

class DownSampleNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(DownSampleNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class CustomModel(nn.Module):
    def __init__(self, model_config,hooks=None,device = 'cuda:1'):
        super(CustomModel, self).__init__()
        # Store the layers in an OrderedDict
        layers = generate_model_from_config(model_config)
        self.layers = nn.ModuleDict(OrderedDict({f'layer_{i}': layer for i, layer in enumerate(layers)}))
        self.resnet = nn.Sequential(*list(self.layers['layer_0'].children()))
        self.fc = nn.Sequential(*list(self.layers.values())[1:])
        self.hooks = hooks or {}
        self.pool = None
        self.downsample1 = DownSampleNorm(128,64,kernel_size=1,stride=1)
        self.downsample2 = DownSampleNorm(256,128,kernel_size=1,stride=1)
        self.downsample1= weight_init(self.downsample1)
        self.downsample2= weight_init(self.downsample2)
        self.device = device

    def forward(self, x,hooks = [4,6]):

        first = x[0]
        second = x[1]
        third = x[2]
        print(first.shape,second.shape,third.shape)
        first=first.to(self.device)
        # print(first.shape)
        for  idx,layer in enumerate(self.resnet):
            if idx ==0:
                x = first
            elif idx == hooks[0]:
                second=second.to(self.device)
                x_add = self.downsample1(second)
                if x_add.shape != x.shape:
                    # print(x_add.shape,x.shape)
                    self.pool  = torch.nn.AdaptiveAvgPool2d(x.shape[2:])
                    x_add = self.pool(x_add)
                    # print("Adaptive Pooling",x_add.shape)

                x = x+x_add
                del second,first,x_add
            elif idx == hooks[1]:
                third=third.to(self.device)
                x_add = self.downsample2(third)
                if x_add.shape != x.shape:
                    # print(x_add.shape,x.shape)
                    self.pool  = torch.nn.AdaptiveAvgPool2d(x.shape[2:])
                    x_add = self.pool(x_add)
                    # print("Adaptive Pooling",x_add.shape)
                x = x+x_add
                del third,x_add
            x = layer(x)
            # print("-------------------")
            # print(x.shape)
            # if idx ==0:
            #     print(f"Concatting at layer {idx},{first.shape}")
            #     x = layer(first)
            # elif idx == hooks[0]:
            #     print(f"Concatting at layer {idx},{x.shape},{second.shape}")
            #     x = torch.cat((x,second),dim=0)
            #     x = layer(x)
            # elif idx == hooks[1]:
            #     print(f"Concatting at layer {idx},{x.shape},{third.shape}")
            #     x = torch.cat((x,third),dim=0)
            #     x = layer(x)
        x = self.fc(x)
        return x
    
class GenericInjection(nn.Module):
    def __init__(self, model_config,hooks=None,device = 'cuda:1'):
        super(GenericInjection, self).__init__()
        # Store the layers in an OrderedDict
        layers = generate_model_from_config(model_config)
        self.layers = nn.ModuleDict(OrderedDict({f'layer_{i}': layer for i, layer in enumerate(layers)}))
        self.resnet = nn.Sequential(*list(self.layers['layer_0'].children()))
        self.fc = nn.Sequential(*list(self.layers.values())[1:])
        self.hooks = hooks or {}
        self.pool = None
        self.downsample1 = DownSampleNorm(128,64,kernel_size=1,stride=1)
        self.downsample2 = DownSampleNorm(256,128,kernel_size=1,stride=1)
        self.downsample3 = DownSampleNorm(256,64,kernel_size=1,stride=1)
        self.downsample1= weight_init(self.downsample1)
        self.downsample2= weight_init(self.downsample2)
        self.device = device
        self.hooks = [4,6] #currently manually changed [1,2]#
    def get_tensor_list(self,x,mode):
        if mode == "EML":
            first = x[0]
            second = x[1]
            third = x[2]
        elif mode == "EL":
            first = x[0]
            third = x[2]
            second = None
        elif mode == "EM":
            first = x[0]
            second = x[1]
            third = None
        elif mode == "ML":
            first = None
            second = x[1]
            third = x[2]

        return first,second,third
        

    def forward(self, x,mode='EML'):
        hooks = self.hooks
        first,second,third = self.get_tensor_list(x,mode)
        if first is not None:
            for  idx,layer in enumerate(self.resnet):
                if idx ==0:
                    x = first
                    first = first.to(self.device)
                elif idx == hooks[0] and second is not None:
                    second=second.to(self.device)
                    x_add = self.downsample1(second)
                    if x_add.shape != x.shape:
                        self.pool  = torch.nn.AdaptiveAvgPool2d(x.shape[2:])
                        x_add = self.pool(x_add)
                    x = x+x_add
                    del second,first,x_add
                elif idx == hooks[1] and third is not None:
                    third=third.to(self.device)
                    x_add = self.downsample2(third)
                    if x_add.shape != x.shape:
                        # print(x_add.shape,x.shape)
                        self.pool  = torch.nn.AdaptiveAvgPool2d(x.shape[2:])
                        x_add = self.pool(x_add)
                        # print("Adaptive Pooling",x_add.shape)
                    x = x+x_add
                    del third,x_add
                x = x.to(self.device)
                x = layer(x)
        else:
            # hooks = [4]
            for idx,layer in enumerate(self.resnet):
                if idx ==0:
                    x = second
                    x = x.to(self.device)
                elif idx == hooks[0] and third is not None:
                    third=third.to(self.device)
                    # print(third.shape)
                    # print(x.shape)
                    x_add = self.downsample3(third)
                    if x_add.shape != x.shape:
                        self.pool  = torch.nn.AdaptiveAvgPool2d(x.shape[2:])
                        x_add = self.pool(x_add)
                    x = x+x_add
                    del second,first,x_add
                x = x.to(self.device)
                x = layer(x)
            
        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import yaml
import os
from collections import OrderedDict
from utils.utils import weight_init,generate_model_from_config
class DownSampleNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True):
        super(DownSampleNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class EarlyFusionAdaptive(nn.Module):
    def __init__(self, model_config,hooks=None,device = 'cuda:1'):
        super(EarlyFusionAdaptive, self).__init__()
        # Store the layers in an OrderedDict
        layers = generate_model_from_config(model_config)
        self.layers = nn.ModuleDict(OrderedDict({f'layer_{i}': layer for i, layer in enumerate(layers)}))
        self.resnet = nn.Sequential(*list(self.layers['layer_0'].children()))
        self.fc = nn.Sequential(*list(self.layers.values())[1:])
        self.hooks = hooks or {}
        self.pool = None
        self.downsample1 = DownSampleNorm(256,256,kernel_size=1,stride=1)
        self.downsample2 = DownSampleNorm(128,256,kernel_size=1,stride=1)
        self.downsample1= weight_init(self.downsample1)
        self.downsample2= weight_init(self.downsample2)
        self.device = device

    def forward(self, x,hooks = [4,6]):

        first = x[0]
        second = x[1]
        third = x[2]
        first=first.to(self.device)
        second=second.to(self.device)
        third=third.to(self.device)
        first = self.downsample1(first)
        second = self.downsample2(second)
    
        pooler = torch.nn.AdaptiveAvgPool2d(third.shape[2:])
        # print("Before", first.shape)
        first = pooler(first)
        second = pooler(second)
        # print(first.shape,second.shape,third.shape)
        third = first + second + third

        x = self.resnet(third)
        x = self.fc(x)
        return x