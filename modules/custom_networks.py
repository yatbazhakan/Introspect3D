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
class CustomModel(nn.Module):
    def __init__(self, model_config,hooks=None):
        super(CustomModel, self).__init__()
        # Store the layers in an OrderedDict
        layers = generate_model_from_config(model_config)
        self.layers = nn.ModuleDict(OrderedDict({f'layer_{i}': layer for i, layer in enumerate(layers)}))
        self.resnet = nn.Sequential(*list(self.layers['layer_0'].children()))
        self.fc = nn.Sequential(*list(self.layers.values())[1:])
        self.hooks = hooks or {}
        self.downsample1 = DownSampleNorm(128,64,kernel_size=1,stride=1)
        self.downsample2 = DownSampleNorm(256,128,kernel_size=1,stride=1)
        self.downsample1= weight_init(self.downsample1)
        self.downsample2= weight_init(self.downsample2)

    def forward(self, x,hooks = [4,6]):

        first = x[0]
        second = x[1]
        third = x[2]
        first=first.to('cuda:0')
        second=second.to('cuda:0')
        third=third.to('cuda:0')
        for  idx,layer in enumerate(self.resnet):
            if idx ==0:
                x = first
            elif idx == hooks[0]:
                x_add = self.downsample1(second)
                x = x+x_add
            elif idx == hooks[1]:
                x_add = self.downsample2(third)
                x = x+x_add

            x = layer(x)
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