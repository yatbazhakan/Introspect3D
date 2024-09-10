from torch.nn import Linear
import torch.nn.functional as F
try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn import global_mean_pool
except:
    print("torch_geometric not installed")
    pass
from utils.utils import generate_model_from_config
import torch
import yaml
import os
from definitions import ROOT_DIR
class GCN(torch.nn.Module):
    def __init__(self, model_config):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.layers = generate_model_from_config(model_config)
        self.is_list = False
    def forward_pass(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        first_linear = True
        for layer in self.layers:
            if isinstance(layer,GCNConv):
                x = layer(x, edge_index)
            elif isinstance(layer,Linear):
                if first_linear:
                    x = global_mean_pool(x, batch)
                    first_linear = False
                x = layer(x)
            elif isinstance(layer,torch.nn.Module):
                x = layer(x)
            else:
                print(x.shape)
                x = layer(x,batch)
        
        return x
    def forward(self, data):
        if isinstance(data,list):
            self.is_list = True
            result_list =  []
        if self.is_list:
            for d in data:
                result_list.append(self.forward_pass(d))
            return result_list
        else:
            return self.forward_pass(data)