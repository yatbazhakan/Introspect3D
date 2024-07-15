from typing import List, Union, Tuple, Dict, Any
import numpy as np
import os
import pickle
import torch
from definitions import ROOT_DIR
from utils.utils import load_detection_model, check_and_create_folder

try:
    from mmdet3d.apis import inference_detector, init_model
except:
    print("Could not load mmdet3d, some packages will be missing")
    pass
d = 1 
class Activations:

    def __init__(self,
                 config,extract=True) -> None:
        self.model = load_detection_model(config)
        # print(self.model)
        self.config = config
        method = config['method']
        self.save_dir = method['save_dir']
        self.extension =  method['extension']
        self.save_input = method['save_input']
        self.hooks = []
        self.activation_list= []
        #TODO: better way to do this, indexes are not used right now, save activation must be generalized
        if extract:
            print("Hook initialized")
            # if self.save_input:
            #     input_hool = self.model.backbone.register_forward_hook(self.register_prerpocessing_output)
            self.hook_layers = method['hook']['indexes']
            for i,index in enumerate(self.hook_layers):
                extract = method['hook']['extract_from'][i]
                if extract:
                    hook = eval(f'self.model.{method["hook"]["layer"]}._modules["{index}"].register_forward_hook(self.register_activation_output)')
                else:
                    hook = eval(f'self.model.{method["hook"]["layer"]}._modules["{index}"].register_forward_hook(self.register_activation_input)')
                self.hooks.append(hook)
                # self.hooks = eval(f'self.model.{method["hook"]["layer"]}.register_forward_hook(self.debug_activation)')
        else:
            self.hook = None
            
        check_and_create_folder(os.path.join(ROOT_DIR,self.save_dir,"features"))

    def __call__(self, x,name):
        self.save_name = name.split('/')[-1]
        self.gradients = []
        self.activations = []
        return inference_detector(self.model, x)
    def debug_activation(self, module, input, output):
        print("Debugging")
        print(output.shape)
        print(output)
    def save_multi_layer_activation(self,split=False):
        # print("Saving activations")
        # if self.extension != "":
        global d 
        if split:
            if split not in self.save_dir:
                self.save_dir = os.path.join(self.save_dir,split)
                # os.makedirs(self.save_dir,exist_ok=True)
                # os.makedirs(os.path.join(self.save_dir,"features"),exist_ok=True)

        save_name = self.save_name.replace(self.extension,'.pkl') #should be more generic currently depends on image, maybe jsut remove extension
        # else:
        # print("Saving", save_name, "to", self.save_dir)
            # save_name = self.save_name + ".pkl" # A fix that might create an isssue, hoping that extension param will fix automatically
        if d == 1:

            print("Saving", save_name, "to", self.save_dir)
            d = 0

        # print(len(self.activation_list))
        with open(os.path.join(self.save_dir,"features" ,save_name), 'wb') as f:
            pickle.dump(self.activation_list, f)
        # print(save_name,self.save_dir)
        #np.save(os.path.join(self.save_dir,"features" ,save_name), self.activation_list)
        del self.activation_list
        self.activation_list = []
    def save_multi_layer_activation_sparse(self,split=False):
        # print("Saving activations")
        # if self.extension != "":
        global d 
        if split:
            if split not in self.save_dir:
                self.save_dir = os.path.join(self.save_dir,split)
                # os.makedirs(self.save_dir,exist_ok=True)
                # os.makedirs(os.path.join(self.save_dir,"features"),exist_ok=True)

        save_name = self.save_name.replace(self.extension,'.pt')
        if '.pt' not in save_name:
            save_name = save_name + ".pt"
        #should be more generic currently depends on image, maybe jsut remove extension
        # else:
        # print("Saving", save_name, "to", self.s   ave_dir)
            # save_name = self.save_name + ".pkl" # A fix that might create an isssue, hoping that extension param will fix automatically
        if d == 1:

            print("Saving", save_name, "to", self.save_dir)
            d = 0

        # print(len(self.activation_list))
        sparse_list=self.make_sparse(self.activation_list)
        torch.save(sparse_list,os.path.join(self.save_dir,"features" ,save_name))
        # with open(os.path.join(self.save_dir,"features" ,save_name), 'wb') as f:
        #     pickle.dump(sparse_list, f)
        # torch.save(sparse_list,os.path.join(self.save_dir,"features" ,save_name))
        # print(save_name,self.save_dir)
        #np.save(os.path.join(self.save_dir,"features" ,save_name), self.activation_list)
        del self.activation_list
        self.activation_list = []
    def make_sparse(self, activation_list):
        if len(activation_list) > 3:
            print("Activation list length",len(activation_list))
        sparse_list = []
        for activation in activation_list:
            num_zeros = np.sum(activation == 0)
            # Calculate the total number of elements
            total_elements = np.prod(activation.shape)
            s = num_zeros/total_elements
            if s > 0.9:
                sparse_list.append(torch.from_numpy(activation).to_sparse().to('cuda:0'))
            else:
                sparse_list.append(torch.from_numpy(activation).to('cuda:0'))
        return sparse_list
    def clear_activation(self):
        del self.activation_list
        self.activation_list = []
    def register_prerpocessing_output(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        # print(len(output))
        print(len(input),input[0].shape)

        last_output = input[0].detach().cpu().numpy()
    def register_activation_output(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        # print(len(output))
        # last_output = output#.detach().cpu() #TODO: generalize this
        last_output = output.detach().cpu().numpy() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        # print(last_output.shape)
        # print("-------------------")
        last_output = np.squeeze(last_output)
        # last_output = torch.squeeze(last_output)
        self.activation_list.append(last_output)

    def register_activation_input(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        # last_output = input[0]# .detach().cpu() #TODO: generalize this
        last_output = input[0].detach().cpu().numpy() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        last_output = np.squeeze(last_output)
        # last_output = torch.squeeze(last_output)
        self.activation_list.append(last_output)

    def save_backbone_output(self,module, input, output): #Original implementation
        # print(output[0].shape,output[1].shape)
        last_output = output[self.hook_layer].detach().cpu() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        # last_output = np.squeeze(last_output)
        last_output = torch.squeeze(last_output)

        save_name = self.save_name.replace(self.extension,'.npy') #should be more generic currently depends on image, maybe jsut remove extension
        # print(save_name,self.save_dir)
        # np.save(os.path.join(self.save_dir,"features" ,save_name), last_output)
    def clear(self):
        self.activations = []
        self.gradients = []
        for hook in self.hooks:
            hook.remove()

class E2EActivations:

    def __init__(self,
                 config,extract=True) -> None:
        self.model = load_detection_model(config)
        # print(self.model)
        self.config = config
        method = config['method']
        self.hooks = []
        self.activation_list= []
        #TODO: better way to do this, indexes are not used right now, save activation must be generalized
        if extract:
            print("Hook initialized")
            # if self.save_input:
            #     input_hool = self.model.backbone.register_forward_hook(self.register_prerpocessing_output)
            self.hook_layers = method['hook']['indexes']
            for i,index in enumerate(self.hook_layers):
                extract = method['hook']['extract_from'][i]
                if extract:
                    hook = eval(f'self.model.{method["hook"]["layer"]}._modules["{index}"].register_forward_hook(self.register_activation_output)')
                else:
                    hook = eval(f'self.model.{method["hook"]["layer"]}._modules["{index}"].register_forward_hook(self.register_activation_input)')
                self.hooks.append(hook)
                # self.hooks = eval(f'self.model.{method["hook"]["layer"]}.register_forward_hook(self.debug_activation)')
        else:
            self.hook = None
            
    def __call__(self, x,name):
        self.gradients = []
        self.activations = []
        print("x shape",x.shape)
        return inference_detector(self.model, x)
    def debug_activation(self, module, input, output):
        print("Debugging")
        print(output.shape)
        print(output)
    
    def register_prerpocessing_output(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        # print(len(output))
        print(len(input),input[0].shape)

        last_output = input[0].detach().cpu().numpy()
    def register_activation_output(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        # print(len(output))
        print("O",len(output))
        last_output = output.detach().cpu().numpy() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        # print(last_output.shape)
        # print("-------------------")
        last_output = np.squeeze(last_output)
        self.activation_list.append(last_output)

    def register_activation_input(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        print("I", len(input),input[0].shape)
        last_output = input[0].detach().cpu().numpy() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        last_output = np.squeeze(last_output)
        self.activation_list.append(last_output)

    def clear(self):
        self.activations = []
        self.gradients = []
        for hook in self.hooks:
            hook.remove()