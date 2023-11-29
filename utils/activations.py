from typing import List, Union, Tuple, Dict, Any
import numpy as np
import os
import pickle
from definitions import ROOT_DIR
try:
    from mmdet3d.apis import inference_detector, init_model
    from utils.utils import load_detection_model, check_and_create_folder
except:
    print("DIfferent environment, some packages will be missing")
    pass
class Activations:

    def __init__(self,
                 config,extract=True) -> None:
        self.model = load_detection_model(config)
        self.config = config
        method = config['method']
        self.save_dir = method['save_dir']
        self.extension =  method['extension']
        self.hooks = []
        self.activation_list= []
        #TODO: better way to do this, indexes are not used right now, save activation must be generalized
        if extract:
            print("Hook initialized")
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
    def save_multi_layer_activation(self):
        save_name = self.save_name.replace(self.extension,'.pkl') #should be more generic currently depends on image, maybe jsut remove extension
        print("Saving", save_name, "to", self.save_dir)
        with open(os.path.join(self.save_dir,"features" ,save_name), 'wb') as f:
            pickle.dump(self.activation_list, f)
        # print(save_name,self.save_dir)
        #np.save(os.path.join(self.save_dir,"features" ,save_name), self.activation_list)
        del self.activation_list
        self.activation_list = []
    def register_activation_output(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        # print(len(output))
        last_output = output.detach().cpu().numpy() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        last_output = np.squeeze(last_output)
        self.activation_list.append(last_output)

    def register_activation_input(self,module, input, output):
        # print(output[0].shape,output[1].shape)
        last_output = input[0].detach().cpu().numpy() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        last_output = np.squeeze(last_output)
        self.activation_list.append(last_output)

    def save_backbone_output(self,module, input, output): #Original implementation
        # print(output[0].shape,output[1].shape)
        last_output = output[self.hook_layer].detach().cpu().numpy() #TODO: generalize this
        # print("Last output shape",last_output.shape)
        last_output = np.squeeze(last_output)
        save_name = self.save_name.replace(self.extension,'.npy') #should be more generic currently depends on image, maybe jsut remove extension
        # print(save_name,self.save_dir)
        np.save(os.path.join(self.save_dir,"features" ,save_name), last_output)
    def clear(self):
        self.activations = []
        self.gradients = []
        for hook in self.hooks:
            hook.remove()