from typing import List, Union, Tuple, Dict, Any
import numpy as np
import os

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
        #TODO: better way to do this, indexes are not used right now, save activation must be generalized
        if extract:
            print("Hook initialized")
            self.hook_layer = method['hook']['layer_index']
            self.hook = eval(f'self.model.{method["hook"]["layer"]}.register_forward_hook(self.save_activation)')
        else:
            self.hook = None
            
        check_and_create_folder(os.path.join(ROOT_DIR,self.save_dir,"features"))

    def __call__(self, x,name):
        self.save_name = name.split('/')[-1]
        self.gradients = []
        self.activations = []
        return inference_detector(self.model, x)
    
    def save_activation(self,module, input, output):
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
        self.hook.remove()