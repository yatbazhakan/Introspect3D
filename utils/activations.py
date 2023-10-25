from typing import List, Union, Tuple, Dict, Any
import numpy as np
import os
from mmdet3d.apis import inference_detector, init_model
from utils.utils import load_detection_model, check_and_create_folder
from definitions import ROOT_DIR
class Activations:

    def __init__(self,
                 config,extract=True) -> None:
        self.model = load_detection_model(config)
        self.config = config
        method = config['method']
        self.save_dir = method['save_dir']
        #TODO: better way to do this, indexes are not used right now, save activation must be generalized
        if extract:
            self.hook = eval(f'self.model.{method["hook"]["layer"]}.register_forward_hook(self.save_activation)')
        else:
            self.hook = None
            
        check_and_create_folder(os.path.join(ROOT_DIR,self.save_dir,"features"))

    def __call__(self, x,name):
        self.save_name = name
        self.gradients = []
        self.activations = []
        return inference_detector(self.model, x)
    
    def save_activation(self,module, input, output):
        last_output = output[2].detach().cpu().numpy()
        last_output = np.squeeze(last_output)
        save_name = self.save_name.replace('.png','.npy') #should be more generic currently depends on image, maybe jsut remove extension
        np.save(os.path.join(self.save_dir,"features" ,save_name), last_output)

    def clear(self):
        self.activations = []
        self.gradients = []
        self.hook.remove()