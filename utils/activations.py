from typing import List, Union, Tuple, Dict, Any
import numpy as np
import os
from mmdet3d.apis import inference_detector, init_model

class Activations:

    def __init__(self,model,
                 config) -> None:
        self.model = model
        self.save_dir = config['save_dir']
        #TODO: better way to do this, indexes are not used right now, save activation must be generalized
        self.hook = eval(f'self.model.{config["hook"]["layer"]}.register_forward_hook(self.save_activation)')
        
    def __call__(self, x,name):
        self.save_name = name
        self.gradients = []
        self.activations = []
        return inference_detector(self.model, x)
    
    def save_activation(self,module, input, output):
        last_output = output[2].detach().cpu().numpy()
        last_output = np.squeeze(last_output)
        np.save(os.path.join(self.save_dir,"features" ,self.save_name + '.npy'), last_output)
        return output