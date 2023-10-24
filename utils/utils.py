from mmdet3d.utils import register_all_modules
from mmdet3d.apis import inference_detector, init_model
import os
from utils.config import Config

def load_detection_model(config: Config):
    """Loads the detection model."""
    root_dir = config['model']['root_dir']
    checkpoint_full_path = os.path.join(root_dir,config['model']['checkpoint_dir'], config['model']['checkpoint'])
    model_config_full_path = os.path.join(root_dir,config['model']['config_dir'], config['model']['config'])
    model = init_model(model_config_full_path, checkpoint_full_path, config['device'])
    model.eval()
    return model
