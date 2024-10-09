from typing import Any, Dict, Union
import yaml

import os
class Config:
    @staticmethod
    def join(loader, node):
        """Concatenates sequence of values."""
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """Loads and returns YAML file as dictionary."""
        yaml.add_constructor('!join', Config.join)
        try:
            with open(path) as file:
                return yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {path} not found.")
        except yaml.YAMLError:
            raise yaml.YAMLError(f"Error in configuration file {path}.")

    def __init__(self, path: str) -> None:
        """Initialize the configuration object."""
        self.config_dict: Dict[str, Any] = self.load_yaml(path)
        self.verbose: bool = self.config_dict.get('verbose', False)
        self.exp_name: str = self.config_dict.get('experiment_name', 'default_exp_name')
        self.wandb: Dict[str, Any] = self.config_dict.get('wandb', {})
        #Separation of these is needed!
        self.detection: Dict[str, Any] = self.config_dict.get('detection', {})
        self.extraction: Dict[str, Any] = self.config_dict.get('extraction', {})
        self.introspection: Dict[str, Any] = self.config_dict.get('introspection', {})
