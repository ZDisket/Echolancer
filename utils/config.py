import yaml
import os
import random
import numpy as np
import torch

def load_config(config_path):
    """
    Load configuration from a YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    """
    Save configuration to a YAML file.
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get the device to use for training (CUDA if available, otherwise CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_config(config, updates):
    """
    Update configuration with new values.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            update_config(config[key], value)
        else:
            config[key] = value
    return config