"""
Hard Hat / PPE Detection - Utility Functions
=============================================
Common utility functions for PPE detection project.
"""

from pathlib import Path
import yaml
import torch


def load_config(data_yaml_path):
    """Load dataset configuration from data.yaml."""
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_class_names(data_yaml_path):
    """Get class names from data.yaml."""
    config = load_config(data_yaml_path)
    return config.get('names', ['head', 'helmet', 'person'])


def get_device():
    """Get available device (GPU if available, else CPU)."""
    return 0 if torch.cuda.is_available() else 'cpu'


def get_batch_size(gpu_memory_gb=None):
    """Get recommended batch size based on GPU memory."""
    if gpu_memory_gb is None:
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            return 2
    
    if gpu_memory_gb >= 12:
        return 16
    elif gpu_memory_gb >= 8:
        return 12
    elif gpu_memory_gb >= 6:
        return 8
    else:
        return 4


def find_latest_model(runs_dir='runs/detect'):
    """Find the latest trained model."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return None
    
    # Look for PPE detection models
    ppe_dirs = [d for d in runs_path.glob('*') if d.is_dir() and 'ppe' in d.name.lower()]
    if not ppe_dirs:
        return None
    
    # Sort by modification time
    ppe_dirs = sorted(ppe_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
    
    for ppe_dir in ppe_dirs:
        best_weights = ppe_dir / 'weights' / 'best.pt'
        if best_weights.exists():
            return best_weights
    
    return None



