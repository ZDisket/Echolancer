"""
Echolancer Standalone Package
=============================

This is a standalone implementation of the Echolancer model that can be 
trained and used for inference independently of the larger codebase.

Modules:
- model: Contains the Echolancer model definition and loss functions
- utils: Utility functions for model creation, parameter counting, and checkpoint handling
- data: Data loading and preprocessing utilities

For training, use train.py
For inference, use infer.py
"""

__version__ = "1.0.0"
__author__ = "Echolancer Team"

# Import main components for easy access
try:
    # Try relative imports first (when used as a package)
    from .model import Echolancer, EcholancerLoss
    from .utils import get_model, get_param_num
except ImportError:
    # Fall back to absolute imports (when running scripts directly)
    from model import Echolancer, EcholancerLoss
    from utils import get_model, get_param_num

__all__ = [
    "Echolancer",
    "EcholancerLoss",
    "get_model",
    "get_param_num"
]