import unittest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import shutil
import yaml

# Add the parent directory to the path so we can import the modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import get_model, get_param_num, save_checkpoint, load_checkpoint
from utils.config import load_config, set_seed, get_device
from model.loss import EcholancerLoss

class TestTrainingUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        print(f"Created temporary test directory: {self.test_dir}")
        
        # Tiny model parameters for VRAM efficiency
        self.tiny_params = {
            'vocab_size': 50,
            'encoder_hidden': 64,
            'encoder_head': 2,
            'encoder_layer': 2,
            'decoder_hidden': 64,
            'decoder_layer': 2,
            'decoder_head': 2,
            'encoder_dropout': 0.1,
            'decoder_dropout': 0.1,
            'mel_channels': 40,
            'emotion_channels': 64,
            'speaker_channels': 16,
            'multi_speaker': False,
            'n_speaker': 1,
            'use_alibi': True,
            'alibi_alpha': 1.0,
            'activation': 'relu'
        }
        
        # Create test config files
        self.model_config_path = os.path.join(self.test_dir, 'test_model.yaml')
        self.train_config_path = os.path.join(self.test_dir, 'test_train.yaml')
        
        # Model config
        model_config = {
            'model': {
                'vocab_size': 50,
                'mel_channels': 40,
                'emotion_channels': 64,
                'speaker_channels': 16,
                'multi_speaker': False,
                'n_speaker': 1,
                'use_alibi': True,
                'alibi_alpha': 1.0,
                'activation': 'relu',
                'encoder': {
                    'hidden': 64,
                    'head': 2,
                    'layer': 2,
                    'dropout': 0.1,
                    'forward_expansion': 2
                },
                'decoder': {
                    'hidden': 64,
                    'head': 2,
                    'layer': 2,
                    'dropout': 0.1
                }
            }
        }
        
        # Train config
        train_config = {
            'optimizer': {
                'type': 'adamw',
                'learning_rate': 0.0001,
                'weight_decay': 0.01,
                'eps': 1e-06,
                'betas': [0.9, 0.98],
                'grad_clip_thresh': 1.0,
                'grad_acc_step': 1
            },
            'scheduler': {
                'type': 'warmup_cosine',
                'warmup_steps': 100
            },
            'loss': {
                'pos_weight': 8.0
            },
            'training': {
                'batch_size': 2,
                'epochs': 2,
                'seed': 1234,
                'save_step': 50,
                'log_step': 10,
                'val_step': 25,
                'test_step': 50,
                'eval_step': 50
            },
            'early_stopping': {
                'patience': 3,
                'min_delta': 0.001
            }
        }
        
        # Save configs
        with open(self.model_config_path, 'w') as f:
            yaml.dump(model_config, f)
        
        with open(self.train_config_path, 'w') as f:
            yaml.dump(train_config, f)
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)
        print(f"Removed temporary test directory: {self.test_dir}")
    
    def test_get_model(self):
        """Test get_model function."""
        print("Testing get_model...")
        
        # Create model
        model = get_model(**self.tiny_params)
        
        # Check that model is created
        self.assertIsNotNone(model)
        self.assertIsInstance(model, nn.Module)
        
        # Check that model has expected components
        self.assertTrue(hasattr(model, 'encoder'))
        self.assertTrue(hasattr(model, 'decoder'))
        self.assertTrue(hasattr(model, 'emotion_encoder'))
        
        # Check parameter count
        param_count = get_param_num(model)
        self.assertGreater(param_count, 0)
        
        print(f"  Model parameter count: {param_count}")
        print("  get_model: PASSED")
    
    def test_save_and_load_checkpoint(self):
        """Test save_checkpoint and load_checkpoint functions."""
        print("Testing save/load checkpoint...")
        
        # Create model and optimizer
        model = get_model(**self.tiny_params)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.test_dir, 'test_checkpoint')
        save_checkpoint(model, optimizer, checkpoint_path, step=100)
        
        # Check that checkpoint file was created
        checkpoint_file = f"{checkpoint_path}_100.pth"
        self.assertTrue(os.path.exists(checkpoint_file))
        print("  Save checkpoint: PASSED")
        
        # Load checkpoint
        model_new = get_model(**self.tiny_params)
        optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=0.0001)
        model_loaded, optimizer_loaded = load_checkpoint(model_new, optimizer_new, checkpoint_file, get_device())
        
        # Check that model was loaded
        self.assertIsNotNone(model_loaded)
        self.assertIsNotNone(optimizer_loaded)
        print("  Load checkpoint: PASSED")
    
    def test_config_loading(self):
        """Test config loading functions."""
        print("Testing config loading...")
        
        # Load model config
        model_config = load_config(self.model_config_path)
        self.assertIsNotNone(model_config)
        self.assertIn('model', model_config)
        print("  Load model config: PASSED")
        
        # Load train config
        train_config = load_config(self.train_config_path)
        self.assertIsNotNone(train_config)
        self.assertIn('optimizer', train_config)
        print("  Load train config: PASSED")
        
        # Test set_seed
        set_seed(1234)
        print("  set_seed: PASSED")
    
    def test_device_detection(self):
        """Test device detection."""
        print("Testing device detection...")
        
        device = get_device()
        self.assertIsNotNone(device)
        self.assertIsInstance(device, torch.device)
        print(f"  Detected device: {device}")
        print("  get_device: PASSED")
    
    def test_loss_function_creation(self):
        """Test creation of loss functions."""
        print("Testing loss function creation...")
        
        # Test EcholancerLoss
        loss_fn = EcholancerLoss()
        self.assertIsNotNone(loss_fn)
        self.assertIsInstance(loss_fn, nn.Module)
        print("  EcholancerLoss creation: PASSED")

if __name__ == '__main__':
    unittest.main()