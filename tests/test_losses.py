import unittest
import torch
import torch.nn as nn
import numpy as np
import os

# Add the parent directory to the path so we can import the modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.loss import EcholancerLoss

class TestLossFunctions(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def test_echolancer_loss(self):
        """Test EcholancerLoss function."""
        print("Testing EcholancerLoss...")
        
        echolancer_loss = EcholancerLoss()
        
        # Create mock batch data (similar to what the model produces)
        batch = (
            ["sample_1", "sample_2"],  # ids
            ["text 1", "text 2"],      # raw_texts
            torch.randint(0, 1, (2,)).to(self.device),  # speakers
            torch.randint(0, 50, (2, 10)).to(self.device),  # texts
            torch.tensor([10, 8]).to(self.device),  # src_lens
            torch.randint(0, 1010, (2, 15)).to(self.device),  # mels
            torch.tensor([15, 12]).to(self.device),  # mel_lens
            torch.randn(2, 768).to(self.device)  # em_hidden
        )
        
        # Create mock model output (similar to what the model produces)
        model_out = (
            torch.zeros(2, 10, dtype=torch.bool).to(self.device),  # text_mask
            torch.zeros(2, 15, dtype=torch.bool).to(self.device),  # mel_mask
            torch.randn(2, 1, 9, 10).to(self.device),  # attn_logprob
            torch.zeros(2, 9, dtype=torch.bool).to(self.device),  # x_mask_in (should match logits time dimension)
            torch.randn(2, 9, 1010).to(self.device),  # logits 
            torch.randint(0, 1010, (2, 10)).to(self.device)  # indices_gt
        )
        
        # Compute loss
        losses = echolancer_loss(batch, model_out)
        
        # Check that we get the expected number of losses
        self.assertEqual(len(losses), 3)
        
        # Check that all losses are scalars
        for loss in losses:
            self.assertIsInstance(loss.item(), float)
            self.assertGreaterEqual(loss.item(), 0.0)
        
        print("  EcholancerLoss: PASSED")

if __name__ == '__main__':
    unittest.main()