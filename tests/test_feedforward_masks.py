import unittest
import torch
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.echolancer import FeedForward

class TestFeedForwardMasks(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def test_feedforward_with_2d_mask(self):
        """Test FeedForward with 2D sequence mask."""
        ff = FeedForward(
            d_model=64,
            d_ff=128,
            dropout=0.1,
            activation='relu'
        ).to(self.device)

        # Create test tensor
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64).to(self.device)

        # Test with 2D mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(self.device)
        mask[:, 8:] = True  # Mask last 2 positions
        output_with_mask = ff(x, mask)

        # Check that masked positions are zero
        self.assertTrue(torch.allclose(output_with_mask[:, 8:, :], torch.zeros_like(output_with_mask[:, 8:, :])))
        print("  FeedForward with 2D mask: PASSED")

    def test_feedforward_with_4d_attention_mask(self):
        """Test FeedForward with 4D attention mask."""
        ff = FeedForward(
            d_model=64,
            d_ff=128,
            dropout=0.1,
            activation='relu'
        ).to(self.device)

        # Create test tensor
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64).to(self.device)

        # Test with 4D attention mask (similar to what Transformer layers use)
        # Create a mask where the last 2 positions are padded (False = padded)
        seq_mask_2d = torch.ones(batch_size, seq_len, dtype=torch.bool).to(self.device)
        seq_mask_2d[:, 8:] = False  # Last 2 positions are padded
        
        # Convert to 4D attention mask format (True = valid positions)
        mask_4d = seq_mask_2d.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_len, seq_len)
        mask_4d = mask_4d & mask_4d.transpose(2, 3)  # Symmetric mask
        
        output_with_mask = ff(x, mask_4d)

        # Check that masked positions are zero
        self.assertTrue(torch.allclose(output_with_mask[:, 8:, :], torch.zeros_like(output_with_mask[:, 8:, :])))
        print("  FeedForward with 4D attention mask: PASSED")

    def test_feedforward_without_mask(self):
        """Test FeedForward without mask."""
        ff = FeedForward(
            d_model=64,
            d_ff=128,
            dropout=0.1,
            activation='relu'
        ).to(self.device)

        # Create test tensor
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64).to(self.device)

        # Test without mask
        output_no_mask = ff(x)
        self.assertEqual(output_no_mask.shape, (batch_size, seq_len, 64))
        print("  FeedForward without mask: PASSED")

if __name__ == '__main__':
    unittest.main()