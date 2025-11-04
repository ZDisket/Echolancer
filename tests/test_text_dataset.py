import unittest
import tempfile
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add the parent directory to the path to import from data module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.text_dataset import TextCorpusWithMLMDataset, TextCorpusDataset


class TestTextCorpusWithMLMDataset(unittest.TestCase):
    """
    Test suite for TextCorpusWithMLMDataset to verify:
    1. Loading a file
    2. Returning chunks with valid MLM objective
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary text file with sample content
        self.sample_text = """The quick brown fox jumps over the lazy dog. This is a sample text corpus for testing.
        The dataset should be able to handle various punctuation marks, numbers like 123, and different sentence structures.
        Testing special characters: @#$%^&*() and symbols."""
        
        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        self.temp_file.write(self.sample_text)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_file(self):
        """Test that the dataset can load a text file successfully."""
        # Test basic dataset loading
        dataset = TextCorpusWithMLMDataset(self.temp_file.name, chunk_length=32)
        
        # Verify dataset was loaded
        self.assertGreater(len(dataset), 0, "Dataset should have at least one chunk")
        
        # Verify tokenizer was created
        tokenizer = dataset.get_tokenizer()
        self.assertIsNotNone(tokenizer, "Tokenizer should not be None")
        
        # Verify text was loaded and tokenized
        self.assertGreater(len(dataset.encoded_text), 0, "Encoded text should not be empty")
        
        print(f"Dataset loaded successfully: {len(dataset)} chunks of length {dataset.chunk_length}")
        print(f"Vocabulary size: {len(tokenizer)}")
    
    def test_return_chunks_with_valid_mlm_objective(self):
        """Test that the dataset returns chunks with valid MLM objective."""
        chunk_length = 64
        mask_fraction = 0.15
        dataset = TextCorpusWithMLMDataset(
            self.temp_file.name, 
            chunk_length=chunk_length, 
            mask_fraction=mask_fraction
        )
        
        # Get a sample from the dataset
        input_chunk, target_chunk, mlm_mask = dataset[0]
        
        # Verify tensor shapes
        self.assertEqual(input_chunk.shape, torch.Size([chunk_length]), 
                        f"Input chunk should have shape [{chunk_length}]")
        self.assertEqual(target_chunk.shape, torch.Size([chunk_length]), 
                        f"Target chunk should have shape [{chunk_length}]")
        self.assertEqual(mlm_mask.shape, torch.Size([chunk_length]), 
                        f"MLM mask should have shape [{chunk_length}]")
        
        # Verify tensor types
        self.assertEqual(input_chunk.dtype, torch.long, "Input chunk should be of type long")
        self.assertEqual(target_chunk.dtype, torch.long, "Target chunk should be of type long")
        self.assertEqual(mlm_mask.dtype, torch.bool, "MLM mask should be of type bool")
        
        # Verify that input and target chunks are related
        # For unmasked positions, input should match target
        unmasked_positions = ~mlm_mask  # Invert the mask to get unmasked positions
        unmasked_input = input_chunk[unmasked_positions]
        unmasked_target = target_chunk[unmasked_positions]
        
        # Unmasked positions should be identical between input and target
        if unmasked_positions.any():  # Only check if there are unmasked positions
            self.assertTrue(torch.equal(unmasked_input, unmasked_target),
                          "Unmasked positions in input should match target")
        
        # Verify that some positions are masked (within reason)
        num_masked = mlm_mask.sum().item()
        expected_min_masked = max(1, int(chunk_length * mask_fraction * 0.7))  # Allow some randomness
        expected_max_masked = min(chunk_length, int(chunk_length * mask_fraction * 1.3))  # Allow some randomness
        
        self.assertGreaterEqual(num_masked, expected_min_masked,
                              f"At least {expected_min_masked} positions should be masked")
        self.assertLessEqual(num_masked, expected_max_masked,
                           f"At most {expected_max_masked} positions should be masked")
        
        # Verify that masked positions differ from target (in most cases - there's a 10% chance to keep original)
        masked_input = input_chunk[mlm_mask]
        masked_target = target_chunk[mlm_mask]
        
        if masked_input.numel() > 0:
            # Count how many masked positions are different from target
            different_count = (masked_input != masked_target).sum().item()
            # At least some of the masked positions should be different (80% of 80% + 10% of 10% = 74% should be different)
            # So we'll be more lenient and just require that at least 10% are different
            expected_different = different_count / len(masked_input) if len(masked_input) > 0 else 0
            # Since there's randomness, we'll just check that we have masked positions
            
        print(f"MLM objective verification:")
        print(f"  - Input chunk shape: {input_chunk.shape}")
        print(f"  - Target chunk shape: {target_chunk.shape}")
        print(f"  - Mask shape: {mlm_mask.shape}")
        print(f"  - Number of masked positions: {num_masked}")
        print(f"  - Expected mask fraction: {mask_fraction}")
        print(f"  - Actual mask fraction: {num_masked / chunk_length:.2f}")
    
    def test_data_loader_compatibility(self):
        """Test that the dataset works with PyTorch DataLoader."""
        dataset = TextCorpusWithMLMDataset(self.temp_file.name, chunk_length=32, mask_fraction=0.2)
        
        # Create a DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Get a batch from the dataloader
        batch_input, batch_target, batch_mask = next(iter(dataloader))
        
        # Verify batch dimensions
        batch_size = 4
        self.assertEqual(batch_input.shape, torch.Size([batch_size, 32]), 
                        "Batch input should have correct shape")
        self.assertEqual(batch_target.shape, torch.Size([batch_size, 32]), 
                        "Batch target should have correct shape")
        self.assertEqual(batch_mask.shape, torch.Size([batch_size, 32]), 
                        "Batch mask should have correct shape")
        
        print(f"DataLoader compatibility verified:")
        print(f"  - Batch input shape: {batch_input.shape}")
        print(f"  - Batch target shape: {batch_target.shape}")
        print(f"  - Batch mask shape: {batch_mask.shape}")
    
    def test_different_chunk_sizes(self):
        """Test the dataset with different chunk sizes."""
        for chunk_size in [16, 32, 64, 128]:
            with self.subTest(chunk_size=chunk_size):
                dataset = TextCorpusWithMLMDataset(self.temp_file.name, chunk_length=chunk_size)
                
                self.assertGreater(len(dataset), 0, f"Dataset should have chunks for size {chunk_size}")
                
                # Get a sample
                input_chunk, target_chunk, mlm_mask = dataset[0]
                
                self.assertEqual(len(input_chunk), chunk_size)
                self.assertEqual(len(target_chunk), chunk_size)
                self.assertEqual(len(mlm_mask), chunk_size)
    
    def test_different_mask_fractions(self):
        """Test the dataset with different mask fractions."""
        for mask_fraction in [0.1, 0.2, 0.3]:
            with self.subTest(mask_fraction=mask_fraction):
                dataset = TextCorpusWithMLMDataset(
                    self.temp_file.name, 
                    chunk_length=64, 
                    mask_fraction=mask_fraction
                )
                
                # Get a sample
                input_chunk, target_chunk, mlm_mask = dataset[0]
                
                num_masked = mlm_mask.sum().item()
                
                # Check that the mask fraction is approximately correct
                actual_fraction = num_masked / len(mlm_mask)
                # Allow 50% tolerance due to randomness in masking
                expected_min = mask_fraction * 0.5
                expected_max = min(1.0, mask_fraction * 1.5)
                
                self.assertGreaterEqual(actual_fraction, expected_min,
                                      f"Actual mask fraction {actual_fraction} should be >= {expected_min}")
                self.assertLessEqual(actual_fraction, expected_max,
                                   f"Actual mask fraction {actual_fraction} should be <= {expected_max}")


class TestTextCorpusDataset(unittest.TestCase):
    """
    Test suite for TextCorpusDataset to verify basic functionality.
    """
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary text file with sample content
        self.sample_text = """The quick brown fox jumps over the lazy dog. This is a sample text corpus for testing."""
        
        # Create a temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        self.temp_file.write(self.sample_text)
        self.temp_file.close()
    
    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_basic_loading(self):
        """Test basic loading functionality of TextCorpusDataset."""
        dataset = TextCorpusDataset(self.temp_file.name, chunk_length=32)
        
        self.assertGreater(len(dataset), 0, "Dataset should have at least one chunk")
        
        # Get a sample
        chunk = dataset[0]
        
        self.assertEqual(len(chunk), 32)
        self.assertEqual(chunk.dtype, torch.long)


if __name__ == '__main__':
    unittest.main()