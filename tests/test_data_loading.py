import unittest
import torch
import numpy as np
import os
import tempfile
import shutil

# Add the parent directory to the path so we can import the modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import EcholancerDataset, get_data_loader, create_dummy_data_files

class TestDataLoading(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        print(f"Created temporary test directory: {self.test_dir}")
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)
        print(f"Removed temporary test directory: {self.test_dir}")
    
    def test_create_dummy_data_files(self):
        """Test creating dummy data files."""
        print("Testing create_dummy_data_files...")
        
        # Create dummy data files
        num_samples = 5
        create_dummy_data_files(self.test_dir, num_samples=num_samples, max_text_len=20, max_mel_len=100)
        
        # Check that files were created
        files = os.listdir(self.test_dir)
        self.assertEqual(len(files), num_samples)
        
        # Check that all files are .pt files
        for file in files:
            self.assertTrue(file.endswith('.pt'))
        
        print("  Created dummy data files: PASSED")
    
    def test_echolancer_dataset(self):
        """Test EcholancerDataset class."""
        print("Testing EcholancerDataset...")
        
        # Create dummy data files first
        num_samples = 5
        create_dummy_data_files(self.test_dir, num_samples=num_samples, max_text_len=20, max_mel_len=100)
        
        # Create dataset
        dataset = EcholancerDataset(self.test_dir)
        
        # Check dataset length
        self.assertEqual(len(dataset), num_samples)
        print("  Dataset length: PASSED")
        
        # Check dataset items
        for i in range(len(dataset)):
            item = dataset[i]
            self.assertIn('text', item)
            self.assertIn('mel', item)
            self.assertIn('speaker', item)
            self.assertIn('emotion', item)
            self.assertIn('text_len', item)
            self.assertIn('mel_len', item)
            
            # Check data types
            self.assertIsInstance(item['text'], np.ndarray)
            self.assertIsInstance(item['mel'], np.ndarray)
            self.assertIsInstance(item['speaker'], (int, np.integer))
            self.assertIsInstance(item['emotion'], np.ndarray)
            self.assertIsInstance(item['text_len'], (int, np.integer))
            self.assertIsInstance(item['mel_len'], (int, np.integer))
            
            # Check shapes
            self.assertGreater(len(item['text']), 0)
            self.assertGreater(len(item['mel']), 0)
            self.assertEqual(len(item['emotion']), 768)  # BERT hidden size
        
        print("  Dataset items: PASSED")
    
    def test_get_data_loader(self):
        """Test get_data_loader function."""
        print("Testing get_data_loader...")
        
        # Create dummy data files first
        num_samples = 8
        create_dummy_data_files(self.test_dir, num_samples=num_samples, max_text_len=20, max_mel_len=100)
        
        # Create data loader
        batch_size = 3
        loader = get_data_loader(self.test_dir, batch_size=batch_size)
        
        # Check that we can iterate through the loader
        batch_count = 0
        total_items = 0
        for batch in loader:
            batch_count += 1
            total_items += len(batch[0])  # batch[0] contains file paths/ids
            
            # Check batch structure
            self.assertEqual(len(batch), 8)  # (ids, raw_texts, speakers, texts, text_lens, mels, mel_lens, emotions)
            
            # Check data types
            self.assertIsInstance(batch[2], torch.Tensor)  # speakers
            self.assertIsInstance(batch[3], torch.Tensor)  # texts
            self.assertIsInstance(batch[4], torch.Tensor)  # text_lens
            self.assertIsInstance(batch[5], torch.Tensor)  # mels
            self.assertIsInstance(batch[6], torch.Tensor)  # mel_lens
            self.assertIsInstance(batch[7], torch.Tensor)  # emotions
            
            # Check batch dimensions
            batch_actual_size = batch[3].size(0)
            self.assertLessEqual(batch_actual_size, batch_size)
            
            # Check text and mel dimensions
            self.assertEqual(batch[3].dim(), 2)  # (batch_size, text_len)
            self.assertEqual(batch[5].dim(), 2)  # (batch_size, mel_len)
            self.assertEqual(batch[7].dim(), 2)  # (batch_size, emotion_dim)
            
            # Check that lengths are positive integers (basic sanity check)
            for i in range(batch_actual_size):
                recorded_text_len = batch[4][i].item()
                recorded_mel_len = batch[6][i].item()
                
                # Basic sanity checks - lengths should be positive
                self.assertGreater(recorded_text_len, 0)
                self.assertGreater(recorded_mel_len, 0)
                
                # Check that lengths don't exceed tensor dimensions
                self.assertLessEqual(recorded_text_len, batch[3].size(1))
                self.assertLessEqual(recorded_mel_len, batch[5].size(1))
        
        # Check that we processed all items
        self.assertEqual(total_items, num_samples)
        print("  Data loader iteration: PASSED")
        
        # Check batch count (should be ceil(num_samples/batch_size))
        expected_batches = (num_samples + batch_size - 1) // batch_size
        self.assertEqual(batch_count, expected_batches)
        print("  Batch count: PASSED")

if __name__ == '__main__':
    unittest.main()