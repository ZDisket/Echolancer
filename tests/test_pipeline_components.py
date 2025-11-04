import unittest
import torch
import torch.nn as nn
import numpy as np
import os
import tempfile
import shutil

# Add the parent directory to the path so we can import the modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock the pipeline components for testing
class MockBERTFrontEnd:
    """Mock BERT frontend for testing."""
    def __init__(self, is_cuda=False, model_name="answerdotai/ModernBERT-base"):
        self.is_cuda = is_cuda
        self.model_name = model_name
        print(f"Mock EmbeddingFE initialized with {model_name}")
    
    def infer(self, text):
        """Mock inference method."""
        # Return mock emotion encodings
        batch_size = 1
        seq_len = len(text.split())  # Simple word count
        encoded_layers = torch.randn(batch_size, seq_len, 768)
        pooled = torch.randn(batch_size, 768)
        
        if self.is_cuda:
            encoded_layers = encoded_layers.cuda()
            pooled = pooled.cuda()
            
        return encoded_layers, pooled

class MockNeuCodecFE:
    """Mock neural audio codec frontend for testing."""
    def __init__(self, is_cuda=False, model_name="neuphonic/distill-neucodec"):
        self.is_cuda = is_cuda
        self.model_name = model_name
        self.sampling_rate = 24000
        print(f"Mock NeuCodecFE initialized with {model_name}")
    
    def encode_audio(self, audio_tensor):
        """Mock audio encoding method."""
        # Return mock discrete codes
        batch_size, channels, audio_len = audio_tensor.shape
        code_len = audio_len // 100  # Simplified length conversion
        codes = torch.randint(0, 1024, (batch_size, channels, code_len))
        return codes
    
    def decode_codes(self, codes):
        """Mock codes decoding method."""
        # Return mock reconstructed audio
        batch_size, channels, code_len = codes.shape
        audio_len = code_len * 100  # Simplified length conversion
        audio = torch.randn(batch_size, channels, audio_len)
        return audio
    
    def infer(self, audio_tensor):
        """Mock inference method."""
        codes = self.encode_audio(audio_tensor)
        reconstructed = self.decode_codes(codes)
        return reconstructed

class TestPipelineComponents(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def test_mock_bert_frontend(self):
        """Test mock BERT frontend."""
        print("Testing MockBERTFrontEnd...")
        
        # Test CPU version
        bert_cpu = MockBERTFrontEnd(is_cuda=False)
        text = "This is a test sentence."
        encoded_layers, pooled = bert_cpu.infer(text)
        
        self.assertIsInstance(encoded_layers, torch.Tensor)
        self.assertIsInstance(pooled, torch.Tensor)
        self.assertEqual(encoded_layers.shape[1], len(text.split()))  # Word count
        self.assertEqual(pooled.shape[1], 768)
        print("  MockBERTFrontEnd (CPU): PASSED")
        
        # Test GPU version if available
        if torch.cuda.is_available():
            bert_gpu = MockBERTFrontEnd(is_cuda=True)
            encoded_layers_gpu, pooled_gpu = bert_gpu.infer(text)
            
            self.assertEqual(encoded_layers_gpu.device.type, 'cuda')
            self.assertEqual(pooled_gpu.device.type, 'cuda')
            print("  MockBERTFrontEnd (GPU): PASSED")
    
    def test_mock_neucodec_fe(self):
        """Test mock NeuCodec frontend."""
        print("Testing MockNeuCodecFE...")
        
        # Test CPU version
        neucodec_cpu = MockNeuCodecFE(is_cuda=False)
        audio = torch.randn(1, 1, 24000)  # (batch, channels, samples) - 1 second at 24kHz
        codes = neucodec_cpu.encode_audio(audio)
        reconstructed = neucodec_cpu.decode_codes(codes)
        
        self.assertIsInstance(codes, torch.Tensor)
        self.assertIsInstance(reconstructed, torch.Tensor)
        print("  MockNeuCodecFE (CPU): PASSED")
        
        # Test GPU version if available
        if torch.cuda.is_available():
            neucodec_gpu = MockNeuCodecFE(is_cuda=True)
            audio_gpu = audio.cuda()
            codes_gpu = neucodec_gpu.encode_audio(audio_gpu)
            reconstructed_gpu = neucodec_gpu.decode_codes(codes_gpu)
            
            self.assertEqual(codes_gpu.device.type, 'cuda')
            self.assertEqual(reconstructed_gpu.device.type, 'cuda')
            print("  MockNeuCodecFE (GPU): PASSED")
    
    def test_pipeline_integration(self):
        """Test integration of all pipeline components."""
        print("Testing pipeline integration...")
        
        # Initialize all components
        bert_model = MockBERTFrontEnd(is_cuda=(self.device.type == 'cuda'))
        neucodec_fe = MockNeuCodecFE(is_cuda=(self.device.type == 'cuda'))
        
        # Test sentences
        test_sentences = ["This is a test sentence.", "Another example."]
        test_speakers = [0, 1]
        
        # Process each sentence
        for sentence, speaker_id in zip(test_sentences, test_speakers):
            # Step 1: Get emotion encoding from BERT
            em_blocks, em_hidden = bert_model.infer(sentence)
            
            # Step 2: Generate discrete audio codes with mock Echolancer model
            seq_len = len(sentence.split())
            token_ids = torch.randint(0, 1000, (1, seq_len))  # Mock token IDs
            
            # Step 3: Generate mock discrete audio codes
            codes = torch.randint(0, 1024, (1, 1, seq_len * 10))  # Mock discrete audio codes
            
            # Step 4: Convert discrete codes to audio using NeuCodec
            audio = neucodec_fe.decode_codes(codes)
            
            # Verify outputs
            self.assertIsInstance(em_blocks, torch.Tensor)
            self.assertIsInstance(em_hidden, torch.Tensor)
            self.assertIsInstance(codes, torch.Tensor)
            self.assertIsInstance(audio, torch.Tensor)
            
            # Check shapes
            self.assertEqual(em_blocks.shape[1], seq_len)  # Same as word count
            self.assertEqual(em_hidden.shape[1], 768)  # BERT hidden size
            self.assertEqual(codes.shape[0], 1)  # Batch size
            self.assertEqual(audio.shape[0], 1)  # Batch size
        
        print("  Pipeline integration: PASSED")

if __name__ == '__main__':
    unittest.main()