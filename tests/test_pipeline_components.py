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
        print(f"Mock BERTFrontEnd initialized with {model_name}")
    
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

class MockScriptedPreEncoder:
    """Mock scripted pre-encoder for testing."""
    def __init__(self, model_dir, device='cpu'):
        self.device = torch.device(device)
        self.config = {'model': {'mel_channels': 40}}
        print(f"Mock ScriptedPreEncoder initialized on {device}")
    
    @property
    def mel_channels(self):
        return self.config['model']['mel_channels']
    
    def decode(self, indices, lengths=None):
        """Mock decode method."""
        batch_size, seq_len = indices.shape[:2]
        mel_channels = self.mel_channels
        spectrogram = torch.randn(batch_size, seq_len, mel_channels)
        return spectrogram.to(self.device)

class MockISTFTNetFE:
    """Mock ISTFTNet frontend for testing."""
    def __init__(self, gen=None, stft=None):
        self.sampling_rate = 22050
        print("Mock ISTFTNetFE initialized")
    
    def infer(self, x):
        """Mock inference method."""
        # Return mock audio waveform
        batch_size, seq_len = x.shape[:2]
        audio_len = seq_len * 256  # Assuming 256 hop size
        audio = torch.randn(batch_size, audio_len)
        return audio.numpy()

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
    
    def test_mock_scripted_preencoder(self):
        """Test mock scripted pre-encoder."""
        print("Testing MockScriptedPreEncoder...")
        
        # Test CPU version
        preenc_cpu = MockScriptedPreEncoder('.', device='cpu')
        indices = torch.randint(0, 1000, (1, 50, 8))  # (batch, seq_len, quantizers)
        spectrogram = preenc_cpu.decode(indices)
        
        self.assertIsInstance(spectrogram, torch.Tensor)
        self.assertEqual(spectrogram.shape, (1, 50, 40))  # (batch, seq_len, mel_channels)
        print("  MockScriptedPreEncoder (CPU): PASSED")
        
        # Test GPU version if available
        if torch.cuda.is_available():
            preenc_gpu = MockScriptedPreEncoder('.', device='cuda')
            indices_gpu = indices.cuda()
            spectrogram_gpu = preenc_gpu.decode(indices_gpu)
            
            self.assertEqual(spectrogram_gpu.device.type, 'cuda')
            print("  MockScriptedPreEncoder (GPU): PASSED")
    
    def test_mock_istftnet_fe(self):
        """Test mock ISTFTNet frontend."""
        print("Testing MockISTFTNetFE...")
        
        istftnet = MockISTFTNetFE()
        spectrogram = torch.randn(1, 100, 40)  # (batch, seq_len, mel_channels)
        audio = istftnet.infer(spectrogram)
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.shape, (1, 100 * 256))  # Assuming 256 hop size
        print("  MockISTFTNetFE: PASSED")
    
    def test_pipeline_integration(self):
        """Test integration of all pipeline components."""
        print("Testing pipeline integration...")
        
        # Initialize all components
        bert_model = MockBERTFrontEnd(is_cuda=(self.device.type == 'cuda'))
        pre_encoder = MockScriptedPreEncoder('.', device=self.device)
        istftnet = MockISTFTNetFE()
        
        # Test sentences
        test_sentences = ["This is a test sentence.", "Another example."]
        test_speakers = [0, 1]
        
        # Process each sentence
        for sentence, speaker_id in zip(test_sentences, test_speakers):
            # Step 1: Get emotion encoding from BERT
            em_blocks, em_hidden = bert_model.infer(sentence)
            
            # Step 2: Generate tokens with mock Echolancer model
            seq_len = len(sentence.split())
            token_ids = torch.randint(0, 1000, (1, seq_len))  # Mock token IDs
            
            # Step 3: Decode indices to spectrogram
            indices = torch.randint(0, 1000, (1, seq_len, 8))  # Mock indices
            spectrogram = pre_encoder.decode(indices)
            
            # Step 4: Convert spectrogram to audio
            audio = istftnet.infer(spectrogram)
            
            # Verify outputs
            self.assertIsInstance(em_blocks, torch.Tensor)
            self.assertIsInstance(em_hidden, torch.Tensor)
            self.assertIsInstance(spectrogram, torch.Tensor)
            self.assertIsInstance(audio, np.ndarray)
            
            # Check shapes
            self.assertEqual(em_blocks.shape[1], seq_len)  # Same as word count
            self.assertEqual(em_hidden.shape[1], 768)  # BERT hidden size
            self.assertEqual(spectrogram.shape[1], seq_len)  # Same sequence length
            self.assertEqual(audio.shape[1], seq_len * 256)  # Assuming 256 hop size
        
        print("  Pipeline integration: PASSED")

if __name__ == '__main__':
    unittest.main()