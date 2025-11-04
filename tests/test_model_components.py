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

from model.echolancer import (
    Echolancer, MultiHeadAttention, FeedForward,
    TransformerEncoderLayer, TransformerDecoderLayer,
    TextEncoder, AudioDecoderAR,
    expand_self_attention_mask, expand_masks2
)
from model.loss import EcholancerLoss

class TestModelComponents(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
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
    
    def test_multihead_attention(self):
        """Test MultiHeadAttention with ALiBi support."""
        print("Testing MultiHeadAttention...")
        
        # Test self-attention with combined linear
        mha_self = MultiHeadAttention(
            d_model=64, 
            num_heads=2, 
            dropout=0.1, 
            use_alibi=True,
            use_combined_linear=True
        ).to(self.device)
        
        # Test cross-attention with separate linear
        mha_cross = MultiHeadAttention(
            d_model=64, 
            num_heads=2, 
            dropout=0.1, 
            use_alibi=False,
            use_combined_linear=False
        ).to(self.device)
        
        # Create test tensors
        batch_size, seq_len_q, seq_len_k = 2, 10, 15
        query = torch.randn(batch_size, seq_len_q, 64).to(self.device)
        key = torch.randn(batch_size, seq_len_k, 64).to(self.device)
        value = torch.randn(batch_size, seq_len_k, 64).to(self.device)
        mask = torch.ones(batch_size, 1, seq_len_q, seq_len_k, dtype=torch.bool).to(self.device)
        
        # Test self-attention
        output_self = mha_self(query, query, query, mask)
        self.assertEqual(output_self.shape, (batch_size, seq_len_q, 64))
        print("  Self-attention with combined linear: PASSED")
        
        # Test cross-attention
        output_cross = mha_cross(query, key, value, mask)
        self.assertEqual(output_cross.shape, (batch_size, seq_len_q, 64))
        print("  Cross-attention with separate linear: PASSED")
    
    def test_feedforward(self):
        """Test FeedForward with sequence masking."""
        print("Testing FeedForward...")
        
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
        
        # Test with mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(self.device)
        mask[:, 8:] = True  # Mask last 2 positions
        output_with_mask = ff(x, mask)
        
        # Check that masked positions are zero
        self.assertTrue(torch.allclose(output_with_mask[:, 8:, :], torch.zeros_like(output_with_mask[:, 8:, :])))
        print("  FeedForward with mask: PASSED")
    
    def test_transformer_layers(self):
        """Test Transformer encoder and decoder layers."""
        print("Testing Transformer layers...")
        
        # Test encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=64,
            num_heads=2,
            d_ff=128,
            dropout=0.1,
            alibi_alpha=1.0,
            use_alibi=True,
            activation='relu'
        ).to(self.device)
        
        # Test decoder layer
        decoder_layer = TransformerDecoderLayer(
            d_model=64,
            num_heads=2,
            d_ff=128,
            dropout=0.1,
            alibi_alpha=1.0,
            use_alibi=True,
            activation='relu'
        ).to(self.device)
        
        # Create test tensors
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, 64).to(self.device)
        memory = torch.randn(batch_size, seq_len, 64).to(self.device)
        # Create masks
        seq_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(self.device)
        attn_mask = expand_self_attention_mask(seq_mask)
        
        # Test encoder layer
        output_encoder = encoder_layer(x, attn_mask, seq_mask)
        self.assertEqual(output_encoder.shape, (batch_size, seq_len, 64))
        print("  Transformer encoder layer: PASSED")
        
        # Test decoder layer
        cross_attn_mask = expand_masks2(seq_mask, seq_mask)
        output_decoder = decoder_layer(x, memory, attn_mask, cross_attn_mask, seq_mask)
        self.assertEqual(output_decoder.shape, (batch_size, seq_len, 64))
        print("  Transformer decoder layer: PASSED")
    
    def test_text_encoder(self):
        """Test TextEncoder component."""
        print("Testing TextEncoder...")
        
        encoder = TextEncoder(
            vocab_size=50,
            embed_size=64,
            num_heads=2,
            num_layers=2,
            forward_expansion=2,
            dropout=0.1,
            emotion_channels=64,
            speaker_channels=16,
            alibi_alpha=1.0,
            use_alibi=True
        ).to(self.device)
        
        # Create test tensors
        batch_size, seq_len = 2, 10
        token_ids = torch.randint(0, 50, (batch_size, seq_len)).to(self.device)
        x_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(self.device)
        encoded_em = torch.randn(batch_size, 64).to(self.device)
        spk_emb = torch.randn(batch_size, 1, 16).to(self.device)
        
        # Test encoder
        output = encoder(token_ids, x_mask, encoded_em, spk_emb)
        self.assertEqual(output.shape, (batch_size, seq_len, 64))
        print("  TextEncoder: PASSED")
    
    def test_spectrogram_decoder(self):
        """Test AudioDecoderAR component."""
        print("Testing AudioDecoderAR...")
        
        decoder = AudioDecoderAR(
            encoder_channels=64,
            mel_channels=40,
            filter_channels=64,
            depth=2,
            heads=2,
            dropout=0.1,
            speaker_channels=16,
            dec_type="transformer",
            alibi_alpha=1.0,
            use_alibi=True
        ).to(self.device)
        
        # Create test tensors
        batch_size, seq_len_x, seq_len_y = 2, 10, 15
        x = torch.randint(0, 1010, (batch_size, seq_len_x)).to(self.device)
        x_mask = torch.zeros(batch_size, seq_len_x, dtype=torch.bool).to(self.device)
        y = torch.randn(batch_size, seq_len_y, 64).to(self.device)
        y_mask = torch.zeros(batch_size, seq_len_y, dtype=torch.bool).to(self.device)
        spk_emb = torch.randn(batch_size, 1, 16).to(self.device)
        
        # Test decoder
        logits, attn_logprob, x_mask_out, _ = decoder(x, x_mask, y, y_mask, spk_emb)
        self.assertEqual(logits.shape, (batch_size, seq_len_x-1, 1010))  # -1 because of shifting
        print("  AudioDecoderAR: PASSED")
    
    def test_echolancer_model(self):
        """Test full Echolancer model."""
        print("Testing Echolancer model...")
        
        model = Echolancer(**self.tiny_params).to(self.device)
        
        # Create test tensors
        batch_size, text_len, mel_len = 2, 10, 15
        speakers = torch.randint(0, 1, (batch_size,)).to(self.device)
        texts = torch.randint(0, 50, (batch_size, text_len)).to(self.device)
        src_lens = torch.tensor([text_len, text_len-2]).to(self.device)
        mels = torch.randint(0, 1010, (batch_size, mel_len)).to(self.device)
        mel_lens = torch.tensor([mel_len, mel_len-3]).to(self.device)
        em_hidden = torch.randn(batch_size, 768).to(self.device)
        
        # Test forward pass
        model_out = model(speakers, texts, src_lens, mels, mel_lens, em_hidden)
        self.assertEqual(len(model_out), 6)
        print("  Echolancer forward pass: PASSED")
        
        # Test inference
        token_outputs = model.infer(speakers[:1], texts[:1], src_lens[:1], em_hidden[:1])
        self.assertEqual(token_outputs.shape[0], 1)  # Batch size
        print("  Echolancer inference: PASSED")

if __name__ == '__main__':
    unittest.main()