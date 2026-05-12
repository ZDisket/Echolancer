import torch
import torch.nn as nn
import sys
import os

# Add model directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.echolancer import MultiHeadAttention

def test_attention_kvcache():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 256
    num_heads = 8
    batch_size = 2
    seq_len = 10
    
    # ALiBi usually helps show issues
    attn = MultiHeadAttention(d_model, num_heads, dropout=0.0, causal=True, use_alibi=True).to(device)
    attn.eval()
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # 1. Normal forward
    with torch.no_grad():
        out_normal, _, _ = attn(x, x, x)
    
    print(f"Normal forward output shape: {out_normal.shape}")
    
    # 2. KV cache forward (full sequence at once)
    with torch.no_grad():
        # First call with None cache
        out_cache_init, kv_cache, cache_seqlens = attn(x, x, x, kv_cache=None, cache_seqlens=None)
        
    print(f"Cache init output shape: {out_cache_init.shape}")
    print(f"KV cache shapes: {kv_cache[0].shape}, {kv_cache[1].shape}")
    print(f"Cache seqlens: {cache_seqlens}")
    
    # Check if outputs match (they should if causal masking is consistent)
    diff = (out_normal - out_cache_init).abs().max().item()
    print(f"Difference between normal and cache-init: {diff}")
    
    # 3. Incremental decoding
    with torch.no_grad():
        kv_cache = None
        cache_seqlens = None
        outputs = []
        
        for i in range(seq_len):
            q_step = x[:, i:i+1]
            out_step, kv_cache, cache_seqlens = attn(q_step, q_step, q_step, kv_cache=kv_cache, cache_seqlens=cache_seqlens)
            outputs.append(out_step)
            
        out_incremental = torch.cat(outputs, dim=1)
        
    diff_inc = (out_normal - out_incremental).abs().max().item()
    print(f"Difference between normal and incremental: {diff_inc}")
    
    if diff_inc < 1e-4:
        print("SUCCESS: MultiHeadAttention KV cache outputs match normal causal forward!")
    else:
        print("FAILURE: MultiHeadAttention KV cache outputs do not match normal causal forward.")

def test_decoder_kvcache():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 512
    batch_size = 2
    seq_len = 10
    
    decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff, dropout=0.0, disable_cross_attn=True).to(device)
    decoder.eval()
    
    x = torch.randn(batch_size, seq_len, d_model).to(device)
    memory = torch.randn(batch_size, 5, d_model).to(device)
    cond = torch.randn(batch_size, 1, 32).to(device) # dummy cond if needed
    
    # 1. Normal forward
    with torch.no_grad():
        out_normal, _, _ = decoder(x, memory, None, None)
    
    print(f"Decoder normal forward output shape: {out_normal.shape}")
    
    # 2. KV cache forward (full sequence)
    with torch.no_grad():
        out_cache_init, kv_caches, cache_seqlens = decoder(x, memory, None, None, kv_caches=None, cache_seqlens=None)
    
    diff_init = (out_normal - out_cache_init).abs().max().item()
    print(f"Decoder difference between normal and cache-init: {diff_init}")

    # 3. Incremental decoding
    with torch.no_grad():
        kv_caches = None
        cache_seqlens = None
        outputs = []
        
        for i in range(seq_len):
            q_step = x[:, i:i+1]
            # ffn_seq_mask should be (B, 1) for the current step if provided
            step_ffn_mask = None # or some slice of a larger mask
            out_step, kv_caches, cache_seqlens = decoder(q_step, memory, None, None, ffn_seq_mask=step_ffn_mask, kv_caches=kv_caches, cache_seqlens=cache_seqlens)
            outputs.append(out_step)
            
        out_incremental = torch.cat(outputs, dim=1)
        
    diff_inc = (out_normal - out_incremental).abs().max().item()
    print(f"Decoder difference between normal and incremental: {diff_inc}")
    
    if diff_inc < 1e-4:
        print("SUCCESS: TransformerDecoder KV cache outputs match normal causal forward!")
    else:
        print("FAILURE: TransformerDecoder KV cache outputs do not match normal causal forward.")

if __name__ == "__main__":
    print("--- Testing MultiHeadAttention ---")
    test_attention_kvcache()
    print("\n--- Testing TransformerDecoder ---")
    try:
        from model.echolancer import TransformerDecoder
        test_decoder_kvcache()
    except ImportError as e:
        print(f"Error importing TransformerDecoder: {e}")
