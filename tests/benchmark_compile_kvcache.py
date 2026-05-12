import torch
import torch.nn as nn
import time
import os
import sys
from tqdm import tqdm

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.echolancer import TransformerDecoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_compile(device_name="cuda"):
    if device_name != "cuda" or not torch.cuda.is_available():
        print("CUDA not available or requested device is not CUDA. torch.compile is best tested on GPU.")
        return
    
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"\n{'='*20} Benchmarking torch.compile (dynamic=True) on {device.upper()} {'='*20}")
    
    # Model configuration - Large (~472M params)
    d_model = 1280
    num_heads = 16
    num_layers = 24
    d_ff = 5120
    batch_size = 1
    
    prefill_len = 100
    max_len = 1024
    gen_len = max_len - prefill_len
    
    decoder = TransformerDecoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,
        disable_cross_attn=True
    ).to(device=device, dtype=dtype)
    decoder.eval()
    
    params = count_parameters(decoder)
    
    # Dummy data
    x_prefill = torch.randn(batch_size, prefill_len, d_model).to(device=device, dtype=dtype)
    memory = torch.randn(batch_size, 1, d_model).to(device=device, dtype=dtype)
    
    print(f"--- Benchmark Details ---")
    print(f"Layers: {num_layers}, Hidden Dim: {d_model}")
    print(f"Parameters: {params / 1e6:.2f}M")
    print(f"Prefill: {prefill_len} tokens")
    print(f"Generation: {gen_len} steps")
    print("-" * 30)

    # 1. Eager Mode (KV Cache)
    print("Running Eager Mode KV Cache...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        out_inc, kv_caches, cache_seqlens = decoder(x_prefill, memory, None, None, kv_caches=None, cache_seqlens=None)
        curr_token = out_inc[:, -1:]
        for _ in tqdm(range(gen_len - 1), desc="Eager Mode"):
            out_step, kv_caches, cache_seqlens = decoder(curr_token, memory, None, None, kv_caches=kv_caches, cache_seqlens=cache_seqlens)
            curr_token = out_step
            
    torch.cuda.synchronize()
    eager_time = time.time() - start_time
    eager_tok_per_sec = gen_len / eager_time
    print(f"Eager Time: {eager_time:.4f}s ({eager_tok_per_sec:.2f} tokens/s)")

    # 2. torch.compile (KV Cache)
    print("\nCompiling model with torch.compile(dynamic=True)...")
    compiled_decoder = torch.compile(decoder, dynamic=True)
    
    print("Warming up compiled model (this may take a minute)...")
    # Warmup with different lengths to trigger dynamic shape handling
    with torch.no_grad():
        # Short sequence
        w_prefill = torch.randn(batch_size, 10, d_model).to(device=device, dtype=dtype)
        out_w, w_kv, w_sl = compiled_decoder(w_prefill, memory, None, None)
        _ = compiled_decoder(out_w[:, -1:], memory, None, None, kv_caches=w_kv, cache_seqlens=w_sl)
        
        # Long sequence (pre-benchmark size)
        out_w, w_kv, w_sl = compiled_decoder(x_prefill, memory, None, None)
        _ = compiled_decoder(out_w[:, -1:], memory, None, None, kv_caches=w_kv, cache_seqlens=w_sl)
    
    print("Running compiled KV Cache...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        out_inc, kv_caches, cache_seqlens = compiled_decoder(x_prefill, memory, None, None, kv_caches=None, cache_seqlens=None)
        curr_token = out_inc[:, -1:]
        for _ in tqdm(range(gen_len - 1), desc="Compiled Mode"):
            out_step, kv_caches, cache_seqlens = compiled_decoder(curr_token, memory, None, None, kv_caches=kv_caches, cache_seqlens=cache_seqlens)
            curr_token = out_step
            
    torch.cuda.synchronize()
    compile_time = time.time() - start_time
    compile_tok_per_sec = gen_len / compile_time
    print(f"Compile Time: {compile_time:.4f}s ({compile_tok_per_sec:.2f} tokens/s)")

    # Summary
    speedup = eager_time / compile_time
    print("-" * 30)
    print(f"Speedup from torch.compile: {speedup:.2f}x")
    if speedup > 1.1:
        print("SUCCESS: torch.compile provides a measurable speedup!")
    else:
        print("NOTE: Speedup is minor. Check if Triton kernels are being generated correctly.")

if __name__ == "__main__":
    benchmark_compile("cuda")
