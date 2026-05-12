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

def benchmark_decoder(device_name="cuda"):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark.")
        return
    
    device = device_name
    # Use bfloat16 for better performance and realistic testing if on CUDA
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
    
    print(f"\n{'='*20} Benchmarking on {device.upper()} ({dtype_str}) {'='*20}")
    
    if device == "cuda":
        # Strategy for ~500M parameters:
        # 12 * L * d_model^2
        # For d_model=1280, L=24: 12 * 24 * 1280^2 = 471.8M
        d_model = 1280
        num_heads = 16
        num_layers = 24
        d_ff = 5120
    else:
        # Very small model for CPU
        d_model = 128
        num_heads = 4
        num_layers = 4
        d_ff = 512
        
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
    print(f"Total Sequence Length: {max_len}")
    print("-" * 30)

    # 1. Without KV Cache (Re-feed everything)
    print("Running Baseline (No KV Cache)...")
    # Warmup
    with torch.no_grad():
        _ = decoder(x_prefill, memory, None, None)
    
    if device == "cuda": torch.cuda.synchronize()
    start_time = time.time()
    
    full_sequence = x_prefill
    with torch.no_grad():
        for i in tqdm(range(gen_len), desc="Baseline (No KV Cache)"):
            out, _, _ = decoder(full_sequence, memory, None, None)
            new_token = out[:, -1:]
            full_sequence = torch.cat([full_sequence, new_token], dim=1)
            
    if device == "cuda": torch.cuda.synchronize()
    baseline_time = time.time() - start_time
    baseline_tok_per_sec = gen_len / baseline_time
    print(f"Baseline Time: {baseline_time:.4f}s ({baseline_tok_per_sec:.2f} tokens/s)")

    # 2. With KV Cache
    print("\nRunning KV Cache Optimized...")
    if device == "cuda": torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        # Prefill step
        out_inc, kv_caches, cache_seqlens = decoder(x_prefill, memory, None, None, kv_caches=None, cache_seqlens=None)
        
        curr_token = out_inc[:, -1:]
        for i in tqdm(range(gen_len - 1), desc="KV Cache Optimized"):
            out_step, kv_caches, cache_seqlens = decoder(curr_token, memory, None, None, kv_caches=kv_caches, cache_seqlens=cache_seqlens)
            curr_token = out_step
            
    if device == "cuda": torch.cuda.synchronize()
    kv_cache_time = time.time() - start_time
    kv_tok_per_sec = gen_len / kv_cache_time
    print(f"KV Cache Time: {kv_cache_time:.4f}s ({kv_tok_per_sec:.2f} tokens/s)")

    # Summary
    speedup = baseline_time / kv_cache_time
    print("-" * 30)
    print(f"Speedup on {device}: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_decoder("cuda")
    benchmark_decoder("cpu")
