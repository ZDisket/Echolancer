import torch
import torch.nn as nn
import time
import os
import sys
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.echolancer import TransformerDecoder

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def trace_decoder(device_name="cuda"):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, skipping GPU trace.")
        return
    
    device = device_name
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    print(f"\n{'='*20} Profiling on {device.upper()} {'='*20}")
    
    # Use a large model to see kernel details clearly
    d_model = 1280
    num_heads = 16
    num_layers = 24
    d_ff = 5120
    batch_size = 1
    
    prefill_len = 100
    # We only need a few steps for a trace
    gen_len = 5 
    
    decoder = TransformerDecoder(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,
        disable_cross_attn=True
    ).to(device=device, dtype=dtype)
    decoder.eval()
    
    # Dummy data
    x_prefill = torch.randn(batch_size, prefill_len, d_model).to(device=device, dtype=dtype)
    memory = torch.randn(batch_size, 1, d_model).to(device=device, dtype=dtype)

    # 1. Trace Baseline (No KV Cache)
    print("Tracing Baseline (No KV Cache)...")
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = decoder(x_prefill, memory, None, None)

    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        full_sequence = x_prefill
        with torch.no_grad():
            for i in range(gen_len):
                with record_function(f"baseline_step_{i}"):
                    out, _, _ = decoder(full_sequence, memory, None, None)
                    new_token = out[:, -1:]
                    full_sequence = torch.cat([full_sequence, new_token], dim=1)
    
    trace_file_baseline = f"trace_baseline_{device}.json"
    prof.export_chrome_trace(trace_file_baseline)
    print(f"Baseline trace saved to {trace_file_baseline}")

    # 2. Trace With KV Cache
    print("\nTracing KV Cache Optimized...")
    # Warmup
    with torch.no_grad():
        out_inc, kv_caches, cache_seqlens = decoder(x_prefill, memory, None, None)
        _ = decoder(out_inc[:, -1:], memory, None, None, kv_caches=kv_caches, cache_seqlens=cache_seqlens)

    with profile(activities=activities, record_shapes=True, with_stack=True) as prof:
        with torch.no_grad():
            with record_function("kv_cache_prefill"):
                out_inc, kv_caches, cache_seqlens = decoder(x_prefill, memory, None, None, kv_caches=None, cache_seqlens=None)
            
            curr_token = out_inc[:, -1:]
            for i in range(gen_len):
                with record_function(f"kv_cache_step_{i}"):
                    out_step, kv_caches, cache_seqlens = decoder(curr_token, memory, None, None, kv_caches=kv_caches, cache_seqlens=cache_seqlens)
                    curr_token = out_step
            
    trace_file_kv = f"trace_kvcache_{device}.json"
    prof.export_chrome_trace(trace_file_kv)
    print(f"KV Cache trace saved to {trace_file_kv}")
    
    print("\nProfiling complete. You can open these .json files in chrome://tracing or perfetto.dev")

if __name__ == "__main__":
    # Prioritize CUDA for detailed analysis if available
    if torch.cuda.is_available():
        trace_decoder("cuda")
    else:
        trace_decoder("cpu")
