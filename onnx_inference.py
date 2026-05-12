"""
ONNX Export and Inference Example for Echolancer

This script demonstrates:
1. How to export the Echolancer model to ONNX
2. How to run inference with ONNX Runtime using KV caching
3. Numpy-based top-p and temperature sampling
"""

import torch
import numpy as np
import argparse
from pathlib import Path

# For export
from model.echolancer import Echolancer, EcholancerONNX


def export_to_onnx(
    model: Echolancer,
    output_path: str,
    max_len: int = 2048,
    opset_version: int = 18
):
    """
    Export Echolancer model to ONNX format.
    
    Args:
        model: Trained Echolancer model
        output_path: Path to save the .onnx file
        max_len: Maximum sequence length for KV cache
        opset_version: ONNX opset version
    """
    model.eval()
    
    # Wrap model for ONNX export
    model_onnx = EcholancerONNX(model, max_len=max_len)
    
    # Create example inputs for tracing
    B = 1
    T = 10  # Example prefill length
    
    tokens = torch.randint(0, 100, (B, T), dtype=torch.long)
    k_cache, v_cache, cache_len = model_onnx.create_cache(B, dtype=torch.float32)
    
    # Speaker ID (for multi-speaker) or None
    spk_id = torch.zeros(B, dtype=torch.long)
    
    print(f"Exporting to {output_path}...")
    print(f"  Cache shape: {model_onnx.get_cache_shape(B)}")
    print(f"  Num layers: {model_onnx.num_layers}")
    print(f"  Num KV heads: {model_onnx.num_kv_heads}")
    print(f"  Head dim: {model_onnx.d_k}")
    
    torch.onnx.export(
        model_onnx,
        (tokens, k_cache, v_cache, cache_len, spk_id),
        output_path,
        input_names=["tokens", "k_cache", "v_cache", "cache_len", "spk_id"],
        output_names=["logits", "k_cache_out", "v_cache_out", "cache_len_out"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq_len"},
            "k_cache": {1: "batch", 2: "max_len"},      # max_len is now dynamic
            "v_cache": {1: "batch", 2: "max_len"},      # max_len is now dynamic
            "cache_len": {0: "batch"},
            "spk_id": {0: "batch"},
            "logits": {0: "batch", 1: "seq_len"},
            "k_cache_out": {1: "batch", 2: "max_len"},  # max_len is now dynamic
            "v_cache_out": {1: "batch", 2: "max_len"},  # max_len is now dynamic
            "cache_len_out": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    
    print(f"Export complete: {output_path}")
    return model_onnx  # Return for cache shape info


# ============================================================================
# ONNX Runtime Inference
# ============================================================================

def sample_top_p(logits: np.ndarray, temperature: float = 1.0, top_p: float = 0.9) -> np.ndarray:
    """
    Sample from logits using temperature and top-p (nucleus) sampling.
    
    Args:
        logits: Raw logits (B, vocab_size)
        temperature: Temperature for softmax (higher = more random)
        top_p: Cumulative probability threshold for nucleus sampling
        
    Returns:
        Sampled token IDs (B,)
    """
    # Apply temperature
    logits = logits / temperature
    
    # Softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    batch_size = probs.shape[0]
    samples = np.zeros(batch_size, dtype=np.int64)
    
    for b in range(batch_size):
        p = probs[b]
        
        # Sort by probability descending
        sorted_indices = np.argsort(p)[::-1]
        sorted_probs = p[sorted_indices]
        
        # Cumulative sum
        cumsum = np.cumsum(sorted_probs)
        
        # Find cutoff index where cumsum exceeds top_p
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        cutoff_idx = min(cutoff_idx, len(sorted_probs))
        
        # Truncate and renormalize
        top_indices = sorted_indices[:cutoff_idx]
        top_probs = sorted_probs[:cutoff_idx]
        top_probs = top_probs / np.sum(top_probs)
        
        # Sample
        chosen = np.random.choice(top_indices, p=top_probs)
        samples[b] = chosen
    
    return samples


class EcholancerONNXInference:
    """
    ONNX Runtime inference wrapper for Echolancer.
    """
    
    def __init__(
        self, 
        onnx_path: str,
        num_layers: int,
        num_kv_heads: int,
        d_k: int,
        max_len: int = 2048,
        provider: str = "CPUExecutionProvider"
    ):
        """
        Args:
            onnx_path: Path to the .onnx model file
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads per layer
            d_k: Head dimension
            max_len: Maximum sequence length for cache
            provider: ONNX Runtime provider ("CUDAExecutionProvider" or "CPUExecutionProvider")
        """
        import onnxruntime as ort
        
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.d_k = d_k
        self.max_len = max_len
        
        # Create session
        providers = [provider]
        if provider == "CUDAExecutionProvider":
            providers.append("CPUExecutionProvider")  # Fallback
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Loaded ONNX model: {onnx_path}")
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
    
    def create_cache(self, batch_size: int = 1, dtype=np.float32):
        """Create empty KV cache tensors."""
        shape = (self.num_layers, batch_size, self.max_len, self.num_kv_heads, self.d_k)
        k_cache = np.zeros(shape, dtype=dtype)
        v_cache = np.zeros(shape, dtype=dtype)
        cache_len = np.zeros(batch_size, dtype=np.int32)
        return k_cache, v_cache, cache_len
    
    def forward(
        self,
        tokens: np.ndarray,
        k_cache: np.ndarray,
        v_cache: np.ndarray,
        cache_len: np.ndarray,
        spk_id: np.ndarray
    ):
        """
        Run one forward pass.
        
        Args:
            tokens: Token IDs (B, T) - can be prefill (T > 1) or decode (T = 1)
            k_cache: Key cache (num_layers, B, max_len, num_kv_heads, d_k)
            v_cache: Value cache (num_layers, B, max_len, num_kv_heads, d_k)
            cache_len: Current cache length (B,)
            spk_id: Speaker IDs (B,)
            
        Returns:
            logits, k_cache, v_cache, cache_len
        """
        outputs = self.session.run(
            self.output_names,
            {
                "tokens": tokens,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "cache_len": cache_len,
                "spk_id": spk_id
            }
        )
        
        return outputs[0], outputs[1], outputs[2], outputs[3]
    
    def generate(
        self,
        prompt_tokens: np.ndarray,
        spk_id: np.ndarray,
        max_new_tokens: int = 500,
        temperature: float = 0.8,
        top_p: float = 0.9,
        eos_token_id: int = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.
        
        Args:
            prompt_tokens: Initial token sequence (B, T_prompt)
            spk_id: Speaker IDs (B,)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling threshold
            eos_token_id: Stop generation when this token is produced
            verbose: Print progress
            
        Returns:
            Generated token sequence (B, T_prompt + T_generated)
        """
        batch_size = prompt_tokens.shape[0]
        
        # Initialize cache
        k_cache, v_cache, cache_len = self.create_cache(batch_size)
        
        # Prefill: process entire prompt at once
        if verbose:
            print(f"Prefilling {prompt_tokens.shape[1]} tokens...")
        
        logits, k_cache, v_cache, cache_len = self.forward(
            prompt_tokens.astype(np.int64),
            k_cache,
            v_cache,
            cache_len,
            spk_id.astype(np.int64)
        )
        
        # Start with prompt tokens
        all_tokens = [prompt_tokens]
        
        # Decode loop
        if verbose:
            print(f"Generating up to {max_new_tokens} tokens...")
        
        for step in range(max_new_tokens):
            # Sample from last position
            next_logits = logits[:, -1, :]  # (B, vocab_size)
            next_token = sample_top_p(next_logits, temperature, top_p)  # (B,)
            next_token = next_token.reshape(batch_size, 1)  # (B, 1)
            
            all_tokens.append(next_token)
            
            # Check for EOS
            if eos_token_id is not None and np.all(next_token == eos_token_id):
                if verbose:
                    print(f"EOS reached at step {step + 1}")
                break
            
            # Forward with single token
            logits, k_cache, v_cache, cache_len = self.forward(
                next_token.astype(np.int64),
                k_cache,
                v_cache,
                cache_len,
                spk_id.astype(np.int64)
            )
            
            if verbose and (step + 1) % 100 == 0:
                print(f"  Generated {step + 1} tokens...")
        
        # Concatenate all tokens
        result = np.concatenate(all_tokens, axis=1)
        
        if verbose:
            print(f"Generation complete: {result.shape[1]} total tokens")
        
        return result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Echolancer ONNX Export/Inference")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    parser.add_argument("--infer", action="store_true", help="Run ONNX inference")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--onnx-path", type=str, default="echolancer.onnx", help="ONNX model path")
    parser.add_argument("--max-len", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling threshold")
    
    args = parser.parse_args()
    
    if args.export:
        # Example: Export model
        # You would load your actual model and config here
        print("=" * 50)
        print("ONNX EXPORT EXAMPLE")
        print("=" * 50)
        print("""
To export your model, load it first:

    from model.echolancer import Echolancer, EcholancerONNX
    
    model = Echolancer(
        vocab_size=...,
        decoder_hidden=1024,
        decoder_layer=24,
        decoder_head=16,
        # ... other config
    )
    model.load_state_dict(torch.load("checkpoint.pt"))
    
    export_to_onnx(model, "echolancer.onnx", max_len=2048)
""")
    
    if args.infer:
        # Example: Run inference
        print("=" * 50)
        print("ONNX INFERENCE EXAMPLE")
        print("=" * 50)
        
        # These values must match your exported model
        # You can save them alongside the ONNX file
        infer = EcholancerONNXInference(
            onnx_path=args.onnx_path,
            num_layers=24,      # From your model config
            num_kv_heads=4,     # From your model config
            d_k=64,             # From your model config (hidden_dim / num_heads)
            max_len=args.max_len,
            provider="CPUExecutionProvider"  # or "CUDAExecutionProvider"
        )
        
        # Example generation
        prompt = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)  # Your encoded prompt
        spk_id = np.array([0], dtype=np.int64)  # Speaker 0
        
        output = infer.generate(
            prompt,
            spk_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=None  # Set to your EOS token ID
        )
        
        print(f"Generated tokens: {output}")
