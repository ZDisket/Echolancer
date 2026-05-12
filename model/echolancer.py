import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func
    from flash_attn import flash_attn_with_kvcache
    FLASH_AVAILABLE = True
    FLASH_KVCACHE_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    FLASH_KVCACHE_AVAILABLE = False
    print("WARNING: Flash Attention not available! Falling back to manual attention.")

try:
    # Fallback: flash_attn might be available but not kvcache (older versions)
    if FLASH_AVAILABLE and not FLASH_KVCACHE_AVAILABLE:
        from flash_attn import flash_attn_with_kvcache
        FLASH_KVCACHE_AVAILABLE = True
except ImportError:
    pass

if FLASH_AVAILABLE:
    try:
        from flash_attn.bert_padding import pad_input, unpad_input
    except ImportError:
        print("WARNING: FlashAttention available, but failed to import BERT padding functions. Varlen will fail!")

# Import Transformer Engine if available
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe
    TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    print("Transformer Engine not available.")


class LoRALayer(nn.Module):
    """
    LoRA layer that can be inserted into existing Linear layers
    """
    def __init__(self, original_layer, rank=16, alpha=16, dropout=0.0, scale=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.scale = scale
        
        # Create low-rank matrices A and B
        # Linear.weight shape is (out_features, in_features)
        out_features, in_features = original_layer.weight.shape
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize A with random normal and B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B is initialized to zero to start with no effect
        
        self.scaling = alpha / rank * scale

    def forward(self, x):
        # Original linear transformation
        original_out = self.original_layer(x)
        
        # LoRA transformation: (x @ A) @ B
        x_dropped = self.dropout(x)
        lora_out = (x_dropped @ self.lora_A) @ self.lora_B
        lora_out = lora_out * self.scaling
        
        return original_out + lora_out


def apply_lora_to_linear(linear_layer, rank=16, alpha=16, dropout=0.0, scale=1.0):
    """
    Apply LoRA to a Linear layer by returning a LoRALayer that wraps it
    """
    return LoRALayer(linear_layer, rank, alpha, dropout, scale)

class CausalConv1d(nn.Module):
    """
    1D Causal Convolution + GELU + LayerNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.1, use_te=False):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(out_channels) # LayerNorm expects (batch, seq_len, features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2) # (batch, features, seq_len) for Conv1d
        x = self.conv(x)
        x = x[..., :-self.padding] # Remove padding to ensure causality
        x = x.transpose(1, 2) # (batch, seq_len, features) back for LayerNorm
        x = self.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class NoOpCanon(nn.Module):
    """No-op placeholder that matches CanonLayer signature."""
    def forward(self, x, mask=None):
        return torch.zeros_like(x)


class CanonLayer(nn.Module):
    """Causal depthwise convolution branch for local token mixing."""
    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)
        nn.init.zeros_(self.conv.weight)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        x_perm = x.transpose(1, 2)  # (B, D, T)
        x_pad = F.pad(x_perm, (self.kernel_size - 1, 0))  # causal padding
        out = self.conv(x_pad).transpose(1, 2)  # (B, T, D)
        if mask is not None:
            out = out.masked_fill(mask.unsqueeze(-1), 0.0)
        return self.scale * out


class AdaLayerNorm(nn.Module):
    """
    Drop-in for nn.LayerNorm that supports optional conditioning (AdaLN/FiLM),
    while keeping parameter names `weight` and `bias` so old LayerNorm checkpoints load.
    """
    def __init__(self, hidden_dim, cond_dim=0, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias   = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        # conditioning projection to [gamma|beta]
        if self.cond_dim > 0:
            self.to_gamma_beta = nn.Linear(cond_dim, 2 * hidden_dim)
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, cond=None, cond_scale=1.5):
        # Plain LayerNorm (same numerics as nn.LayerNorm)
        x = F.layer_norm(x, (self.hidden_dim,), self.weight, self.bias, self.eps)

        if self.cond_dim == 0:
            return x

        if cond.dim() == 2:
            cond = cond.unsqueeze(1)   # (B,1,C)
        cond = cond / (cond.norm(dim=-1, keepdim=True) + 1e-6)

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)  # (B,1,H) each
        if cond_scale != 1.0:
            gamma = gamma * cond_scale
            beta  = beta  * cond_scale

        # broadcast over time if x is (B,T,H)
        return x * (1 + gamma) + beta


# Utility functions
def sequence_mask(max_length, x_lengths):
    """
    Make a bool sequence mask
    :param max_length: Max length of sequences
    :param x_lengths: Tensor (batch,) indicating sequence lengths
    :return: Bool tensor size (batch, max_length) where True is padded and False is valid
    """
  #  print(f"Making mask max len {max_length}, first 5 lens: {x_lengths[:5]}")
    mask = torch.arange(max_length).expand(len(x_lengths), max_length).to(x_lengths.device)
    mask = mask >= x_lengths.unsqueeze(1)
    return mask

def expand_self_attention_mask(x_mask):
    """
    Expand True=padded masks into an attention mask for self-attention.
    :param x_mask: Mask of x size (batch, seq_len), where True is padded
    :return: Attention mask for MultiHeadAttention
    """
    if x_mask is None:
        return None
    x_mask_expanded = x_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
    attention_mask = x_mask_expanded.expand(-1, -1, x_mask.size(1), -1)  # Shape: (batch_size, 1, seq_len, seq_len)
    attention_mask = ~attention_mask  # True=padded => True=valid
    return attention_mask

def expand_masks2(x_mask, y_mask):
    """
    Expand True=padded masks into an attention mask.
    Inputs can be different or the same.
    :param x_mask: Mask of x size (batch, seq_len), where True indicates padded positions.
    :param y_mask: Mask of y size (batch, seq_2_len), where True indicates padded positions.
    :return: Attention mask for MultiHeadAttention, where True indicates valid positions.
    """
    x_mask_expanded = x_mask.unsqueeze(1).unsqueeze(3)  # Shape: (batch_size, 1, seq_len, 1)
    y_mask_expanded = y_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_2_len)
    # Combine masks: If either token is padded, mark the pair as padded.
    attention_mask = x_mask_expanded | y_mask_expanded  # True if padded in either sequence
    attention_mask = ~attention_mask  # Invert: now True indicates valid positions.
    return attention_mask


def get_te_linear_layer(in_features, out_features, bias=True, use_te=False):
    """
    Get a Linear layer, using Transformer Engine if available and use_te is True.
    """
    if use_te and TRANSFORMER_ENGINE_AVAILABLE:
        return te.Linear(in_features, out_features, bias=bias)
    else:
        return nn.Linear(in_features, out_features, bias=bias)

# Emotion encoder
class EmotionEncoder(nn.Module):
    """
    Emotion encoder that progressively downsamples BERT embeddings to emotion channels.
    
    Args:
        input_size: Size of input BERT embeddings (typically 768)
        emotion_channels: Size of output emotion embeddings
        hidden_sizes: List of hidden layer sizes for progressive downsampling
        dropout: Dropout rate for all layers
    """
    def __init__(self, input_size=768, emotion_channels=256, hidden_sizes=[512, 384], dropout=0.5, use_te=False):
        super(EmotionEncoder, self).__init__()
        self.input_size = input_size
        self.emotion_channels = emotion_channels
        
        # Build the downsampling stack
        layers = []
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        # Add final projection to emotion channels
        layers.append(nn.Linear(prev_size, emotion_channels))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: Input BERT embeddings (B, input_size)
            
        Returns:
            Encoded emotion embeddings (B, emotion_channels)
        """
        return self.encoder(x)

# Basic building blocks
class NormalizedEmbedding(nn.Module):
    """
    Embedding + LayerNorm + Dropout
    """

    def __init__(self, num_embeddings, embedding_dim, dropout=0.1, norm=True, use_te=False):
        super(NormalizedEmbedding, self).__init__()
        self.use_te = use_te
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class FiLM(nn.Module):
    def __init__(self, d_model, d_cond):
        super().__init__()
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(d_cond, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2)
        )

    def forward(self, h, cond):  # h: (B,T,C), cond: (B,d_cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)  # (B,C),(B,C)
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with sequence masking support
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu', use_te=False, d_cond=0, cond_init_scale=1.0,
                 lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0):
        super(FeedForward, self).__init__()
        self.use_te = use_te
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale
        
        self.linear1 = get_te_linear_layer(d_model, d_ff, bias=False, use_te=use_te)
        self.linear2 = get_te_linear_layer(d_ff, d_model, bias=False, use_te=use_te)
        self.d_cond = d_cond
        self.dropout = nn.Dropout(dropout)
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'swiglu':
            # For SwiGLU, we need to adjust the dimensions
            # We'll use 2/3 of d_ff to match common implementations
            self.linear1 = get_te_linear_layer(d_model, 2 * (d_ff // 2), bias=False, use_te=use_te)  # Split for SwiGLU
            self.linear2 = get_te_linear_layer(d_ff // 2, d_model, bias=False, use_te=use_te)
            self.activation = self._swiglu
        elif activation.lower() == 'relu2':
            self.activation = self._relu2
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Apply LoRA to linear layers if rank > 0
        if self.lora_rank > 0:
            self.linear1 = apply_lora_to_linear(self.linear1, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)
            self.linear2 = apply_lora_to_linear(self.linear2, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)

        if self.d_cond > 0:
            self.film = FiLM(d_model, d_cond)

    def _swiglu(self, x):
        """SwiGLU activation function"""
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

    def _relu2(self, x):
        """ReLU squared activation function"""
        return F.relu(x) ** 2

    def apply_mask(self, x, mask):
        """
        Apply mask to tensor.
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Boolean mask, can be:
                  - 2D shape (batch_size, seq_len) where True indicates padded positions
                  - 4D attention mask (batch_size, 1, seq_len, seq_len) where True indicates valid positions
        Returns:
            Masked tensor
        """
        if mask is not None:
            # Handle different mask formats
            if mask.dim() == 2:
                # Standard sequence mask (batch_size, seq_len)
                # Expand mask to match x dimensions: (batch_size, seq_len, d_model)
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x = x.masked_fill(mask_expanded, 0.0)
            elif mask.dim() == 4:
                # Attention mask (batch_size, 1, seq_len, seq_len)
                # For sequence-level masking, we can use the diagonal elements
                # which represent self-attention (token attending to itself)
                # Extract diagonal: (batch_size, 1, seq_len) -> (batch_size, seq_len)
                seq_mask = torch.diagonal(mask, dim1=2, dim2=3).squeeze(1)
                # seq_mask is True for valid positions, we need True for padded positions
                seq_mask_padded = ~seq_mask
                # Expand mask to match x dimensions: (batch_size, seq_len, d_model)
                mask_expanded = seq_mask_padded.unsqueeze(-1).expand_as(x)
                x = x.masked_fill(mask_expanded, 0.0)
        return x

    def forward(self, x, mask=None, cond=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Boolean mask of shape (batch_size, seq_len) where True indicates padded positions
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.d_cond > 0:
            x = self.film(x, cond.squeeze(1))

        # First linear transformation
        x = self.linear1(x)
        x = self.apply_mask(x, mask)
        
        # Apply activation
        x = self.activation(x)

        x = self.apply_mask(x, mask)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Second linear transformation
        x = self.linear2(x)
        x = self.apply_mask(x, mask)
        
        return x

class SimpleCrossAttention(nn.Module):
    def __init__(self, d_model, d_att, dropout=0.0, causal=False,
                 lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale

        self.W_q = nn.Linear(d_model, d_att, bias=False)
        self.W_k = nn.Linear(d_model, d_att, bias=False)
        self.W_v = nn.Linear(d_model, d_att, bias=False)
        self.W_o = nn.Linear(d_att, d_model, bias=False)

        # Apply LoRA if rank > 0
        if self.lora_rank > 0:
            self.W_q = apply_lora_to_linear(self.W_q, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)
            self.W_k = apply_lora_to_linear(self.W_k, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)
            self.W_v = apply_lora_to_linear(self.W_v, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)
            self.W_o = apply_lora_to_linear(self.W_o, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(d_att)

    def forward(self, query, key, value, mask=None):
        B, T_q, _ = query.shape
        T_k = key.size(1)

        Q = self.W_q(query)                  # (B, T_q, d_att)
        K = self.W_k(key)                    # (B, T_k, d_att)
        V = self.W_v(value)                  # (B, T_k, d_att)

        # Raw scores in compute dtype (likely bf16 under autocast)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, T_q, T_k)

        # Build a boolean keep-mask of shape (B, T_q, T_k)
        if mask is not None:
            if mask.dim() != 4 or mask.size(1) != 1 or mask.size(2) != T_q or mask.size(3) != T_k:
                raise ValueError("Mask must have shape (B, 1, T_q, T_k).")
            keep = mask[:, 0].to(torch.bool)
        else:
            keep = torch.ones(B, T_q, T_k, dtype=torch.bool, device=scores.device)

        if self.causal:
            # Combine with causal constraint
            causal = torch.tril(torch.ones(T_q, T_k, dtype=torch.bool, device=scores.device))
            keep = keep & causal  # broadcast over B

        # Mark masked entries as -inf (still in compute dtype)
        scores = scores.masked_fill(~keep, float("-inf"))

        # Detect rows that are fully masked to avoid NaNs in logsumexp
        row_has_any = keep.any(dim=-1, keepdim=True)  # (B, T_q, 1)

        # Do softmax math in fp32 for stability, then cast back
        scores_f32 = scores.float()
        attn_logprobs = F.log_softmax(scores_f32, dim=-1)

        # For fully-masked rows, set logprobs to 0 (⇒ probs=0 everywhere); output becomes zero vector
        attn_logprobs = torch.where(row_has_any, attn_logprobs, torch.zeros_like(attn_logprobs))

        attention = torch.exp(attn_logprobs).to(scores.dtype)  # (B, T_q, T_k)

        attention = self.dropout(attention)
        x = torch.matmul(attention, V)            # (B, T_q, d_att)
        x = self.W_o(x)                           # (B, T_q, d_model)

        # Cast logprobs back to compute dtype to match expectations
        return x, attn_logprobs.to(scores.dtype), None


# Attention modules (simplified versions)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, alibi_alpha=1.0, use_alibi=False, use_combined_linear=True, num_kv_heads=None, start_i_increment=0, causal=False, use_te=False,
                 lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_query_groups = self.num_heads // self.num_kv_heads
        self.d_k = d_model // num_heads
        self.use_alibi = use_alibi
        self.alibi_alpha = alibi_alpha
        self.use_combined_linear = use_combined_linear # INOP
        self.start_i_increment = start_i_increment
        self.backend = "flash" if FLASH_AVAILABLE else "manual"
        self.causal = causal
        self.use_te = use_te
        self._is_export = False  # ONNX export mode

        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale

        # Use separate linear layers for Q, K, and V
        # The Muon paper says it performs better when the QKV projs are separate (I don't use Muon btw)
        # Regardless, fusing the QKV proj because you're concerned about performance is like going to McDonald's, ordeing
        # 2 big macs with large fries, and asking for diet Coke because you're on a diet. The big FLOPs lie in the FFNs
        # and attention operations, the QKV projs are irrelevant.
        self.W_q = get_te_linear_layer(d_model, d_model, bias=False, use_te=use_te)
        self.W_k = get_te_linear_layer(d_model, self.num_kv_heads * self.d_k, bias=False, use_te=use_te)
        self.W_v = get_te_linear_layer(d_model, self.num_kv_heads * self.d_k, bias=False, use_te=use_te)
        
        self.W_o = get_te_linear_layer(d_model, d_model, bias=False, use_te=use_te)

        # Apply LoRA if rank > 0
        if self.lora_rank > 0:
            self.W_q = apply_lora_to_linear(self.W_q, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)
            self.W_k = apply_lora_to_linear(self.W_k, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)
            self.W_v = apply_lora_to_linear(self.W_v, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)
            self.W_o = apply_lora_to_linear(self.W_o, self.lora_rank, self.lora_alpha, self.lora_dropout, self.lora_scale)

        self.dropout = nn.Dropout(dropout)
        
        if self.use_alibi:
            self.register_buffer('alibi_slopes', self._get_alibi_slopes(start_i_increment))

    @property
    def is_export(self):
        return self._is_export

    @is_export.setter
    def is_export(self, value):
        self._is_export = value

    def _get_alibi_slopes(self, start_i_increment=0):
        """
        Get the slopes for ALiBi attention biases with layer scaling.
        """
        # Calculate slopes with layer-scaled ALiBi
        slopes = torch.tensor([
            2 ** (-self.alibi_alpha * (i + start_i_increment)) 
            for i in range(1, self.num_heads + 1)
        ])
        return slopes

    def _get_alibi_bias(self, seq_len_q: int, seq_len_k: int, *, device=None, dtype=torch.float32):
        """
        Returns additive ALiBi bias of shape (1, H, Tq, Tk) in fp32.
        Uses bias[q,k] = slope_h * (k - q), which is <= 0 on/left of the diagonal (causal region).
        """
        device = device if device is not None else next(self.parameters()).device

        # per-head slopes (H,)
        slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1).to(dtype=torch.float32)  # (1,H,1,1)

        # positions
        q_idx = torch.arange(seq_len_q, device=device, dtype=dtype).view(1, 1, seq_len_q, 1)
        k_idx = torch.arange(seq_len_k, device=device, dtype=dtype).view(1, 1, 1, seq_len_k)

        bias = slopes * (k_idx - q_idx)  # (1,H,Tq,Tk), typically <= 0 on/left of diagonal
        return bias  # fp32

    def forward(self, query, key, value, mask=None, kv_cache="NOT_PROVIDED", cache_seqlens="NOT_PROVIDED"):
        if kv_cache != "NOT_PROVIDED" or cache_seqlens != "NOT_PROVIDED":
            # Call forward_with_kvcache if either was provided
            real_kv_cache = None if kv_cache == "NOT_PROVIDED" else kv_cache
            real_cache_seqlens = None if cache_seqlens == "NOT_PROVIDED" else cache_seqlens
            return self.forward_with_kvcache(query, key=key, value=value, kv_cache=real_kv_cache, cache_seqlens=real_cache_seqlens)

        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Always use separate projections for Q, K, V
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Reshape and transpose for multi-head attention
        # Query reshape: (batch_size, seq_len_q, num_heads, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)

        # Key and Value reshape: (batch_size, seq_len_k, num_kv_heads, d_k)
        K = K.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        if self.backend == "flash":
            if mask is None:
                x = self._forward_flashattention(K, Q, V, batch_size, mask, seq_len_k, seq_len_q)
            else:
                x = self._forward_flashattention_varlen(K, Q, V, batch_size, mask, seq_len_k, seq_len_q)
        else:
            x = self._forward_manual(K, Q, V, batch_size, mask, seq_len_k, seq_len_q)

        # For self-attention in transformer blocks, return the attention output,
        # and None for KV cache values if not used.
        return x, None, None

    def _forward_manual(self, K, Q, V, batch_size, mask, seq_len_k, seq_len_q):
        """
        Q,K,V shape assumption: (B, Hq/Hk, T, Dh) *before* this call.
        mask: expected shape (B, 1, Tq, Tk) with 1/True = keep, 0/False = block.
        """
        B = batch_size
        device = Q.device
        # Ensure tensors are the same dtype
        d_k = Q.size(-1)

        # ---- GQA expansion (repeat KV to match query heads) ----
        if self.num_kv_heads != self.num_heads:
            K = K.repeat_interleave(self.num_query_groups, dim=1)  # (B, H, Tk, Dh)
            V = V.repeat_interleave(self.num_query_groups, dim=1)  # (B, H, Tk, Dh)

        # ---- Scaled dot-product scores in fp32 ----
        # (B, H, Tq, Dh) @ (B, H, Dh, Tk) -> (B, H, Tq, Tk)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scale = 1.0 / (d_k ** 0.5)
        scores = (scores * scale).float()  # upcast for numerical stability

        # ---- ALiBi (additive bias) ----
        if self.use_alibi:
            # Build (1, H, Tq, Tk) and expand to (B, H, Tq, Tk)
            alibi_bias = self._get_alibi_bias(seq_len_q, seq_len_k)
            # _get_alibi_bias should already return shape (1, H, Tq, Tk); expand to batch
            if alibi_bias.dim() == 4 and alibi_bias.size(0) == 1:
                alibi_bias = alibi_bias.expand(B, -1, -1, -1)
            scores = scores + alibi_bias  # still fp32

        # ---- External mask (additive -inf where blocked) ----
        if mask is not None:
            # Expect mask shape (B, 1, Tq, Tk); True/1=keep, False/0=block
            if mask.dim() != 4 or mask.size(0) != B or mask.size(2) != seq_len_q or mask.size(3) != seq_len_k:
                raise ValueError(f"Mask must be (B,1,Tq,Tk); got {tuple(mask.size())}")
            if mask.size(1) != 1:
                raise ValueError(f"Mask head dim must be 1 for broadcast; got {mask.size(1)}")

            if mask.dtype != torch.bool:
                # Treat nonzero as keep
                mask_bool = mask != 0
            else:
                mask_bool = mask

            neg_inf = torch.finfo(torch.float32).min
            scores = scores.masked_fill(~mask_bool, neg_inf)

        # ---- Causal mask (if you don't already encode causality in 'mask') ----
        if self.causal:
            # Causal over (Tq, Tk)
            causal = torch.ones((seq_len_q, seq_len_k), device=device, dtype=torch.bool).tril()
            causal = causal.view(1, 1, seq_len_q, seq_len_k)  # (1,1,Tq,Tk)
            neg_inf = torch.finfo(torch.float32).min
            scores = scores.masked_fill(~causal, neg_inf)

        # ---- Softmax in fp32, then cast back ----
        attn = torch.softmax(scores, dim=-1)
        attn = attn.to(Q.dtype)
        attn = self.dropout(attn)

        # ---- Attention apply ----
        x = torch.matmul(attn, V)  # (B,H,Tq,Dh)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # (B,Tq,H*Dh)
        x = self.W_o(x)
        return x

    def _forward_flashattention(self, K, Q, V, batch_size, mask, seq_len_k, seq_len_q):
        # Q, K, V come in as (B, nheads{,_k}, T{_q,_k}, d_k) from the caller
        # flash_attn_func expects (B, T, nheads{,_k}, d_k)
        q = Q.transpose(1, 2).contiguous()  # (B, T_q, nheads,  d_k)
        k = K.transpose(1, 2).contiguous()  # (B, T_k, nheads_k, d_k)
        v = V.transpose(1, 2).contiguous()  # (B, T_k, nheads_k, d_k)

        before_dtype = q.dtype

        alibi = None
        if self.use_alibi:
            # FlashAttention requires fp32 slopes; they must match the number of Q heads
            alibi = self.alibi_slopes.to(device=q.device, dtype=torch.float32)

        out = flash_attn_func(
            q.bfloat16(), k.bfloat16(), v.bfloat16(),
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=None,
            causal=self.causal,
            window_size=(-1, -1),
            alibi_slopes=alibi,
            deterministic=False,
        )  # (B, T_q, nheads, d_k)

        out = out.reshape(batch_size, seq_len_q, self.d_model).to(dtype=before_dtype)  # (B, T_q, d_model)
        out = self.W_o(out)
        return out

    @torch._dynamo.disable # torch.compile doesn't like me.
    def _forward_flashattention_varlen(self, K, Q, V, batch_size, mask, seq_len_k, seq_len_q):
        # K, Q, V: (B, nheads{,_k}, T{_k,_q}, d_k)
        # mask: (B, H_or_1, T_q, T_k); 1 = keep, 0 = pad
        # Returns: (B, T_q, d_model) projected by W_o, like _forward_flashattention

        # Reorder to (B, T*, nheads*, d_k) for unpad_input
        q = Q.transpose(1, 2).contiguous()  # (B, T_q, nheads,   d_k)
        k = K.transpose(1, 2).contiguous()  # (B, T_k, nheads_k, d_k)
        v = V.transpose(1, 2).contiguous()  # (B, T_k, nheads_k, d_k)

        before_dtype = q.dtype
        device = q.device

        # ---- Build per-sequence keep masks for Q and K from a 4D attention mask ----
        # mask could be broadcasted over heads; merge head dim with any()
        # result m has shape (B, T_q, T_k), dtype=bool
        if mask.size(1) != 1:
            m = mask.ne(0).any(dim=1)
        else:
            m = mask[:, 0].ne(0)

        # q_keep: keep a query position if it attends to at least one valid key
        # k_keep: keep a key position if at least one query attends to it
        q_keep = m.any(dim=-1)  # (B, T_q), bool
        k_keep = m.any(dim=-2)  # (B, T_k), bool

        # ---- Unpad Q, K, V to varlen layout ----
        # unpad_input expects attention_mask with 1 = keep, 0 = drop
        q_unpad, q_idx, cu_q, max_sq_q, _ = unpad_input(q, q_keep.to(device=device))
        k_unpad, k_idx, cu_k, max_sq_k, _ = unpad_input(k, k_keep.to(device=device))
        v_unpad, _, _, _, _ = unpad_input(v, k_keep.to(device=device))

        # Shapes now:
        # q_unpad: (total_q, nheads,   d_k)
        # k_unpad: (total_k, nheads_k, d_k)
        # v_unpad: (total_k, nheads_k, d_k)
        # cu_q, cu_k: (B + 1,) int32 cumulative lengths

        cu_q = cu_q.to(device=device, dtype=torch.int32)
        cu_k = cu_k.to(device=device, dtype=torch.int32)

        # ALiBi: fp32 slopes, either (nheads,) or (B, nheads). We have (nheads,)
        alibi = None
        if self.use_alibi:
            alibi = self.alibi_slopes.to(device=device, dtype=torch.float32)

        out_unpad = flash_attn_varlen_func(
            q_unpad.contiguous().bfloat16(), k_unpad.contiguous().bfloat16(), v_unpad.contiguous().bfloat16(),
            cu_q, cu_k,
            int(max_sq_q), int(max_sq_k),
            dropout_p=self.dropout.p if self.training else 0.0,
            softmax_scale=None,
            causal=self.causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=alibi,
            deterministic=False,
            return_attn_probs=False,
            block_table=None,
        )  # (total_q, nheads, d_k)

        # ---- Pad back to (B, T_q, nheads, d_k) then project ----
        out_padded = pad_input(out_unpad, q_idx, batch_size, seq_len_q)  # (B, T_q, nheads, d_k)
        out = out_padded.reshape(batch_size, seq_len_q, self.d_model)  # (B, T_q, d_model)
        out = out.to(dtype=before_dtype)
        out = self.W_o(out)  # (B, T_q, d_model)
        return out

    def forward_with_kvcache(self, query, key=None, value=None, kv_cache=None, cache_seqlens=None):
        """
        Forward pass with KV cache for efficient autoregressive inference.
        
        Args:
            query: Input tensor (B, T_new, d_model) - typically T_new=1 for incremental decoding
            key: Optional input tensor for K projection (B, T_new, d_model)
            value: Optional input tensor for V projection (B, T_new, d_model)
            kv_cache: Tuple of (k_cache, v_cache) each with shape (B, max_seq_len, num_kv_heads, d_k)
                      or None for first step
            cache_seqlens: Tensor (B,) indicating current sequence lengths in cache, or None
            
        Returns:
            output: Attention output (B, T_new, d_model)
            new_kv_cache: Updated (k_cache, v_cache) tuple
            new_cache_seqlens: Updated sequence lengths
        """
        batch_size = query.size(0)
        seq_len_new = query.size(1)
        device = query.device
        dtype = query.dtype
        
        # Use query if key/value are not provided (self-attention)
        if key is None: key = query
        if value is None: value = query

        # Project query, key, value for new tokens
        Q = self.W_q(query)  # (B, T_new, d_model)
        K_new = self.W_k(key)  # (B, T_new, num_kv_heads * d_k)
        V_new = self.W_v(value)  # (B, T_new, num_kv_heads * d_k)
        
        # Reshape for attention: (B, T_new, num_heads/num_kv_heads, d_k)
        Q = Q.view(batch_size, seq_len_new, self.num_heads, self.d_k)
        K_new = K_new.view(batch_size, seq_len_new, self.num_kv_heads, self.d_k)
        V_new = V_new.view(batch_size, seq_len_new, self.num_kv_heads, self.d_k)
        
        # Initialize or update KV cache
        if kv_cache is None:
            # First step: initialize cache with reasonable max length
            max_cache_len = 2048  # Default max length, will expand if needed
            k_cache = torch.zeros(batch_size, max_cache_len, self.num_kv_heads, self.d_k, 
                                  device=device, dtype=dtype)
            v_cache = torch.zeros(batch_size, max_cache_len, self.num_kv_heads, self.d_k,
                                  device=device, dtype=dtype)
            cache_seqlens = torch.zeros(batch_size, device=device, dtype=torch.int32)
        else:
            k_cache, v_cache = kv_cache
            # Expand cache if needed
            if cache_seqlens.max() + seq_len_new > k_cache.size(1):
                if self._is_export:
                    raise RuntimeError("KV cache overflow in export mode. Pre-allocate a larger cache buffer.")
                new_max_len = k_cache.size(1) * 2
                new_k_cache = torch.zeros(batch_size, new_max_len, self.num_kv_heads, self.d_k,
                                          device=device, dtype=dtype)
                new_v_cache = torch.zeros(batch_size, new_max_len, self.num_kv_heads, self.d_k,
                                          device=device, dtype=dtype)
                new_k_cache[:, :k_cache.size(1)] = k_cache
                new_v_cache[:, :v_cache.size(1)] = v_cache
                k_cache = new_k_cache
                v_cache = new_v_cache
        
        # Use flash_attn_with_kvcache if available (most efficient)
        if FLASH_KVCACHE_AVAILABLE and self.backend == "flash":
            # Get ALiBi slopes if used
            alibi = None
            if self.use_alibi:
                alibi = self.alibi_slopes.to(device=device, dtype=torch.float32)
            
            # flash_attn_with_kvcache updates cache in-place and returns attention output
            out = flash_attn_with_kvcache(
                q=Q.bfloat16(),
                k_cache=k_cache.bfloat16(),
                v_cache=v_cache.bfloat16(),
                k=K_new.bfloat16(),
                v=V_new.bfloat16(),
                cache_seqlens=cache_seqlens,
                causal=self.causal,
                alibi_slopes=alibi,
            )  # (B, T_new, num_heads, d_k)
            
            # Update sequence lengths
            new_cache_seqlens = cache_seqlens + seq_len_new
            
            # Reshape and project output
            out = out.to(dtype=dtype)
            out = out.reshape(batch_size, seq_len_new, self.d_model)
            out = self.W_o(out)
            
            # Note: k_cache and v_cache were updated in-place by flash_attn_with_kvcache
            return out, (k_cache.to(dtype), v_cache.to(dtype)), new_cache_seqlens
        
        else:
            # Manual fallback: update cache and compute attention
            # Update cache with new K, V
            if self._is_export:
                # ONNX-friendly: vectorized cache update using scatter
                # Build write positions: (B, T_new, 1, 1) broadcast to (B, T_new, H, D)
                seq_offsets = torch.arange(seq_len_new, device=device, dtype=cache_seqlens.dtype).view(1, -1, 1, 1)
                write_positions = cache_seqlens.view(-1, 1, 1, 1) + seq_offsets  # (B, T_new, 1, 1)
                write_positions = write_positions.expand(-1, -1, self.num_kv_heads, self.d_k)  # (B, T_new, H, D)
                
                k_cache.scatter_(1, write_positions, K_new)
                v_cache.scatter_(1, write_positions, V_new)
            else:
                # Original Python loop (faster in eager mode)
                for b in range(batch_size):
                    start_idx = cache_seqlens[b].item()
                    end_idx = start_idx + seq_len_new
                    k_cache[b, start_idx:end_idx] = K_new[b]
                    v_cache[b, start_idx:end_idx] = V_new[b]
            
            new_cache_seqlens = cache_seqlens + seq_len_new
            
            # Get max_seq_len - avoid .item() in export mode
            if self._is_export:
                max_seq_len = new_cache_seqlens.max()  # Keep as tensor
            else:
                max_seq_len = new_cache_seqlens.max().item()
            
            # Get valid K, V from cache: (B, max_seq_len, num_kv_heads, d_k)
            K_full = k_cache[:, :max_seq_len]
            V_full = v_cache[:, :max_seq_len]
            
            # Transpose for attention: (B, num_heads/num_kv_heads, T, d_k)
            Q_t = Q.transpose(1, 2)  # (B, num_heads, T_new, d_k)
            K_t = K_full.transpose(1, 2)  # (B, num_kv_heads, max_seq_len, d_k)
            V_t = V_full.transpose(1, 2)  # (B, num_kv_heads, max_seq_len, d_k)
            
            # GQA expansion
            if self.num_kv_heads != self.num_heads:
                K_t = K_t.repeat_interleave(self.num_query_groups, dim=1)
                V_t = V_t.repeat_interleave(self.num_query_groups, dim=1)
            
            # Compute attention scores
            d_k = self.d_k
            scores = torch.matmul(Q_t, K_t.transpose(-2, -1))  # (B, H, T_new, max_seq_len)
            scale = 1.0 / (d_k ** 0.5)
            scores = (scores * scale).float()
            
            # ALiBi bias
            if self.use_alibi:
                # Per-batch ALiBi bias calculation for variable sequence lengths
                k_idx = torch.arange(max_seq_len, device=device, dtype=torch.float32).view(1, 1, 1, -1)
                q_idx = torch.arange(seq_len_new, device=device, dtype=torch.float32).view(1, 1, -1, 1)
                
                # Add batch-specific query offset: (B, 1, 1, 1)
                q_pos_offset = (new_cache_seqlens - seq_len_new).view(-1, 1, 1, 1).to(torch.float32)
                q_idx = q_idx + q_pos_offset
                
                slopes = self.alibi_slopes.view(1, self.num_heads, 1, 1).to(dtype=torch.float32)
                alibi_bias = slopes * (k_idx - q_idx) # (B, H, T_new, max_seq_len)
                scores = scores + alibi_bias

            # Causal mask: query at position q can attend to keys at positions <= q
            if self.causal:
                # Create mask for incremental positions
                q_pos = torch.arange(seq_len_new, device=device).view(1, 1, -1, 1)
                q_pos = q_pos + (new_cache_seqlens - seq_len_new).view(-1, 1, 1, 1)  # Add offset per batch
                k_pos = torch.arange(max_seq_len, device=device).view(1, 1, 1, -1)
                causal_mask = q_pos >= k_pos  # (B, 1, T_new, max_seq_len)
                neg_inf = torch.finfo(torch.float32).min
                scores = scores.masked_fill(~causal_mask, neg_inf)
            
            # Softmax and apply attention
            attn = torch.softmax(scores, dim=-1).to(Q.dtype)
            out = torch.matmul(attn, V_t)  # (B, H, T_new, d_k)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_new, self.d_model)
            out = self.W_o(out)
            
            return out, (k_cache, v_cache), new_cache_seqlens


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i_increment=0, use_te=False):
        super(TransformerEncoderLayer, self).__init__()
        self.use_te = use_te
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, alibi_alpha, use_alibi, use_combined_linear=True, 
                                          num_kv_heads=num_kv_heads, start_i_increment=start_i_increment, use_te=use_te)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation, use_te=use_te)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, seq_mask):
        # 1. Self-attention (pre-norm)
        residual = x
        x = self.norm1(x)  # PRE-norm
        x, _, _ = self.self_attn(x, x, x, attn_mask)
        x = residual + self.dropout(x)

        # 2. Feed-forward (pre-norm)
        residual = x
        x = self.norm2(x)  # PRE-norm
        x = self.ffn(x, seq_mask)
        x = residual + self.dropout(x)

        return x, None, None

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i=0, use_te=False):
        super(TransformerEncoder, self).__init__()
        self.use_te = use_te
        # Calculate scaling factor to prevent start_i from exceeding 32 in final layer
        alibi_scaling_fac = max(1, ((num_layers - 1) * num_heads) // (32 - start_i)) if num_layers > 1 else 1
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, alibi_alpha, 
                                   use_alibi, activation, num_kv_heads, 
                                   start_i_increment=start_i + ((i * num_heads) // alibi_scaling_fac),
                                   use_te=use_te)
            for i in range(num_layers)
        ])

    def forward(self, x, attn_mask, seq_mask):
        for layer in self.layers:
            x, _, _ = layer(x, attn_mask, seq_mask)
        return x, None, None

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i_increment=0,
                 cross_attn_type="full", disable_cross_attn=False, use_te=False, d_cond=0, 
                 lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False,
                 use_canon_a=False, use_canon_c=False, canon_kernel_size=4):
        super(TransformerDecoderLayer, self).__init__()
        self.disable_cross_attn = disable_cross_attn  # Option to disable cross attention
        self.use_te = use_te
        self.use_macaron = use_macaron
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale
        self._is_export = False
        self.ffn_scale = 1.0
        self.use_canon_a = use_canon_a
        self.use_canon_c = use_canon_c

        # Canon layers for local token mixing
        self.canon_a = CanonLayer(d_model, canon_kernel_size) if self.use_canon_a else None
        self.canon_c = CanonLayer(d_model, canon_kernel_size) if self.use_canon_c else None
        
        if self.use_macaron:
            d_ff = d_ff // 2 # halven FFN scale so that we have the same amount of parameters
            self.ffn0 = FeedForward(d_model, d_ff, dropout, activation, d_cond=0, use_te=use_te,
                                           lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale)
            self.norm0 = AdaLayerNorm(d_model, cond_dim=d_cond)
            self.ffn_scale = 0.5

        
        self.self_attn = MultiHeadAttention(d_model, num_heads, 0.0, alibi_alpha, use_alibi, use_combined_linear=True,
                                          num_kv_heads=num_kv_heads, start_i_increment=start_i_increment, causal=True, use_te=use_te, 
                                          lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale)  # Self-attention uses GQA
        self.cross_attn_type = cross_attn_type

        if not self.disable_cross_attn:

            if self.cross_attn_type == "full":
                self.cross_attn = MultiHeadAttention(d_model, num_heads, 0.0, alibi_alpha=1.0, use_alibi=False, use_combined_linear=False,
                                                   num_kv_heads=num_kv_heads, start_i_increment=0, use_te=use_te,
                                                   lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale)
            elif self.cross_attn_type == "monotonic":
                self.cross_attn = SimpleCrossAttention(d_model, 128, dropout, causal=False, 
                                                       lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale)
            else:
                raise ValueError(f"Invalid cross attention type: {self.cross_attn_type}")

            self.norm2 = AdaLayerNorm(d_model, cond_dim=d_cond)
        else:
            self.norm2 = nn.Identity()
            self.cross_attn = nn.Identity()


        self.ffn = FeedForward(d_model, d_ff, dropout, activation, d_cond=0, use_te=use_te,
                               lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale)
        self.norm1 = AdaLayerNorm(d_model, cond_dim=d_cond)

        self.norm3 = AdaLayerNorm(d_model, cond_dim=d_cond)
        self.dropout = nn.Dropout(dropout)
        self.last_logprobs = None

    @property
    def is_export(self):
        return self._is_export

    @is_export.setter
    def is_export(self, value):
        self._is_export = value
        self.self_attn.is_export = value
        if hasattr(self.cross_attn, 'is_export'):
            self.cross_attn.is_export = value

    def forward(self, x, memory, cond, self_attn_mask, cross_attn_mask, ffn_seq_mask, kv_cache=None, cache_seqlens=None):
        # 1. Self-attention (pre-norm)
        residual = x
        
        if self.use_macaron:
            x = self.norm0(x, cond=cond)
            x = self.ffn0(x, ffn_seq_mask, cond=cond)
            x = residual + self.ffn_scale * x
            residual = x

        normed = self.norm1(x, cond=cond)  # PRE-norm
        if self.use_canon_a:
            x = residual + self.canon_a(normed, ffn_seq_mask)
            residual = x

        if self.disable_cross_attn:  # keep the bug-fix comment #1
            self_attn_mask = None

        x, new_kv_cache, new_cache_seqlens = self.self_attn(normed, normed, normed, self_attn_mask, kv_cache=kv_cache, cache_seqlens=cache_seqlens)
        x = residual + x

        # 2. Cross-attention (pre-norm)
        if not self.disable_cross_attn:
            residual = x
            x = self.norm2(x)  # PRE-norm
            
            x, logprobs_or_new_kv, maybe_seqlens = self.cross_attn(x, memory, memory, cross_attn_mask)
            x = residual + x

            # If cross_attn is MultiHeadAttention, logprobs_or_new_kv is None
            # If cross_attn is SimpleCrossAttention, it is attn_logprobs
            if isinstance(self.cross_attn, SimpleCrossAttention):
                self.last_logprobs = logprobs_or_new_kv

        # 3. Feed-forward (pre-norm)
        residual = x

        normed = self.norm3(x, cond=cond)  # PRE-norm
        if self.use_canon_c:
            x = residual + self.canon_c(normed, ffn_seq_mask)
            residual = x

        x = self.ffn(normed, ffn_seq_mask, cond=cond)

        x = residual + self.ffn_scale * x

        return x, new_kv_cache, new_cache_seqlens



class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None,
                 start_i=0, disable_cross_attn=False, use_te=False, d_cond=0, lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False,
                 use_canon_a=False, use_canon_c=False, canon_kernel_size=4):
        super(TransformerDecoder, self).__init__()
        self.use_te = use_te

        # Calculate scaling factor to prevent start_i from exceeding 32 in final layer
        alibi_scaling_fac = max(1, ((num_layers - 1) * num_heads) // (32 - start_i)) if num_layers > 1 else 1

        if disable_cross_attn:
            print("Decoder disable cross attention")

        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale
        self._is_export = False

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, alibi_alpha, 
                                   use_alibi, activation, num_kv_heads,
                                   start_i_increment=start_i + ((i * num_heads) // alibi_scaling_fac),
                                   disable_cross_attn=disable_cross_attn,
                                   use_te=use_te, cross_attn_type="full", d_cond=d_cond,
                                   lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale, use_macaron=use_macaron,
                                   use_canon_a=use_canon_a, use_canon_c=use_canon_c, canon_kernel_size=canon_kernel_size)
            for i in range(num_layers)
        ])
        self.disable_cross_attn = disable_cross_attn

    @property
    def is_export(self):
        return self._is_export

    @is_export.setter
    def is_export(self, value):
        self._is_export = value
        for layer in self.layers:
            layer.is_export = value

    def forward(self, x, memory, cond, self_attn_mask, cross_attn_mask=None, ffn_seq_mask=None, kv_caches=None, cache_seqlens=None):
        new_kv_caches = []
        new_seqlens = cache_seqlens 
        
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            # Pass original cache_seqlens to each layer, but collect outputs
            x, new_cache, layer_new_seqlens = layer(x, memory, cond, self_attn_mask, cross_attn_mask, ffn_seq_mask, 
                                               kv_cache=layer_kv_cache, cache_seqlens=cache_seqlens)
            new_kv_caches.append(new_cache)
            new_seqlens = layer_new_seqlens
            
        return x, new_kv_caches, new_seqlens

# Model components
# Unused for now
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, forward_expansion, dropout, 
                 emotion_channels=256, speaker_channels=0, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i=0,
                 pretraining_mode=False, use_te=False):
        super().__init__()
        self.embed = None
        self.encoder = TransformerEncoder(embed_size, num_heads, num_layers, embed_size * forward_expansion, dropout, alibi_alpha, use_alibi, activation, num_kv_heads, start_i, use_te=use_te)
        self.emotion_channels = emotion_channels
        self.speaker_channels = speaker_channels
        self.pretraining_mode = pretraining_mode  # New parameter for pretraining mode
        self.use_te = use_te

    #    if self.speaker_channels > 0:
     #       self.spk_cond = nn.Linear(speaker_channels, embed_size)

        # For MLM pretraining, add output projection layer
        if self.pretraining_mode:
            self.mlm_head = None

    def forward(self, token_ids, x_mask, encoded_em, spk_emb=None, mlm_mask=None):
        # Guard against invalid token IDs
        
        # Embed token_ids
        x = self.embed(token_ids)  # Shape: (batch, max_seq_len, embed_size)

      #  if self.speaker_channels > 0 and spk_emb is not None:
       #     x = x + self.spk_cond(spk_emb)

        if self.emotion_channels > 0 and encoded_em is not None:
            # Expand emotion encoding to match sequence length
            encoded_em_expanded = encoded_em.unsqueeze(1).expand(-1, x.size(1), -1)
            x[:, :, :self.emotion_channels] = encoded_em_expanded

        # Convert 2D mask to 4D attention mask for transformer encoder
        attn_mask = expand_self_attention_mask(x_mask)
        
        # Pass both attention mask and sequence mask to transformer encoder
        x, _, _ = self.encoder(x, attn_mask, x_mask)
        
        return x, None, None

        # In pretraining mode with MLM, return logits for masked positions
        if self.pretraining_mode and mlm_mask is not None:
            # Apply MLM head to get logits for all positions
            logits = self.mlm_head(x)  # (batch, seq_len, vocab_size)
            # Only return logits for masked positions
            return x, logits, mlm_mask
        else:
            return x


class AudioDecoderAR(nn.Module):
    def __init__(self, encoder_channels, codebook_size, filter_channels, depth, heads, dropout=0.1,
                 speaker_channels=0, dec_type="transformer", alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i=0,
                 pretraining_mode=False, use_te=False, vocab_offset=0, lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False,
                 use_canon_a=False, use_canon_c=False, canon_kernel_size=4):
        super().__init__()

        self.encoder_channels = encoder_channels
        self.filter_channels = filter_channels
        self.codebook_size = codebook_size
        self.n_embeds = self.codebook_size + 32 # so that next number is divisible by 16 and 8
        self.speaker_channels = speaker_channels
        self.vocab_offset = vocab_offset
        self.bos_token_id = (self.codebook_size + self.vocab_offset) + 1
        self.eos_token_id = (self.codebook_size + self.vocab_offset) + 2
        self.pad_token_id = (self.codebook_size + self.vocab_offset) + 3
        self.needs_proj = None

        # For inference.
        self.vocab_min = self.vocab_offset
        self.vocab_max = self.eos_token_id + 1

        self.dec_type = dec_type.lower()
        self.decoder_type = self.dec_type
        self.use_alibi = use_alibi
        self.alibi_alpha = alibi_alpha
        self.pretraining_mode = pretraining_mode  # New parameter to enable pretraining mode
        self.use_te = use_te

        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale
        self._is_export = False

        self.spk_cond = None
        if self.dec_type == "transformer":
            self.embed = None
            #self.prenet = CausalConv1d(self.filter_channels, self.filter_channels, 3, 1, False)
            self.dec = TransformerDecoder(filter_channels, heads, depth,
                                        filter_channels * 4, dropout, alibi_alpha, use_alibi, activation, num_kv_heads, start_i,
                                        disable_cross_attn=True, use_te=use_te, d_cond=speaker_channels,
                                        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale, use_macaron=use_macaron,
                                        use_canon_a=use_canon_a, use_canon_c=use_canon_c, canon_kernel_size=canon_kernel_size)

            self.out_proj = None
        else:
            raise RuntimeError(f"Invalid decoder type: {self.dec_type}")

        self.gate_proj = nn.Identity()  # no sigmoid, we use BCEWithLogitsLoss
        self.g_drop = nn.Dropout(0.1)

    @property
    def is_export(self):
        return self._is_export

    @is_export.setter
    def is_export(self, value):
        self._is_export = value
        self.dec.is_export = value

    def forward(self, x, x_mask, y=None, y_mask=None, spk_emb=None, kv_cache=None, cache_seqlens=None):
        """
        Autoregressive next-token prediction for discrete token generation.
        
        Args:
            x: Embedded text (B, seq_len_x, Cemb) for teacher-forcing during training
            x_mask: Boolean mask (B, seq_len_x) where True indicates padded positions
            y: Encoded text representations (B, seq_len_y, d_model), or None in pretraining mode
            y_mask: Boolean mask (B, seq_len_y) where True indicates padded positions, or None in pretraining mode
            spk_emb: Speaker embedding (B, 1, speaker_channels) or None
            kv_cache: Optional list of KV caches for transformer layers
            cache_seqlens: Optional tensor of sequence lengths for KV cache
            
        Returns:
            Tuple containing:
                - indices_pred: Predicted token logits (B, seq_len_x-1, vocab_size)
                - kv_cache: Updated KV cache list
                - cache_seqlens: Updated cache sequence lengths
        """
        B, L = x.size()

        # Create attention masks
        self_attn_mask = expand_self_attention_mask(x_mask)
            
        # Embed input tokens
        x = self.embed(x)

        if x_mask is not None:
            x = x.masked_fill(x_mask.unsqueeze(-1), 0.0)


        # Decoder forward pass
        if self.decoder_type == "transformer":
            dec_out, new_kv_cache, new_cache_seqlens = self.dec(x, None, spk_emb, self_attn_mask, None, 
                                                               kv_caches=kv_cache, cache_seqlens=cache_seqlens)
            
            # Final projection
            indices_pred = self.out_proj(dec_out)  # (B, L-1, vocab_size)
            
            return indices_pred, new_kv_cache, new_cache_seqlens


    def infer(self, max_length=1000, spk_emb=None, temperature=0.8, top_p=1.0, input_tokens=None):
        """
        Unconditional inference for discrete token generation without encoder input.
        
        Args:
            max_length: Maximum length of generated token sequence
            spk_emb: Speaker embedding (B, 1, speaker_channels) or None
            temperature: Temperature for sampling (higher = more random)
            top_p: Top-p (nucleus) sampling threshold (1.0 = no top-p sampling)
            
        Returns:
            Generated discrete token IDs (B, T_gen)
        """
        B = spk_emb.size(0) if spk_emb is not None else 1  # Default to batch size 1 if no speaker embedding
        device = spk_emb.device if spk_emb is not None else torch.device('cuda')

        # Initialize with a "beginning of sequence" token
        if input_tokens is None: #lol
            decoder_input = torch.full((B, 1),
                                       fill_value=self.bos_token_id,
                                       dtype=torch.long,
                                       device=device)
        else:
            decoder_input = input_tokens

        token_outputs = []  # To store generated token IDs
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        # Audio vocabulary invalid mask (B, V). Prevent non-audio tokens from being generated.
        invalid_mask = torch.ones(1, self.embed.num_embeddings, dtype=torch.bool, device=device)
        invalid_mask[:, self.vocab_min:self.vocab_max + 1] = False

        for t in range(max_length):
            if t == max_length - 1:
                print("Warning! Reached max decoder steps.")

            current_length = decoder_input.size(1)
            # In inference mode, no positions are padded.
            x_mask = None

            # Run forward pass without encoder input (y and y_mask are None in pretraining mode)
            indices_pred = self.forward(
                decoder_input, x_mask, None, None, spk_emb=spk_emb)

            # Get the logits for the last token.
            logits = indices_pred[:, -1, :]  # (B, vocab_size)

            next_token = self.sample(logits, temperature, top_p, invalid_mask=invalid_mask)

            token_outputs.append(next_token)

            # Append the sampled token to the decoder input.
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break

        # Concatenate predictions along the time dimension.
        token_outputs = torch.cat(token_outputs, dim=1)  # (B, seq_len)
        return token_outputs

    def sample(self, logits, temperature, top_p, prev_tokens=None, repetition_penalty=1.05, invalid_mask=None):
        if invalid_mask is not None:
            logits = logits.masked_fill(invalid_mask, float('-inf'))

        # --- Repetition penalty ---
        if prev_tokens is not None and repetition_penalty != 1.0:
            for b in range(logits.size(0)):
                for tok in prev_tokens[b].tolist():
                    if self.vocab_min <= tok <= self.vocab_max:
                        logits[b, tok] /= repetition_penalty

        # Temperature scaling
        logits = logits / temperature

        # Top-p (nucleus) sampling
        if top_p < 1.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


def copy_linear(dst, src):
    if dst.weight.shape != src.weight.shape:
        return
    assert dst.weight.shape == src.weight.shape, f"weight shape {dst.weight.shape} != {src.weight.shape}"
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None:
            assert dst.bias.shape == src.bias.shape, f"bias shape {dst.bias.shape} != {src.bias.shape}"
            dst.bias.copy_(src.bias)
        elif dst.bias is not None and src.bias is None:
            torch.nn.init.zeros_(dst.bias)


def copy_layernorm(dst_ln, src_ln):
    # shape/affine must match
    assert isinstance(dst_ln, nn.LayerNorm) and isinstance(src_ln, nn.LayerNorm)
    assert dst_ln.normalized_shape == src_ln.normalized_shape
    assert dst_ln.elementwise_affine == src_ln.elementwise_affine

    # match epsilon (numerical behavior)
    dst_ln.eps = src_ln.eps

    # copy params if affine
    if dst_ln.elementwise_affine:
        with torch.no_grad():
            dst_ln.weight.copy_(src_ln.weight)
            dst_ln.bias.copy_(src_ln.bias)

# Main model
class Echolancer(nn.Module):
    """ Echolancer:
        Decoder-only model for processing concatenated sequences"""

    def __init__(self, vocab_size, decoder_hidden, decoder_layer, decoder_head, 
                 decoder_dropout=0.1, emotion_channels=256,
                 speaker_channels=0, multi_speaker=False, n_speaker=0,
                 alibi_alpha=1.0, use_alibi=False, activation='relu',
                 vq_token_mode=False, vq_vocab_size=1024, 
                 decoder_kv_heads=None, decoder_start_i=0,
                 emotion_input_size=768, emotion_hidden_sizes=[512, 384], emotion_dropout=0.1,
                 pretraining_mode=False, use_te=False, zero_shot_mode=False,
                 lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False,
                 use_canon_a=False, use_canon_c=False, canon_kernel_size=4):
        super(Echolancer, self).__init__()
        self.emotion_channels = emotion_channels
        self.speaker_channels = speaker_channels
        self.use_alibi = use_alibi
        self.alibi_alpha = alibi_alpha
        self.text_vocab_size = vocab_size
        self.vq_token_mode = vq_token_mode  # New parameter for zero-shot mode
        self.vq_vocab_size = vq_vocab_size  # Vocabulary size for VQ tokens
        self.use_te = use_te
        self.zero_shot_mode = zero_shot_mode
        
        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale
        self._is_export = False

        
        if self.emotion_channels > 0:
            # Progressive downsampling emotion encoder
            self.emotion_encoder = EmotionEncoder(
                input_size=emotion_input_size,
                emotion_channels=emotion_channels,
                hidden_sizes=emotion_hidden_sizes,
                dropout=emotion_dropout
            )
        else:
            print("No emotion conditioning")
            self.emotion_encoder = None

        self.decoder = AudioDecoderAR(
            decoder_hidden,  # Use decoder_hidden instead of encoder_hidden
            self.vq_vocab_size,
            decoder_hidden,  # Also use decoder_hidden for filter_channels
            decoder_layer,
            decoder_head,
            decoder_dropout,
            speaker_channels=self.speaker_channels,
            dec_type="transformer",
            alibi_alpha=alibi_alpha,
            use_alibi=use_alibi,
            activation=activation,
            num_kv_heads=decoder_kv_heads,
            start_i=decoder_start_i,
            pretraining_mode=pretraining_mode,
            use_te=use_te,
            vocab_offset=self.text_vocab_size,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_scale=lora_scale,
            use_macaron=use_macaron,
            use_canon_a=use_canon_a,
            use_canon_c=use_canon_c,
            canon_kernel_size=canon_kernel_size,
        )

        self.combined_vocab_size = vocab_size + self.decoder.n_embeds # decoder adds special tokens.
        self.embedding_channels = decoder_hidden  # Use decoder_hidden
        self.combined_emb = nn.Embedding(self.combined_vocab_size, self.embedding_channels)
        self.combined_head = nn.Linear(self.embedding_channels, self.combined_vocab_size, bias=False)

        self.combined_head.weight = self.combined_emb.weight # Tie weights

        self.spk_norm = nn.LayerNorm(speaker_channels) if (vq_token_mode or zero_shot_mode) else nn.Identity()
        self.decoder.out_proj = self.combined_head
        self.decoder.embed = self.combined_emb

        self.last_logprobs = None
        self.speaker_emb = None
        if multi_speaker and n_speaker > 0 and not (vq_token_mode or zero_shot_mode):
            # Only use speaker embedding in traditional mode
            self.speaker_emb = nn.Embedding(n_speaker, self.speaker_channels)
        
        # Add VQ token embedding and encoder for zero-shot mode
        if vq_token_mode:
            # Embedding layer for VQ tokens
            # Use a reasonable default size if speaker_channels is 0
            embedding_dim = speaker_channels if speaker_channels > 0 else 256
            self.vq_token_emb = nn.Embedding(vq_vocab_size + 16, embedding_dim)
            # Small transformer encoder to process VQ tokens into speaker embeddings
            self.vq_encoder = TransformerEncoder(
                embedding_dim,  # d_model
                4,  # num_heads (smaller than main encoder)
                3,  # num_layers (smaller than main encoder)
                embedding_dim * 2,  # d_ff
                decoder_dropout,  # Use decoder_dropout instead of encoder_dropout
                alibi_alpha,
                use_alibi,
                activation,
                start_i=4
            )
            # Final projection layer to get correct speaker embedding size
            if speaker_channels > 0:
                self.vq_final_proj = nn.Linear(embedding_dim, speaker_channels)
            else:
                self.vq_final_proj = nn.Linear(embedding_dim, 256)  # Default size
            # Add a pooling layer to convert sequence to single embedding
            self.vq_pooling = nn.AdaptiveAvgPool1d(1)

        self.apply_xavier_uniform_init()

    @property
    def is_export(self):
        return self._is_export

    @is_export.setter
    def is_export(self, value):
        self._is_export = value
        self.decoder.is_export = value

    def forward(self, sequence, seq_lens, spk_ids=None, em_hidden=None, kv_cache=None, cache_seqlens=None):
        """
        Forward pass of decoder-only Echolancer for concatenated sequences.
        
        Args:
            sequence: Concatenated token sequence (B, T_total) containing both text and audio tokens
            seq_lens: Sequence lengths (B,) for the concatenated sequences
            mask: Optional attention mask (B, T_total) where True indicates padded positions
            spk_ids: Speaker IDs (B,) or speaker embeddings (B, speaker_channels) - optional
            em_hidden: Emotion embeddings (B, emotion_dim) - optional
            
            kv_cache: Optional list of KV caches for transformer layers
            cache_seqlens: Optional tensor of sequence lengths for KV cache
            
        Returns:
            Tuple of (logits, new_kv_cache, new_cache_seqlens)
        """
        B, T_total = sequence.size()
        
        spk_emb = self.get_spk_cond(spk_ids)

        # Process emotion encoding if applicable
        encoded_emotion = self.emotion_encoder(em_hidden) if self.emotion_channels > 0 and em_hidden is not None else None

        # Create attention mask if not provided
        x_mask = sequence_mask(T_total, seq_lens) if seq_lens is not None else None


        logits, new_kv_cache, new_cache_seqlens = self.decoder(
            sequence, x_mask, spk_emb=spk_emb, kv_cache=kv_cache, cache_seqlens=cache_seqlens
        )

        return logits, new_kv_cache, new_cache_seqlens

    def embed_sequence(self, seq_lens, sequence):
        """
        Embed a concatenated sequence for processing.
        
        Args:
            seq_lens: Sequence lengths (B,) 
            sequence: Token sequence (B, T_total)
            
        Returns:
            Embedded sequence (B, T_total, embedding_dim)
        """
        seq_mask = sequence_mask(sequence.size(1), seq_lens)
        embedded = self.combined_emb(sequence)  # (B, len, Cemb)
        # Apply mask to padded positions if needed
        embedded = embedded.masked_fill(seq_mask.unsqueeze(-1), 0)
        return seq_mask, embedded

    def infer(self, seq_start, spk_ids=None, em_hidden=None, max_length=1000, temperature=0.8, top_p=1.0):
        """
        Autoregressive inference for the decoder-only Echolancer model.
        
        Args:
            seq_start: Starting sequence (B, T_start) - initial tokens to condition generation
            seq_lens: Sequence lengths (B,) for the starting sequences
            spk_ids: Speaker IDs (B,) or speaker embeddings (B, speaker_channels) - optional
            em_hidden: Emotion embeddings (B, emotion_dim) - optional
            max_length: Maximum length of generated sequence
            temperature: Temperature for sampling (higher = more random)
            top_p: Top-p (nucleus) sampling threshold (1.0 = no top-p sampling)
            
        Returns:
            Generated token sequence (B, T_gen)
        """

        device = seq_start.device

        spk_emb = self.get_spk_cond(spk_ids)

        # Process emotion encoding if applicable
        if self.emotion_channels > 0 and em_hidden is not None:
            encoded_emotion = self.emotion_encoder(em_hidden)
        else:
            encoded_emotion = None

        token_outputs = self.decoder.infer(
            input_tokens=seq_start,
            max_length=max_length,
            spk_emb=spk_emb,
            temperature=temperature,
            top_p=top_p
        )

        return token_outputs

    def get_spk_cond(self, spk_ids):
        # Handle speaker embedding based on mode
        if self.vq_token_mode and spk_ids is not None:
            # Zero-shot mode: use spk_ids as VQ tokens to generate speaker embedding
            spk_emb = self.vq_tokens_to_speaker_emb(spk_ids, None)  # Use the provided spk_ids as VQ tokens
        elif self.speaker_emb is not None and spk_ids is not None:
            # Traditional mode: use speaker ID to lookup embedding
            batch_size = spk_ids.size(0)
            spk_emb = self.speaker_emb(spk_ids).view(batch_size, 1, -1)
        elif self.zero_shot_mode and spk_ids is not None:
            # Zero-shot mode: use provided speaker embeddings
            batch_size = spk_ids.size(0)
            spk_emb = spk_ids.view(batch_size, 1, -1)  # (B, 1, speaker_channels)
            spk_emb = F.normalize(spk_emb, p=2, dim=-1)  # normalize along last dim
        else:
            # No speaker conditioning
            spk_emb = None

        spk_emb = self.spk_norm(spk_emb)

        return spk_emb

    def vq_tokens_to_speaker_emb(self, vq_tokens, vq_token_lens=None):
        """
        Convert VQ tokens to speaker embedding using a small transformer encoder.
        
        Args:
            vq_tokens: VQ token IDs (B, T_vq)
            vq_token_lens: VQ token sequence lengths (B,) - optional
            
        Returns:
            Speaker embedding (B, 1, speaker_channels)
        """
        if not self.vq_token_mode:
            raise ValueError("VQ token mode is not enabled for this model")

        # Embed VQ tokens
        vq_emb = self.vq_token_emb(vq_tokens)  # (B, T_vq, speaker_channels)
        
        # Create mask if lengths provided
        if vq_token_lens is not None:
            vq_mask = sequence_mask(vq_emb.size(1), vq_token_lens)  # (B, T_vq)
            sa_mask = expand_self_attention_mask(vq_mask)  # (B, 1, T_vq, T_vq)
            seq_mask = vq_mask  # Sequence mask for the transformer encoder
        else:
            vq_mask = None
            sa_mask = None
            seq_mask = torch.zeros(vq_emb.size(0), vq_emb.size(1), dtype=torch.bool, device=vq_emb.device)  # No padding mask
            
        # Encode with transformer
        encoded_vq = self.vq_encoder(vq_emb, sa_mask, seq_mask)  # (B, T_vq, speaker_channels)
        
        # Pool to get single embedding
        if vq_mask is not None:
            # Masked average pooling
            # Set padded positions to zero
            mask_expanded = vq_mask.unsqueeze(-1).expand_as(encoded_vq)
            encoded_vq = encoded_vq.masked_fill(mask_expanded, 0.0)
            # Compute mean of non-padded positions
            lengths_expanded = vq_token_lens.float().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            pooled = encoded_vq.sum(dim=1, keepdim=True) / lengths_expanded  # (B, 1, speaker_channels)
        else:
            # Simple average pooling
            pooled = encoded_vq.mean(dim=1, keepdim=True)  # (B, 1, speaker_channels)
            
        return pooled

    def apply_xavier_uniform_init(self):
        """
        Apply Xavier uniform initialization to transformer components, embeddings, and linear layers.
        This helps with gradient flow and training stability, especially for deeper networks.
        """
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        self.apply(init_fn)

    def apply_zero_init(self):
        """
        Apply zero initialization to transformer components, embeddings, and linear layers.

        If you use this function, Noam Shazeer himself will shoot you. Bad for training.
        """

        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.zeros_(m.weight)
            elif isinstance(m, nn.Conv1d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_fn)

    def enable_lora(self):
        """Enable LoRA adaptation by setting requires_grad=True for LoRA parameters"""
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True

    def disable_lora(self):
        """Disable LoRA adaptation by setting requires_grad=False for LoRA parameters"""
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = False

    def merge_lora_weights(self):
        """Merge LoRA weights into the original weights"""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                # Calculate the LoRA update: A @ B scaled appropriately
                lora_update = (module.lora_A @ module.lora_B) * module.scaling
                
                # Add the update to the original weight
                with torch.no_grad():
                    module.original_layer.weight += lora_update.t()  # Transpose to match dimensions
                    
                # Zero out the LoRA parameters since they're now merged
                module.lora_A.zero_()
                module.lora_B.zero_()

    def unmerge_lora_weights(self):
        """Un-merge LoRA weights from the original weights"""
        for module in self.modules():
            if isinstance(module, LoRALayer):
                # Calculate the LoRA update that was previously added
                lora_update = (module.lora_A @ module.lora_B) * module.scaling
                
                # Subtract the update from the original weight
                with torch.no_grad():
                    module.original_layer.weight -= lora_update.t()  # Transpose to match dimensions

    def get_lora_parameters(self):
        """Get all LoRA parameters for optimizer"""
        lora_params = []
        for name, param in self.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                lora_params.append(param)
        return lora_params

    def get_non_lora_parameters(self):
        """Get all non-LoRA parameters (frozen) for optimizer"""
        non_lora_params = []
        for name, param in self.named_parameters():
            if 'lora_A' not in name and 'lora_B' not in name:
                non_lora_params.append(param)
        return non_lora_params


class EcholancerONNX(nn.Module):
    """
    ONNX-exportable wrapper for Echolancer with flat KV cache tensors.
    
    This wrapper handles conversion between flat cache tensors (ONNX-friendly)
    and the internal per-layer cache format.
    
    Cache shapes:
        k_cache: (num_layers, B, max_len, num_kv_heads, d_k)
        v_cache: (num_layers, B, max_len, num_kv_heads, d_k)
        cache_len: (B,) - current sequence length in cache
    
    Usage:
        # Wrap existing model
        model_onnx = EcholancerONNX(model, max_len=2048)
        model_onnx.eval()
        
        # Export to ONNX
        torch.onnx.export(model_onnx, (tokens, k_cache, v_cache, cache_len, spk_emb), "model.onnx", ...)
        
        # At runtime, same function works for prefill (T > 1) and decode (T = 1)
    """
    
    def __init__(self, model: Echolancer, max_len: int = 2048):
        super().__init__()
        self.model = model
        self.max_len = max_len
        
        # Enable export mode (propagates to all attention layers)
        self.model.is_export = True
        
        # Cache dimensions from model structure
        decoder = self.model.decoder.dec
        first_layer = decoder.layers[0]
        self.num_layers = len(decoder.layers)
        self.num_kv_heads = first_layer.self_attn.num_kv_heads
        self.d_k = first_layer.self_attn.d_k
    
    def get_cache_shape(self, batch_size: int = 1):
        """Returns the shape for k_cache and v_cache tensors."""
        return (self.num_layers, batch_size, self.max_len, self.num_kv_heads, self.d_k)
    
    def create_cache(self, batch_size: int = 1, device='cpu', dtype=torch.float32):
        """
        Create empty cache tensors for initialization.
        
        Returns:
            k_cache: (num_layers, B, max_len, num_kv_heads, d_k)
            v_cache: (num_layers, B, max_len, num_kv_heads, d_k)
            cache_len: (B,)
        """
        shape = self.get_cache_shape(batch_size)
        k_cache = torch.zeros(shape, device=device, dtype=dtype)
        v_cache = torch.zeros(shape, device=device, dtype=dtype)
        cache_len = torch.zeros(batch_size, device=device, dtype=torch.int32)
        return k_cache, v_cache, cache_len
    
    def forward(
        self,
        tokens,      # (B, T) - input token IDs
        k_cache,     # (num_layers, B, max_len, num_kv_heads, d_k)
        v_cache,     # (num_layers, B, max_len, num_kv_heads, d_k)
        cache_len,   # (B,) - current position in cache
        spk_id=None  # (B,) speaker ID for multi-speaker, or (B, 1, speaker_channels) embedding for zero-shot
    ):
        """
        Unified forward for both prefill and decode.
        
        Args:
            tokens: Token IDs (B, T) - T can be any length
            k_cache: Key cache (num_layers, B, max_len, num_kv_heads, d_k)
            v_cache: Value cache (num_layers, B, max_len, num_kv_heads, d_k)
            cache_len: Current cache length per batch (B,)
            spk_id: Speaker input. For multi-speaker models with fixed embeddings,
                    pass speaker ID (B,) which will be looked up via the embedding table.
                    For zero-shot mode, pass speaker embedding (B, 1, speaker_channels) directly.
            
        Returns:
            logits: (B, T, vocab_size)
            k_cache: Updated key cache (same shape as input)
            v_cache: Updated value cache (same shape as input)
            cache_len: Updated cache length (B,)
        """
        B, T = tokens.size()
        device = tokens.device
        dtype = k_cache.dtype
        
        # Handle speaker conditioning
        # If model has speaker embedding table and input is 1D, look up the embedding
        if self.model.speaker_emb is not None and spk_id is not None:
            if spk_id.dim() == 1:
                # spk_id is (B,) - look up embedding
                spk_cond = self.model.speaker_emb(spk_id).view(B, 1, -1)  # (B, 1, speaker_channels)
                spk_cond = self.model.spk_norm(spk_cond)
            else:
                # Already an embedding tensor
                spk_cond = spk_id
        elif spk_id is not None:
            # Zero-shot mode or direct embedding input
            spk_cond = spk_id
        else:
            spk_cond = None
        
        # Convert flat caches to per-layer list format
        # Each layer expects: (k, v) where k, v are (B, max_len, num_kv_heads, d_k)
        kv_caches = []
        for layer_idx in range(self.num_layers):
            k_layer = k_cache[layer_idx]  # (B, max_len, num_kv_heads, d_k)
            v_layer = v_cache[layer_idx]  # (B, max_len, num_kv_heads, d_k)
            kv_caches.append((k_layer, v_layer))
        
        # Create sequence lengths (all tokens are valid, no padding during inference)
        seq_lens = cache_len + T
        
        # Call the model's forward with KV cache
        logits, new_kv_caches, new_cache_len = self.model(
            tokens, 
            seq_lens=None,  # No padding mask needed for inference
            spk_ids=spk_cond,  # Pass processed speaker conditioning
            em_hidden=None,
            kv_cache=kv_caches,
            cache_seqlens=cache_len
        )
        
        # Convert per-layer caches back to flat tensors
        # Stack along layer dimension
        new_k_cache = torch.stack([kv[0] for kv in new_kv_caches], dim=0)
        new_v_cache = torch.stack([kv[1] for kv in new_kv_caches], dim=0)
        
        return logits, new_k_cache, new_v_cache, new_cache_len
