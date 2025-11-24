import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_varlen_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("WARNING: Flash Attention not available! Falling back to manual attention.")

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
        return x, attn_logprobs.to(scores.dtype)


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

    def forward(self, query, key, value, mask=None):
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

        # For self-attention in transformer blocks, only return the attention output,
        # not a tuple with logprobs (that's used only for cross-attention)
        return x

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
        x = self.self_attn(x, x, x, attn_mask)
        x = residual + self.dropout(x)

        # 2. Feed-forward (pre-norm)
        residual = x
        x = self.norm2(x)  # PRE-norm
        x = self.ffn(x, seq_mask)
        x = residual + self.dropout(x)

        return x

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
            x = layer(x, attn_mask, seq_mask)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i_increment=0,
                 cross_attn_type="full", disable_cross_attn=False, use_te=False, d_cond=0, 
                 lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False):
        super(TransformerDecoderLayer, self).__init__()
        self.disable_cross_attn = disable_cross_attn  # Option to disable cross attention
        self.use_te = use_te
        self.use_macaron = use_macaron
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_scale = lora_scale
        self.ffn_scale = 1.0
        
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

    def forward(self, x, memory, cond, self_attn_mask, cross_attn_mask, ffn_seq_mask):
        # 1. Self-attention (pre-norm)
        residual = x
        
        if self.use_macaron:
            x = self.norm0(x, cond=cond)
            x = self.ffn0(x, ffn_seq_mask, cond=cond)
            x = residual + self.ffn_scale * x
            residual = x

        x = self.norm1(x, cond=cond)  # PRE-norm
        if self.disable_cross_attn:  # keep the bug-fix comment #1
            self_attn_mask = None

        x = self.self_attn(x, x, x, self_attn_mask)  # Always returns a tuple (output, logprobs_or_none)
        x = residual + x

        # 2. Cross-attention (pre-norm)
        if not self.disable_cross_attn:
            residual = x
            x = self.norm2(x)  # PRE-norm
            x, attn_logprobs = self.cross_attn(x, memory, memory, cross_attn_mask)
            x = residual + x

            self.last_logprobs = attn_logprobs


        # 3. Feed-forward (pre-norm)
        residual = x
        x = self.norm3(x, cond=cond)  # PRE-norm
        x = self.ffn(x, ffn_seq_mask, cond=cond)

        x = residual + self.ffn_scale * x

        return x



class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None,
                 start_i=0, disable_cross_attn=False, use_te=False, d_cond=0, lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False):
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

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, alibi_alpha, 
                                   use_alibi, activation, num_kv_heads,
                                   start_i_increment=start_i + ((i * num_heads) // alibi_scaling_fac),
                                   disable_cross_attn=disable_cross_attn,
                                   use_te=use_te, cross_attn_type="full", d_cond=d_cond,
                                   lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale, use_macaron=use_macaron)
            for i in range(num_layers)
        ])
        self.disable_cross_attn = disable_cross_attn

    def forward(self, x, memory, cond, self_attn_mask, cross_attn_mask=None, ffn_seq_mask=None):
        for layer in self.layers:
            x = layer(x, memory, cond, self_attn_mask, cross_attn_mask, ffn_seq_mask)
        return x

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
        x = self.encoder(x, attn_mask, x_mask)

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
                 pretraining_mode=False, use_te=False, vocab_offset=0, lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False):
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

        self.spk_cond = None
        if self.dec_type == "transformer":
            self.embed = None
            #self.prenet = CausalConv1d(self.filter_channels, self.filter_channels, 3, 1, False)
            self.dec = TransformerDecoder(filter_channels, heads, depth,
                                        filter_channels * 4, dropout, alibi_alpha, use_alibi, activation, num_kv_heads, start_i,
                                        disable_cross_attn=True, use_te=use_te, d_cond=speaker_channels,
                                        lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, lora_scale=lora_scale, use_macaron=use_macaron)

            self.out_proj = None
        else:
            raise RuntimeError(f"Invalid decoder type: {self.dec_type}")

        self.gate_proj = nn.Identity()  # no sigmoid, we use BCEWithLogitsLoss
        self.g_drop = nn.Dropout(0.1)

    def forward(self, x, x_mask, y=None, y_mask=None, spk_emb=None):
        """
        Autoregressive next-token prediction for discrete token generation.
        
        Args:
            x: Embedded text (B, seq_len_x, Cemb) for teacher-forcing during training
            x_mask: Boolean mask (B, seq_len_x) where True indicates padded positions
            y: Encoded text representations (B, seq_len_y, d_model), or None in pretraining mode
            y_mask: Boolean mask (B, seq_len_y) where True indicates padded positions, or None in pretraining mode
            spk_emb: Speaker embedding (B, 1, speaker_channels) or None

        Returns:
            Tuple containing:
                - indices_pred: Predicted token logits (B, seq_len_x-1, vocab_size)
                - gate_pred: Gate predictions (B, seq_len_x-1, 1)
                - attn_logprob: Attention log probabilities (B, 1, seq_len_x-1, seq_len_y) or zeros in pretraining
                - x_mask_in: Input mask for decoder (B, seq_len_x-1)
                - hidden_out: Decoder hidden states (B, seq_len_x-1, d_model)
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
            dec_out = self.dec(x, None, spk_emb, self_attn_mask, None)
            
            # Final projection
            indices_pred = self.out_proj(dec_out)  # (B, L-1, vocab_size)
            
            return indices_pred


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
                 lora_rank=0, lora_alpha=16, lora_dropout=0.0, lora_scale=1.0, use_macaron=False):
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

    def forward(self, sequence, seq_lens, spk_ids=None, em_hidden=None):
        """
        Forward pass of decoder-only Echolancer for concatenated sequences.
        
        Args:
            sequence: Concatenated token sequence (B, T_total) containing both text and audio tokens
            seq_lens: Sequence lengths (B,) for the concatenated sequences
            mask: Optional attention mask (B, T_total) where True indicates padded positions
            spk_ids: Speaker IDs (B,) or speaker embeddings (B, speaker_channels) - optional
            em_hidden: Emotion embeddings (B, emotion_dim) - optional
            
        Returns:
            Tuple of outputs for loss computation
        """
        B, T_total = sequence.size()
        
        spk_emb = self.get_spk_cond(spk_ids)

        # Process emotion encoding if applicable
        encoded_emotion = self.emotion_encoder(em_hidden) if self.emotion_channels > 0 and em_hidden is not None else None

        # Create attention mask if not provided
        x_mask = sequence_mask(T_total, seq_lens) if seq_lens is not None else None


        logits = self.decoder(
            sequence, x_mask, None, None, spk_emb=spk_emb,
        )

        return logits

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