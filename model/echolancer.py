import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Utility functions
def sequence_mask(max_length, x_lengths):
    """
    Make a bool sequence mask
    :param max_length: Max length of sequences
    :param x_lengths: Tensor (batch,) indicating sequence lengths
    :return: Bool tensor size (batch, max_length) where True is padded and False is valid
    """
    mask = torch.arange(max_length).expand(len(x_lengths), max_length).to(x_lengths.device)
    mask = mask >= x_lengths.unsqueeze(1)
    return mask

def expand_self_attention_mask(x_mask):
    """
    Expand True=padded masks into an attention mask for self-attention.
    :param x_mask: Mask of x size (batch, seq_len), where True is padded
    :return: Attention mask for MultiHeadAttention
    """
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
    def __init__(self, input_size=768, emotion_channels=256, hidden_sizes=[512, 384], dropout=0.1):
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

    def __init__(self, num_embeddings, embedding_dim, dropout=0.1, norm=True):
        super(NormalizedEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim) if norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with sequence masking support
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        elif activation.lower() == 'swiglu':
            # For SwiGLU, we need to adjust the dimensions
            # We'll use 2/3 of d_ff to match common implementations
            self.linear1 = nn.Linear(d_model, 2 * (d_ff // 2))  # Split for SwiGLU
            self.linear2 = nn.Linear(d_ff // 2, d_model)
            self.activation = self._swiglu
        elif activation.lower() == 'relu2':
            self.activation = self._relu2
        else:
            raise ValueError(f"Unsupported activation: {activation}")

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

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Boolean mask of shape (batch_size, seq_len) where True indicates padded positions
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
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

# Attention modules (simplified versions)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, alibi_alpha=1.0, use_alibi=False, use_combined_linear=True, num_kv_heads=None, start_i_increment=0):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_query_groups = self.num_heads // self.num_kv_heads
        self.d_k = d_model // num_heads
        self.use_alibi = use_alibi
        self.alibi_alpha = alibi_alpha
        self.use_combined_linear = use_combined_linear
        self.start_i_increment = start_i_increment

        if use_combined_linear:
            # Use a single linear layer for all projections (more efficient for self-attention)
            # Input size: d_model, Output size: d_model * 3 (for Q, K, V)
            self.W_combined = nn.Linear(d_model, d_model + 2 * self.num_kv_heads * self.d_k)
        else:
            # Use separate linear layers (better for cross-attention)
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)
            self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        
        self.W_o = nn.Linear(d_model, d_model)

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

    def _get_alibi_bias(self, seq_len_q, seq_len_k):
        """
        Generate ALiBi attention bias matrix.
        """
        # Create position indices
        q_indices = torch.arange(seq_len_q, device=self.alibi_slopes.device).unsqueeze(1)  # (seq_len_q, 1)
        k_indices = torch.arange(seq_len_k, device=self.alibi_slopes.device).unsqueeze(0)  # (1, seq_len_k)
        
        # Calculate position differences
        relative_positions = q_indices - k_indices  # (seq_len_q, seq_len_k)
        
        # Expand to (num_heads, seq_len_q, seq_len_k)
        relative_positions = relative_positions.unsqueeze(0).expand(
            self.num_heads, seq_len_q, seq_len_k
        )
        
        # Apply slopes
        alibi_bias = self.alibi_slopes.unsqueeze(1).unsqueeze(2) * relative_positions
        
        return alibi_bias  # (num_heads, seq_len_q, seq_len_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        if self.use_combined_linear and query is key and key is value:
            # Self-attention with combined linear: Q, K, V all come from the same input
            if hasattr(self, 'W_combined'):
                qkv = self.W_combined(query)
                Q = qkv[..., :self.d_model]
                K = qkv[..., self.d_model:self.d_model + self.num_kv_heads * self.d_k]
                V = qkv[..., self.d_model + self.num_kv_heads * self.d_k:]
            else:
                # Fallback to separate projections
                Q = self.W_q(query)
                K = self.W_k(key)
                V = self.W_v(value)
        else:
            # Cross-attention or separate linear mode: use separate projections
            if hasattr(self, 'W_combined') and not self.use_combined_linear:
                # Fallback to separate projections from combined linear if needed
                Q = self.W_combined(query)[..., :self.d_model]
                K = self.W_combined(key)[..., self.d_model:self.d_model + self.num_kv_heads * self.d_k]
                V = self.W_combined(value)[..., self.d_model + self.num_kv_heads * self.d_k:]
            else:
                # Use separate linear layers
                Q = self.W_q(query)
                K = self.W_k(key)
                V = self.W_v(value)

        # Reshape and transpose for multi-head attention
        # Query reshape: (batch_size, seq_len_q, num_heads, d_k)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        
        # Key and Value reshape: (batch_size, seq_len_k, num_kv_heads, d_k)
        K = K.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat K and V for grouped query attention
        # Each KV head is repeated num_query_groups times
        if self.num_kv_heads != self.num_heads:
            # Repeat K and V to match the number of query heads
            K = K.repeat_interleave(self.num_query_groups, dim=1)
            V = V.repeat_interleave(self.num_query_groups, dim=1)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Add ALiBi bias if enabled
        if self.use_alibi:
            alibi_bias = self._get_alibi_bias(seq_len_q, seq_len_k)
            alibi_bias = alibi_bias.unsqueeze(0)  # Add batch dimension
            scores = scores + alibi_bias

        if mask is not None:
            # Ensure mask has the right shape for broadcasting with scores
            # scores: (batch_size, num_heads, seq_len_q, seq_len_k)
            # mask should be: (batch_size, 1, seq_len_q, seq_len_k) to broadcast to all heads
            
            # Handle different mask dimensions
            if mask.dim() == 2:
                # 2D mask: (batch_size, seq_len) - convert to 4D attention mask
                # This is likely a sequence mask, need more context to convert properly
                # For now, we'll assume it's for query sequence and expand appropriately
                mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
                mask = mask.expand(-1, -1, seq_len_q, -1)  # (batch_size, 1, seq_len_q, seq_len)
            elif mask.dim() == 3:
                # 3D mask: (batch_size, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len_q, seq_len_k)
            elif mask.dim() == 4 and mask.size(1) != 1:
                # 4D mask but wrong head dimension
                mask = mask[:, :1, :, :]  # Take first head and broadcast
            
            # Expand last two dimensions to match scores if needed
            if mask.size(-2) != seq_len_q or mask.size(-1) != seq_len_k:
                # Create a new mask with correct dimensions
                new_mask = torch.ones(batch_size, 1, seq_len_q, seq_len_k, dtype=mask.dtype, device=mask.device)
                # Copy values from original mask (assuming they're compatible)
                min_seq_q = min(mask.size(-2), seq_len_q)
                min_seq_k = min(mask.size(-1), seq_len_k)
                if mask.dim() == 4:
                    new_mask[:, :, :min_seq_q, :min_seq_k] = mask[:, :, :min_seq_q, :min_seq_k]
                mask = new_mask
            
            scores = scores.masked_fill(mask == 0, -1e9)

        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply attention to values
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Output projection
        x = self.W_o(x)

        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i_increment=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, alibi_alpha, use_alibi, use_combined_linear=True, 
                                          num_kv_heads=num_kv_heads, start_i_increment=start_i_increment)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask, seq_mask):
        # Self attention (use 4D attention mask)
        attn_out = self.self_attn(x, x, x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed forward (use 2D sequence mask)
        ffn_out = self.ffn(x, seq_mask)
        x = self.norm2(x + self.dropout(ffn_out))

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i=0):
        super(TransformerEncoder, self).__init__()
        # Calculate scaling factor to prevent start_i from exceeding 32 in final layer
        alibi_scaling_fac = max(1, ((num_layers - 1) * num_heads) // (32 - start_i)) if num_layers > 1 else 1
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, alibi_alpha, 
                                   use_alibi, activation, num_kv_heads, 
                                   start_i_increment=start_i + ((i * num_heads) // alibi_scaling_fac))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask, seq_mask):
        for layer in self.layers:
            x = layer(x, attn_mask, seq_mask)
        return self.norm(x)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i_increment=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, alibi_alpha, use_alibi, use_combined_linear=True, 
                                          num_kv_heads=num_kv_heads, start_i_increment=start_i_increment)  # Self-attention uses GQA
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, alibi_alpha=1.0, use_alibi=False, use_combined_linear=False, 
                                           num_kv_heads=num_kv_heads, start_i_increment=0)  # Cross-attention uses separate linear layers and GQA
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_attn_mask, cross_attn_mask, ffn_seq_mask):
        # Self attention
        attn_out = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross attention
        attn_out = self.cross_attn(x, memory, memory, cross_attn_mask)
        x = self.norm2(x + self.dropout(attn_out))

        # Feed forward (use sequence mask)
        ffn_out = self.ffn(x, ffn_seq_mask)
        x = self.norm3(x + self.dropout(ffn_out))

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i=0):
        super(TransformerDecoder, self).__init__()
        # Calculate scaling factor to prevent start_i from exceeding 32 in final layer
        alibi_scaling_fac = max(1, ((num_layers - 1) * num_heads) // (32 - start_i)) if num_layers > 1 else 1
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, alibi_alpha, 
                                   use_alibi, activation, num_kv_heads,
                                   start_i_increment=start_i + ((i * num_heads) // alibi_scaling_fac))
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, self_attn_mask, cross_attn_mask, ffn_seq_mask):
        for layer in self.layers:
            x = layer(x, memory, self_attn_mask, cross_attn_mask, ffn_seq_mask)
        return self.norm(x)

# Model components
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, forward_expansion, dropout, 
                 emotion_channels=256, speaker_channels=0, alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i=0):
        super().__init__()
        self.embed = NormalizedEmbedding(vocab_size, embed_size)
        self.encoder = TransformerEncoder(embed_size, num_heads, num_layers, embed_size * forward_expansion, dropout, alibi_alpha, use_alibi, activation, num_kv_heads, start_i)
        self.emotion_channels = emotion_channels
        self.speaker_channels = speaker_channels

        if self.speaker_channels > 0:
            self.spk_cond = nn.Linear(speaker_channels, embed_size)

    def forward(self, token_ids, x_mask, encoded_em, spk_emb=None):
        # Embed token_ids
        x = self.embed(token_ids)  # Shape: (batch, max_seq_len, embed_size)

        if self.speaker_channels > 0 and spk_emb is not None:
            x = x + self.spk_cond(spk_emb)

        if self.emotion_channels > 0 and encoded_em is not None:
            # Expand emotion encoding to match sequence length
            encoded_em_expanded = encoded_em.unsqueeze(1).expand(-1, x.size(1), -1)
            x[:, :, :self.emotion_channels] = encoded_em_expanded

        # Convert 2D mask to 4D attention mask for transformer encoder
        attn_mask = expand_self_attention_mask(x_mask)
        
        # Pass both attention mask and sequence mask to transformer encoder
        x = self.encoder(x, attn_mask, x_mask)

        return x

class SpectrogramDecoderAR(nn.Module):
    def __init__(self, encoder_channels, mel_channels, filter_channels, depth, heads, dropout=0.1,
                 speaker_channels=0, dec_type="transformer", alibi_alpha=1.0, use_alibi=False, activation='relu', num_kv_heads=None, start_i=0):
        super().__init__()

        self.encoder_channels = encoder_channels
        self.filter_channels = filter_channels
        self.codebook_size = 1000
        self.n_embeds = self.codebook_size + 10
        self.speaker_channels = speaker_channels
        self.bos_token_id = self.codebook_size + 1
        self.eos_token_id = self.codebook_size + 2
        self.pad_token_id = self.codebook_size + 3

        self.dec_type = dec_type.lower()
        self.decoder_type = self.dec_type
        self.use_alibi = use_alibi
        self.alibi_alpha = alibi_alpha

        self.spk_cond = nn.Linear(speaker_channels, filter_channels) if speaker_channels > 0 else None

        if self.dec_type == "transformer":
            self.embed = nn.Embedding(self.n_embeds, self.filter_channels)
            self.dec = TransformerDecoder(filter_channels, heads, depth, 
                                        filter_channels * 4, dropout, alibi_alpha, use_alibi, activation, num_kv_heads, start_i)
            self.out_proj = nn.Linear(filter_channels, self.n_embeds)
        else:
            raise RuntimeError(f"Invalid decoder type: {self.dec_type}")

        self.gate_proj = nn.Identity()  # no sigmoid, we use BCEWithLogitsLoss
        self.g_drop = nn.Dropout(0.1)

    def forward(self, x, x_mask, y, y_mask, spk_emb=None, shift_tokens=True):
        """
        Autoregressive next-token prediction for discrete token generation.
        
        Args:
            x: Input discrete token IDs (B, seq_len_x) for teacher-forcing during training
            x_mask: Boolean mask (B, seq_len_x) where True indicates padded positions
            y: Encoded text representations (B, seq_len_y, d_model)
            y_mask: Boolean mask (B, seq_len_y) where True indicates padded positions
            spk_emb: Speaker embedding (B, 1, speaker_channels) or None
            shift_tokens: Whether to shift input tokens for autoregressive prediction
            
        Returns:
            Tuple containing:
                - indices_pred: Predicted token logits (B, seq_len_x-1, vocab_size)
                - gate_pred: Gate predictions (B, seq_len_x-1, 1)
                - attn_logprob: Attention log probabilities (B, 1, seq_len_x-1, seq_len_y)
                - x_mask_in: Input mask for decoder (B, seq_len_x-1)
                - hidden_out: Decoder hidden states (B, seq_len_x-1, d_model)
        """
        B, L = x.size()
        spk_res = None

        if shift_tokens:
            # Create the shifted sequence [x_0, ..., x_{L-2}] for autoregressive prediction
            x = x[:, :-1]  # Shape: (B, L-1)
            # Create the corresponding mask for the shifted sequence
            x_mask = x_mask[:, :-1]  # Shape: (B, L-1)

        # Create attention masks
        self_attn_mask = expand_self_attention_mask(x_mask)  # self-attention mask
        cross_attn_mask = expand_masks2(x_mask, y_mask.bool())  # cross-attention mask

        # Embed input tokens
        x = self.embed(x)

        # Apply speaker conditioning if provided
        if self.speaker_channels > 0 and spk_emb is not None:
            spk_res = self.spk_cond(spk_emb)
            x = x + spk_res

        # Decoder forward pass
        if self.decoder_type == "transformer":
            # Pass attention masks and sequence masks to transformer decoder
            dec_out = self.dec(x, y, self_attn_mask, cross_attn_mask, x_mask)
            
            # Final projections to token logits
            indices_pred = self.out_proj(dec_out)  # (B, L-1, vocab_size)
            
            # Dummy attention logprobs for compatibility
            attn_logprob = torch.zeros(B, 1, indices_pred.size(1), y.size(1), device=x.device)
            
            return indices_pred, attn_logprob, x_mask, dec_out

    def infer(self, y, y_mask, spk_emb=None, max_length=1000, temperature=0.8, top_p=1.0):
        """
        Autoregressive inference for discrete token generation.
        
        Args:
            y: Encoded text representations (B, T_text, d_model)
            y_mask: Text attention mask (B, T_text) where True indicates valid positions
            spk_emb: Speaker embedding (B, 1, speaker_channels) or None
            max_length: Maximum length of generated token sequence
            temperature: Temperature for sampling (higher = more random)
            top_p: Top-p (nucleus) sampling threshold (1.0 = no top-p sampling)
            
        Returns:
            Generated discrete token IDs (B, T_gen)
        """
        B = y.size(0)
        device = y.device

        # Initialize with a "beginning of sequence" token
        decoder_input = torch.full((B, 1),
                                   fill_value=self.bos_token_id,
                                   dtype=torch.long,
                                   device=device)

        token_outputs = []  # To store generated token IDs
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_length):
            if t == max_length - 1:
                print("Warning! Reached max decoder steps.")

            current_length = decoder_input.size(1)
            # In inference mode, no positions are padded.
            x_mask = torch.zeros(B, current_length, dtype=torch.bool, device=device)

            # Run forward pass
            indices_pred, attn_logprob, _, hidden_out = self.forward(
                decoder_input, x_mask, y, y_mask, spk_emb=spk_emb, shift_tokens=False)

            # Get the logits for the last token.
            logits = indices_pred[:, -1, :]  # (B, vocab_size)

            # Temperature scaling.
            logits = logits / temperature

            # Top-p (nucleus) sampling
            if top_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token that exceeds top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))

            probs = F.softmax(logits, dim=-1)
            # Sample the next token.
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            token_outputs.append(next_token)

            # Append the sampled token to the decoder input.
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break

        # Concatenate predictions along the time dimension.
        token_outputs = torch.cat(token_outputs, dim=1)  # (B, seq_len)
        return token_outputs

# Main model
class Echolancer(nn.Module):
    """ Echolancer:
        Transformer-TTS based model: NAR Encoder, AR Decoder"""

    def __init__(self, vocab_size, encoder_hidden, encoder_head, encoder_layer, 
                 decoder_hidden, decoder_layer, decoder_head, encoder_dropout=0.1, 
                 decoder_dropout=0.1, mel_channels=80, emotion_channels=256, 
                 speaker_channels=0, multi_speaker=False, n_speaker=0,
                 alibi_alpha=1.0, use_alibi=False, activation='relu',
                 vq_token_mode=False, vq_vocab_size=1024, 
                 encoder_kv_heads=None, decoder_kv_heads=None,
                 encoder_start_i=0, decoder_start_i=0,
                 emotion_input_size=768, emotion_hidden_sizes=[512, 384], emotion_dropout=0.1):
        super(Echolancer, self).__init__()
        self.emotion_channels = emotion_channels
        self.speaker_channels = speaker_channels
        self.use_alibi = use_alibi
        self.alibi_alpha = alibi_alpha
        self.vq_token_mode = vq_token_mode  # New parameter for zero-shot mode
        self.vq_vocab_size = vq_vocab_size  # Vocabulary size for VQ tokens

        self.encoder = TextEncoder(
            vocab_size,
            encoder_hidden,
            encoder_head,
            encoder_layer,
            4,  # forward_expansion
            encoder_dropout,
            emotion_channels=self.emotion_channels,
            speaker_channels=self.speaker_channels,
            alibi_alpha=alibi_alpha,
            use_alibi=use_alibi,
            activation=activation,
            num_kv_heads=encoder_kv_heads,
            start_i=encoder_start_i
        )
        
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
            
        # Remove pre_encoder as it's not part of the model itself
        self.pre_encoder = None

        self.decoder = SpectrogramDecoderAR(
            encoder_hidden,
            mel_channels,
            decoder_hidden,
            decoder_layer,
            decoder_head,
            decoder_dropout,
            speaker_channels=self.speaker_channels,
            dec_type="transformer",
            alibi_alpha=alibi_alpha,
            use_alibi=use_alibi,
            activation=activation,
            num_kv_heads=decoder_kv_heads,
            start_i=decoder_start_i
        )

        self.last_logprobs = None
        self.speaker_emb = None
        if multi_speaker and n_speaker > 0 and not vq_token_mode:
            # Only use speaker embedding in traditional mode
            self.speaker_emb = nn.Embedding(n_speaker, self.speaker_channels)
        
        # Add VQ token embedding and encoder for zero-shot mode
        if vq_token_mode:
            # Embedding layer for VQ tokens
            self.vq_token_emb = nn.Embedding(vq_vocab_size, speaker_channels)
            # Small transformer encoder to process VQ tokens into speaker embeddings
            self.vq_encoder = TransformerEncoder(
                speaker_channels,  # d_model
                2,  # num_heads (smaller than main encoder)
                2,  # num_layers (smaller than main encoder)
                speaker_channels * 2,  # d_ff
                encoder_dropout,
                alibi_alpha,
                use_alibi,
                activation
            )
            # Add a pooling layer to convert sequence to single embedding
            self.vq_pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, speakers, texts, src_lens, mels, mel_lens, em_hidden=None):
        """
        Forward pass of Echolancer, training.
        
        Args:
            speakers: Speaker IDs (B,) - used in traditional mode
            texts: Text token IDs (B, T_text)
            src_lens: Text sequence lengths (B,)
            mels: Mel token IDs (B, T_mel) - used for teacher forcing during training
                  In zero-shot mode, these are VQ tokens for speaker encoding
            mel_lens: Mel sequence lengths (B,)
                  In zero-shot mode, these are lengths for VQ tokens
            em_hidden: Emotion embeddings (B, emotion_dim)
            
        Returns:
            Tuple of outputs for loss computation
        """
        # Handle speaker embedding based on mode
        if self.vq_token_mode:
            # Zero-shot mode: use VQ tokens (mels) to generate speaker embedding
            # In zero-shot mode, mels contains VQ tokens that we process directly
            spk_emb = self.vq_tokens_to_speaker_emb(mels, mel_lens)
        elif self.speaker_emb is not None:
            # Traditional mode: use speaker ID to lookup embedding
            spk_emb = self.speaker_emb(speakers).unsqueeze(1)  # (B, 1, speaker_channels)
        else:
            # No speaker conditioning
            spk_emb = None
            
        encoded_emotion = self.emotion_encoder(em_hidden) if self.emotion_channels > 0 and em_hidden is not None else None

        text_mask = sequence_mask(texts.size(1), src_lens)
        mel_mask = sequence_mask(mels.size(1), mel_lens)

        encoded_text = self.encoder(texts, text_mask, encoded_emotion, spk_emb)
        
        # For discrete token prediction, we predict the next token given the current sequence
        logits, attn_logprob, x_mask_in, _ = self.decoder(mels, mel_mask, encoded_text, text_mask, spk_emb=spk_emb)
        self.last_logprobs = attn_logprob

        return (
            text_mask,  # Text attention mask
            mel_mask,  # Mel attention mask
            attn_logprob,  # Attention log probabilities
            x_mask_in,  # Input mask for decoder
            logits,  # Predicted logits for next tokens
            mels,  # Ground truth tokens (shifted for loss)
        )

    def infer(self, speakers, texts, src_lens, em_hidden=None, max_length=1000, gate_threshold=0.5, vq_tokens=None, vq_token_lens=None, temperature=0.8, top_p=1.0):
        """
        Autoregressive inference for the Echolancer model.
        
        Args:
            speakers: Speaker IDs (B,) - used in traditional mode
            texts: Text token IDs (B, T_text)
            src_lens: Text sequence lengths (B,)
            em_hidden: Emotion embeddings (B, emotion_dim)
            max_length: Maximum length of generated mel token sequence
            gate_threshold: Threshold for gate prediction to stop generation
            vq_tokens: VQ tokens (B, T_vq) - used in zero-shot mode
            vq_token_lens: VQ token sequence lengths (B,) - used in zero-shot mode
            temperature: Temperature for sampling (higher = more random)
            top_p: Top-p (nucleus) sampling threshold (1.0 = no top-p sampling)
            
        Returns:
            Tuple containing:
                - token_outputs: Generated discrete mel token IDs (B, T_gen)
                - gate_outputs: Gate predictions (B, T_gen, 1)
        """
        device = texts.device

        # Handle speaker embedding based on mode
        if self.vq_token_mode and vq_tokens is not None:
            # Zero-shot mode: use VQ tokens to generate speaker embedding
            spk_emb = self.vq_tokens_to_speaker_emb(vq_tokens, vq_token_lens)
        elif self.speaker_emb is not None:
            # Traditional mode: use speaker ID to lookup embedding
            spk_emb = self.speaker_emb(speakers).unsqueeze(1)  # (B, 1, speaker_channels)
        else:
            # No speaker conditioning
            spk_emb = None

        # Process emotion encoding if applicable
        if self.emotion_channels > 0 and em_hidden is not None:
            encoded_emotion = self.emotion_encoder(em_hidden)
        else:
            encoded_emotion = None

        text_mask = sequence_mask(texts.size(1), src_lens)

        # Encode the text using the text encoder.
        encoded_text = self.encoder(texts, text_mask, encoded_emotion, spk_emb)

        # Now hand off the encoded text to the decoder's inference method.
        token_outputs = self.decoder.infer(
            encoded_text, text_mask, 
            max_length=max_length,
            spk_emb=spk_emb,
            temperature=temperature,
            top_p=top_p
        )

        return token_outputs
    
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
        else:
            vq_mask = None
            sa_mask = None
            
        # Encode with transformer
        encoded_vq = self.vq_encoder(vq_emb, sa_mask)  # (B, T_vq, speaker_channels)
        
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