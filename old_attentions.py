from math import sqrt
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
from rotary_embedding_torch import RotaryEmbedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
import torchbnn
from torchbnn import BayesConv1d
from .attblocks import *
from .subatts import *

try:
    from .block.rwkv7_time_mix import RWKV7TimeMix
    from .block.rwkv7_block_config_map import RWKV7BlockConfigMap
except ImportError:
    print("RWKV not available")

flex_attention_available = False

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_available = True
except ImportError:
    print("FlexAttention not available.")

if flex_attention_available:
    flex_attention = torch.compile(flex_attention, dynamic=True)
    create_block_mask = torch.compile(create_block_mask, dynamic=True)


flash_attention_available = False

def is_rocm():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(0).lower()
    return "amd" in device_name or "gfx" in device_name  # ROCm GPUs often have 'gfx' codes

if is_rocm():
    os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
    print("ROCm detected. FlashAttention will use Triton backend.")

try:
    from flash_attn import flash_attn_varlen_func
    flash_attention_available = True
except ImportError:
    print("Flash Attention not available.")

if not flash_attention_available and not flex_attention_available:
    print("Warning: No efficient attention variants found. You should probably get one if you're operating at any serious scale.")



class PartialConv1d(torch.nn.Conv1d):
    """
    Zero padding creates a unique identifier for where the edge of the data is, such that the model can almost always identify
    exactly where it is relative to either edge given a sufficient receptive field. Partial padding goes to some lengths to remove
    this affect.
    """

    def __init__(self, *args, **kwargs):
        super(PartialConv1d, self).__init__(*args, **kwargs)
        weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.register_buffer("weight_maskUpdater", weight_maskUpdater, persistent=False)
        slide_winsize = torch.tensor(self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2])
        self.register_buffer("slide_winsize", slide_winsize, persistent=False)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            self.register_buffer('bias_view', bias_view, persistent=False)
        # caching part
        self.last_size = (-1, -1, -1)

        update_mask = torch.ones(1, 1, 1)
        self.register_buffer('update_mask', update_mask, persistent=False)
        mask_ratio = torch.ones(1, 1, 1)
        self.register_buffer('mask_ratio', mask_ratio, persistent=False)
        self.partial: bool = True

    def calculate_mask(self, input: torch.Tensor, mask_in: Optional[torch.Tensor]):
        with torch.no_grad():
            if mask_in is None:
                mask = torch.ones(1, 1, input.shape[2], dtype=input.dtype, device=input.device)
            else:
                mask = mask_in
            update_mask = F.conv1d(
                mask,
                self.weight_maskUpdater,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            # for mixed precision training, change 1e-8 to 1e-6
            mask_ratio = self.slide_winsize / (update_mask + 1e-6)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio.to(update_mask), update_mask)
            return torch.mul(input, mask), mask_ratio, update_mask

    def forward_aux(self, input: torch.Tensor, mask_ratio: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 3

        raw_out = self._conv_forward(input, self.weight, self.bias)

        if self.bias is not None:
            output = torch.mul(raw_out - self.bias_view, mask_ratio) + self.bias_view
            output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        return output

    @torch.jit.ignore
    def forward_with_cache(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_cache = not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export())
        cache_hit = use_cache and mask_in is None and self.last_size == input.shape
        if cache_hit:
            mask_ratio = self.mask_ratio
            update_mask = self.update_mask
        else:
            input, mask_ratio, update_mask = self.calculate_mask(input, mask_in)
            if use_cache:
                # if a mask is input, or tensor shape changed, update mask ratio
                self.last_size = tuple(input.shape)
                self.update_mask = update_mask
                self.mask_ratio = mask_ratio
        return self.forward_aux(input, mask_ratio, update_mask)

    def forward_no_cache(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.partial:
            input, mask_ratio, update_mask = self.calculate_mask(input, mask_in)
            return self.forward_aux(input, mask_ratio, update_mask)
        else:
            if mask_in is not None:
                input = torch.mul(input, mask_in)
            return self._conv_forward(input, self.weight, self.bias)

    def forward(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.partial:
            return self.forward_with_cache(input, mask_in)
        else:
            if mask_in is not None:
                input = torch.mul(input, mask_in).to(input.device)
            return self._conv_forward(input, self.weight, self.bias)


def reduce_mask(mask):
    """
    Reduce an attention mask to a normal one
    :param mask: Attention mask shape (batch, 1, seq_length, seq_length)

    :return: Reduced mask size (batch, 1, seq_length)
    """
    reduced_mask = mask[:, 0, :, 0].unsqueeze(1)
    return reduced_mask


class SwiGLUConvFFN(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            kernel_size: int = 3,
            drop: float = 0.0,
            bias: bool = True,
            causal: bool = False,
            act="swiglu",
            conv_att=False,
    ):
        """
        Initializes the SwiGLU feed-forward network with Conv1D layers.

        Parameters:
            in_features (int): Input dimension of the FFN.
            hidden_features (int, optional): Inner dimension of the FFN. Defaults to in_features.
            out_features (int, optional): Output dimension of the FFN. Defaults to in_features.
            kernel_size (int, optional): Kernel size for convolution layers. Defaults to 3. Can also pass a 2-elem list
            drop (float, optional): Dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to use bias in convolution layers. Defaults to True.
            causal (bool, optional): Whether to use causal padding. Defaults to False.
            act: What activation to use. Options are "swiglu", "relu2", "relu", and "aptx"
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        valid_acts = ["swiglu", "relu2", "aptx", "relu", "dprelu", "aptxs1", "relugtz", "relugt"]

        if act not in valid_acts:
            raise ValueError(f"Unknown activation {act}. Valid activations are {valid_acts}")

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size, 1]

        self.kernel_size = kernel_size
        self.causal = causal
        self.drop = nn.Dropout(drop)
        self.act = act

        # wall of if statements
        # I swear im not yanderedev
        if act == "swiglu":
            self.act_fn = self._swiglu
        elif act == "relu2":
            self.act_fn = self._relu2
        elif act == "aptx" or act == "aptxs1":
            self.aptx = APTx(trainable=True) if act == "aptx" else APTxS1(trainable=True)
            self.act_fn = self._aptx
        elif act == "relu":
            self.act_fn = self._relu
        elif act == "dprelu":
            self.dprelu = DPReLU()
            self.act_fn = self._dprelu
        elif act == "relugt":
            self.dprelu = ReLUGT()
            self.act_fn = self._dprelu
        elif act == "relugtz":
            self.dprelu = ReLUGT()
            self.act_fn = self._relugtz

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding


        self.has_gating = act in ["swiglu", "relugtz"]

        # In a test I found out that nn.Linear is 19% faster than Conv1D with kernel size 1
        # (probably because it calls some special CUDA kernel)
        # so this is worth the extra complexity
        if kernel_size[0] == 1 and kernel_size[1] == 1:
            expand = 2 * hidden_features if self.has_gating else hidden_features

            self.lin1 = nn.Linear(in_features, expand, bias=bias)
            self.lin2 = nn.Linear(hidden_features, out_features, bias=bias)
            self.forward = self.forward_dense
            conv_att = False
        else:
            self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size[0], bias=bias)

            # Efficient gating: Instead of doubling the expansion like in normal FFNs which can be expensive
            # for convs, we keep a separate gate_proj which is dense.
            if self.has_gating:
                self.gate_proj = nn.Linear(in_features, hidden_features, bias=bias)


            self.conv2 = nn.Conv1d(hidden_features, out_features, kernel_size[1], bias=bias)


        self.lwa = MaskedCBAM1d(hidden_features) if conv_att else None

    def _swiglu(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = F.silu(x1) * x2
        return x

    def _relugtz(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = self.dprelu(x1) * x2
        return x

    def _relu2(self, x):
        return F.relu(x) ** 2

    def _aptx(self, x):
        return self.aptx(x)

    def _relu(self, x):
        return F.relu(x)

    def _dprelu(self, x):
        return self.dprelu(x)

    def _causal_padding(self, x: torch.Tensor, kernel_size) -> torch.Tensor:
        """
        Applies causal padding to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Padded tensor.
        """
        if kernel_size == 1:
            return x
        pad_left = kernel_size - 1
        pad_right = 0
        return F.pad(x, (pad_left, pad_right))

    def _same_padding(self, x: torch.Tensor, kernel_size) -> torch.Tensor:
        """
        Applies same padding to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).

        Returns:
            torch.Tensor: Padded tensor.
        """
        if kernel_size == 1:
            return x
        pad_left = (kernel_size - 1) // 2
        pad_right = kernel_size // 2
        return F.pad(x, (pad_left, pad_right))

    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[Tensor, Tensor]:
        """
        Applies a mask to the input tensor.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, seq_length).
            mask (torch.Tensor): Mask tensor of shape (batch_size, 1, 1, seq_length).

        Returns:
            torch.Tensor: Masked input tensor of shape (batch_size, channels, seq_length).
            torch.Tensor: Mask.
        """
        batch_size, channels, seq_length = x.shape

        if mask.shape == (batch_size, 1, seq_length):
            x = x.masked_fill(mask, 0)
            return x, mask

        if mask is not None:
            assert mask.shape == (batch_size, 1, 1, seq_length), f"Mask shape mismatch: {mask.shape}"
            mask = mask.squeeze(1)  # Reduce to (batch_size, 1, seq_length)
            x = x * mask

            mask = mask.bool()
            mask = ~mask
        return x, mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the SwiGLU Conv1D feed-forward network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, in_features).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, seq_length, seq_length), where True is include and False exclude.
            OR  (batch_size, 1, seq_length) where True is exclude and False include

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, out_features).
        """
        # Transpose for Conv1D (batch_size, channels, seq_length)
        x = x.transpose(1, 2)

        # Apply mask before the first convolution
        x, c_mask = self.apply_mask(x, mask)

        x12 = self.conv1(self.padding(x, self.kernel_size[0])) # (B, C, L)

        if self.lwa is not None:
            x12 = self.lwa(x12, c_mask)

        if self.has_gating: # x = (B, C, L) => (B, L, C) => (B, C, L)
            gate = self.gate_proj(x.transpose(1,2)).transpose(1,2) 
            # Our gating functions take in a single tensor then chunk them channel wise
            x12 = torch.cat([x12, gate], dim=1) # (


        hidden = self.act_fn(x12)

        hidden = self.drop(hidden)

        # Apply mask before the second convolution
        hidden, _ = self.apply_mask(hidden, mask)

        out = self.conv2(self.padding(hidden, self.kernel_size[1]))

        # Transpose back to (batch_size, seq_length, out_features)
        return out.transpose(1, 2)

    def forward_dense(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the SwiGLU linear feed-forward network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, in_features).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, 1, seq_length, seq_length), where True is include and False exclude.
            OR  (batch_size, 1, seq_length) where True is exclude and False include

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, out_features).
        """
        # Transpose for masking
        x = x.transpose(1, 2)

        x, _ = self.apply_mask(x, mask)

        # Transpose back for linear
        x = x.transpose(1, 2)

        x12 = self.lin1(x)

        hidden = self.act_fn(x12.transpose(1, 2)).transpose(1, 2)
        hidden = self.drop(hidden)

        hidden = hidden.transpose(1, 2)

        hidden, _ = self.apply_mask(hidden, mask)

        hidden = hidden.transpose(1, 2)

        out = self.lin2(hidden)
        return out


class DynamicALiBi(nn.Module):
    def __init__(self, num_heads, min_val=0.8, max_val=1.2):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.heads = num_heads

        # Initialize with values near 1.0, ensuring a stable start
        self.alibi_betas = nn.Parameter(torch.ones(num_heads) * 1.0)

        # Register a hook to enforce constraints after every optimization step
        self.alibi_betas.register_hook(
            lambda grad: grad * ((self.alibi_betas >= self.min_val) & (self.alibi_betas <= self.max_val)).float())

    def forward(self):
        # Ensure alibi_betas stays within the valid range
        with torch.no_grad():
            self.alibi_betas.clamp_(self.min_val, self.max_val)
        return self.alibi_betas.view(1, self.heads, 1, 1)


class MultiHeadAttention(nn.Module):
    """
    Modern Multi Head Attention. Contains:

    num_persistent: "Augmenting Self-attention with Persistent Memory" (https://arxiv.org/abs/1907.01470)
    use_talking_heads: "Talking-Heads Attention" (https://arxiv.org/abs/2003.02436)
    use_alibi: "Attention with Linear Biases" (https://ofir.io/train_short_test_long.pdf)
    rma_inp_dim: Recurrent Memory Attention (my invention). Per-head dim for projection, if necessary.
    weighted_heads: Weighted Heads Attention. Keep trainable scalar weights for each head, which are used to multiply just
    before the final projection, in order to allow the model to dynamically prioritize heads (Decreases performance, don't use)
    dynamic_alibi: Dynamic ALiBi. Keep per-head trainable multipliers to dynamically adjust the slopes as it trains.

    Supports three backends:
    manual: Naive PyTorch impl; flex: FlexAttention (broken); flash: FlashAttention. Default is auto, will pick best.

    If num_persistent > 0, we call this an AllAttention layer.

    """

    def __init__(self, embed_size, heads, alibi_alpha=1.0, start_i_increment=0, use_alibi=True, use_talking_heads=True,
                 num_persistent=0, rma_inp_dim=None, weighted_heads=False, dynamic_alibi=False, qk_rmsnorm=False, backend="auto",
                 gqa_groups="auto", causal=False):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.use_alibi = use_alibi
        self.dynamic_alibi = dynamic_alibi
        self.qk_rmsnorm = qk_rmsnorm
        backend = "manual"
        self.backend = backend
        self.causal = causal

        if backend == "auto":
            
            if flash_attention_available:
                self.backend = "flash"
                
            if flex_attention_available:
                self.backend = "flex"

        if self.backend == "flash":
            self.dynamic_alibi = False

        if gqa_groups == "auto":
            if 8 > heads:
                gqa_groups = 1
            else:
                gqa_groups = 2 
                
        self.gqa_groups = gqa_groups
        # --- GQA setup ---
        if gqa_groups > 1:
            if heads % gqa_groups != 0:
                 raise ValueError(f"Number of heads ({heads}) must be divisible by gqa_groups ({gqa_groups})")
            self.num_kv_heads = heads // gqa_groups
            self.num_queries_per_kv = gqa_groups
            self.backend = "manual"
        else:
            self.num_kv_heads = heads
            self.num_queries_per_kv = 1
            self.gqa_groups = 0

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.queries = nn.Linear(embed_size, heads * self.head_dim, bias=False)
        self.keys = nn.Linear(embed_size, self.num_kv_heads * self.head_dim, bias=False)
        self.values = nn.Linear(embed_size, self.num_kv_heads * self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.alibi_alpha = alibi_alpha
        self.use_talking_heads = False
        self.start_i_increment = start_i_increment
        self.num_persistent = num_persistent
        self.weighted_heads = weighted_heads


        if self.backend != "manual":
            self.use_talking_heads = False
            self.num_persistent = 0
            self.rma_inp_dim = None
            self.weighted_heads = False

        if self.qk_rmsnorm:
            self.q_norm = RMSNorm(self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, bias=False)

        if self.use_alibi:
            # Precompute ALiBi slopes
            self.slopes = torch.tensor(
                [2 ** (-self.alibi_alpha * (i + self.start_i_increment)) for i in range(1, self.heads + 1)],
                dtype=torch.float32).view(1, self.heads, 1, 1)

            if self.dynamic_alibi:
                # if we just make it a naive parameter it can learn extreme/nonsense values
                # and break the attention (i learned the hard way)
                self.alibi_adapter = DynamicALiBi(self.heads, 0.8, 1.25)

        if self.use_talking_heads:  # Talking heads: x-transformers version (using Conv2d instead of Linear)
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias=False)

        if self.num_persistent > 0:
            # persistent vectors:
            # (num_persistent, 1, head_dim)
            self.persistent_keys = nn.Parameter(torch.randn(self.num_persistent, 1, self.head_dim))
            self.persistent_values = nn.Parameter(torch.randn(self.num_persistent, 1, self.head_dim))

            # Initialize persistent vectors
            nn.init.kaiming_uniform_(self.persistent_keys, a=sqrt(self.num_persistent))
            nn.init.kaiming_uniform_(self.persistent_values, a=sqrt(self.num_persistent))

            if rma_inp_dim is not None:
                self.rma_k_proj = GatedRetention(rma_inp_dim, self.head_dim)
                self.rma_v_proj = GatedRetention(rma_inp_dim, self.head_dim)

        if self.weighted_heads:
            self.head_weights = nn.Parameter(torch.ones(self.heads))

    def calculate_alibi_bias(self, device, query_len, key_len, current_persistent):
        if not self.use_alibi:
            return torch.zeros(1, device=device)

        current_slopes = self.slopes.to(device)

        if self.dynamic_alibi:
            alibi_betas = self.alibi_adapter()
            current_slopes *= alibi_betas

        if self.backend == "flash":
            return current_slopes.view(self.heads) # FA only takes slopes

        t_q = torch.arange(query_len, device=device)
        t_k = torch.arange(key_len, device=device)
        alibi_bias = (t_q.view(1, 1, -1, 1) - t_k.view(1, 1, 1, -1)).abs()

        alibi_bias = -alibi_bias * current_slopes
        if current_persistent > 0:
            alibi_bias = F.pad(alibi_bias, (0, current_persistent), "constant", 0).to(device)
        return alibi_bias

    # --- GQA Helper (remains the same) ---
    def _repeat_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (N, S, H_kv, Dk)
        # Output: (N, S, H_q, Dk)
        if self.num_queries_per_kv == 1:
            return x
        # Directly repeat each key/value head self.num_queries_per_kv times along the head dimension.
        x = x.repeat_interleave(self.num_queries_per_kv, dim=2)
        return x

    def attention_forward(self, queries, keys, values, mask, alibi_bias, current_persistent, N, query_len, key_len):
        keys, values = self._repeat_kv_heads(keys), self._repeat_kv_heads(values)

        # Compute energy using einsum, simplifying matrix multiplication across batches and heads
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if self.use_talking_heads:
            energy = self.pre_softmax_talking_heads(energy)

        if self.use_alibi:
            energy += alibi_bias

        if mask is not None:
            if current_persistent > 0:
                # Extend mask to include persistent vectors (always unmasked)
                extended_mask = F.pad(mask, (0, current_persistent), value=1)
                extended_mask = extended_mask.expand(N, self.heads, query_len, key_len + current_persistent)
                mask = extended_mask
            energy = energy.masked_fill(mask == 0, float("-1e6"))

        attention_weights = F.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        if self.use_talking_heads:
            attention = self.post_softmax_talking_heads(attention_weights)
        else:
            attention = attention_weights

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        if self.weighted_heads:  # (batch, len, n_heads, head_dim)
            out = out * self.head_weights.view(1, 1, -1, 1)

        return out, attention_weights

    # --- inside MultiHeadAttention ----------------------------------------------
    def flex_attention_forward(
            self, queries, keys, values,
            mask, alibi_bias,
            N, query_len, key_len  # key_len == original sequence length
    ):
        """
        Flex‑Attention backend, **no persistent memory**.
        Returns: out (B, Q, H, D), attention_weights=None
        """

        # ------------------------------------------------------------------ #
        # 1)  Layout:  FlexAttention wants (B, H, S, D)
        # ------------------------------------------------------------------ #
        q = queries.permute(0, 2, 1, 3).contiguous()  # (B, H, Q, D)
        k = keys.permute(0, 2, 1, 3).contiguous()  # (B, Hkv, K, D)
        v = values.permute(0, 2, 1, 3).contiguous()  # (B, Hkv, K, D)
        # (Hkv == self.num_kv_heads; FlexAttention handles GQA natively)

        kv_len_total = k.shape[2]  # == key_len (no memory)

        # ------------------------------------------------------------------ #
        # 2)  ALiBi bias – inject per‑element via score_mod
        # ------------------------------------------------------------------ #
        if self.use_alibi:
            alibi_ = alibi_bias.squeeze(0)  # (H, Q, K)

            def score_mod(score, b, h, q_idx, kv_idx):
                # `score` is a scalar; add scalar bias.
                return score + alibi_[h, q_idx, kv_idx]
        else:
            score_mod = None  # FlexAttention treats None as identity

        # ------------------------------------------------------------------ #
        # 3)  Boolean mask ➜ BlockMask    (shape B×1×Q×K, True = keep)
        # ------------------------------------------------------------------ #
        block_mask = None
        if mask is not None:
            # mask: (B,1,Q,K) with True=valid
            # 1) Precompute per‑batch sequence lengths:
            q_lens = mask.any(dim=-1).squeeze(1).sum(dim=-1)  # (B,)
            k_lens = mask.any(dim=-2).squeeze(1).sum(dim=-1)  # (B,)
            
            causal = self.causal  # capture the flag


            # 2) mask_mod purely by tensor comparisons:
            def mask_mod(b, h, q_idx, kv_idx):
                # ensure integer indices
               # b      = b.to(torch.long)
                #q_idx  = q_idx.to(torch.long)
                #kv_idx = kv_idx.to(torch.long)
            
                # check within real lengths
                valid_q = q_idx  < q_lens[b]   # (…,) bool tensor
                valid_k = kv_idx < k_lens[b]   # (…,) bool tensor
            
                base_mask = valid_q & valid_k
            
                # apply causal: only allow kv positions ≤ q position
                if causal:
                    return base_mask & (kv_idx <= q_idx)
                return base_mask

            block_mask = create_block_mask(
                mask_mod,
                B=N, H=self.heads,
                Q_LEN=query_len, KV_LEN=kv_len_total,
                device=q.device
            )

        # ------------------------------------------------------------------ #
        # 4)  Call FlexAttention
        # ------------------------------------------------------------------ #
        out = flex_attention(q, k, v,
                             score_mod=score_mod,
                             block_mask=block_mask)  # (B, H, Q, D)

        # ------------------------------------------------------------------ #
        # 5)  Back to (B, Q, H, D) + optional weighted heads
        # ------------------------------------------------------------------ #
        out = out.permute(0, 2, 1, 3)  # (B, Q, H, D)

        if self.weighted_heads:
            out = out * self.head_weights.view(1, 1, -1, 1)

        return out, None

    def flash_attention_forward(self, queries, keys, values, mask, N, query_len, key_len):
            """
            Computes attention using flash_attn_varlen_func.
    
            Args:
                queries: (N, query_len, self.heads, self.head_dim)
                keys: (N, key_len_total, self.num_kv_heads, self.head_dim)
                values: (N, key_len_total, self.num_kv_heads, self.head_dim)
                mask: Optional (N, 1, query_len, key_len_total), boolean mask. True is valid.
                      Used to derive sequence lengths. Assumed extended for persistent mem if used.
                N: Batch size.
                query_len: Padded query sequence length.
                key_len: Padded key sequence length (including persistent memory if applicable).
    
            Returns:
                out: (N, query_len, self.heads, self.head_dim) - Attention output.
                attention_weights: None (Flash Attention does not return weights by default).
            """
            if not flash_attention_available:
                raise RuntimeError("Flash Attention backend selected but flash_attn is not available.")

            queries = queries.to(values.dtype)
            keys = keys.to(values.dtype)
    
            # --- Prepare inputs for flash_attn_varlen_func ---
    
            # 1. Derive sequence lengths and cumulative sequence lengths (cu_seqlens) from the mask
            if mask is not None:
                # Ensure mask is boolean
                mask = mask.bool()
                # Derive query lengths: A query position is valid if it's part of the original sequence (before padding).
                # We assume the mask correctly identifies valid Q positions along the Q dimension.
                # Check if any key is valid for a given query position.
                # Mask shape: (N, 1, Q, K_total)
                q_padding_mask = mask.any(dim=-1).squeeze(1) # Shape: (N, Q) True for valid tokens
                q_seqlens = q_padding_mask.sum(dim=-1, dtype=torch.int32) # Shape: (N,) Actual lengths
    
                # Derive key lengths: A key position is valid if it's part of the original sequence.
                # Check if the key position is valid along the K dimension.
                # Since the mask is (N, 1, Q, K), we need to know valid K positions independent of Q.
                # A simple way is to assume a K position is valid if *any* Q attends to it.
                # However, it's more robust to get valid K positions directly if possible.
                # If the original unpadded key length is known, use that. Here, we infer from the mask.
                # We check if a key position is attended by *any* valid query position.
                k_padding_mask = mask.any(dim=-2).squeeze(1) # Shape: (N, K_total) True for valid tokens
                k_seqlens = k_padding_mask.sum(dim=-1, dtype=torch.int32) # Shape: (N,) Actual lengths
    
                # Handle cases where a sequence might be fully masked (length 0)
                q_seqlens = torch.clamp(q_seqlens, min=0)
                k_seqlens = torch.clamp(k_seqlens, min=0)
    
                max_seqlen_q = query_len # Use padded length as max_seqlen for flash_attn
                max_seqlen_k = key_len   # Use padded length as max_seqlen for flash_attn
                # If q_seqlens.max() > 0 else query_len # Get max actual length if needed, but flash needs padded max
                # k_seqlens.max().item() if k_seqlens.max() > 0 else key_len
    
            else:
                # No mask provided: Assume all sequences in the batch have the full padded length.
                q_seqlens = torch.full((N,), query_len, dtype=torch.int32, device=queries.device)
                k_seqlens = torch.full((N,), key_len, dtype=torch.int32, device=queries.device)
                
                max_seqlen_q = query_len
                max_seqlen_k = key_len
                # Create boolean masks indicating all tokens are valid
                q_padding_mask = torch.ones(N, query_len, dtype=torch.bool, device=queries.device)
                k_padding_mask = torch.ones(N, key_len, dtype=torch.bool, device=queries.device)
                
    
            # Calculate cumulative sequence lengths (required by flash_attn_varlen_func)
            # Shape: (batch_size + 1,)
            cu_seqlens_q = F.pad(torch.cumsum(q_seqlens, dim=0, dtype=torch.int32), (1, 0))
            cu_seqlens_k = F.pad(torch.cumsum(k_seqlens, dim=0, dtype=torch.int32), (1, 0))
         #   print(q_seqlens, k_seqlens, max_seqlen_q, max_seqlen_k)
    
            # 2. Reshape Q, K, V from (N, SeqLen, NumHeads, HeadDim) to (TotalTokens, NumHeads, HeadDim)
            # Need to handle padding: select only the non-padded tokens.
    
            # Reshape N, S, H, D -> (N*S), H, D first
            q_reshaped = queries.reshape(-1, self.heads, self.head_dim)
            k_reshaped = keys.reshape(-1, self.num_kv_heads, self.head_dim)
            v_reshaped = values.reshape(-1, self.num_kv_heads, self.head_dim)
    
            # Create flat boolean masks for selecting valid tokens from reshaped tensors
            # These masks have shape (N * query_len,) and (N * key_len,) respectively
            indices_q = q_padding_mask.flatten() # True where token is valid
            indices_k = k_padding_mask.flatten() # True where token is valid
    
            # Select non-padded tokens using the boolean masks
            # q_unpadded shape: (total_actual_q_tokens, H, D)
            q_unpadded = q_reshaped[indices_q]
            # k_unpadded shape: (total_actual_k_tokens, Hkv, D)
            k_unpadded = k_reshaped[indices_k]
            # v_unpadded shape: (total_actual_k_tokens, Hkv, D)
            v_unpadded = v_reshaped[indices_k] # Use same indices for K and V
    
    
            # 3. Prepare ALiBi slopes
            alibi_slopes_param = None
            if self.use_alibi:
                alibi_slopes_param = self.calculate_alibi_bias(q_reshaped.device, query_len, key_len, 0).unsqueeze(0).repeat(N, 1)
    
            # --- Call flash_attn_varlen_func ---
            # dropout_p=0.0 as dropout is usually applied elsewhere / inference
            # softmax_scale=None lets flash attention use the default 1/sqrt(head_dim)
            # Note: flash_attn expects total number of tokens to match cu_seqlens_q/k[-1]
            if q_unpadded.shape[0] != cu_seqlens_q[-1].item():
                 raise ValueError(f"Mismatch in query token count: q_unpadded has {q_unpadded.shape[0]}, expected {cu_seqlens_q[-1]}")
            if k_unpadded.shape[0] != cu_seqlens_k[-1].item():
                 raise ValueError(f"Mismatch in key token count: k_unpadded has {k_unpadded.shape[0]}, expected {cu_seqlens_k[-1]}")
    
            # Handle case with zero-length sequences if any seqlen is 0
            if q_seqlens.min() == 0 or k_seqlens.min() == 0:
                # Flash attention might error with zero lengths. Need to handle manually.
                # If all sequences have length 0, return zeros.
                # If only some have length 0, need careful handling.
                # For simplicity, if any sequence has 0 length, we might need to skip flash attn or handle carefully.
                # A practical approach: if total tokens is 0, return zeros. Otherwise, proceed but be wary.
                if q_unpadded.shape[0] == 0:
                    # No query tokens, output should be zeros matching query shape
                    out = torch.zeros_like(queries) # Zeros with shape (N, query_len, H, D)
                    return out, None
    
                # If k/v tokens are zero, but q tokens exist, output should also be zero
                if k_unpadded.shape[0] == 0:
                     out = torch.zeros_like(queries)
                     return out, None
                # If some sequences have 0 length but not all, flash attention *should* handle this via cu_seqlens.
    
    
            flash_out, _, attn_weights = flash_attn_varlen_func(
                q=q_unpadded,
                k=k_unpadded,
                v=v_unpadded,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q, # Use padded length
                max_seqlen_k=max_seqlen_k, # Use padded length
                dropout_p=0.0, # Set actual dropout rate if needed during training
                softmax_scale=None,
                causal=self.causal,
                alibi_slopes=alibi_slopes_param,
                return_attn_probs=True # We don't need attention weights
            )
            # flash_out shape: (total_actual_q_tokens, self.heads, self.head_dim)
    
            # --- Reshape output back to (N, query_len, self.heads, self.head_dim) ---
            # Need to scatter the unpadded output back into a padded tensor using the boolean mask.
            # Create a zero tensor with the shape of the padded, reshaped queries before token selection
            out_padded = torch.zeros_like(q_reshaped) # Shape (N*query_len, H, D)
            # Use the boolean mask `indices_q` to place the results
            out_padded[indices_q] = flash_out
            # Reshape back to (N, query_len, H, D)
            out = out_padded.reshape(N, query_len, self.heads, self.head_dim)
    
            # Return output in the expected format and None for weights
            return out, attn_weights # (N, query_len, heads, head_dim), None

    def forward(self, values, keys, queries, mask=None, recurr_persistent=None, return_weights=False):
        """
        Do attention
        :param values: Values
        :param keys: Keys
        :param queries: Queries
        :param mask: Attention mask size (batch, 1, query_len, key_len), bool mask where True is valid
        :param recurr_persistent: Packed tuple (keys, values) of recurrent persistent memory
        :return: Attentioned tensor
        """
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # 1. Project queries, keys, values
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 2. Reshape Q, K, V
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_kv_heads, self.head_dim)
        values = values.reshape(N, value_len, self.num_kv_heads, self.head_dim)

        if self.qk_rmsnorm:
            keys = self.k_norm(keys)
            queries = self.q_norm(queries)

        current_persistent = self.num_persistent

        if current_persistent > 0:
            p_keys = self.persistent_keys
            p_values = self.persistent_values

            if recurr_persistent is not None:
                recurr_keys, recurr_values = recurr_persistent

                recurr_keys = self.rma_k_proj(recurr_keys)
                recurr_values = self.rma_v_proj(recurr_values)

                # Concat the recurrent ones before ours along the seq dim
                p_keys = torch.cat([recurr_keys, p_keys], dim=0)
                p_values = torch.cat([recurr_values, p_values], dim=0)

                current_persistent = p_keys.size(0)

            expanded_persistent_keys = p_keys.unsqueeze(0).expand(N, -1, self.heads, -1)
            expanded_persistent_values = p_values.unsqueeze(0).expand(N, -1, self.heads, -1)

            # Concatenate persistent vectors to keys and values
            keys = torch.cat([keys, expanded_persistent_keys], dim=1)
            values = torch.cat([values, expanded_persistent_values], dim=1)

        alibi_bias = self.calculate_alibi_bias(keys.device, query_len, key_len, current_persistent)
        backend = self.backend
        
        if not self.training:
            backend = "manual"

        if backend == "flex":
            out, attention_weights = self.flex_attention_forward(queries, keys, values, mask, alibi_bias, N, query_len,
                                                                 key_len)
        elif backend == "manual":
            out, attention_weights = self.attention_forward(queries, keys, values, mask, alibi_bias, current_persistent,
                                                            N,
                                                            query_len, key_len)
        elif backend == "flash":
            out, attention_weights = self.flash_attention_forward(queries, keys, values, mask, N, query_len, key_len)

        out = out.reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)

        if not return_weights:
            return out
        else:
            return out, attention_weights


def expand_masks(x_mask, y_mask):
    """
    Expand True=padded masks into an attention mask.
    Inputs can be different or the same
    :param x_mask: Mask of x size (batch, seq_len), where True is padded
    :param y_mask: Mask of y size (batch, seq_2_len), where True is padded
    :return: Attention mask for MultiHeadAttention
    """
    x_mask_expanded = x_mask.unsqueeze(1).unsqueeze(3)  # Shape: (batch_size, 1, mel_len, 1)
    y_mask_expanded = y_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, duration_len)
    # Combine masks using broadcasting
    attention_mask = x_mask_expanded & y_mask_expanded  # Shape: (batch_size, 1, mel_len, duration_len)
    attention_mask = ~attention_mask  # True=padded => True=valid
    return attention_mask


def expand_self_attention_mask(mask):
    """
    Turn a bool mask into a Self attention mask
    :param mask: Bool sequence mask, True=padding size (batch, max_length)
    :return: Self attention mask size (batch, 1, seq_len, seq_len), True=valid
    """
    valid = ~mask
    attn_mask = valid.unsqueeze(1) & valid.unsqueeze(2)
    return attn_mask.unsqueeze(1)




class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha=1.0, start_i_increment=0,
                 kernel_size=3, act="swiglu", rma_mem_dim=0, conv_att=False, talking_heads=True,
                 coarse_fine=False, dynamic_alibi=False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.use_rma = rma_mem_dim > 0
        self.coarse_fine = coarse_fine

        self.attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha,
                                            start_i_increment=start_i_increment, num_persistent=rma_mem_dim,
                                            rma_inp_dim=embed_size // heads if self.use_rma else 0,
                                            use_talking_heads=talking_heads,
                                            dynamic_alibi=dynamic_alibi, causal=False,
                                            gqa_groups=0, backend="auto")

        if self.coarse_fine:
            self.coarse_attention = MultiHeadAttention(embed_size, 1, alibi_alpha=alibi_alpha,
                                                       start_i_increment=start_i_increment, num_persistent=0,
                                                       rma_inp_dim=0,
                                                       use_talking_heads=False,  causal=False)
            self.norm3 = nn.LayerNorm(embed_size)

        self.feed_forward = SwiGLUConvFFN(
            in_features=embed_size,
            hidden_features=forward_expansion * embed_size,
            out_features=embed_size,
            kernel_size=kernel_size,
            drop=dropout,
            act=act,
            conv_att=conv_att,
            bias=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, conv_mask=None, mem_kv=None, coarse_features=None, coarse_mask=None):
        # Compute normalized x for coarse attention if needed, using the original x
        if self.coarse_fine:
            norm_x_coarse = self.norm3(x)

        # Primary attention
        norm_x_primary = self.norm1(x)
        attn_output = self.attention(norm_x_primary, norm_x_primary, norm_x_primary, mask, mem_kv)
        x = x + self.dropout(attn_output)

        # Coarse attention (if applicable)
        if self.coarse_fine:
            coarse_fine_attn_mask = expand_masks(conv_mask.squeeze(1), coarse_mask.squeeze(1))
            coarse_attn_output = self.coarse_attention(coarse_features, coarse_features, norm_x_coarse,
                                                       coarse_fine_attn_mask)
            x = x + self.dropout(coarse_attn_output)

        # Feed-forward
        norm_x_ff = self.norm2(x)
        ff_output = self.feed_forward(norm_x_ff, mask if conv_mask is None else conv_mask)
        x = x + self.dropout(ff_output)

        # Return persistent key-value pairs if using RMA
        kv_ret = (self.attention.persistent_keys, self.attention.persistent_values) if self.use_rma else None
        return x, kv_ret


class RWKVFormerLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha=1.0, start_i_increment=0,
                 kernel_size=1, act="swiglu", conv_att=False, talking_heads=True, dynamic_alibi=False, config_map=None):
        super(RWKVFormerLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.time_mix = RWKV7TimeMix(config_map)
        self.time_mix.reset_parameters()

        self.cross_attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha,
                                                  start_i_increment=start_i_increment, num_persistent=0,
                                                  rma_inp_dim=0, use_talking_heads=talking_heads,
                                                  dynamic_alibi=dynamic_alibi,
                                                  use_alibi=False, causal=False)

        self.feed_forward = SwiGLUConvFFN(
            in_features=embed_size,
            hidden_features=forward_expansion * embed_size,
            out_features=embed_size,
            kernel_size=kernel_size,
            drop=dropout,
            act=act,
            conv_att=conv_att,
            bias=True,
        )
        self.last_weights = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask, cross_attn_mask, conv_mask=None, v_first_val=None):
        # Primary attention
        norm_x = self.norm1(x)

        attn_output, _, _, v_first_val = self.time_mix(norm_x.bfloat16(), None, None, v_first_val)
        attn_output = attn_output.masked_fill(conv_mask.transpose(1,2), 0)
        x = x + self.dropout(attn_output)

        # Cross-attention
        norm_x = self.norm2(x)
        cross_attn_output, cross_attn_weights = self.cross_attention(y, y, norm_x, mask=cross_attn_mask, return_weights=True)
        x = x + self.dropout(cross_attn_output)

        # Feed-forward
        norm_x_ff = self.norm3(x)
        ff_output = self.feed_forward(norm_x_ff, mask if conv_mask is None else conv_mask)
        x = x + self.dropout(ff_output)

        # cache last attn weights
        self.last_weights = cross_attn_weights

        return x, v_first_val


class RWKVFormerDecoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0, start_i=0
                 , act="swiglu", talking_heads=True, dynamic_alibi=False, alibi_scaling_fac=None):
        super().__init__()
        self.use_conv_att = False
        if alibi_scaling_fac is None:
            alibi_scaling_fac = 1

        use_flexattn_on_ca = [True] * num_layers
        use_flexattn_on_ca[0] = False
        use_flexattn_on_ca[-1] = False
        self.decoder_layers = nn.ModuleList()

        rwkv_cfg = RWKV7BlockConfigMap(num_hidden_layers=num_layers, hidden_size=embed_size)
        rwkv_cfg.dtype = "bfloat16"
        rwkv_cfg.head_size = 32

        for i in range(num_layers):
            layer_cfg = rwkv_cfg
            layer_cfg.layer_id = i
            self.decoder_layers.append(
                RWKVFormerLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha=alibi_alpha,
                                start_i_increment=start_i + ((i * heads) // alibi_scaling_fac), kernel_size=1,
                                act=act, talking_heads=talking_heads,
                                dynamic_alibi=dynamic_alibi, config_map=layer_cfg)
            )


    def forward(self, x, y, mask, cross_attn_mask, conv_mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
            y: Input tensor of shape (batch_size, seq_2_length, embed_size)
            mask: Mask tensor of shape (batch_size, 1, seq_length, seq_length) or similar.
            cross_attn_mask: Mask tensor of shape (batch_size, 1, seq_length, seq_2_length)
            conv_mask: Convolutional mask size (batch, 1, seq_length) where True is padded and False is valid

            Note: mask does not need to be causal, this call automatically does that.
        Returns:
            The output of the last encoder layer.
        """
        mask = make_mask_causal(mask)
        v_first_val = None

        for i, layer in enumerate(self.decoder_layers):
            x, layer_v_first_val = layer(x, y, mask, cross_attn_mask, conv_mask, v_first_val)  # Here x serves as query, key, and value
            if i == 0:
                v_first_val = layer_v_first_val

        return x



class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha=1.0, start_i_increment=0,
                 kernel_size=1, act="swiglu", conv_att=False, talking_heads=True, dynamic_alibi=False, flex_attention=True, use_ca=True):
        super(TransformerDecoderLayer, self).__init__()
        self.use_ca = use_ca
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size) if self.use_ca else nn.Identity()
        self.norm3 = nn.LayerNorm(embed_size)
        print(start_i_increment)
        

        self.attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha,
                                            start_i_increment=start_i_increment, num_persistent=0,
                                            rma_inp_dim=0, use_talking_heads=talking_heads,
                                            dynamic_alibi=dynamic_alibi, causal=True)


        if self.use_ca:
            self.cross_attention = MultiHeadAttention(embed_size, heads, alibi_alpha=alibi_alpha,
                                                      start_i_increment=start_i_increment, num_persistent=0,
                                                      rma_inp_dim=0, use_talking_heads=talking_heads,
                                                      dynamic_alibi=dynamic_alibi,
                                                      use_alibi=False, causal=False, backend="auto" if flex_attention else "manual")

        self.feed_forward = SwiGLUConvFFN(
            in_features=embed_size,
            hidden_features=forward_expansion * embed_size,
            out_features=embed_size,
            kernel_size=kernel_size,
            drop=dropout,
            act=act,
            conv_att=conv_att,
            bias=True,
        )
        self.last_weights = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, mask, cross_attn_mask, conv_mask=None, residual=None):
        # Primary attention
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, norm_x, norm_x, mask, None)
        x = x + self.dropout(attn_output)

        # Cross-attention
        if self.use_ca:  
            norm_x = self.norm2(x)
            cross_attn_output, cross_attn_weights = self.cross_attention(y, y, norm_x, mask=cross_attn_mask, return_weights=True)
            x = x + self.dropout(cross_attn_output)
        else:
            cross_attn_weights = None; cross_attn_output = None

        if residual is not None:
            x = x + self.dropout(residual)

        # Feed-forward
        norm_x_ff = self.norm3(x)
        ff_output = self.feed_forward(norm_x_ff, mask if conv_mask is None else conv_mask)
        x = x + self.dropout(ff_output)

        # cache last attn weights
        self.last_weights = cross_attn_weights

        return x, cross_attn_weights, cross_attn_output


class RNNFormerLayer(nn.Module): # redundant arguments kept for easy interface with TransformerDecoder class
    def __init__(self, embed_size, heads, forward_expansion, dropout, alibi_alpha=1.0, start_i_increment=0,
                 kernel_size=1, act="swiglu", conv_att=False, talking_heads=True, dynamic_alibi=False, flex_attention=True):
        super(RNNFormerLayer, self).__init__()
        # Layer norms for stability
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.last_weights = None

        # Transformer-style cross-attention
        self.cross_attention = MultiHeadAttention(
            embed_size, heads,
            alibi_alpha=alibi_alpha,
            start_i_increment=start_i_increment,
            num_persistent=0, rma_inp_dim=0,
            use_talking_heads=talking_heads,
            dynamic_alibi=dynamic_alibi,
            use_alibi=False,
        )
        rnn_type = "LSTM"
        # RNN block: choose between GRU or LSTM
        if rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(embed_size, embed_size, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_size, embed_size, batch_first=True)

    def run_rnn(self, x, conv_mask):
        # Pack padded sequence
        # max(x_lengths) is less than the seq_len dimension in x, often by 5 or 10
        # and the LSTM outputs a tensor of max(x_lengths) in its seq_len dimension
        # therefore, we save the original seq_len dimension for padding back later
        x_seq_len_orig = x.size(1)

        # conv_mask: shape (batch, 1, seq_length), where True = padded, False = valid
        # Step 1: Squeeze to shape (batch, seq_length)
        mask = conv_mask.squeeze(1)
        # Step 2: Invert the mask: valid tokens are now True
        valid_mask = ~mask
        # Step 3: Count valid tokens along the sequence dimension
        x_lengths = valid_mask.sum(dim=1)


        x = pack_padded_sequence(x, x_lengths.detach().cpu(),
                                 # pack_padded_sequence demands that the lengths tensor be on the CPU
                                 batch_first=True, enforce_sorted=False)
        # LSTM pass
        x, _ = self.rnn(x)
        # Unpack the sequence
        x, lens_unpacked = pad_packed_sequence(x, batch_first=True,
                                               total_length=x_seq_len_orig)  # x_lstm:  (batch, seq_len, lstm_channels)

        return x

    def forward(self, x, y, mask, cross_attn_mask, conv_mask=None):
        """
        x: Decoder input of shape (batch, seq_len, embed_size)
        encoder_out: Encoder output of shape (batch, src_seq_len, embed_size)
        cross_attn_mask: Optional mask for cross-attention
        """
        # Transformer-style cross-attention:
        norm_x = self.norm1(x)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            y, y, norm_x, mask=cross_attn_mask, return_weights=True
        )
        x = x + self.dropout(cross_attn_output)

        # RNN update:
        norm_x = self.norm2(x)
        # By default, the initial hidden state is zero
        rnn_output = self.run_rnn(norm_x, conv_mask)
        x = x + self.dropout(rnn_output)

        # cache last attn weights
        self.last_weights = cross_attn_weights

        return x


def make_mask_causal(mask: torch.Tensor) -> torch.Tensor:
    """
    Given a self-attention mask of shape (batch_size, 1, seq_length, seq_length)
    where True indicates a valid (non-padded) token, this function modifies the mask
    to enforce causality (only allow attending to the current and previous positions),
    while ensuring that no positions are turned True if they weren't already valid.

    Args:
        mask (torch.Tensor): Input mask tensor with shape (batch_size, 1, seq_length, seq_length),
                             where True indicates valid and False indicates padded.

    Returns:
        torch.Tensor: A new causal mask of the same shape.
    """
    batch_size, _, seq_length, _ = mask.shape
    device = mask.device
    # Create a causal mask: lower triangular matrix where positions (i,j) are True if j <= i.
    causal_mask = torch.tril(torch.ones(seq_length, seq_length, dtype=torch.bool, device=device))
    # Expand causal_mask dimensions to match mask shape: (1, 1, seq_length, seq_length)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    # Combine the original mask with the causal mask.
    # Only positions that are valid in the original mask AND allowed by causality remain True.
    new_mask = mask & causal_mask
    return new_mask


class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0, start_i=0
                 , act="swiglu", talking_heads=True, dynamic_alibi=False, alibi_scaling_fac=None, use_ca=True):
        super().__init__()
        self.use_conv_att = False
        self.use_ca = use_ca
        if alibi_scaling_fac is None:
            # select a scaling factor so that the final layer never sees a start_i greater than 32
            alibi_scaling_fac = round(max(1, ((num_layers - 1) * heads) // (32 - start_i)))

        use_flexattn_on_ca = [True] * num_layers
        use_flexattn_on_ca[0] = False
        use_flexattn_on_ca[-1] = False

        if not self.use_ca:
            use_ca_lay = [False] * num_layers
            
            use_ca_lay[0] = True
            #use_ca_lay[1] = True
        else:
            use_ca_lay = [True] * num_layers
        
        self.decoder_layers = nn.ModuleList([  # Layer-Scaled ALiBi
            TransformerDecoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha=alibi_alpha,
                                    start_i_increment=start_i + ((i * heads) // alibi_scaling_fac), kernel_size=1,
                                    act=act, talking_heads=talking_heads,
                                    dynamic_alibi=dynamic_alibi, flex_attention=use_flexattn_on_ca, use_ca=use_ca_lay[i])
            for i in range(num_layers)
        ])

    def forward(self, x, y, mask, cross_attn_mask, conv_mask=None, residuals=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
            y: Input tensor of shape (batch_size, seq_2_length, embed_size)
            mask: Mask tensor of shape (batch_size, 1, seq_length, seq_length) or similar.
            cross_attn_mask: Mask tensor of shape (batch_size, 1, seq_length, seq_2_length)
            conv_mask: Convolutional mask size (batch, 1, seq_length) where True is padded and False is valid

            Note: mask does not need to be causal, this call automatically does that.
        Returns:
            The output of the last encoder layer.
        """
        mask = make_mask_causal(mask)
        attn_per_layer = []            # collect (B, H, L1, L2)
        
        # turn list of pairs into a dict for O(1) lookup
        cleaned = [pair for pair in (residuals or []) if pair is not None]
        residual_dict = {lid: res for lid, res in cleaned}

        for i, layer in enumerate(self.decoder_layers):
            if i in residual_dict:
                x = x + residual_dict[i]
                
            x, cross_attn_weights, ca_output = layer(x, y, mask, cross_attn_mask, conv_mask, None)  # Here x serves as query, key, and value
            
            if cross_attn_weights is not None:
                # cross_attn_weights: (B, H, L1, L2)
                if cross_attn_weights.size(1) > 1:
                    # keep only the last head (shape becomes (B, 1, L1, L2))
                    cross_attn_weights = cross_attn_weights[:, -1:, :, :]
                attn_per_layer.append(cross_attn_weights)

        if len(attn_per_layer):
            all_attn = torch.cat(attn_per_layer, dim=1)
        else:
            all_attn = None
        
        return x, all_attn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0, start_i=0,
                 kernel_size=3, act="swiglu", rma_mem_dim=0, conv_att=False, multi_scale=False, talking_heads=True,
                 coarse_fine=False,
                 dynamic_alibi=False, alibi_scaling_fac=None):
        super().__init__()
        self.use_conv_att = conv_att
        self.coarse_fine = coarse_fine
        if alibi_scaling_fac is None:
            alibi_scaling_fac = round(max(1, ((num_layers - 1) * heads) // (32 - start_i)))

        # Our design is coarse fine attention for all layers except the first.
        coarse_fine_vec = [self.coarse_fine] * num_layers
        # if coarse_fine=True, this will be all True except for the first layer (what we want)
        coarse_fine_vec[0] = False

        self.encoder_layers = nn.ModuleList([  # Layer-Scaled ALiBi
            TransformerEncoderLayer(embed_size, heads, forward_expansion, dropout, alibi_alpha=alibi_alpha,
                                    start_i_increment=start_i + ((i * heads) // alibi_scaling_fac),
                                    kernel_size=[kernel_size[i], 1] if multi_scale else kernel_size, act=act,
                                    rma_mem_dim=rma_mem_dim, conv_att=self.use_conv_att and i == num_layers - 1,
                                    talking_heads=talking_heads, coarse_fine=coarse_fine_vec[i],
                                    dynamic_alibi=dynamic_alibi)
            for i in range(num_layers)
        ])
        self.head_dim = embed_size // heads
        self.rma_mem_dim = rma_mem_dim
        self.dropout = nn.Dropout(dropout)
        self.use_rma = self.rma_mem_dim > 0

        if self.use_rma:
            self.kv_proj = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
            )

        if self.coarse_fine:
            self.coarse_projs = nn.ModuleList(
                [
                    nn.Sequential(nn.Conv1d(embed_size, embed_size, 5, 2),
                                  nn.ReLU(),
                                  nn.Dropout(0.1))
                    for _ in range(num_layers - 1)
                ]
            )

    def forward(self, x, mask, conv_mask=None, residuals=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
            mask: Mask tensor of shape (batch_size, 1, seq_length, seq_length) or similar.
            conv_mask: Convolutional mask size (batch, 1, seq_length) where True is padded and False is valid
        Returns:
            The output of the last encoder layer.
        """
        # Pass the input through each encoder layer in sequence

        recurr_keys = None
        recurr_values = None
        coarse_x, coarse_mask = x, conv_mask
        # turn list of pairs into a dict for O(1) lookup
        cleaned = [pair for pair in (residuals or []) if pair is not None]
        residual_dict = {lid: res for lid, res in cleaned}


        for i, layer in enumerate(self.encoder_layers):
            
            if i in residual_dict:
                x = x + residual_dict[i]
                
            x, current_kv = layer(x, mask, conv_mask, (recurr_keys, recurr_values) if recurr_keys is not None else None,
                                  coarse_x, coarse_mask)  # Here x serves as query, key, and value

            # Break at the last layer after processing;
            # due to the design of coarse_fine, we have n_layers - 1 projections, which will trigger an IndexError
            # when it tries to process coarse features after the last layer
            if i == len(self.encoder_layers) - 1:
                break

            if self.coarse_fine:
                # Neat 'lil trick: Max pooling with same args as the conv will compress the mask for us
                coarse_mask = F.max_pool1d(conv_mask.float(), kernel_size=5, stride=2).bool()

                coarse_x = self.coarse_projs[i](
                    x.transpose(1, 2)
                ).masked_fill(coarse_mask, 0).transpose(1, 2)

            if self.use_rma:
                key_r, val_r = current_kv

                # prevent backpropagation into the previous layers
                # otherwise, it tries to optimize each attention layer for the next
                # faster and better loss
                key_r = self.kv_proj(key_r.detach())
                val_r = self.kv_proj(val_r.detach())

                key_r = reduce_sequence_length(key_r)
                val_r = reduce_sequence_length(val_r)

                # Collect recurrent key-values
                recurr_keys = key_r if recurr_keys is None else torch.cat([recurr_keys, key_r], dim=0)
                recurr_values = val_r if recurr_values is None else torch.cat([recurr_values, val_r], dim=0)

        return x


def make_conv(bayesian, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, WN=True):
    if bayesian:
        return BayesConv1d(0.0, 0.1, in_channels, out_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation)
    else:
        if WN:
            return weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        else:
            return nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=stride, padding=padding, dilation=dilation)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_se=False,
                 reduction=16, bayesian=False, use_swiglu=False, use_aptx=False, use_cbam=False):
        """
        Initialize TemporalBlock for TCN
        :param n_inputs: Number of input channels
        :param n_outputs: Output channels
        :param kernel_size: Kernel size
        :param stride: Stride of convs
        :param dilation: Dilation
        :param padding: Padding
        :param dropout: Dropout
        :param use_se: Use Squeeze-Excite attention
        :param reduction: Reduction for Squeeze-Excite, if enabled
        :param bayesian: Use Bayesian convs, for nondeterminism. Will use LayerNorm instead of weight normalization
        :param use_swiglu: Use SwiGLU for the final activation
        :param use_aptx: Use APTx for the acts
        :param use_cbam: Use CBAM at the final and at residual.
        """
        super(TemporalBlock, self).__init__()

        self.use_swiglu = use_swiglu

        self.conv1 = make_conv(bayesian, n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.ln1 = TransposeLayerNorm(n_outputs) if bayesian else nn.Identity()
        self.relu1 = APTx() if use_aptx else nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        n_outputs_orig = n_outputs
        n_final = n_outputs

        if self.use_swiglu and n_inputs == n_outputs:
            n_final = n_inputs * 2

        self.conv2 = make_conv(bayesian, n_outputs, n_final, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.ln2 = TransposeLayerNorm(n_final) if bayesian else nn.Identity()
        self.relu2 = APTx() if use_aptx else nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.ln1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.ln2, self.relu2, self.dropout2)

        n_outputs = n_final

        self.downsample = make_conv(bayesian, n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

        if use_cbam:
            use_se = False

        self.se_block = SEBlock1D(n_outputs, reduction) if use_se else nn.Identity()
        self.cbam_block = CBAM(n_outputs, reduction) if use_cbam else nn.Identity()
        self.res_cbam = CBAM(n_outputs, reduction) if use_cbam else nn.Identity()
        self.drop = nn.Dropout(0.1)

        if use_swiglu and n_inputs == n_outputs_orig:
            self.relu = SwiGLUCNN()
        else:
            self.relu = APTx() if use_aptx else nn.ReLU()

    def forward(self, x, mask=None):
        """
        Forward pass through the Temporal Block
        :param x: Tensor size (batch, in_channels, seq_len)
        :param mask: Bool mask size (batch, 1, seq_len), where True is padded and False is valid.
                    If not passed, will assume all sequence is valid.
        :return: Processed tensor size (batch, out_channels, seq_len)
        """

        if mask is None:
            mask = torch.zeros((x.size(0), 1, x.size(2))).bool().to(x.device)

        self.chomp1.set_mask(mask)
        self.chomp2.set_mask(mask)

        out = self.net(x).masked_fill(mask, 0)

        res = self.downsample(x).masked_fill(mask, 0)
        out = out + self.res_cbam(res).masked_fill(mask, 0)
        out = self.drop(out)

        # Only one of these will be valid
        out = self.se_block(out).masked_fill(mask, 0)
        out = self.cbam_block(out).masked_fill(mask, 0)

        out = self.drop(out)

        out = self.relu(out).masked_fill(mask, 0)
        return out


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_growth="exp", use_se=False,
                 bayesian=False):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size] * num_levels

        for i in range(num_levels):

            if dilation_growth == "exp":
                dilation_size = 2 ** i
            elif dilation_growth == "mul":
                dilation_size = max(1, 2 * i)
            elif dilation_growth == "add":
                dilation_size = i + 1
            else:
                raise RuntimeError(f"Unknown dilation growth type {dilation_growth}")

            k_size = kernel_size[i]
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, k_size, stride=1, dilation=dilation_size,
                                     padding=(k_size - 1) * dilation_size, dropout=dropout, use_se=use_se,
                                     bayesian=bayesian)]

        self.network = nn.Sequential(*layers)

    def forward(self, x, mask):
        """
        :param x: Tensor size (batch, in_channels, seq_len)
        :param mask: Bool mask size (batch, 1, seq_len), where True is padded and False is valid.
                    If not passed, will assume all sequence is valid.
        :return: Processed tensor size (batch, out_channels, seq_len)
        """

        # Must do: Refactor the Sequential into a ModuleList; we're doing this because transfer learning
        for layer in self.network:
            x = layer(x, mask)

        return x


def mask_to_causal_attention_mask(mask):
    """
    Turn a bool mask into a causal attention mask
    :param mask: Bool sequence mask, True=padding size (batch, max_length)
    :return: Causal attention mask size (batch, 1, seq_len, seq_len), True=valid
    """
    batch_size, seq_len = mask.shape
    # Create a lower triangular matrix of ones
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=0).to(mask.device)
    # Expand dimensions to fit the attention mask shape (batch, 1, seq_len, seq_len)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    # Combine the causal mask with the input mask
    attention_mask = mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3) & causal_mask
    # Flip the mask, our attention uses True=valid
    attention_mask = ~attention_mask
    return attention_mask


class CausalConv1d(nn.Conv1d):
    """
    A 1D convolution layer that applies causal padding on the left side.
    For a kernel size k and dilation d, it pads the input with d*(k-1) zeros on the left.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        # Compute the required left-padding for causal convolution
        self.causal_padding = dilation * (kernel_size - 1)
        # Remove any padding passed in kwargs and use padding=0 (we handle it manually)
        kwargs.pop('padding', None)
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation, padding=0, **kwargs)

    def forward(self, x):
        # Pad only on the left: (left, right)
        if self.causal_padding > 0:
            x = F.pad(x, (self.causal_padding, 0))
        return super().forward(x)


from torch.nn.utils.parametrizations import weight_norm

class CausalConv1da(nn.Conv1d):
    """
    1-D *causal* convolution with optional weight-norm already applied.

    Padding rule
    ------------
    For kernel size *k* and dilation *d* the layer pads
    **d × (k − 1)** zeros on the *left* so that every output sample
    depends only on current and past inputs.

    Parameters
    ----------
    in_channels      : int
    out_channels     : int
    kernel_size      : int
    dilation         : int, default=1
    bias             : bool, default=True
    use_weight_norm  : bool, default=True
    **kwargs         : any extra Conv1d arguments (stride, groups, etc.)
    """
    def __init__(
        self,
        in_channels:   int,
        out_channels:  int,
        kernel_size:   int,
        dilation:      int  = 1,
        bias:          bool = True,
        use_weight_norm: bool = True,
        **kwargs
    ):
        # amount of left-padding to keep the op causal
        self.causal_padding = dilation * (kernel_size - 1)

        # Conv1d handles **no** padding; we pad manually in forward()
        kwargs.pop("padding", None)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
            **kwargs
        )

        # register weight-norm on this module's own .weight
        if use_weight_norm:
            nn.utils.weight_norm(self, name="weight")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal_padding:
            x = F.pad(x, (self.causal_padding, 0))  # (left, right)
        return super().forward(x)

from torch.nn.utils.parametrizations import weight_norm

class ResidualBlock1D(nn.Module):
    """
    Conv1D+Squeeze-Excite+LayerNorm residual block for sequence modeling with optional masking.

    Accepts an optional x_mask (batch, 1, len) bool Tensor where padded elements are True.
    If provided, the mask is applied with .masked_fill() before each activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.3, act="relu", causal=False, norm="layer"):
        super(ResidualBlock1D, self).__init__()

        assert norm in ["weight", "layer", "instance"], f"Unknown normalization type {norm}, must be 'weight', 'layer', or 'instance'"

        if causal:
            self.conv1 = CausalConv1da(in_channels, out_channels, kernel_size, dilation=dilation, use_weight_norm = norm == "weight")
            self.conv2 = CausalConv1da(out_channels, out_channels, kernel_size, dilation=dilation,  use_weight_norm = norm == "weight")
            self.cbam = None
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same")
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding="same")
            self.cbam = CBAM1D(out_channels)

        if norm == "weight":
            if not causal:
                self.conv1 = weight_norm(self.conv1)
                self.conv2 = weight_norm(self.conv2)

            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        elif norm == "layer":
            self.norm1 = TransposeLayerNorm(out_channels)
            self.norm2 = TransposeLayerNorm(out_channels)
        else:
            self.norm1 = nn.InstanceNorm1d(out_channels, affine=True)
            self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)

        if act == "taptx":
            self.relu = APTx(trainable=True)
        elif act == "aptx":
            self.relu = APTx()
        elif act == "relu":
            self.relu = nn.ReLU()
        else:
            raise RuntimeError(f"Unknown activation: {act}")
            
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, x_mask=None):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.norm1(out)
        # Apply mask before the first activation if provided
        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.norm2(out)
        if self.cbam is not None:
            out = self.cbam(out, x_mask)
        out += residual
        # Apply mask before the second activation if provided
        if x_mask is not None:
            out = out.masked_fill(x_mask, 0)
        out = self.relu(out)
        out = self.dropout(out)
        return out


def reduce_sequence_length(input_tensor):
    """
    Reduces the sequence length of a tensor by half using 1D max pooling.

    Args:
    input_tensor (torch.Tensor): Input tensor of shape (seq_len, N, dim).

    Returns:
    torch.Tensor: Output tensor with reduced sequence length, shape (seq_len//2, N, dim).
    """
    seq_len, N, dim = input_tensor.shape

    # Ensure seq_len is even for the 1D max pooling to reduce by half
    assert seq_len % 2 == 0, "Sequence length should be even for halving with max pooling"

    # Permute the tensor to (N, dim, seq_len) for 1D max pooling
    input_tensor_permuted = input_tensor.permute(1, 2, 0)  # (N, dim, seq_len)

    # Apply 1D max pooling with kernel size 2 and stride 2
    output_tensor_permuted = F.max_pool1d(input_tensor_permuted, kernel_size=2, stride=2)

    # Permute back to original dimensions (seq_len//2, N, dim)
    output_tensor = output_tensor_permuted.permute(2, 0, 1)  # (seq_len//2, N, dim)

    return output_tensor


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)[:, :, :-self.padding]


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, normalization='layer', act="aptx",
                 dropout=0.5, causal=True):
        super(ConvReluNorm, self).__init__()
        self.causal = causal

        if self.causal:
            self.causal_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        else:
            self.causal_conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation, padding="same")

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "aptx":
            self.act = APTx()
        elif act == "taptx":
            self.act = APTx(trainable=True)
        elif act == "dprelu":
            self.act = DPReLU()

        self.drop = nn.Dropout(dropout)

        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        elif normalization == 'layer':
            self.norm = TransposeLayerNorm(out_channels)
        elif normalization == "":
            self.norm = nn.Identity()
        else:
            raise ValueError("Normalization type must be either 'batch', 'layer', or an empty string (none)")

    def forward(self, x, x_mask):
        x = self.causal_conv(x).masked_fill(x_mask, 0)
        x = self.act(x).masked_fill(x_mask, 0)
        x = self.norm(x).masked_fill(x_mask, 0)
        x = self.drop(x)
        return x


class NeoTCNAttention(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=[2, 2, 2], dropout=0.2, att_dropout=0.3, heads=[2, 2, 2],
                 alibi_alpha=1.25, start_i_increment=1, dilation_growth="", act="aptx", bayesian=False,
                 integrated=False, conv_att="se"):
        super(NeoTCNAttention, self).__init__()

        self.layers = nn.ModuleList()
        self.att_layers = nn.ModuleList()
        self.out_channels = num_channels[-1]
        self.n_heads = heads
        self.attn_dropout = nn.Dropout(att_dropout)

        if len(heads) != len(num_channels):
            raise ValueError("The length of heads must be equal to the length of num_channels")
        if len(kernel_size) != len(num_channels):
            raise ValueError("The length of kernel_size must be equal to the length of num_channels")

        current_channels = num_inputs

        for level, (out_channels, num_heads, k_size) in enumerate(zip(num_channels, heads, kernel_size)):
            is_last = level == len(num_channels) - 1
            curr_i_increment = start_i_increment + (level * num_heads)

            if dilation_growth == "exp":
                dilation_size = 2 ** level
            elif dilation_growth == "mul":
                dilation_size = max(1, 2 * level)
            elif dilation_growth == "add":
                dilation_size = level + 1
            elif dilation_growth == "":
                dilation_size = 1
            else:
                raise RuntimeError(f"Unknown dilation growth type {dilation_growth}")

            # pre-attention arrangement
            self.att_layers.append(
                MultiHeadAttention(current_channels, num_heads, alibi_alpha=alibi_alpha,
                                   start_i_increment=curr_i_increment,
                                   num_persistent=16)
                if num_heads > 0 else nn.Identity()  # append an identity so it still occupies an index i in the list

            )
            self.layers.append(
                ConvReluNorm(current_channels, out_channels, k_size, dilation_size, act=act, dropout=dropout,
                             causal=False))

            current_channels = out_channels  # The output of the current block is the input for the next

        if conv_att == "se":
            self.conv_att = MaskedSEBlock1D(out_channels)
        elif conv_att == "cbam":
            self.conv_att = MaskedCBAM1d(out_channels)
        else:
            self.conv_att = None

    def forward(self, x, mask, inp_channel_last=True):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_length, channels) if inp_channel_last=True, else (batch, channels, seq)
            mask: Mask tensor of shape (batch_size, seq_length), where True is invalid (padding) and False is valid
        """

        att_mask = mask_to_causal_attention_mask(mask)
        mask = mask.unsqueeze(1)

        if inp_channel_last:
            x = x.transpose(1, 2)  # (batch, channels, seq)

        for i, layer in enumerate(self.layers):

            if self.n_heads[i] > 0:
                x = x.transpose(1, 2)  # (batch, seq, channels)
                x_att = self.att_layers[i](x, x, x, att_mask)
                x_att = self.attn_dropout(x_att)
                x += x_att
                x = x.transpose(1, 2)  # (batch, channels, seq)

            x = layer(x, mask)

        if self.conv_att is not None:
            x = self.conv_att(x, mask)

        if inp_channel_last:
            x = x.transpose(1, 2)  # (batch, seq, channels)

        return x


