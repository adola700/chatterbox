"""
Perceiver Resampler — Compresses variable-length speech conditioning into fixed-length tokens.

Used by T3CondEnc to compress 150 speech conditioning embeddings (6s of reference audio
at 25 tokens/sec) into 32 fixed-length conditioning tokens via cross-attention + self-attention.

Architecture
============
    Input:  (B, 150, 1024)  — speech conditioning embeddings from T3.speech_emb + speech_pos_emb
    Output: (B, 32, 1024)   — compressed conditioning tokens prepended to transformer input

    Step 1 — Cross-attention:
        Q = 32 learned query tokens (1, 32, 1024) → expanded to (B, 32, 1024)
        K, V = input speech embeddings (B, 150, 1024)
        → pre_att = (B, 32, 1024)

    Step 2 — Self-attention:
        Q = K = V = pre_att (B, 32, 1024)
        → output = (B, 32, 1024)

Components
----------
    Perceiver
        pre_attention_query : nn.Parameter(1, 32, 1024) — learned query tokens
        attn : AttentionBlock2 — shared attention block used for both cross and self attention

    AttentionBlock2
        norm   : LayerNorm(1024)
        to_q   : Linear(1024, 1024)
        to_k   : Linear(1024, 1024)
        to_v   : Linear(1024, 1024)
        attention : AttentionQKV(4 heads, head_dim=256, flash=True)
        proj_out  : Linear(1024, 1024)
        Residual connection: output = x1 + proj_out(attention(q, k, v))

    AttentionQKV
        4 heads, head_dim=256, scale=256^-0.5
        Flash attention (torch.backends.cuda.sdp_kernel) when available
        Fallback: manual scaled dot-product with einsum

    RelativePositionBias (optional, NOT used in default Perceiver config)
        Logarithmic relative position bucketing for attention bias.
"""
# Copyright (c) 2025 Resemble AI
# Author: Manmay Nakhashi
# MIT License
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)


class AttentionQKV(nn.Module):
    def __init__(self, n_heads, head_dim, dropout_rate=0.1, scale=None, flash=False):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.flash = flash
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.flash_config = self.setup_flash_config() if flash else None

    def setup_flash_config(self):
        # Setup flash attention configuration
        flash_config = {
            'enable_flash': True,
            'enable_math': True,
            'enable_mem_efficient': True
        }
        return flash_config

    def forward(self, q, k, v, mask=None):
        q, k, v = [self.split_heads(tensor) for tensor in [q, k, v]]
        if self.flash:
            out = self.flash_attention(q, k, v, mask=mask)
        else:
            out = self.scaled_dot_product_attention(q, k, v, mask=mask)

        return self.combine_heads(out)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        sim = torch.einsum("bhlt,bhls->bhts", q, k) * self.scale
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        return torch.einsum("bhts,bhls->bhlt", attn, v)

    def flash_attention(self, q, k, v, mask=None):
        config = self.flash_config if self.flash_config else {}
        with torch.backends.cuda.sdp_kernel(**config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout_rate if self.training else 0.
            )
        return out

    def split_heads(self, x):
        bs, length, _ = x.shape
        x = x.view(bs, length, self.n_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def combine_heads(self, x):
        bs, _, length, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(bs, length, -1)


class AttentionBlock2(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other,
    using AttentionQKV and separate linear transformations for Q, K, and V.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        relative_pos_embeddings=False,
        flash_attention=True,
        dropout_rate=0.2,
        scale=None
    ):
        super().__init__()
        self.channels = channels

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = nn.LayerNorm(channels)

        # Separate linear layers for Q, K, and V
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)

        self.attention = AttentionQKV(self.num_heads, channels // self.num_heads, dropout_rate=dropout_rate, flash=flash_attention, scale=scale)

        self.proj_out = nn.Linear(channels, channels)

        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x1, x2, mask=None):
        b1, c1, *spatial1 = x1.shape
        b2, c2, *spatial2 = x2.shape

        x1_norm = self.norm(x1)
        x2_norm = self.norm(x2)

        q = self.to_q(x1_norm)
        k = self.to_k(x2_norm)
        v = self.to_v(x2_norm)

        h = self.attention(q, k, v, mask=mask)
        h = self.proj_out(h)

        return (x1 + h).reshape(b1, c1, *spatial1)


class Perceiver(nn.Module):
    """Perceiver Resampler — compresses speech conditioning via cross-attention + self-attention.

    Inspired by https://arxiv.org/abs/2103.03206

    Default config (used by T3CondEnc):
        pre_attention_query_token = 32   — number of learned query tokens
        pre_attention_query_size = 1024  — dimension of each query
        embedding_dim = 1024             — must match T3 hidden size (n_channels)
        num_attn_heads = 4               — attention heads (head_dim = 1024/4 = 256)

    Forward:
        Input:  (B, 150, 1024) — speech conditioning embeddings
        Output: (B, 32, 1024)  — compressed conditioning tokens

    Parameters:
        pre_attention_query : nn.Parameter(1, 32, 1024) — ~32K params
        attn (AttentionBlock2): ~4.2M params (4× Linear(1024,1024) + LayerNorm)
        Total: ~4.2M parameters
    """
    def __init__(self, pre_attention_query_token=32, pre_attention_query_size=1024, embedding_dim=1024, num_attn_heads=4):
        """
        Args:
            pre_attention_query_token: Number of learned query tokens (default 32).
                Controls output sequence length: input (B, S, D) → output (B, 32, D).
            pre_attention_query_size: Dimension of each query token (default 1024).
                Must match embedding_dim.
            embedding_dim: Transformer hidden dimension (default 1024).
                Must match T3Config.n_channels.
            num_attn_heads: Number of attention heads (default 4).
                head_dim = embedding_dim / num_attn_heads = 256.
        """
        super().__init__()

        # Initialize the pre-attention query parameter
        self.pre_attention_query = torch.nn.Parameter(
            torch.empty(1, pre_attention_query_token, pre_attention_query_size)
        )

        # Calculate the variance for uniform initialization
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (pre_attention_query_token + pre_attention_query_token))

        # Initialize the pre-attention query with uniform distribution
        self.pre_attention_query.data.uniform_(-query_variance, query_variance)

        # Initialize the attention block
        self.attn = AttentionBlock2(embedding_dim, num_attn_heads)

    def forward(self, h):
        """Compress input sequence via cross-attention then self-attention.

        Args:
            h: (B, S, 1024) — input speech conditioning embeddings.
               Typically S=150 (speech_cond_prompt_len from T3Config).

        Returns:
            (B, 32, 1024) — compressed conditioning tokens.

        Steps:
            1. Expand learned queries: (1, 32, 1024) → (B, 32, 1024)
            2. Cross-attention: Q=queries, K=V=h → (B, 32, 1024) + residual
            3. Self-attention: Q=K=V=pre_att → (B, 32, 1024) + residual
        """
        # Expand the pre-attention query to match the batch size of the input
        query_ = self.pre_attention_query.expand(h.shape[0], -1, -1)
        # Apply the first attention mechanism (cross-attention)
        pre_att = self.attn(query_, h)
        # Apply the second attention mechanism (self-attention)
        attn = self.attn(pre_att, pre_att)
        return attn
