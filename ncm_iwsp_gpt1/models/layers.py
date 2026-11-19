"""Transformer pieces used by the decoder and the generator.

Adapted from HazyResearch/ncm ``models/common.py`` and ``models/vae.py``
(MIT, Copyright (c) 2025 hermann kumbong), commit c702742. The decoder
attention is the non-causal block in ``vae.py``. The generator attention is
the masked, RoPE block in ``common.py``. Flex attention, gated MLPs, and
training helpers are omitted.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttrDict:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, inputs):
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-6)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x_f * rms).to(x.dtype) * self.weight


def precompute_rope_cos_sin(positions, head_dim, base=10000):
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def apply_rotary_emb(x, cos, sin):
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class DecoderAttention(nn.Module):
    """Full (non-causal) attention used by the VQ-VAE decoder."""

    def __init__(self, n_embd, n_head, bias, dropout, use_qk_norm):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd {n_embd} not divisible by n_head {n_head}")
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.use_qk_norm = bool(use_qk_norm)
        if self.use_qk_norm:
            head_dim = n_embd // n_head
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x):
        batch, time, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(y))


class GeneratorAttention(nn.Module):
    """Masked attention with optional RoPE, used by the coarse-to-fine generator."""

    def __init__(self, n_embd, n_head, bias, dropout, use_qk_norm):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd {n_embd} not divisible by n_head {n_head}")
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.use_qk_norm = bool(use_qk_norm)
        if self.use_qk_norm:
            head_dim = n_embd // n_head
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)

    def forward(self, x, attn_mask=None, rope_cos_sin=None):
        batch, time, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, time, self.n_head, head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout, use_qk_norm):
        super().__init__()
        # Official vae.Block hardcodes bias=True. Decoder LayerNorm at the
        # output of the VQ-VAE is a separate module with bias=False.
        self.ln_1 = LayerNorm(n_embd, bias=True)
        self.attn = DecoderAttention(n_embd, n_head, bias=True, dropout=dropout, use_qk_norm=use_qk_norm)
        self.ln_2 = LayerNorm(n_embd, bias=True)
        self.mlp = MLP(n_embd, bias=True, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout, use_qk_norm):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = GeneratorAttention(n_embd, n_head, bias=bias, dropout=dropout, use_qk_norm=use_qk_norm)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(self, x, attn_mask=None, rope_cos_sin=None):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask, rope_cos_sin=rope_cos_sin)
        x = x + self.mlp(self.ln_2(x))
        return x


def init_output_projections(module, n_layer: int) -> None:
    std = 0.02 / math.sqrt(2 * max(int(n_layer), 1))
    for name, param in module.named_parameters():
        if name.endswith("c_proj.weight"):
            nn.init.normal_(param, mean=0.0, std=std)
