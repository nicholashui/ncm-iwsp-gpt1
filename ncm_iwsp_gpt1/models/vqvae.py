"""Residual VQ-VAE tokenizer: lookup encoder, multiscale residual codes, one decoder block.

Adapted from HazyResearch/ncm ``models/vqvae.py`` and ``models/vae.py``
(MIT, Copyright (c) 2025 hermann kumbong), commit c702742. ``enc_layers=0`` is
an embedding lookup plus a position table. ``dec_layers=1`` is one pre-norm
block. ``n_heads`` is the attention width of that block, which official configs
name ``n_layers``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ncm_iwsp_gpt1.models.layers import DecoderBlock, LayerNorm
from ncm_iwsp_gpt1.models.quant import MultiscaleResidualQuantizer


class Encoder(nn.Module):
    def __init__(self, vocab_size, context_length, embed_dim, dropout):
        super().__init__()
        self.wte = nn.Embedding(int(vocab_size), int(embed_dim))
        self.wpe = nn.Embedding(int(context_length), int(embed_dim))
        self.drop = nn.Dropout(float(dropout))

    def forward(self, token_ids):
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        return self.drop(self.wte(token_ids) + self.wpe(positions))


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, dropout, use_qk_norm, n_blocks):
        super().__init__()
        self.middle_block = DecoderBlock(embed_dim, n_heads, dropout, use_qk_norm)
        extra = max(0, int(n_blocks) - 1)
        self.extra_blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, n_heads, dropout, use_qk_norm) for _ in range(extra)]
        )
        self.ln_f = LayerNorm(embed_dim, bias=False)
        self.output_layer = nn.Linear(embed_dim, int(vocab_size), bias=False)

    def forward(self, latent):
        hidden = self.middle_block(latent)
        for block in self.extra_blocks:
            hidden = block(hidden)
        return self.output_layer(self.ln_f(hidden))


class VQVAE(nn.Module):
    def __init__(self, tokenizer_cfg: dict, vocab_size: int):
        super().__init__()
        cfg = tokenizer_cfg
        if int(cfg.get("enc_layers", 0)) != 0:
            raise NotImplementedError("this harness pins enc_layers=0 (embedding lookup only)")
        self.embed_dim = int(cfg["embed_dim"])
        self.context_length = int(cfg["context_length"])
        self.in_vocab_size = int(vocab_size)
        self.encoder = Encoder(
            self.in_vocab_size, self.context_length, self.embed_dim, float(cfg.get("dropout", 0.0))
        )
        groups = int(cfg.get("pre_quant_groupnorm", 0) or 0)
        if groups:
            if self.embed_dim % groups != 0:
                raise ValueError(f"pre_quant_groupnorm={groups} must divide embed_dim={self.embed_dim}")
            self.pre_quant_norm = nn.GroupNorm(groups, self.embed_dim, affine=False)
        else:
            self.pre_quant_norm = None
        levels = [int(level) for level in cfg["levels"]]
        codebook = cfg["codebook_size"]
        if isinstance(codebook, (list, tuple)):
            vocab_sizes = [int(size) for size in codebook]
        else:
            vocab_sizes = [int(codebook)] * len(levels)
        resi = cfg.get("quant_resi", {}) or {}
        self.vq = MultiscaleResidualQuantizer(
            vocab_sizes=vocab_sizes,
            embed_dim=self.embed_dim,
            levels=levels,
            quant_resi_ratio=float(resi.get("ratio", 0.5)),
            quant_resi_enabled=bool(resi.get("enabled", True)),
            learned_l1_sampling=bool(cfg.get("learned_l1_sampling", True)),
            learned_l1_cascade=bool(cfg.get("learned_l1_cascade", True)),
            learned_l1_cascade_base=int(cfg.get("learned_l1_cascade_base", 4)),
        )
        self.decoder = Decoder(
            vocab_size=self.in_vocab_size,
            embed_dim=self.embed_dim,
            n_heads=int(cfg["n_heads"]),
            dropout=float(cfg.get("dropout", 0.0)),
            use_qk_norm=bool(cfg.get("use_qk_norm", True)),
            n_blocks=int(cfg.get("dec_layers", 1)),
        )

    def _encode_pre_quant(self, token_ids):
        if token_ids.size(1) != self.context_length:
            raise ValueError(
                f"expected context {self.context_length}, got {token_ids.size(1)}"
            )
        latent = self.encoder(token_ids)
        if self.pre_quant_norm is not None:
            latent = self.pre_quant_norm(latent.transpose(1, 2)).transpose(1, 2)
        return latent

    @torch.no_grad()
    def encode(self, token_ids):
        return self.vq.z_to_idx(self._encode_pre_quant(token_ids))

    @torch.no_grad()
    def decode_logits(self, indices):
        return self.decoder(self.vq.idx_to_z_q(indices))

    @torch.no_grad()
    def reconstruct_logits(self, token_ids):
        indices = self.encode(token_ids)
        return self.decode_logits(indices), indices
