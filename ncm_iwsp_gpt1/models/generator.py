"""Coarse-to-fine next-concept generator.

Adapted from HazyResearch/ncm ``models/next_concept.py`` (MIT, Copyright (c) 2025
hermann kumbong), commit c702742, restricted to this harness: ``predict_level0``
false, block-diagonal attention, RoPE, level encodings, raw-token prefix.
Level 0 is not a transformer target. The evaluator samples it from a unigram.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ncm_iwsp_gpt1.models.layers import GeneratorBlock, LayerNorm, init_output_projections, precompute_rope_cos_sin
from ncm_iwsp_gpt1.prefix_protocol import normalize_prefix_source


def _run_block(hidden, block, attn_mask, rope, segments):
    return block(hidden, attn_mask=attn_mask, rope_cos_sin=rope, segments=segments)


class NextConceptGenerator(nn.Module):
    def __init__(self, vqvae, generator_cfg: dict):
        super().__init__()
        if bool(generator_cfg.get("predict_level0", False)):
            raise ValueError("predict_level0 must stay false")
        if generator_cfg.get("attn_pattern", "block_diagonal") != "block_diagonal":
            raise ValueError("only block_diagonal attention is implemented")
        # A tuple keeps the VQ-VAE out of this module's parameter list.
        self._vqvae = (vqvae,)
        self.levels = list(vqvae.vq.levels)
        self.prefix_len = int(generator_cfg.get("prefix_len", 1))
        if self.prefix_len < 0:
            raise ValueError("prefix_len must be >= 0")
        # eos: phase 1, a constant <|endoftext|> prefix. window: phase 2, the
        # start of the real token row. The mask below does not depend on which.
        self.prefix_source = normalize_prefix_source(generator_cfg.get("prefix_source", "eos"))
        self.n_embd = int(generator_cfg["n_embd"])
        self.n_head = int(generator_cfg["n_head"])
        self.n_layer = int(generator_cfg["n_layer"])
        bias = bool(generator_cfg.get("bias", True))
        dropout = float(generator_cfg.get("dropout", 0.0))
        use_qk_norm = bool(generator_cfg.get("use_qk_norm", True))
        self.use_level_encoding = bool(generator_cfg.get("use_level_encoding", True))
        self.use_rope = bool(generator_cfg.get("use_rope", True))
        rope_base = int(generator_cfg.get("rope_base", 100000))

        target_levels = self.levels[1:]
        if not target_levels:
            raise ValueError("need at least two VQ levels so the generator has a target")
        target_len = sum(target_levels)
        prefix = self.prefix_len
        head_dim = self.n_embd // self.n_head

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Linear(vqvae.embed_dim, self.n_embd, bias=bias),
                "blocks": nn.ModuleList(
                    [
                        GeneratorBlock(self.n_embd, self.n_head, bias, dropout, use_qk_norm)
                        for _ in range(self.n_layer)
                    ]
                ),
                "ln_f": LayerNorm(self.n_embd, bias=bias),
            }
        )
        if self.use_level_encoding:
            self.transformer["wle"] = nn.Embedding(len(target_levels), self.n_embd)

        vocab_sizes = [int(size) for size in vqvae.vq.vqvae_vocab_sizes[1:]]
        self.vocab_sizes = vocab_sizes
        self.lm_heads = nn.ModuleList(
            [nn.Linear(self.n_embd, size, bias=True) for size in vocab_sizes]
        )

        level_ids = torch.cat(
            [torch.full((length,), index) for index, length in enumerate(target_levels)]
        ).view(1, target_len, 1)
        level_ids_t = level_ids.transpose(1, 2)
        target_mask = torch.where(level_ids == level_ids_t, 0.0, float("-inf")).reshape(target_len, target_len)
        top_left = torch.zeros(prefix, prefix)
        top_right = torch.full((prefix, target_len), float("-inf"))
        bottom_left = torch.zeros(target_len, prefix)
        full_mask = torch.cat(
            [
                torch.cat([top_left, top_right], dim=1),
                torch.cat([bottom_left, target_mask], dim=1),
            ],
            dim=0,
        ).view(1, 1, prefix + target_len, prefix + target_len)
        self.register_buffer("attn_mask", full_mask.contiguous(), persistent=False)
        self.register_buffer("lvl_1L", level_ids.view(1, target_len).long().contiguous(), persistent=False)

        if self.use_rope:
            full_pos = torch.cat(
                [torch.arange(prefix)] + [torch.arange(length) for length in target_levels]
            )
            cos, sin = precompute_rope_cos_sin(full_pos, head_dim, rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
            for index, length in enumerate(target_levels):
                pos = torch.cat([torch.arange(prefix), torch.arange(length)])
                cos_k, sin_k = precompute_rope_cos_sin(pos, head_dim, rope_base)
                self.register_buffer(f"rope_cos_gen_{index}", cos_k, persistent=False)
                self.register_buffer(f"rope_sin_gen_{index}", sin_k, persistent=False)

        init_output_projections(self, self.n_layer)
        # Off unless a fine-tune run turns them on. Eval keeps the dense mask.
        self.chunked_block_attn = False
        self.gradient_checkpointing = False

    def forward(self, latents):
        prefix = self.prefix_len
        prefix_latent = latents[:, :prefix, :]
        target_latent = latents[:, prefix:, :]
        prefix_emb = self.transformer.wte(prefix_latent)
        target = self.transformer.wte(target_latent)
        if self.use_level_encoding:
            target = target + self.transformer.wle(self.lvl_1L)
        hidden = torch.cat([prefix_emb, target], dim=1)
        rope = (self.rope_cos, self.rope_sin) if self.use_rope else None
        if self.chunked_block_attn:
            segments = (prefix, tuple(int(level) for level in self.levels[1:]))
            attn_mask = None
        else:
            segments = None
            attn_mask = self.attn_mask
        recompute = bool(self.gradient_checkpointing) and self.training
        for block in self.transformer.blocks:
            if recompute:
                # Bind the block in a default arg so the loop variable is not shared.
                def _recompute(hidden, _block=block, _mask=attn_mask, _rope=rope, _segments=segments):
                    return _run_block(hidden, _block, _mask, _rope, _segments)

                hidden = checkpoint(_recompute, hidden, use_reentrant=False)
            else:
                hidden = block(hidden, attn_mask=attn_mask, rope_cos_sin=rope, segments=segments)
        hidden = self.transformer.ln_f(hidden)
        target_hidden = hidden[:, prefix:, :]
        pieces = []
        offset = 0
        for head, level in zip(self.lm_heads, self.levels[1:]):
            pieces.append(head(target_hidden[:, offset : offset + level, :]))
            offset += level
        return torch.cat(pieces, dim=1)

    def _decode_level_logits(self, target, prefix_emb, head_index):
        hidden = torch.cat([prefix_emb, target], dim=1)
        length = target.shape[1]
        prefix = self.prefix_len
        mask = torch.zeros(1, 1, prefix + length, prefix + length, device=hidden.device)
        mask[:, :, :prefix, prefix:] = float("-inf")
        rope = None
        if self.use_rope:
            rope = (
                getattr(self, f"rope_cos_gen_{head_index}"),
                getattr(self, f"rope_sin_gen_{head_index}"),
            )
        for block in self.transformer.blocks:
            hidden = block(hidden, attn_mask=mask, rope_cos_sin=rope)
        hidden = self.transformer.ln_f(hidden)
        return self.lm_heads[head_index](hidden[:, prefix:, :])

    def _sample(self, logits, temperature, top_k, top_p):
        if temperature is None or float(temperature) <= 0:
            return logits.argmax(dim=-1)
        flat = logits.reshape(-1, logits.shape[-1]) / float(temperature)
        if top_k and int(top_k) > 0:
            kth = torch.topk(flat, min(int(top_k), flat.shape[-1]), dim=-1).values[:, -1, None]
            flat = flat.masked_fill(flat < kth, float("-inf"))
        if top_p and float(top_p) < 1.0:
            sorted_logits, sorted_idx = torch.sort(flat, dim=-1, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) > float(top_p)
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            flat = torch.zeros_like(flat).scatter_(-1, sorted_idx, sorted_logits)
        draw = torch.multinomial(F.softmax(flat, dim=-1), num_samples=1)
        return draw.view(logits.shape[:-1])

    @torch.no_grad()
    def generate(self, level0_idx, prefix_tokens, temperature=0.0, top_k=0, top_p=1.0):
        """Sample levels 1..K-1 conditioned on a provided level-0 code."""
        vqvae = self._vqvae[0]
        vq = vqvae.vq
        batch = prefix_tokens.shape[0]
        device = prefix_tokens.device
        prefix_latent = vqvae.encoder.wte(prefix_tokens)[:, : self.prefix_len, :]
        prefix_emb = self.transformer.wte(prefix_latent)
        if self.use_level_encoding:
            level_pos = self.transformer.wle(self.lvl_1L)
        else:
            level_pos = torch.zeros(1, sum(self.levels[1:]), self.n_embd, device=device)
        running = torch.zeros(batch, self.levels[-1], vqvae.embed_dim, device=device)
        generated = [level0_idx]
        idx_next = level0_idx
        cursor = 0
        for level_idx in range(len(self.levels) - 1):
            level_codes = vq.embeddings[level_idx](idx_next)
            running, pooled = vq.get_next_autoregressive_input(level_idx, running, level_codes)
            length = self.levels[level_idx + 1]
            target = self.transformer.wte(pooled) + level_pos[:, cursor : cursor + length, :]
            logits = self._decode_level_logits(target, prefix_emb, level_idx)
            idx_next = self._sample(logits, temperature, top_k, top_p)
            generated.append(idx_next)
            cursor += length
        return torch.cat(generated, dim=1)
