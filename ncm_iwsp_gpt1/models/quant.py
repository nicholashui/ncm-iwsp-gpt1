"""Multiscale residual vector quantizer, inference only.

Adapted from HazyResearch/ncm ``models/quant.py`` (MIT, Copyright (c) 2025
hermann kumbong), commit c702742. The official training forward updates
codebooks with EMA. This harness does not. Codes are nearest neighbours of a
residual that is downsampled coarse-to-fine. ``quant_resi`` is the Phi blend
``(1-r) h + r Conv(h)`` after each upsample. Level 0 may use the learned
stride cascade that official smoke initializes as average-pool / nearest-repeat.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Phi(nn.Conv1d):
    def __init__(self, embed_dim, ratio):
        super().__init__(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        self._ratio = float(ratio)

    def forward(self, hidden):
        residual = self._ratio
        return hidden * (1.0 - residual) + super().forward(hidden) * residual


class PhiNonShared(nn.ModuleList):
    """Pick the per-level Phi whose tick is closest to ``level / (K-1)``."""

    def __init__(self, layers):
        super().__init__(layers)
        count = len(layers)
        if count == 4:
            start = 1.0 / 3.0 / count
            end = 1.0 - 1.0 / 3.0 / count
        else:
            start = 1.0 / 2.0 / count
            end = 1.0 - 1.0 / 2.0 / count
        if count == 1:
            self.ticks = [0.5]
        else:
            step = (end - start) / (count - 1)
            self.ticks = [start + i * step for i in range(count)]

    def __getitem__(self, at):
        if isinstance(at, int):
            return super().__getitem__(at)
        nearest = min(range(len(self.ticks)), key=lambda i: abs(self.ticks[i] - float(at)))
        return super().__getitem__(nearest)


def _plan_cascade_strides(total_stride, base=4):
    remaining = int(total_stride)
    strides = []
    while remaining > 1:
        factor = min(int(base), remaining)
        while factor > 1 and remaining % factor != 0:
            factor -= 1
        if factor <= 1:
            strides.append(remaining)
            remaining = 1
        else:
            strides.append(factor)
            remaining //= factor
    return strides or [1]


def _make_learned_down(embed_dim, stride):
    conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        index = torch.arange(embed_dim)
        conv.weight[index, index, :] = 1.0 / float(stride)
        conv.bias.zero_()
    return conv


def _make_learned_up(embed_dim, stride):
    conv = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=stride, stride=stride, bias=True)
    with torch.no_grad():
        conv.weight.zero_()
        index = torch.arange(embed_dim)
        conv.weight[index, index, :] = 1.0
        conv.bias.zero_()
    return conv


def _make_cascade(embed_dim, total_stride, base, kind):
    factory = _make_learned_down if kind == "down" else _make_learned_up
    strides = _plan_cascade_strides(total_stride, base)
    if kind == "up":
        strides = list(reversed(strides))
    return nn.Sequential(*[factory(embed_dim, stride) for stride in strides])


class MultiscaleResidualQuantizer(nn.Module):
    def __init__(
        self,
        vocab_sizes,
        embed_dim,
        levels,
        quant_resi_ratio=0.5,
        quant_resi_enabled=True,
        learned_l1_sampling=True,
        learned_l1_cascade=True,
        learned_l1_cascade_base=4,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.levels = [int(level) for level in levels]
        self.vqvae_vocab_sizes = [int(size) for size in vocab_sizes]
        if len(self.vqvae_vocab_sizes) != len(self.levels):
            raise ValueError("one codebook size is required per VQ level")
        if self.levels[-1] < self.levels[0]:
            raise ValueError("the last level is the full latent length")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(size, self.embed_dim) for size in self.vqvae_vocab_sizes]
        )
        # Official codebooks are EMA parameters, not Adam parameters. Here they
        # are ordinary tensors the writer may overwrite. Nothing in forward updates them.
        for embedding in self.embeddings:
            embedding.weight.data.normal_()
            embedding.weight.requires_grad_(False)

        self.quant_resi_enabled = bool(quant_resi_enabled)
        if self.quant_resi_enabled:
            ratio = float(quant_resi_ratio)
            if abs(ratio) <= 1e-6:
                self.quant_resi = None
                self.quant_resi_enabled = False
            else:
                self.quant_resi = PhiNonShared(
                    [Phi(self.embed_dim, ratio) for _ in self.levels]
                )
        else:
            self.quant_resi = None
        self._qr_num_levels = len(self.levels)

        self.learned_l1_sampling = bool(learned_l1_sampling)
        if self.learned_l1_sampling:
            full = self.levels[-1]
            coarse = self.levels[0]
            if coarse <= 0 or full % coarse != 0:
                raise ValueError(f"learned L1 sampling needs L={full} divisible by level0={coarse}")
            stride = full // coarse
            base = int(learned_l1_cascade_base)
            if learned_l1_cascade:
                self.learned_down_l1 = _make_cascade(self.embed_dim, stride, base, "down")
                self.learned_up_l1 = _make_cascade(self.embed_dim, stride, base, "up")
            else:
                self.learned_down_l1 = _make_learned_down(self.embed_dim, stride)
                self.learned_up_l1 = _make_learned_up(self.embed_dim, stride)
        else:
            self.learned_down_l1 = None
            self.learned_up_l1 = None

    def _apply_quant_resi(self, hidden, level_idx):
        if not self.quant_resi_enabled:
            return hidden
        as_channels = hidden.transpose(-1, -2).contiguous()
        at = level_idx / max(self._qr_num_levels - 1, 1)
        mixed = self.quant_resi[at](as_channels)
        return mixed.transpose(-1, -2).contiguous()

    def _distances(self, flat, weight):
        return (
            torch.sum(flat**2, dim=1, keepdim=True)
            + torch.sum(weight**2, dim=1)
            - 2 * torch.matmul(flat, weight.t())
        )

    def _down_for_quant(self, latent, level_idx, target_len):
        channels_first = latent.transpose(-1, -2)
        if level_idx == 0 and self.learned_l1_sampling:
            return self.learned_down_l1(channels_first).transpose(-1, -2).contiguous()
        reduced = F.interpolate(channels_first, size=(target_len,), mode="area")
        return reduced.transpose(-1, -2).contiguous()

    def _up_to_full(self, codes, level_idx, target_len):
        channels_first = codes.transpose(-1, -2)
        if level_idx == 0 and self.learned_l1_sampling:
            return self.learned_up_l1(channels_first).transpose(-1, -2).contiguous()
        expanded = F.interpolate(channels_first, size=(target_len,), mode="linear", align_corners=False)
        return expanded.transpose(-1, -2).contiguous()

    @torch.no_grad()
    def z_to_idx(self, latent):
        latent = latent.float()
        batch, length, channels = latent.shape
        if length != self.levels[-1]:
            raise ValueError(f"latent length {length} != finest level {self.levels[-1]}")
        residual = latent
        index_chunks = []
        last = len(self.levels) - 1
        for level_idx, level in enumerate(self.levels):
            if level_idx != last:
                flat = self._down_for_quant(residual, level_idx, level).reshape(-1, channels)
            else:
                flat = residual.reshape(-1, channels)
            distances = self._distances(flat, self.embeddings[level_idx].weight)
            indices = torch.argmin(distances, dim=1).view(batch, level)
            index_chunks.append(indices)
            quantized = self.embeddings[level_idx](indices)
            if level_idx != last:
                quantized = self._up_to_full(quantized, level_idx, length)
            quantized = self._apply_quant_resi(quantized, level_idx)
            residual = residual - quantized
        return torch.cat(index_chunks, dim=-1)

    @torch.no_grad()
    def idx_to_z_q(self, indices):
        reconstructed = torch.zeros(
            indices.shape[0], self.levels[-1], self.embed_dim, device=indices.device
        )
        offset = 0
        last = len(self.levels) - 1
        for level_idx, level in enumerate(self.levels):
            chosen = self.embeddings[level_idx](indices[:, offset : offset + level])
            if level_idx != last:
                chosen = self._up_to_full(chosen, level_idx, self.levels[-1])
            chosen = self._apply_quant_resi(chosen, level_idx)
            reconstructed = reconstructed + chosen
            offset += level
        return reconstructed

    @torch.no_grad()
    def idx_to_ncm_input(self, indices):
        """Latents the generator reads: each coarser residual, upsampled, then area-pooled to the next level."""
        running = torch.zeros(indices.shape[0], self.levels[-1], self.embed_dim, device=indices.device)
        pieces = []
        offset = 0
        for level_idx in range(len(self.levels) - 1):
            level = self.levels[level_idx]
            chosen = self.embeddings[level_idx](indices[:, offset : offset + level])
            chosen = self._up_to_full(chosen, level_idx, self.levels[-1])
            chosen = self._apply_quant_resi(chosen, level_idx)
            running = running + chosen
            nxt = self.levels[level_idx + 1]
            pooled = F.interpolate(running.transpose(-1, -2), size=(nxt,), mode="area")
            pieces.append(pooled.transpose(-1, -2).contiguous())
            offset += level
        return torch.cat(pieces, dim=1)

    @torch.no_grad()
    def idx_to_zhats(self, indices):
        """Cumulative latent after each level, for qualitative coarse-to-fine decodes."""
        running = torch.zeros(indices.shape[0], self.levels[-1], self.embed_dim, device=indices.device)
        snapshots = []
        offset = 0
        last = len(self.levels) - 1
        for level_idx, level in enumerate(self.levels):
            chosen = self.embeddings[level_idx](indices[:, offset : offset + level])
            if level_idx != last:
                chosen = self._up_to_full(chosen, level_idx, self.levels[-1])
            chosen = self._apply_quant_resi(chosen, level_idx)
            running = running + chosen
            if level_idx != last:
                pooled = F.interpolate(
                    running.transpose(-1, -2), size=(level,), mode="area"
                ).transpose(-1, -2)
                snapshots.append(pooled.contiguous())
            else:
                snapshots.append(running)
            offset += level
        return snapshots

    @torch.no_grad()
    def get_next_autoregressive_input(self, level_idx, running, level_codes):
        if level_idx != len(self.levels) - 1:
            upsampled = self._up_to_full(level_codes, level_idx, self.levels[-1])
            upsampled = self._apply_quant_resi(upsampled, level_idx)
            running = running + upsampled
            nxt = self.levels[level_idx + 1]
            pooled = F.interpolate(running.transpose(-1, -2), size=(nxt,), mode="area")
            return running, pooled.transpose(-1, -2).contiguous()
        level_codes = self._apply_quant_resi(level_codes, level_idx)
        running = running + level_codes
        return running, running
