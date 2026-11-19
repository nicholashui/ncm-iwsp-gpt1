"""Local stub writer for the dry loop. No API calls and no training.

Iteration 0 leaves the seeded initialization alone. Later iterations overwrite
the GPT-2 token embedding with a sinusoid, the codebooks with axis-aligned
prototypes, and the generator input projection with a scaled identity. The
iteration index is a discrete choice, not a loss to be minimized.

This does not download GPT-2 weights and it does not touch NCM-GPT1 checkpoints.
"""

from __future__ import annotations

import torch

MODEL_SHORTHAND = "stub-closed-form"

MODEL_DESCRIPTION = (
    "Iteration 0 is the seeded random initialization and is the baseline. "
    "Later iterations write a sinusoid into the GPT-2 token embedding, one "
    "axis-aligned prototype per codebook row, and a scaled identity into the "
    "generator input projection. No corpus counts, no gradients, and no "
    "downloaded weights."
)

LAST_DESCRIPTION = MODEL_DESCRIPTION


def write_weights(tokenizer, generator, iteration: int = 0) -> None:
    """Closed-form tensor writes. Deterministic given ``iteration``."""
    global LAST_DESCRIPTION
    torch.manual_seed(0)
    step = int(iteration)
    if step <= 0:
        LAST_DESCRIPTION = (
            "Iteration 0 does not assign tensors. The model keeps the seeded "
            "PyTorch initialization. That is the random-init baseline."
        )
        return

    scale = 0.05 * step
    with torch.no_grad():
        table = tokenizer.encoder.wte.weight
        vocab, dim = table.shape
        positions = torch.arange(vocab, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.arange(dim, dtype=torch.float32).unsqueeze(0) + 1.0
        table.copy_(scale * torch.sin(positions * frequencies / dim))

        for level_index, embedding in enumerate(tokenizer.vq.embeddings):
            codebook = embedding.weight
            rows, width = codebook.shape
            codebook.zero_()
            diagonal = torch.arange(rows)
            codebook[diagonal, diagonal % width] = 1.0 + 0.01 * level_index

        projection = generator.transformer.wte
        projection.weight.zero_()
        rank = min(projection.weight.shape)
        eye = torch.arange(rank)
        projection.weight[eye, eye] = 1.0 / step
        if projection.bias is not None:
            projection.bias.zero_()

    LAST_DESCRIPTION = (
        f"Iteration {step} writes a sinusoid of scale {scale:.3f} into the GPT-2 "
        "token embedding, axis-aligned codebook prototypes, and a scaled identity "
        f"generator projection (1/{step}). These are closed forms, not a fit."
    )
