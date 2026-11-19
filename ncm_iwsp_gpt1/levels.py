"""Square residual-VQ schedules used by NCM-GPT1 and by this harness.

NCM-GPT1's 512-token window uses 23 levels: ``1**2 .. 22**2``, then 512
instead of ``23**2`` (529). Those widths sum to 4307 grid cells. Shorter
contexts use the same rule: every perfect square strictly below ``L``, then
``L`` itself.
"""

from __future__ import annotations


def square_levels(context: int) -> list[int]:
    context = int(context)
    if context < 2:
        raise ValueError("context_length must be at least 2")
    levels = []
    index = 1
    while index * index < context:
        levels.append(index * index)
        index += 1
    levels.append(context)
    if len(levels) < 2:
        raise ValueError(f"context {context} does not yield two VQ levels")
    return levels


def grid_cells(levels) -> int:
    return int(sum(int(level) for level in levels))


# Full NCM-iWSP-GPT1 / NCM-GPT1 window. 23**2 is 529, which is not used.
GPT1_CONTEXT = 512
GPT1_LEVELS = square_levels(GPT1_CONTEXT)
GPT1_GRID_CELLS = grid_cells(GPT1_LEVELS)
GPT1_CODEBOOK = 4096
GPT1_VOCAB = 50304
