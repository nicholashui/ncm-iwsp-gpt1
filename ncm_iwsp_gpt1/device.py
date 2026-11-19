"""Choose the torch device for evaluation.

Writers still assign tensors on whatever device the modules are built on
(CPU, then moved). The evaluator moves both modules and packed windows.
"""

from __future__ import annotations

import torch


def resolve_device(name: str | torch.device | None = "auto") -> torch.device:
    """Return a torch device.

    ``auto`` (the default) uses CUDA when it is available, otherwise CPU.
    ``cuda`` requires a working CUDA build and raises if none is present.
    """
    if isinstance(name, torch.device):
        if name.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return name
    requested = "auto" if name is None else str(name).strip().lower()
    if requested in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {name!r}; use auto, cuda, or cpu")
