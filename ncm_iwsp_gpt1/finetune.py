"""Gradient fine-tune of the untied NCM-iWSP-GPT1 generator.

The tokenizer stays frozen. Its encode path is a nearest-neighbour quantizer
with no straight-through estimator, so the optimizer updates the generator
only: untied ``Linear`` heads, no post-upsample convolution. This module is
separate from the writer loop. The writer files still do not train.

Checkpoints, logs, and reports are written only to ``--out``. Paths under
``/kaggle/input``, inside an input file's directory, or inside a v10 / source
export directory are refused.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ncm_iwsp_gpt1.build import build_models
from ncm_iwsp_gpt1.device import resolve_device
from ncm_iwsp_gpt1.dialogue import (
    DIALOGUE_FORMAT,
    DIALOGUE_PACK_FORMAT,
    DIALOGUE_QUALITY_NOTE,
    TRUNCATION_POLICY,
    dialogue_settings,
    normalize_history_placement,
)
from ncm_iwsp_gpt1.packing import refuse_banned_label
from ncm_iwsp_gpt1.prefix_protocol import (
    batch_prefix_tokens,
    legacy_prefix_warning,
    load_generator_state,
    normalize_prefix_source,
    prefix_metadata_problems,
    recorded_prefix,
)

FINETUNE_FORMAT = "ncm-iwsp-gpt1-finetune"
PACKED_FILENAME = "packed_windows.pt"

# Conservative for a 16 GiB T4. NCM-GPT1 v10 (tied heads, ~101.69M) used
# per-card batch 2 with checkpointing at about 3.3 GiB. This generator is
# 154,666,240 parameters because the heads are untied Linear(768, 4096).
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM = 1
DEFAULT_PEAK_LR = 3e-5
DEFAULT_FLOOR_LR = 3e-6
DEFAULT_WARMUP = 100
DEFAULT_MAX_STEPS = 100_000
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_BETAS = (0.9, 0.95)
DEFAULT_ALPHA = 0.0
DEFAULT_MID_AUX = 0.0
DEFAULT_CORRUPT_P_MAX = 0.8
DEFAULT_CORRUPT_P_MIN = 0.2
DEFAULT_VAL_EVERY = 1000
DEFAULT_VAL_WINDOWS = 256
DEFAULT_VAL_BATCH_SIZE = 1
DEFAULT_SAVE_EVERY = 1000
DEFAULT_LOG_EVERY = 50
DEFAULT_WORKERS = 0

FORBIDDEN_DIR_NAMES = frozenset(
    {
        "ncm-gpt1-kaggle-100k",
        "ncm-gpt-1-train",
        "kaggle-v10-weights",
        "ncm-iwsp-gpt1-init",
        "ncm-iwsp-gpt1-finetune-v3",
        "ncm-iwsp-gpt1-prefix-level0",
        "ncm-iwsp-gpt1-finetune-phase2",
    }
)

_FOREIGN_BITS = ("post_upsample_convs.", ".proj.", "log_inv_T")


class FinetuneError(ValueError):
    """The fine-tune run cannot start or cannot continue safely."""


def loss_is_alpha0(alpha, mid_aux) -> bool:
    """True when the optimized scalar is the paper per-position mean."""
    return float(alpha) == 0.0 and float(mid_aux) == 0.0


def loss_note(alpha, mid_aux) -> str:
    """Explain whether ``loss`` may be read next to an alpha=0 curve.

    ``ce_first``, ``ce_mid``, and ``ce_last`` are always raw per-level means.
    ``ce_mid`` is the single middle predicted level (index ``len // 2``).
    ``mid_aux`` averages a different slice, the central third ``[n//3, 2n//3)``.
    """
    if loss_is_alpha0(alpha, mid_aux):
        return (
            "Optimized loss is the alpha=0 mean of per-position cross-entropy "
            "(every predicted code cell has the same weight). mid_aux is 0. "
            "ce_first, ce_mid, and ce_last are the raw means of the first, middle, "
            "and last predicted levels and do not include a mid_aux term. "
            "NCM-GPT1 v10 logged a different scalar once alpha or mid_aux left 0 "
            "(alpha 0.5, and mid_aux 0.5 on later kernels). Compare those runs on "
            "ce_first, ce_mid, and ce_last."
        )
    return (
        f"Optimized loss includes alpha={float(alpha)} and mid_aux={float(mid_aux)}. "
        "Do not compare this loss to alpha=0 curves. "
        "ce_first, ce_mid, and ce_last remain raw per-level means "
        "(ce_mid is index len//2, not the mid_aux central-third band) "
        "and are the numbers to compare."
    )


def describe_run(cfg: dict, options: dict | None = None) -> dict:
    """Plan a run from a model config. Does not build the modules."""
    options = dict(options or {})
    tokenizer = cfg["tokenizer"]
    generator = cfg["generator"]
    alpha = float(options.get("alpha", DEFAULT_ALPHA))
    mid_aux = float(options.get("mid_aux", DEFAULT_MID_AUX))
    levels = [int(level) for level in tokenizer["levels"]]
    prefix_len = int(generator.get("prefix_len", 1))
    prefix_source = normalize_prefix_source(generator.get("prefix_source", "eos"))
    plan = {
        "system": "NCM-iWSP-GPT1",
        "config_name": cfg["name"],
        "context_length": int(tokenizer["context_length"]),
        "levels": levels,
        "n_levels": len(levels),
        "grid_cells": int(sum(levels)),
        "codebook_size": int(tokenizer["codebook_size"]),
        "n_layer": int(generator["n_layer"]),
        "n_embd": int(generator["n_embd"]),
        "n_head": int(generator["n_head"]),
        "predict_level0": False,
        "prefix_len": prefix_len,
        "prefix_source": prefix_source,
        "prefix_pad": "right" if prefix_source == "window" else None,
        "generator_sequence_length": prefix_len + int(sum(levels[1:])),
        "retrain_prefix_head": False,
        "untied_heads": True,
        "post_upsample_conv": False,
        "tokenizer_frozen": True,
        "optimized": "generator",
        "batch_size_per_gpu": int(options.get("batch_size", DEFAULT_BATCH_SIZE)),
        "grad_accum": int(options.get("grad_accum", DEFAULT_GRAD_ACCUM)),
        "peak_lr": float(options.get("peak_lr", DEFAULT_PEAK_LR)),
        "floor_lr": float(options.get("floor_lr", DEFAULT_FLOOR_LR)),
        "warmup": int(options.get("warmup", DEFAULT_WARMUP)),
        "max_steps": int(options.get("max_steps", DEFAULT_MAX_STEPS)),
        "weight_decay": float(options.get("weight_decay", DEFAULT_WEIGHT_DECAY)),
        "alpha": alpha,
        "mid_aux": mid_aux,
        "corrupt_p_max": float(options.get("corrupt_p_max", DEFAULT_CORRUPT_P_MAX)),
        "corrupt_p_min": float(options.get("corrupt_p_min", DEFAULT_CORRUPT_P_MIN)),
        "val_windows": int(options.get("val_windows", DEFAULT_VAL_WINDOWS)),
        "gradient_checkpointing": bool(options.get("gradient_checkpointing", True)),
        "chunked_block_attn": bool(options.get("chunked_block_attn", True)),
        "amp": "fp16+GradScaler on CUDA capability below 8 (T4); bf16 on capability 8+",
        "loss_comparable_to_alpha0": loss_is_alpha0(alpha, mid_aux),
        "loss_note": loss_note(alpha, mid_aux),
        "expected_init_files": ["tokenizer.pt", "generator.pt"],
        "expected_packed_names": [
            "artifacts/packed_windows.pt",
            "packed_windows.pt",
        ],
        "writes": "only the --out directory",
    }
    if prefix_source == "window" and not (
        isinstance(cfg.get("dialogue"), dict) and cfg["dialogue"].get("enabled")
    ):
        plan["hyperparameter_note"] = (
            "peak_lr, floor_lr, warmup, max_steps, batch_size, grad_accum, "
            "corruption, and validation frequency are the existing finetune CLI "
            "defaults. They are not a chosen Phase 2 recipe, and this plan does "
            "not start training. The prefix head is not retrained. A later run "
            "must pass --confirm and must write to a new --out directory."
        )
    return plan


def describe_dialogue_run(cfg: dict, options: dict | None = None) -> dict:
    """Plan a dialogue fine-tune. Does not build modules and does not train.

    Hyperparameters are the existing finetune CLI defaults. They are not an
    approved phase-3 recipe. The dataset id is not in the config.
    """
    if not isinstance(cfg.get("dialogue"), dict) or not cfg["dialogue"].get("enabled"):
        raise FinetuneError(
            "finetune-dialogue needs a phase-3 config with dialogue.enabled true. "
            "Phase 1 and phase 2 stay on finetune."
        )
    plan = describe_run(cfg, options)
    settings = dialogue_settings(cfg)
    plan.update(
        {
            "dialogue": True,
            "checkpoint_format": DIALOGUE_FORMAT,
            "history_placement": settings["history_placement"],
            "truncation": settings["truncation"],
            "pad_side": settings["pad_side"],
            "truncation_policy": TRUNCATION_POLICY,
            "dataset": None,
            "dataset_note": (
                "No dialogue corpus is selected. pack-dialogue and finetune-dialogue "
                "both require --dataset-id at the command line. The id is not read "
                "from this config. data.dataset on the phase-3 yaml is unset."
            ),
            "retrain_prefix_head": False,
            "prefix_head_note": (
                "The prefix head is not retrained. --retrain-prefix-head is an error, "
                "not a silent skip and not a second training loop."
            ),
            "special_role_tokens": False,
            "code_ce_is_chat_quality": False,
            "quality_note": DIALOGUE_QUALITY_NOTE,
            "warm_start": (
                "Pass a phase-2 generator checkpoint (format ncm-iwsp-gpt1-finetune) "
                "or a converted generator.pt as --generator, with --resume none. "
                "Resume continues only a dialogue-format checkpoint in the new --out."
            ),
            "hyperparameter_note": (
                "peak_lr, floor_lr, warmup, max_steps, batch_size, grad_accum, "
                "corruption, and validation frequency are the existing finetune CLI "
                "defaults. They are not an approved Phase 3 recipe, and this plan does "
                "not start training. The tokenizer stays frozen. predict_level0 stays "
                "false. The prefix head is not retrained. A later run must pass "
                "--confirm, --dataset-id, and a new --out directory."
            ),
            "writes": "only the --out directory",
        }
    )
    return plan


def level_weights(levels, alpha: float, device=None, dtype=torch.float32) -> torch.Tensor:
    """``w_k = ell_k^{-alpha} / sum_j ell_j^{1-alpha}``.

    ``sum_k ell_k w_k = 1``. Alpha 0 gives every code cell the same weight.
    """
    lengths = torch.tensor([float(level) for level in levels], dtype=dtype, device=device)
    denom = lengths.pow(1.0 - float(alpha)).sum().clamp(min=1e-12)
    return lengths.pow(-float(alpha)) / denom


def mid_level_slice(n_levels: int) -> tuple[int, int]:
    """Central third of the predicted levels, half-open.

    Twenty-two predicted levels use indices ``[7, 14)``.
    """
    count = int(n_levels)
    if count <= 0:
        return 0, 0
    if count < 3:
        return 0, count
    start = count // 3
    end = (2 * count) // 3
    if end <= start:
        end = min(count, start + 1)
    return start, end


def generator_ce_loss(logits, targets, levels, vocab_sizes, alpha: float = 0.0, mid_aux: float = 0.0) -> dict:
    """Per-level cross-entropy on untied-head logits.

    ``loss`` is the quantity to differentiate. ``ce_first``, ``ce_mid``, and
    ``ce_last`` are raw level means either way. ``length_weighted_ce`` is the
    alpha=0 position mean, also either way.
    """
    levels = [int(level) for level in levels]
    vocab_sizes = [int(size) for size in vocab_sizes]
    if sum(levels) != int(targets.shape[1]):
        raise FinetuneError(
            f"target length {targets.shape[1]} != sum of predicted levels {sum(levels)}"
        )
    if logits.shape[1] != targets.shape[1]:
        raise FinetuneError(
            f"logit length {logits.shape[1]} != target length {targets.shape[1]}"
        )
    weights = level_weights(levels, alpha, device=logits.device)
    elbo_weights = level_weights(levels, 0.0, device=logits.device)
    weighted = logits.new_zeros(())
    elbo = logits.new_zeros(())
    level_sums = []
    level_counts = []
    offset = 0
    for index, (length, vocab) in enumerate(zip(levels, vocab_sizes)):
        piece = logits[:, offset : offset + length, :vocab].float().reshape(-1, vocab)
        gold = targets[:, offset : offset + length].reshape(-1)
        ce = F.cross_entropy(piece, gold, reduction="none")
        weighted = weighted + (ce * weights[index]).sum()
        elbo = elbo + (ce * elbo_weights[index]).sum()
        level_sums.append(ce.sum())
        level_counts.append(int(ce.numel()))
        offset += length
    batch = int(targets.shape[0])
    loss = weighted / batch
    means = [level_sums[index] / level_counts[index] for index in range(len(levels))]
    coef = float(mid_aux or 0.0)
    if coef != 0.0 and means:
        start, end = mid_level_slice(len(means))
        loss = loss + coef * torch.stack(means[start:end]).mean()
    parts = [float(value.detach()) for value in means]
    mid_index = len(parts) // 2
    return {
        "loss": loss,
        "ce_first": parts[0],
        "ce_mid": parts[mid_index],
        "ce_last": parts[-1],
        "ce_mid_index": mid_index,
        "ce_levels": parts,
        "mean_level_ce": float(sum(parts) / len(parts)),
        "length_weighted_ce": float((elbo / batch).detach()),
        "level_sums": [float(value.detach()) for value in level_sums],
        "level_counts": level_counts,
        "weighted_sum": float(weighted.detach()),
        "elbo_sum": float(elbo.detach()),
    }


def corrupt_codes(indices, levels, vocab_sizes, p_max: float, p_min: float):
    """Replace input codes, coarse ``p_max`` down to fine ``p_min``. Labels stay put."""
    out = indices.clone()
    count = len(levels)
    if count == 0:
        return out
    if count == 1:
        rates = [float(p_max)]
    else:
        span = float(p_max) - float(p_min)
        rates = [float(p_max) - span * index / (count - 1) for index in range(count)]
    offset = 0
    for rate, length, vocab in zip(rates, levels, vocab_sizes):
        length = int(length)
        vocab = int(vocab)
        chunk = out[:, offset : offset + length]
        mask = torch.rand(chunk.shape, device=indices.device) < float(rate)
        noise = torch.randint(0, vocab, chunk.shape, device=indices.device)
        out[:, offset : offset + length] = torch.where(mask, noise, chunk)
        offset += length
    return out


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def refuse_finetune_output(out, sources) -> None:
    """Refuse writes into inputs, Kaggle inputs, or known source exports."""
    destination = Path(out).resolve()
    lowered = [part.lower() for part in destination.parts]
    if "kaggle" in lowered:
        index = lowered.index("kaggle")
        if index + 1 < len(lowered) and lowered[index + 1] == "input":
            raise FinetuneError(
                f"refusing to write {destination} under /kaggle/input. "
                "Checkpoints, logs, and reports go to /kaggle/working or another --out "
                "outside the input datasets."
            )
    for part in destination.parts:
        if part.lower() in FORBIDDEN_DIR_NAMES:
            raise FinetuneError(
                f"refusing to write {destination}: '{part}' is a source-weight or v10 "
                "artifact directory. Write to /kaggle/working, or to a new folder, so "
                "the converted init and the v10 export stay unchanged."
            )
    for source in sources or []:
        if source is None:
            continue
        source_path = Path(source).resolve()
        parent = source_path if source_path.is_dir() else source_path.parent
        if destination == source_path or _is_inside(destination, parent):
            raise FinetuneError(
                f"refusing to write {destination} inside {parent}. "
                "That directory holds an input (converted weights or packed windows)."
            )


def _read_blob(path: Path):
    if not path.is_file():
        raise FinetuneError(f"file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _jsonable(value):
    if torch.is_tensor(value):
        if value.numel() > 8:
            return {"tensor_shape": [int(size) for size in value.shape], "dtype": str(value.dtype)}
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _strip_module_prefix(state: dict) -> dict:
    keys = [str(key) for key in state.keys()]
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def state_and_meta(blob, label: str) -> tuple[dict, dict]:
    """Accept ``{model, step, converted_from}`` or a raw state dict."""
    if not isinstance(blob, dict) or not blob:
        raise FinetuneError(f"{label} checkpoint is empty")
    meta = {}
    state = None
    if isinstance(blob.get("model"), dict):
        state = blob["model"]
        for key, value in blob.items():
            if key in {"model", "optimizer", "scaler"}:
                continue
            meta[key] = value
    elif label == "tokenizer" and isinstance(blob.get("tokenizer"), dict):
        state = blob["tokenizer"]
        meta = {key: value for key, value in blob.items() if key not in {"tokenizer", "generator"}}
    elif label == "generator" and isinstance(blob.get("generator"), dict):
        state = blob["generator"]
        meta = {key: value for key, value in blob.items() if key not in {"tokenizer", "generator"}}
    elif all(torch.is_tensor(value) for value in blob.values()):
        state = blob
    if state is None:
        raise FinetuneError(
            f"{label} checkpoint has no 'model' state dict. "
            "Expected the converted init written as {model, step, converted_from}."
        )
    state = _strip_module_prefix(state)
    if not all(torch.is_tensor(value) for value in state.values()):
        raise FinetuneError(f"{label} state dict contains a non-tensor")
    return state, meta


def assert_untied_generator_state(state: dict) -> None:
    """The fine-tune generator is the expanded Linear heads, already converted."""
    foreign = [key for key in state if any(bit in key for bit in _FOREIGN_BITS)]
    if foreign:
        raise FinetuneError(
            "generator checkpoint still has tied-head or post-upsample tensors "
            f"({foreign[0]}). Fine-tuning reads the converted NCM-iWSP-GPT1 init, "
            "whose heads are lm_heads.*.weight with shape (codebook, n_embd). "
            "Use tokenizer.pt and generator.pt from the converted init directory."
        )
    heads = [key for key in state if key.startswith("lm_heads.") and key.endswith(".weight")]
    if not heads:
        raise FinetuneError(
            "generator state has no lm_heads.*.weight. "
            "The converted init stores one untied Linear per predicted level."
        )


def load_module_init(path, module, label: str) -> dict:
    state, meta = state_and_meta(_read_blob(Path(path)), label)
    if label == "generator":
        assert_untied_generator_state(state)
    try:
        if label == "generator":
            load_generator_state(module, state)
        else:
            module.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise FinetuneError(f"{label} init does not match this config: {exc}") from exc
    return meta


def _check_width(windows: torch.Tensor, context_length: int, path: Path) -> None:
    if windows.ndim != 2 or int(windows.shape[1]) != int(context_length):
        raise FinetuneError(
            f"{path} has windows of shape {tuple(windows.shape)}; "
            f"this config expects [N, {int(context_length)}]."
        )


def split_heldout(windows: torch.Tensor, n_val: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic tail, matching the v10 held-out split."""
    count = int(windows.shape[0])
    if count < 2:
        raise FinetuneError(f"need at least 2 packed windows to hold out validation, found {count}")
    hold = int(n_val)
    if hold <= 0:
        raise FinetuneError("val_windows must be positive so validation CE is logged")
    hold = min(hold, count - 1)
    return windows[:-hold], windows[-hold:]


def load_training_windows(path, context_length: int, val_windows: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Load v10 ``packed_windows.pt`` or this repo's visible cache.

    The hidden split of an iWSP cache is not returned and is not trained on.
    v10 files store one ``windows`` tensor; the tail is validation.
    """
    path = Path(path)
    blob = _read_blob(path)
    if not isinstance(blob, dict):
        raise FinetuneError(f"{path} is not a packed-window dict")
    raw_meta = blob.get("meta") if isinstance(blob.get("meta"), dict) else {}
    if raw_meta.get("dialogue") or raw_meta.get("format") == DIALOGUE_PACK_FORMAT:
        raise FinetuneError(
            f"{path} is a dialogue pack (format {DIALOGUE_PACK_FORMAT}). "
            "Use finetune-dialogue. The book-window finetune command will not read it."
        )
    if torch.is_tensor(blob.get("windows")):
        windows = blob["windows"]
        _check_width(windows, context_length, path)
        train, val = split_heldout(windows, val_windows)
        kind = "v10_packed_windows"
    elif torch.is_tensor(blob.get("visible_train")) and torch.is_tensor(blob.get("visible")):
        train = blob["visible_train"]
        val = blob["visible"]
        _check_width(train, context_length, path)
        _check_width(val, context_length, path)
        if train.shape[0] < 1 or val.shape[0] < 1:
            raise FinetuneError(f"{path} needs non-empty visible_train and visible tensors")
        kind = "iwsp_visible_splits"
    else:
        raise FinetuneError(
            f"{path} is not a packed-window file. Expected a v10 dict with tensor "
            f"'{PACKED_FILENAME}' key 'windows' (usually artifacts/{PACKED_FILENAME}), "
            "or this repo's cache with visible_train and visible."
        )
    if raw_meta.get("L") is not None and int(raw_meta["L"]) != int(context_length):
        raise FinetuneError(
            f"{path} meta L={raw_meta['L']} does not match context {int(context_length)}"
        )
    meta = {"kind": kind, "path": str(path)}
    meta.update(_jsonable(raw_meta))
    return train.contiguous(), val.contiguous(), meta


def _check_prefix_rows(prefix, prefix_len, rows, path, label: str) -> None:
    if (
        not torch.is_tensor(prefix)
        or prefix.ndim != 2
        or int(prefix.shape[0]) != int(rows)
        or int(prefix.shape[1]) != int(prefix_len)
    ):
        shape = tuple(prefix.shape) if torch.is_tensor(prefix) else type(prefix).__name__
        raise FinetuneError(
            f"{path} {label} has shape {shape}; expected [{int(rows)}, {int(prefix_len)}]."
        )


def load_dialogue_windows(path, context_length: int, prefix_len: int, dataset_id, placement: str):
    """Load a ``pack-dialogue`` cache. BookCorpus files are refused.

    The dataset id must be passed again at train time. It is not taken from a
    config default. Hidden rows are not returned.
    """
    path = Path(path)
    blob = _read_blob(path)
    if not isinstance(blob, dict):
        raise FinetuneError(f"{path} is not a dialogue-window dict")
    raw_meta = blob.get("meta") if isinstance(blob.get("meta"), dict) else {}
    if raw_meta.get("format") != DIALOGUE_PACK_FORMAT or not raw_meta.get("dialogue"):
        raise FinetuneError(
            f"{path} is not a pack-dialogue cache. BookCorpus packed windows stay on finetune. "
            "finetune-dialogue does not default to a corpus."
        )
    if not str(raw_meta.get("license_note", "")).strip():
        raise FinetuneError(f"{path} is missing a license or authorization note")
    if raw_meta.get("special_role_tokens") or raw_meta.get("add_special_role_tokens"):
        raise FinetuneError(
            f"{path} requested special role tokens. This build keeps GPT-2 BPE and encoder.wte unchanged."
        )
    wanted = str(dataset_id or "").strip()
    if not wanted:
        raise FinetuneError("finetune-dialogue needs --dataset-id. No dialogue corpus is selected.")
    try:
        refuse_banned_label(wanted)
    except ValueError as exc:
        raise FinetuneError(str(exc)) from exc
    found_id = str(raw_meta.get("dataset", "")).strip()
    if found_id != wanted:
        raise FinetuneError(
            f"packed dataset id {found_id!r} does not match --dataset-id {wanted!r}."
        )
    found_placement = normalize_history_placement(raw_meta.get("history_placement"))
    if found_placement != normalize_history_placement(placement):
        raise FinetuneError(
            f"packed history_placement {found_placement!r} does not match the config "
            f"({normalize_history_placement(placement)!r}). Repack, or set the config to that placement."
        )
    if int(raw_meta.get("context_length", context_length)) != int(context_length):
        raise FinetuneError(
            f"{path} context_length {raw_meta.get('context_length')} does not match {int(context_length)}"
        )
    if int(raw_meta.get("prefix_len", -1)) != int(prefix_len):
        raise FinetuneError(
            f"{path} prefix_len {raw_meta.get('prefix_len')} does not match {int(prefix_len)}"
        )
    if not (torch.is_tensor(blob.get("visible_train")) and torch.is_tensor(blob.get("visible"))):
        raise FinetuneError(f"{path} needs visible_train and visible windows")
    train = blob["visible_train"]
    val = blob["visible"]
    train_prefix = blob.get("prefix_visible_train")
    val_prefix = blob.get("prefix_visible")
    _check_width(train, context_length, path)
    _check_width(val, context_length, path)
    _check_prefix_rows(train_prefix, prefix_len, train.shape[0], path, "prefix_visible_train")
    _check_prefix_rows(val_prefix, prefix_len, val.shape[0], path, "prefix_visible")
    if int(train.shape[0]) < 1 or int(val.shape[0]) < 1:
        raise FinetuneError(f"{path} needs non-empty visible_train and visible tensors")
    meta = {"kind": "dialogue_windows", "path": str(path)}
    meta.update(_jsonable(raw_meta))
    return (
        train.contiguous(),
        val.contiguous(),
        train_prefix.contiguous(),
        val_prefix.contiguous(),
        meta,
    )


class _WindowDataset(Dataset):
    def __init__(self, windows: torch.Tensor):
        self.windows = windows

    def __len__(self):
        return int(self.windows.shape[0])

    def __getitem__(self, index):
        return self.windows[index].long()


class _PrefixWindowDataset(Dataset):
    """Window tokens plus an explicit generator prefix. Dialogue runs only."""

    def __init__(self, windows: torch.Tensor, prefixes: torch.Tensor):
        if int(windows.shape[0]) != int(prefixes.shape[0]):
            raise FinetuneError("dialogue prefix rows do not match windows")
        self.windows = windows
        self.prefixes = prefixes

    def __len__(self):
        return int(self.windows.shape[0])

    def __getitem__(self, index):
        return self.windows[index].long(), self.prefixes[index].long()


def _optimizer_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _make_loader(
    windows,
    batch_size: int,
    seed: int,
    rank: int,
    world_size: int,
    workers: int,
    pin: bool,
    shuffle: bool = True,
    prefixes=None,
):
    if prefixes is None:
        dataset = _WindowDataset(windows)
    else:
        dataset = _PrefixWindowDataset(windows, prefixes)
    if len(dataset) < int(batch_size):
        raise FinetuneError(
            f"need at least {int(batch_size)} windows on this rank's loader, found {len(dataset)}"
        )
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(rank))
    kwargs = {
        "batch_size": int(batch_size),
        "num_workers": int(workers),
        "pin_memory": bool(pin),
    }
    if int(world_size) > 1:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=True,
            seed=int(seed),
            drop_last=True,
        )
        return DataLoader(dataset, sampler=sampler, **kwargs), sampler
    return (
        DataLoader(dataset, shuffle=bool(shuffle), drop_last=False, generator=generator, **kwargs),
        None,
    )


def _forever(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise FinetuneError("the training loader produced no batches")
        epoch += 1


def lr_at(step: int, warmup: int, peak: float, floor: float, max_steps: int) -> float:
    """Linear warmup, then cosine decay from peak to floor."""
    step = max(int(step), 0)
    warmup = max(int(warmup), 0)
    peak = float(peak)
    floor = float(floor)
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if int(max_steps) <= warmup:
        return peak
    progress = (step - warmup) / max(int(max_steps) - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return floor + 0.5 * (1.0 + math.cos(math.pi * progress)) * (peak - floor)


def _set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def setup_amp(device: torch.device):
    """bf16 on capability 8+. fp16 and a GradScaler on T4 (sm_75). fp32 on CPU."""
    if device.type != "cuda":
        return nullcontext(), None, "fp32"
    capability = torch.cuda.get_device_capability(device)
    if capability[0] >= 8 and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16), None, "bf16"
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16),
        torch.amp.GradScaler("cuda"),
        "fp16",
    )


def _backward(loss, scaler) -> None:
    if scaler is None:
        loss.backward()
        return
    scaler.scale(loss).backward()


def _step(optimizer, parameters, clip: float, scaler) -> float:
    if scaler is None:
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(clip))
        optimizer.step()
        return float(grad_norm)
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(clip))
    scaler.step(optimizer)
    scaler.update()
    return float(grad_norm)


def _checksum(module) -> float:
    total = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for param in module.parameters():
            total = total + param.detach().double().sum()
    return float(total)


def _unwrap(model):
    return getattr(model, "module", model)


def _batch_tensors(batch, device):
    """A book batch is a token tensor. A dialogue batch is ``(tokens, prefix)``."""
    prefix = None
    if (
        isinstance(batch, (list, tuple))
        and len(batch) == 2
        and torch.is_tensor(batch[0])
        and torch.is_tensor(batch[1])
    ):
        tokens, prefix = batch
    elif torch.is_tensor(batch):
        tokens = batch
    else:
        raise FinetuneError("a fine-tune batch must be a token tensor or (tokens, prefix)")
    non_blocking = device.type == "cuda"
    tokens = tokens.to(device, non_blocking=non_blocking)
    if prefix is not None:
        prefix = prefix.to(device, non_blocking=non_blocking)
    return tokens, prefix


def _prepare_batch(
    vqvae,
    generator,
    tokens,
    eos_id: int,
    corrupt: bool,
    p_max: float,
    p_min: float,
    pad_id: int,
    prefix_tokens=None,
):
    with torch.no_grad():
        indices = vqvae.encode(tokens)
        levels = [int(level) for level in vqvae.vq.levels]
        vocabs = [int(size) for size in vqvae.vq.vqvae_vocab_sizes]
        source = indices
        if corrupt and (float(p_max) > 0.0 or float(p_min) > 0.0):
            source = corrupt_codes(indices, levels, vocabs, p_max, p_min)
        latent = vqvae.vq.idx_to_ncm_input(source)
        if prefix_tokens is None:
            prefix = batch_prefix_tokens(
                tokens,
                int(generator.prefix_len),
                generator.prefix_source,
                int(eos_id),
                int(pad_id),
            )
        else:
            prefix = prefix_tokens.to(dtype=torch.long)
            if prefix.ndim != 2 or int(prefix.shape[0]) != int(tokens.shape[0]):
                raise FinetuneError("dialogue prefix batch does not match the token batch")
            if int(prefix.shape[1]) != int(generator.prefix_len):
                raise FinetuneError(
                    f"dialogue prefix width {int(prefix.shape[1])} != "
                    f"prefix_len {int(generator.prefix_len)}"
                )
        teacher = torch.cat([vqvae.encoder.wte(prefix), latent], dim=1)
        # Level 0 stays out of the generator targets.
        targets = indices[:, levels[0] :]
    return teacher, targets, levels[1:], vocabs[1:]


def _sync_context(model, sync: bool):
    if sync or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def _is_main(rank: int) -> bool:
    return int(rank) == 0


def _barrier(world_size: int) -> None:
    if int(world_size) <= 1:
        return
    import torch.distributed as dist

    dist.barrier()


def _init_process(device_name: str | None):
    if os.environ.get("RANK") is None:
        return resolve_device("auto" if device_name is None else device_name), 0, 1, 0, False
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise FinetuneError("distributed fine-tuning requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    started = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        started = True
    return torch.device(f"cuda:{local_rank}"), rank, world_size, local_rank, started


def _cleanup_process(started: bool) -> None:
    if not started:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _atomic_save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record)) + "\n")


def _prune_steps(out: Path, keep: int = 2) -> None:
    files = [path for path in out.glob("generator_step*.pt") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime)
    for path in files[:-int(keep)]:
        path.unlink(missing_ok=True)


def resolve_resume(resume, out: Path):
    if resume is None or str(resume).lower() in {"", "none", "off"}:
        return None
    if str(resume) == "auto":
        latest = Path(out) / "latest.pt"
        return latest if latest.is_file() else None
    path = Path(resume)
    if not path.is_file():
        raise FinetuneError(f"resume checkpoint not found: {resume}")
    return path


def _find_named(root: Path, name: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob(name) if path.is_file())


def find_converted_pair(root) -> tuple[Path, Path]:
    """Pair ``tokenizer.pt`` with ``generator.pt`` in the same directory.

    A directory that also holds ``convert_report.json`` or ``convert_v10.py``
    wins over a raw export that happens to use the same filenames.
    """
    root = Path(root)
    generators = {path.parent.resolve(): path for path in _find_named(root, "generator.pt")}
    pairs = []
    for tokenizer in _find_named(root, "tokenizer.pt"):
        generator = generators.get(tokenizer.parent.resolve())
        if generator is not None:
            pairs.append((tokenizer, generator))
    if not pairs:
        raise FinetuneError(
            f"no directory under {root} contains both tokenizer.pt and generator.pt. "
            "Upload the converted init (those two files, plus convert_report.json) "
            "as its own Kaggle dataset."
        )

    def rank(pair: tuple[Path, Path]):
        parent = pair[0].parent
        score = 0
        if (parent / "convert_report.json").is_file():
            score -= 10
        if (parent / "convert_v10.py").is_file():
            score -= 5
        lowered = parent.name.lower()
        if "init" in lowered or "iwsp" in lowered:
            score -= 3
        return (score, len(parent.parts), str(parent))

    pairs.sort(key=rank)
    if len(pairs) > 1 and rank(pairs[0])[0] == rank(pairs[1])[0]:
        listed = ", ".join(str(pair[0].parent) for pair in pairs)
        raise FinetuneError(
            "multiple converted tokenizer.pt + generator.pt pairs. "
            f"Pass --tokenizer and --generator explicitly. Found: {listed}"
        )
    return pairs[0]


def find_packed_windows(root) -> Path:
    """Prefer ``artifacts/packed_windows.pt`` over a copy at a dataset root."""
    matches = _find_named(Path(root), PACKED_FILENAME)
    if not matches:
        raise FinetuneError(
            f"no {PACKED_FILENAME} under {root}. "
            "The v10 file lives at artifacts/packed_windows.pt inside the export dataset, "
            "not necessarily beside the raw tokenizer export."
        )

    def rank(path: Path):
        in_artifacts = "artifacts" in {part.lower() for part in path.parts}
        return (0 if in_artifacts else 1, len(path.parts), str(path))

    matches.sort(key=rank)
    return matches[0]


def _cuda_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024**3)


def _evaluate(vqvae, generator, loader, device, eos_id, pad_id, alpha, mid_aux, autocast):
    training = generator.training
    generator.eval()
    sums = None
    counts = None
    weighted = 0.0
    elbo = 0.0
    rows = 0
    for batch in loader:
        tokens, prefix_batch = _batch_tensors(batch, device)
        teacher, targets, levels, vocabs = _prepare_batch(
            vqvae,
            generator,
            tokens,
            eos_id,
            corrupt=False,
            p_max=0.0,
            p_min=0.0,
            pad_id=pad_id,
            prefix_tokens=prefix_batch,
        )
        with torch.no_grad():
            with autocast:
                logits = generator(teacher)
            stats = generator_ce_loss(logits, targets, levels, vocabs, alpha=alpha, mid_aux=0.0)
        if sums is None:
            sums = [0.0] * len(levels)
            counts = [0] * len(levels)
        for index, (part_sum, part_count) in enumerate(zip(stats["level_sums"], stats["level_counts"])):
            sums[index] += part_sum
            counts[index] += part_count
        weighted += stats["weighted_sum"]
        elbo += stats["elbo_sum"]
        rows += int(tokens.shape[0])
    if training:
        generator.train()
    if not rows or sums is None:
        raise FinetuneError("validation loader produced no windows")
    means = [sums[index] / counts[index] for index in range(len(sums))]
    loss = weighted / rows
    if float(mid_aux) != 0.0:
        start, end = mid_level_slice(len(means))
        loss += float(mid_aux) * (sum(means[start:end]) / (end - start))
    mid_index = len(means) // 2
    return {
        "val_ce_first": means[0],
        "val_ce_mid": means[mid_index],
        "val_ce_last": means[-1],
        "val_ce_mid_index": mid_index,
        "val_ce_levels": means,
        "val_loss": float(loss),
        "val_length_weighted_ce": float(elbo / rows),
        "val_mean_level_ce": float(sum(means) / len(means)),
        "val_rows": int(rows),
    }


def run_finetune(
    cfg: dict,
    tokenizer_path,
    generator_path,
    packed_path,
    out_dir,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    peak_lr: float = DEFAULT_PEAK_LR,
    floor_lr: float = DEFAULT_FLOOR_LR,
    warmup: int = DEFAULT_WARMUP,
    max_steps: int = DEFAULT_MAX_STEPS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    betas=DEFAULT_BETAS,
    alpha: float = DEFAULT_ALPHA,
    mid_aux: float = DEFAULT_MID_AUX,
    corrupt_p_max: float = DEFAULT_CORRUPT_P_MAX,
    corrupt_p_min: float = DEFAULT_CORRUPT_P_MIN,
    val_every: int = DEFAULT_VAL_EVERY,
    val_windows: int = DEFAULT_VAL_WINDOWS,
    val_batch_size: int = DEFAULT_VAL_BATCH_SIZE,
    save_every: int = DEFAULT_SAVE_EVERY,
    log_every: int = DEFAULT_LOG_EVERY,
    max_hours: float | None = None,
    reserve_minutes: float = 15.0,
    resume="auto",
    device: str | None = "cpu",
    gradient_checkpointing: bool = True,
    chunked_block_attn: bool = True,
    workers: int = DEFAULT_WORKERS,
    seed: int | None = None,
    dialogue: bool = False,
    dataset_id: str | None = None,
    retrain_prefix_head: bool = False,
) -> dict:
    """Fine-tune the generator from a converted init. Tokenizer weights stay fixed.

    ``dialogue=True`` is the phase-3 path: a pack-dialogue cache, an explicit
    dataset id, and no prefix-head update. The book-window path leaves
    ``dialogue`` false.
    """
    if retrain_prefix_head:
        raise FinetuneError(
            "the prefix head is not retrained. "
            "finetune-dialogue leaves it frozen. "
            "--retrain-prefix-head is an error, not a silent retrain."
        )
    if dialogue:
        if not isinstance(cfg.get("dialogue"), dict) or not cfg["dialogue"].get("enabled"):
            raise FinetuneError(
                "finetune-dialogue needs a phase-3 config with dialogue.enabled true."
            )
        if not str(dataset_id or "").strip():
            raise FinetuneError(
                "finetune-dialogue needs --dataset-id. No dialogue corpus is selected."
            )
    elif isinstance(cfg.get("dialogue"), dict) and cfg["dialogue"].get("enabled"):
        raise FinetuneError(
            "this config enables dialogue. Use finetune-dialogue, which requires a dataset id "
            "and does not retrain the prefix head. The finetune command stays the book-window path."
        )
    sources = [tokenizer_path, generator_path, packed_path]
    refuse_finetune_output(out_dir, sources)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    refuse_finetune_output(out, sources)

    device_t, rank, world_size, local_rank, started_dist = _init_process(device)
    try:
        return _run_on_process(
            cfg,
            tokenizer_path,
            generator_path,
            packed_path,
            out,
            device_t,
            rank,
            world_size,
            local_rank,
            batch_size=int(batch_size),
            grad_accum=int(grad_accum),
            peak_lr=float(peak_lr),
            floor_lr=float(floor_lr),
            warmup=int(warmup),
            max_steps=int(max_steps),
            weight_decay=float(weight_decay),
            grad_clip=float(grad_clip),
            betas=tuple(betas),
            alpha=float(alpha),
            mid_aux=float(mid_aux),
            corrupt_p_max=float(corrupt_p_max),
            corrupt_p_min=float(corrupt_p_min),
            val_every=int(val_every),
            val_windows=int(val_windows),
            val_batch_size=int(val_batch_size),
            save_every=int(save_every),
            log_every=int(log_every),
            max_hours=max_hours,
            reserve_minutes=float(reserve_minutes),
            resume=resume,
            gradient_checkpointing=bool(gradient_checkpointing),
            chunked_block_attn=bool(chunked_block_attn),
            workers=int(workers),
            seed=seed,
            dialogue=bool(dialogue),
            dataset_id=None if dataset_id is None else str(dataset_id),
            retrain_prefix_head=False,
        )
    finally:
        _cleanup_process(started_dist)


def _refuse_resume_prefix(blob: dict, generator) -> None:
    """Resume only when the checkpoint was trained with this prefix protocol.

    A v3 file that omits ``prefix_len`` can resume the phase-1 config. It cannot
    resume a phase-2 config: that file is a warm start (``--resume none``), not
    an optimizer continuation.
    """
    recorded = recorded_prefix(blob)
    if recorded is None:
        if generator.prefix_source == "eos" and int(generator.prefix_len) == 1:
            return
        raise FinetuneError(
            "resume checkpoint does not record prefix_len. "
            "A phase-1 file can resume only prefix_len 1 and prefix_source eos. "
            "Warm-start a phase-2 run with --generator pointing at that file and "
            "--resume none. Do not resume its optimizer into a different prefix."
        )
    problems = prefix_metadata_problems(
        recorded, int(generator.prefix_len), generator.prefix_source
    )
    if problems:
        raise FinetuneError(
            "resume checkpoint prefix does not match this config: "
            + "; ".join(problems)
            + ". Start a new --out with --resume none to warm-start the weights."
        )


def _run_on_process(cfg, tokenizer_path, generator_path, packed_path, out, device, rank, world_size, local_rank, **opt):
    main = _is_main(rank)
    seed = int(cfg["seed"] if opt["seed"] is None else opt["seed"])
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    context = int(cfg["tokenizer"]["context_length"])
    train_prefix = None
    val_prefix = None
    if opt.get("dialogue"):
        settings = dialogue_settings(cfg)
        train_windows, val_windows, train_prefix, val_prefix, packed_meta = load_dialogue_windows(
            packed_path,
            context,
            int(cfg["generator"]["prefix_len"]),
            opt.get("dataset_id"),
            settings["history_placement"],
        )
        plan = describe_dialogue_run(cfg, opt)
        plan["dataset"] = packed_meta.get("dataset")
        plan["history_placement"] = packed_meta.get("history_placement")
    else:
        train_windows, val_windows, packed_meta = load_training_windows(
            packed_path, context, opt["val_windows"]
        )
        plan = describe_run(cfg, opt)
    checkpoint_format = DIALOGUE_FORMAT if opt.get("dialogue") else FINETUNE_FORMAT
    quality_note = DIALOGUE_QUALITY_NOTE if opt.get("dialogue") else None
    plan.update(
        {
            "device": str(device),
            "world_size": int(world_size),
            "packed": packed_meta,
            "train_windows": int(train_windows.shape[0]),
            "val_windows_loaded": int(val_windows.shape[0]),
            "tokenizer_path": str(tokenizer_path),
            "generator_path": str(generator_path),
            "out": str(out),
        }
    )
    note = loss_note(opt["alpha"], opt["mid_aux"])
    comparable = loss_is_alpha0(opt["alpha"], opt["mid_aux"])
    if main:
        _write_json(out / "report.json", plan)
        print(note, flush=True)
        print(
            f"fine-tune NCM-iWSP-GPT1  batch/gpu={opt['batch_size']}  "
            f"checkpointing={opt['gradient_checkpointing']}  "
            f"chunked_attn={opt['chunked_block_attn']}  "
            f"train_windows={train_windows.shape[0]}  val_windows={val_windows.shape[0]}",
            flush=True,
        )

    pretok, vqvae, generator = build_models(cfg, seed=seed)
    torch.manual_seed(seed + rank)
    tok_meta = load_module_init(tokenizer_path, vqvae, "tokenizer")
    gen_meta = load_module_init(generator_path, generator, "generator")
    warmstart = legacy_prefix_warning(gen_meta, generator.prefix_source)
    mismatch = prefix_metadata_problems(
        recorded_prefix(gen_meta), generator.prefix_len, generator.prefix_source
    )
    if mismatch:
        warmstart = (
            "generator init prefix metadata differs from this config ("
            + "; ".join(mismatch)
            + "). Persistent transformer and lm_heads weights are still loaded. "
            "This is a warm start. Resume refuses a different prefix protocol."
        )
    if main and warmstart:
        print(warmstart, flush=True)
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad_(False)
    generator.chunked_block_attn = bool(opt["chunked_block_attn"])
    generator.gradient_checkpointing = bool(opt["gradient_checkpointing"])
    vqvae.to(device)
    generator.to(device)
    tok_checksum = _checksum(vqvae)

    train_parameters = [param for param in generator.parameters() if param.requires_grad]
    if not train_parameters:
        raise FinetuneError("the generator has no trainable parameters")
    optimizer = torch.optim.AdamW(
        train_parameters,
        lr=float(opt["peak_lr"]),
        betas=tuple(float(value) for value in opt["betas"]),
        weight_decay=float(opt["weight_decay"]),
    )
    resume_path = resolve_resume(opt["resume"], out)
    start_step = 0
    scaler_state = None
    if resume_path is not None:
        resume_blob = _read_blob(Path(resume_path))
        if not isinstance(resume_blob, dict) or not isinstance(resume_blob.get("model"), dict):
            raise FinetuneError(f"{resume_path} has no generator model to resume")
        if opt.get("dialogue"):
            if resume_blob.get("format") != DIALOGUE_FORMAT:
                raise FinetuneError(
                    f"{resume_path} format is {resume_blob.get('format')!r}. "
                    f"Dialogue resume expects {DIALOGUE_FORMAT}. "
                    "A phase-2 file is a warm start: pass it as --generator with --resume none."
                )
        elif resume_blob.get("format") not in {None, FINETUNE_FORMAT}:
            raise FinetuneError(
                f"{resume_path} format is {resume_blob.get('format')!r}. "
                f"Resume expects {FINETUNE_FORMAT}."
            )
        resume_state = _strip_module_prefix(resume_blob["model"])
        assert_untied_generator_state(resume_state)
        _refuse_resume_prefix(resume_blob, generator)
        load_generator_state(generator, resume_state)
        if isinstance(resume_blob.get("optimizer"), dict):
            optimizer.load_state_dict(resume_blob["optimizer"])
            _optimizer_to_device(optimizer, device)
        start_step = int(resume_blob.get("step", 0))
        scaler_state = resume_blob.get("scaler")
        if main:
            print(f"resumed generator from {resume_path} at step {start_step}", flush=True)
    gen_checksum = _checksum(generator)
    pin = device.type == "cuda"
    train_loader, train_sampler = _make_loader(
        train_windows,
        opt["batch_size"],
        seed,
        rank,
        world_size,
        opt["workers"],
        pin,
        prefixes=train_prefix,
    )
    val_loader, _ = _make_loader(
        val_windows,
        max(int(opt["val_batch_size"]), 1),
        seed + 1,
        0,
        1,
        0,
        pin,
        shuffle=False,
        prefixes=val_prefix,
    )
    batches = _forever(train_loader, train_sampler)
    autocast, scaler, amp_name = setup_amp(device)
    if scaler is not None and isinstance(scaler_state, dict):
        scaler.load_state_dict(scaler_state)
    if main:
        print(f"amp={amp_name}  device={device}  world_size={world_size}", flush=True)
        if quality_note:
            print(quality_note, flush=True)

    net = generator
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        net = DDP(
            generator,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    max_steps = int(opt["max_steps"])
    deadline = None
    if opt["max_hours"] not in (None, 0):
        deadline = time.monotonic() + float(opt["max_hours"]) * 3600.0 - float(opt["reserve_minutes"]) * 60.0

    def save(step: int, finished: bool) -> None:
        if not main:
            _barrier(world_size)
            return
        module = _unwrap(net)
        payload = {
            "format": checkpoint_format,
            "model": module.state_dict(),
            "step": int(step),
            "optimizer": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "init_step": gen_meta.get("step"),
            "converted_from": gen_meta.get("converted_from"),
            "tokenizer_converted_from": tok_meta.get("converted_from"),
            "config_name": cfg["name"],
            "loss_spec": {
                "alpha": float(opt["alpha"]),
                "mid_aux": float(opt["mid_aux"]),
                "loss_comparable_to_alpha0": comparable,
                "loss_note": note,
            },
            "prefix_len": int(module.prefix_len),
            "prefix_source": str(module.prefix_source),
            "architecture": {
                "untied_heads": True,
                "post_upsample_conv": False,
                "predict_level0": False,
                "prefix_len": int(module.prefix_len),
                "prefix_source": str(module.prefix_source),
                "prefix_pad": "right" if module.prefix_source == "window" else None,
                "chunked_block_attn": bool(opt["chunked_block_attn"]),
                "gradient_checkpointing": bool(opt["gradient_checkpointing"]),
            },
        }
        if opt.get("dialogue"):
            payload["dialogue"] = True
            payload["retrain_prefix_head"] = False
            payload["history_placement"] = packed_meta.get("history_placement")
            payload["dataset"] = packed_meta.get("dataset")
            payload["code_ce_is_chat_quality"] = False
            payload["quality_note"] = quality_note
            payload["tokenizer_frozen"] = True
            payload["predict_level0"] = False
        _atomic_save(out / "latest.pt", payload)
        if int(step) > 0:
            _atomic_save(out / f"generator_step{int(step)}.pt", payload)
            _prune_steps(out, keep=2)
        _write_json(
            out / "status.json",
            {
                "step": int(step),
                "max_steps": max_steps,
                "finished": bool(finished),
                "stage": checkpoint_format,
                "loss_comparable_to_alpha0": comparable,
                "loss_note": note,
            },
        )
        _barrier(world_size)

    def validate(step: int) -> dict | None:
        _barrier(world_size)
        metrics = None
        error = None
        if main:
            try:
                metrics = _evaluate(
                    vqvae,
                    _unwrap(net),
                    val_loader,
                    device,
                    int(pretok.eos_token_id),
                    int(pretok.pad_token_id),
                    opt["alpha"],
                    opt["mid_aux"],
                    autocast,
                )
                record = {
                    "step": int(step),
                    "loss_comparable_to_alpha0": comparable,
                    "loss_note": note,
                    "alpha": float(opt["alpha"]),
                    "mid_aux": float(opt["mid_aux"]),
                    **metrics,
                }
                if quality_note:
                    record["quality_note"] = quality_note
                    record["code_ce_is_chat_quality"] = False
                _append_jsonl(out / "val_log.jsonl", record)
                print(
                    f"val step {step}  loss={metrics['val_loss']:.4f}  "
                    f"ce_first/mid/last="
                    f"{metrics['val_ce_first']:.4f}/{metrics['val_ce_mid']:.4f}/{metrics['val_ce_last']:.4f}  "
                    f"length_weighted_ce={metrics['val_length_weighted_ce']:.4f}",
                    flush=True,
                )
            except Exception as exc:
                error = exc
        if world_size > 1:
            import torch.distributed as dist

            flag = torch.tensor([0 if error is None else 1], device=device, dtype=torch.int32)
            dist.broadcast(flag, src=0)
            if int(flag.item()) != 0:
                if error is not None:
                    raise error
                raise FinetuneError("rank 0 failed during validation")
        elif error is not None:
            raise error
        _barrier(world_size)
        return metrics

    if start_step >= max_steps:
        save(start_step, True)
        return _summary(
            start_step, start_step, True, out, tok_checksum, tok_checksum, gen_checksum, gen_checksum,
            comparable, note, resume_path, gen_meta, checkpoint_format,
        )

    net.train()
    if start_step == 0 or (opt["val_every"] > 0 and start_step % opt["val_every"] == 0):
        validate(start_step)

    stopped = False
    step = start_step
    try:
        while step < max_steps:
            lr = lr_at(step, opt["warmup"], opt["peak_lr"], opt["floor_lr"], max_steps)
            _set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            last_stats = None
            accum = max(int(opt["grad_accum"]), 1)
            for micro in range(accum):
                tokens, prefix_batch = _batch_tensors(next(batches), device)
                teacher, targets, levels, vocabs = _prepare_batch(
                    vqvae,
                    _unwrap(net),
                    tokens,
                    int(pretok.eos_token_id),
                    corrupt=True,
                    p_max=opt["corrupt_p_max"],
                    p_min=opt["corrupt_p_min"],
                    pad_id=int(pretok.pad_token_id),
                    prefix_tokens=prefix_batch,
                )
                sync = micro == accum - 1
                with _sync_context(net, sync):
                    with autocast:
                        logits = net(teacher)
                        stats = generator_ce_loss(
                            logits, targets, levels, vocabs, alpha=opt["alpha"], mid_aux=opt["mid_aux"]
                        )
                        loss = stats["loss"] / accum
                    if not torch.isfinite(loss):
                        raise FinetuneError(f"non-finite loss at step {step}")
                    _backward(loss, scaler)
                last_stats = {
                    "loss": float(stats["loss"].detach()),
                    "ce_first": stats["ce_first"],
                    "ce_mid": stats["ce_mid"],
                    "ce_last": stats["ce_last"],
                    "mean_level_ce": stats["mean_level_ce"],
                    "length_weighted_ce": stats["length_weighted_ce"],
                }
                del logits, loss, stats, teacher, targets
            _step(optimizer, train_parameters, opt["grad_clip"], scaler)
            step += 1
            if main and last_stats is not None and (
                step == start_step + 1 or step % max(int(opt["log_every"]), 1) == 0
            ):
                train_record = {
                    "step": step,
                    "lr": lr,
                    "alpha": float(opt["alpha"]),
                    "mid_aux": float(opt["mid_aux"]),
                    "loss_comparable_to_alpha0": comparable,
                    "loss_note": note,
                    "mem_gb": _cuda_mem_gb(),
                    "amp": amp_name,
                    **last_stats,
                }
                if quality_note:
                    train_record["quality_note"] = quality_note
                    train_record["code_ce_is_chat_quality"] = False
                _append_jsonl(out / "train_log.jsonl", train_record)
                print(
                    f"step {step}/{max_steps}  loss={last_stats['loss']:.4f}  "
                    f"ce_first/mid/last="
                    f"{last_stats['ce_first']:.4f}/{last_stats['ce_mid']:.4f}/{last_stats['ce_last']:.4f}  "
                    f"lr={lr:.2e}  mem={_cuda_mem_gb():.2f}GiB",
                    flush=True,
                )
            if opt["val_every"] > 0 and step % opt["val_every"] == 0:
                validate(step)
            if opt["save_every"] > 0 and step % opt["save_every"] == 0:
                save(step, False)
            stop_for_time = deadline is not None and time.monotonic() >= deadline
            if world_size > 1:
                import torch.distributed as dist

                flag = torch.tensor([1 if stop_for_time else 0], device=device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                stop_for_time = int(flag.item()) == 1
            if stop_for_time:
                stopped = True
                if main:
                    print(f"time budget reached at step {step}; saving", flush=True)
                break
    except KeyboardInterrupt:
        stopped = True
        if main:
            print(f"interrupted at step {step}; saving", flush=True)

    finished = (not stopped) and step >= max_steps
    if opt["val_every"] > 0 and (step % opt["val_every"] != 0):
        validate(step)
    save(step, finished)
    gen_end = _checksum(_unwrap(net))
    tok_end = _checksum(vqvae)
    return _summary(
        start_step, step, finished, out, tok_checksum, tok_end, gen_checksum, gen_end,
        comparable, note, resume_path, gen_meta, checkpoint_format,
    )


def _summary(
    start_step, step, finished, out, tok_start, tok_end, gen_start, gen_end,
    comparable, note, resume_path, gen_meta, checkpoint_format=FINETUNE_FORMAT,
) -> dict:
    summary = {
        "start_step": int(start_step),
        "step": int(step),
        "finished": bool(finished),
        "out": str(out),
        "resumed_from": None if resume_path is None else str(resume_path),
        "tokenizer_checksum_start": tok_start,
        "tokenizer_checksum_end": tok_end,
        "generator_checksum_start": gen_start,
        "generator_checksum_end": gen_end,
        "loss_comparable_to_alpha0": bool(comparable),
        "loss_note": note,
        "init_step": gen_meta.get("step"),
        "converted_from": _jsonable(gen_meta.get("converted_from")),
        "format": checkpoint_format,
    }
    if checkpoint_format == DIALOGUE_FORMAT:
        summary["retrain_prefix_head"] = False
    return summary


def run_dialogue_finetune(
    cfg,
    tokenizer_path,
    generator_path,
    packed_path,
    out_dir,
    *,
    dataset_id: str,
    retrain_prefix_head: bool = False,
    **kwargs,
) -> dict:
    """Phase-3 generator fine-tune. The prefix head is not updated.

    ``dataset_id`` is required and must match the pack-dialogue cache. The
    generator warm-start is whatever ``--generator`` points at, typically a
    phase-2 checkpoint. Writes only go to ``out_dir``.
    """
    return run_finetune(
        cfg,
        tokenizer_path,
        generator_path,
        packed_path,
        out_dir,
        dialogue=True,
        dataset_id=dataset_id,
        retrain_prefix_head=bool(retrain_prefix_head),
        **kwargs,
    )
