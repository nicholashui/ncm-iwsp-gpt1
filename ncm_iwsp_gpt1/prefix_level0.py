"""Prefix level-0 head for NCM-iWSP-GPT1.

One code over the level-0 codebook, conditioned on the real prompt. Token
features are the frozen tokenizer ``encoder.wte`` rows. The head adds its own
position table and attention-pools the first T positions. A plain mean would
ignore order, because the sum of positions does not depend on which token
sits where. ``NextConceptGenerator`` is not this module: ``predict_level0``
stays false, and ``prefix_len`` stays 1. At inference the generator prefix
is still ``<|endoftext|>``.

Training draws T uniformly from 1 through the context length and asks the
head to predict the level-0 code of the full packed window. Only this head
is optimized. The tokenizer and the generator are frozen.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncm_iwsp_gpt1.build import build_models, lock_for_eval
from ncm_iwsp_gpt1.device import resolve_device
from ncm_iwsp_gpt1.finetune import FinetuneError, load_module_init, load_training_windows, refuse_finetune_output

PREFIX_LEVEL0_FORMAT = "ncm-iwsp-gpt1-prefix-level0"
POOLING = "attention_pool_of_wte_plus_learned_position"


class PrefixLevel0Error(ValueError):
    """The prefix head cannot be built, trained, or loaded."""


class PrefixLevel0Head(nn.Module):
    """Logits over the single level-0 codebook, from prompt token embeddings.

    ``embeddings`` are ``encoder.wte(token_ids)`` with no tokenizer position
    added. A short prompt is not tiled out to the context. Learned positions
    live here, at offsets ``0 .. T-1``, because the tokenizer position table
    belongs to a full window this head never builds. A learned query attends
    over those positions so token order can change the pooled vector.
    """

    def __init__(self, embed_dim: int, codebook_size: int, context_length: int):
        super().__init__()
        embed_dim = int(embed_dim)
        codebook_size = int(codebook_size)
        context_length = int(context_length)
        if embed_dim < 1 or context_length < 1 or codebook_size < 2:
            raise ValueError("embed_dim and context_length must be >= 1, and codebook_size >= 2")
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.context_length = context_length
        self.pos = nn.Embedding(context_length, embed_dim)
        self.query = nn.Parameter(torch.empty(embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, codebook_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor | int) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(f"embeddings must be [batch, time, embed_dim], got {tuple(embeddings.shape)}")
        batch, time, dim = embeddings.shape
        if dim != self.embed_dim:
            raise ValueError(f"embedding width {dim} != head embed_dim {self.embed_dim}")
        if time > self.context_length:
            raise ValueError(
                f"prompt length {time} is longer than context_length {self.context_length}"
            )
        if time < 1:
            raise ValueError("prompt embeddings must contain at least one token")
        lengths = _as_lengths(lengths, batch, time, embeddings.device)
        positions = torch.arange(time, device=embeddings.device)
        hidden = embeddings + self.pos(positions).unsqueeze(0)
        pooled = attention_pool(hidden, self.query, lengths)
        return self.proj(self.norm(pooled))


def head_parameter_count(embed_dim: int, codebook_size: int, context_length: int) -> int:
    """Parameter count of ``PrefixLevel0Head`` for these three sizes."""
    embed_dim = int(embed_dim)
    codebook_size = int(codebook_size)
    context_length = int(context_length)
    return (
        context_length * embed_dim
        + embed_dim
        + 2 * embed_dim
        + embed_dim * codebook_size
        + codebook_size
    )


def attention_pool(hidden: torch.Tensor, query: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Softmax pool over the first ``lengths[i]`` positions.

    Scores are the dot product of each position with ``query``. A mean would
    add the same position sum for every assignment of tokens, so order would
    drop out. Attention weights depend on which token sits at which offset.
    """
    time = int(hidden.shape[1])
    positions = torch.arange(time, device=hidden.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    scores = torch.matmul(hidden, query)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bt,btd->bd", weights, hidden)


def _as_lengths(lengths, batch: int, time: int, device) -> torch.Tensor:
    if isinstance(lengths, int):
        if lengths < 1 or lengths > time:
            raise ValueError(f"lengths must be in 1..{time}, got {lengths}")
        return torch.full((batch,), int(lengths), dtype=torch.long, device=device)
    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths, dtype=torch.long)
    lengths = lengths.to(device=device, dtype=torch.long).reshape(-1)
    if int(lengths.shape[0]) != int(batch):
        raise ValueError(f"lengths has {int(lengths.shape[0])} rows, embeddings have {batch}")
    if bool(torch.any(lengths < 1) or torch.any(lengths > time)):
        raise ValueError(f"lengths must be in 1..{time}")
    return lengths


def sample_prefix_lengths(batch: int, context_length: int, generator=None, device=None) -> torch.Tensor:
    """Uniform draw of T from 1 through ``context_length``, one per row.

    Short prompts and full windows are the same distribution. T is never 0.
    """
    context_length = int(context_length)
    batch = int(batch)
    if context_length < 1:
        raise ValueError("context_length must be positive")
    if batch < 1:
        raise ValueError("batch must be positive")
    if generator is None and device is not None:
        return torch.randint(1, context_length + 1, (batch,), device=device)
    lengths = torch.randint(1, context_length + 1, (batch,), generator=generator)
    if device is not None:
        lengths = lengths.to(device)
    return lengths


def embed_prompt_tokens(wte: nn.Embedding, token_ids: torch.Tensor) -> torch.Tensor:
    """Frozen ``encoder.wte`` rows. The tokenizer position table is not added."""
    return wte(token_ids)


def sample_logits(logits: torch.Tensor, temperature=0.0, top_k=0, top_p=1.0) -> torch.Tensor:
    """One class per row. Temperature 0 is argmax, the same rule as the generator."""
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
    probs = F.softmax(flat, dim=-1)
    draw = torch.multinomial(probs, num_samples=1)
    return draw.view(logits.shape[:-1])


def describe_prefix_run(cfg: dict, options: dict | None = None) -> dict:
    """Plan a prefix-head run. Does not build modules or read packed windows.

    This command stays on the phase-1 guard: ``prefix_len`` 1 and an
    end-of-text generator prefix. Phase 2 does not retrain the head here.
    """
    _phase1_config(cfg)
    options = dict(options or {})
    tokenizer = cfg["tokenizer"]
    generator = cfg["generator"]
    embed_dim = int(tokenizer["embed_dim"])
    codebook_size = int(tokenizer["codebook_size"])
    context_length = int(tokenizer["context_length"])
    max_steps = options.get("max_steps")
    return {
        "system": "NCM-iWSP-GPT1",
        "command": "train-prefix-level0",
        "config_name": cfg["name"],
        "optimized": "prefix_level0_head",
        "tokenizer_frozen": True,
        "generator_frozen": True,
        "predict_level0": False,
        "prefix_len": int(generator["prefix_len"]),
        "phase1_requires_prefix_len": 1,
        "context_length": context_length,
        "embed_dim": embed_dim,
        "codebook_size": codebook_size,
        "level0_len": int(tokenizer["levels"][0]),
        "head_parameters": head_parameter_count(embed_dim, codebook_size, context_length),
        "pooling": POOLING,
        "t_sampling": "uniform integer from 1 through context_length, one draw per row",
        "target": "level-0 code of the full packed window",
        "features": "frozen encoder.wte of the first T tokens, plus a learned position and attention pool in the head",
        "generator_prefix_at_inference": "<|endoftext|>",
        "batch_size": int(options.get("batch_size", 8)),
        "lr": float(options.get("lr", 1e-3)),
        "max_steps": None if max_steps is None else int(max_steps),
        "writes": "only the --out directory",
        "does_not_submit_kaggle": True,
        "packed_windows_not_in_repo": True,
        "note": (
            "Phase 1 only. The generator is not fine-tuned and prefix_len stays 1. "
            "The goal is continuation of a short prompt, not a chatbot answer. "
            "If short-prefix level-0 cross-entropy stays near the unigram, this head "
            "is not enough. That comparison is logged. Phase 2 is a separate config "
            "and does not retrain this head."
        ),
        "retrain_for_phase2": False,
    }


def _phase1_config(cfg: dict) -> None:
    generator = cfg["generator"]
    tokenizer = cfg["tokenizer"]
    if generator.get("predict_level0", False):
        raise PrefixLevel0Error(
            "predict_level0 must stay false. The prefix head is a separate module, "
            "not a generator target."
        )
    if int(generator.get("prefix_len", 1)) != 1:
        raise PrefixLevel0Error(
            "phase 1 keeps generator prefix_len at 1 so existing weights still see "
            "<|endoftext|>. Lengthening the prefix is phase 2."
        )
    if int(tokenizer["levels"][0]) != 1:
        raise PrefixLevel0Error("the prefix head predicts a single level-0 code; levels[0] must be 1")
    if isinstance(tokenizer["codebook_size"], (list, tuple)):
        raise PrefixLevel0Error("codebook_size must be one integer")


def _read_blob(path: Path):
    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PrefixLevel0Error(f"could not read {path}: {exc}") from exc


def read_prefix_blob(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise PrefixLevel0Error(f"prefix head not found: {path}")
    blob = _read_blob(path)
    if not isinstance(blob, dict) or blob.get("format") != PREFIX_LEVEL0_FORMAT:
        found = blob.get("format") if isinstance(blob, dict) else type(blob).__name__
        raise PrefixLevel0Error(
            f"{path} is not a {PREFIX_LEVEL0_FORMAT} checkpoint (found {found!r}). "
            "Train one with `python -m ncm_iwsp_gpt1 train-prefix-level0`."
        )
    if not isinstance(blob.get("model"), dict) or not blob["model"]:
        raise PrefixLevel0Error(f"{path} is missing the prefix-head state dict")
    return blob


def prefix_blob_matches(blob: dict, embed_dim: int, codebook_size: int, context_length: int) -> None:
    problems = []
    if int(blob.get("embed_dim", -1)) != int(embed_dim):
        problems.append(f"embed_dim {blob.get('embed_dim')} != {int(embed_dim)}")
    if int(blob.get("codebook_size", -1)) != int(codebook_size):
        problems.append(f"codebook_size {blob.get('codebook_size')} != {int(codebook_size)}")
    if int(blob.get("context_length", -1)) != int(context_length):
        problems.append(f"context_length {blob.get('context_length')} != {int(context_length)}")
    if problems:
        raise PrefixLevel0Error("prefix head does not match this config: " + "; ".join(problems))


def _strip_module_prefix(state: dict) -> dict:
    keys = [str(key) for key in state.keys()]
    if keys and all(key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def load_prefix_head(blob: dict, embed_dim: int, codebook_size: int, context_length: int) -> PrefixLevel0Head:
    prefix_blob_matches(blob, embed_dim, codebook_size, context_length)
    head = PrefixLevel0Head(embed_dim, codebook_size, context_length)
    state = _strip_module_prefix(blob["model"])
    if not all(torch.is_tensor(value) for value in state.values()):
        raise PrefixLevel0Error("prefix head state dict contains a non-tensor")
    try:
        head.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise PrefixLevel0Error(f"prefix head state does not match this config: {exc}") from exc
    return head


def save_prefix_head(
    path,
    head: PrefixLevel0Head,
    step: int,
    *,
    embed_dim: int,
    codebook_size: int,
    context_length: int,
    config_name: str,
    extra: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": PREFIX_LEVEL0_FORMAT,
        "model": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "step": int(step),
        "config_name": str(config_name),
        "embed_dim": int(embed_dim),
        "codebook_size": int(codebook_size),
        "context_length": int(context_length),
        "level0_len": 1,
        "pooling": POOLING,
        "predict_level0": False,
        "prefix_len": 1,
        "generator_updated": False,
        "tokenizer_updated": False,
    }
    for key, value in dict(extra or {}).items():
        if key in {"format", "model"}:
            continue
        payload[key] = value
    torch.save(payload, path)
    return path


def load_windows_for_level0(path, context_length: int) -> torch.Tensor:
    """Windows used only to count level 0.

    A v10 ``packed_windows.pt`` stores ``windows``. This repo's cache stores
    ``visible_train``. The hidden split is not read.
    """
    path = Path(path)
    if not path.is_file():
        raise PrefixLevel0Error(f"packed windows not found: {path}")
    blob = _read_blob(path)
    if not isinstance(blob, dict):
        raise PrefixLevel0Error(f"{path} is not a packed-window dict")
    if torch.is_tensor(blob.get("windows")):
        windows = blob["windows"]
    elif torch.is_tensor(blob.get("visible_train")):
        windows = blob["visible_train"]
    else:
        raise PrefixLevel0Error(
            f"{path} has no windows tensor. Expected a v10 key 'windows' "
            "or this repo's visible_train."
        )
    if windows.ndim != 2 or int(windows.shape[1]) != int(context_length):
        raise PrefixLevel0Error(
            f"{path} has windows of shape {tuple(windows.shape)}; "
            f"expected [N, {int(context_length)}]."
        )
    if int(windows.shape[0]) < 1:
        raise PrefixLevel0Error(f"{path} has no windows")
    meta = blob.get("meta") if isinstance(blob.get("meta"), dict) else {}
    if meta.get("L") is not None and int(meta["L"]) != int(context_length):
        raise PrefixLevel0Error(f"{path} meta L={meta['L']} does not match context {int(context_length)}")
    return windows.long().contiguous()


@torch.no_grad()
def level0_code_counts(vqvae, windows: torch.Tensor, max_windows: int, batch_size: int = 8) -> torch.Tensor:
    """Add the level-0 codes of up to ``max_windows`` rows into a count vector."""
    codebook = int(vqvae.vq.vqvae_vocab_sizes[0])
    taken = windows[: int(max_windows)].long()
    if int(taken.shape[0]) < 1:
        raise PrefixLevel0Error("no windows available to count level 0")
    device = next(vqvae.parameters()).device
    counts = torch.zeros(codebook, dtype=torch.float64)
    step = max(int(batch_size), 1)
    for start in range(0, int(taken.shape[0]), step):
        batch = taken[start : start + step].to(device)
        codes = vqvae.encode(batch)[:, 0].detach().cpu().long()
        counts += torch.bincount(codes, minlength=codebook).to(dtype=torch.float64)
    return counts


def _clone_params(module) -> list[torch.Tensor]:
    return [param.detach().cpu().clone() for param in module.parameters()]


def _params_equal(left, right) -> bool:
    if len(left) != len(right):
        return False
    return all(torch.equal(before, after) for before, after in zip(left, right))


def _forbid_frozen_grad(*modules) -> None:
    for module in modules:
        for param in module.parameters():
            if param.grad is not None:
                raise PrefixLevel0Error("tokenizer or generator received a gradient; only the prefix head trains")


def train_step(head, optimizer, vqvae, tokens: torch.Tensor, grad_clip: float = 1.0) -> dict:
    """One update of the head. ``tokens`` is a batch of full windows.

    The target is the level-0 code of the whole window. T is drawn after the
    target is fixed, so a short prefix is scored against the same code as the
    full window. Embeddings are detached.
    """
    head.train()
    tokens = tokens.long()
    if tokens.ndim != 2 or int(tokens.shape[1]) < 1:
        raise PrefixLevel0Error(f"training windows must be [batch, context], got {tuple(tokens.shape)}")
    context = int(tokens.shape[1])
    with torch.no_grad():
        embeddings = embed_prompt_tokens(vqvae.encoder.wte, tokens).detach()
        target = vqvae.encode(tokens)[:, 0].long()
    lengths = sample_prefix_lengths(int(tokens.shape[0]), context, device=tokens.device)
    optimizer.zero_grad(set_to_none=True)
    logits = head(embeddings, lengths)
    if int(logits.shape[-1]) <= int(target.max()):
        raise PrefixLevel0Error("level-0 target is outside the head codebook")
    loss = F.cross_entropy(logits, target)
    if not torch.isfinite(loss):
        raise PrefixLevel0Error("prefix-head loss is not finite")
    loss.backward()
    _forbid_frozen_grad(vqvae)
    grad_norm = torch.nn.utils.clip_grad_norm_(list(head.parameters()), float(grad_clip))
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(grad_norm),
        "mean_T": float(lengths.detach().float().mean().cpu()),
        "lengths": lengths.detach().cpu(),
        "target": target.detach().cpu(),
    }


@torch.no_grad()
def prefix_ce(head, vqvae, windows: torch.Tensor, length: int, batch_size: int = 8) -> float:
    """Mean cross-entropy of a fixed prefix length against the full-window level 0."""
    was_training = head.training
    head.eval()
    total = 0.0
    count = 0
    device = next(head.parameters()).device
    windows = windows.long()
    length = int(length)
    step = max(int(batch_size), 1)
    for start in range(0, int(windows.shape[0]), step):
        batch = windows[start : start + step].to(device)
        embeddings = embed_prompt_tokens(vqvae.encoder.wte, batch)
        target = vqvae.encode(batch)[:, 0].long()
        lengths = torch.full((int(batch.shape[0]),), length, dtype=torch.long, device=device)
        logits = head(embeddings, lengths)
        total += float(F.cross_entropy(logits, target, reduction="sum").detach().cpu())
        count += int(batch.shape[0])
    head.train(was_training)
    return total / max(count, 1)


@torch.no_grad()
def unigram_nll(counts: torch.Tensor, vqvae, windows: torch.Tensor, batch_size: int = 8) -> float:
    """Add-one unigram negative log likelihood of level 0 on ``windows``."""
    codebook = int(counts.numel())
    probs = (counts.double() + 1.0) / (counts.double().sum() + codebook)
    logp = torch.log(probs)
    total = 0.0
    count = 0
    device = next(vqvae.parameters()).device
    windows = windows.long()
    step = max(int(batch_size), 1)
    for start in range(0, int(windows.shape[0]), step):
        batch = windows[start : start + step].to(device)
        target = vqvae.encode(batch)[:, 0].detach().cpu().long()
        total += float((-logp[target]).sum())
        count += int(target.numel())
    return total / max(count, 1)


def _move_optimizer_state(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_prefix_level0(
    cfg: dict,
    tokenizer_path,
    packed_path,
    out,
    generator_path=None,
    max_steps: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    seed: int | None = None,
    device: str | torch.device | None = "auto",
    log_every: int = 10,
    val_every: int = 50,
    save_every: int = 100,
    unigram_windows: int = 256,
    val_windows: int = 32,
    grad_clip: float = 1.0,
    resume=None,
) -> dict:
    """Train only the prefix head. Does not submit a job and does not write beside inputs.

    ``packed_windows.pt`` is not in this repository. Callers pass a local file.
    A missing file raises. This function does not download BookCorpus.
    """
    _phase1_config(cfg)
    if int(max_steps) < 1:
        raise PrefixLevel0Error("--max-steps must be >= 1")
    if int(batch_size) < 1:
        raise PrefixLevel0Error("--batch-size must be >= 1")
    if float(lr) <= 0:
        raise PrefixLevel0Error("--lr must be positive")
    if int(log_every) < 1 or int(val_every) < 1 or int(save_every) < 1:
        raise PrefixLevel0Error("--log-every, --val-every, and --save-every must be >= 1")
    out = Path(out)
    tokenizer_path = Path(tokenizer_path)
    packed_path = Path(packed_path)
    generator_path = None if generator_path is None else Path(generator_path)
    resume = None if resume is None or str(resume).strip() in {"", "none"} else Path(str(resume))
    sources = [tokenizer_path, packed_path]
    if generator_path is not None:
        sources.append(generator_path)
    try:
        refuse_finetune_output(out, sources)
        if resume is not None:
            try:
                resume.resolve().relative_to(out.resolve())
            except ValueError:
                refuse_finetune_output(out, [resume])
    except FinetuneError as exc:
        raise PrefixLevel0Error(str(exc)) from exc
    if out.exists() and not out.is_dir():
        raise PrefixLevel0Error(f"--out is a file: {out}")
    out.mkdir(parents=True, exist_ok=True)

    seed_value = int(cfg["seed"] if seed is None else seed)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    resolved = resolve_device(device)
    _pretok, vqvae, generator = build_models(cfg)
    try:
        load_module_init(tokenizer_path, vqvae, "tokenizer")
        if generator_path is not None:
            load_module_init(generator_path, generator, "generator")
    except FinetuneError as exc:
        raise PrefixLevel0Error(str(exc)) from exc
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    vqvae.to(resolved)
    generator.to(resolved)
    tokenizer_before = _clone_params(vqvae)
    generator_before = _clone_params(generator)

    embed_dim = int(vqvae.embed_dim)
    codebook_size = int(vqvae.vq.vqvae_vocab_sizes[0])
    context_length = int(vqvae.context_length)
    head = PrefixLevel0Head(embed_dim, codebook_size, context_length).to(resolved)
    start = 0
    resume_blob = None
    if resume is not None:
        resume_blob = read_prefix_blob(resume)
        prefix_blob_matches(resume_blob, embed_dim, codebook_size, context_length)
        state = _strip_module_prefix(resume_blob["model"])
        try:
            head.load_state_dict(state, strict=True)
        except (RuntimeError, KeyError) as exc:
            raise PrefixLevel0Error(f"prefix head state does not match this config: {exc}") from exc
        start = int(resume_blob.get("step", 0))
        if start >= int(max_steps):
            raise PrefixLevel0Error(
                f"checkpoint is already at step {start}, which is >= --max-steps {int(max_steps)}. "
                "--max-steps is the step to reach, not an extra count."
            )
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    if resume_blob is not None and isinstance(resume_blob.get("optimizer"), dict):
        optimizer.load_state_dict(resume_blob["optimizer"])
        _move_optimizer_state(optimizer, resolved)
    head_before = _clone_params(head)
    head_ids = {id(param) for param in head.parameters()}
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) not in head_ids:
                raise PrefixLevel0Error("optimizer contains a parameter that is not part of the prefix head")

    try:
        train, val, meta = load_training_windows(packed_path, context_length, int(val_windows))
    except FinetuneError as exc:
        raise PrefixLevel0Error(str(exc)) from exc
    if int(train.shape[0]) < int(batch_size):
        raise PrefixLevel0Error(
            f"need at least {int(batch_size)} training windows, found {int(train.shape[0])}. "
            "Lower --batch-size or pass a larger packed file."
        )

    counts = level0_code_counts(vqvae, train, max_windows=int(unigram_windows), batch_size=int(batch_size))
    _forbid_frozen_grad(vqvae, generator)
    log_path = out / "train_log.jsonl"
    latest = out / "prefix_level0.pt"
    last = None
    short_t = min(4, context_length)
    for step in range(start + 1, int(max_steps) + 1):
        index = torch.randint(0, int(train.shape[0]), (int(batch_size),))
        tokens = train.index_select(0, index).to(resolved)
        last = train_step(head, optimizer, vqvae, tokens, grad_clip=float(grad_clip))
        _forbid_frozen_grad(vqvae, generator)
        row = {
            "step": step,
            "loss": last["loss"],
            "grad_norm": last["grad_norm"],
            "mean_T": last["mean_T"],
            "lr": float(lr),
        }
        if step % int(val_every) == 0:
            row["val_ce_T1"] = prefix_ce(head, vqvae, val, 1, batch_size=int(batch_size))
            row["val_unigram_nll"] = unigram_nll(counts, vqvae, val, batch_size=int(batch_size))
        if step % int(log_every) == 0 or step == int(max_steps):
            _append_jsonl(log_path, row)
            print(json.dumps(row), flush=True)
        if step % int(save_every) == 0:
            save_prefix_head(
                out / f"prefix_level0_step{step}.pt",
                head,
                step,
                embed_dim=embed_dim,
                codebook_size=codebook_size,
                context_length=context_length,
                config_name=cfg["name"],
                extra={"optimizer": optimizer.state_dict()},
            )

    if last is None:
        raise PrefixLevel0Error("training produced no steps")
    head.eval()
    val_ce_t1 = prefix_ce(head, vqvae, val, 1, batch_size=int(batch_size))
    val_ce_t4 = prefix_ce(head, vqvae, val, short_t, batch_size=int(batch_size))
    val_ce_full = prefix_ce(head, vqvae, val, context_length, batch_size=int(batch_size))
    val_unigram = unigram_nll(counts, vqvae, val, batch_size=int(batch_size))
    _forbid_frozen_grad(vqvae, generator)
    if not _params_equal(tokenizer_before, _clone_params(vqvae)):
        raise PrefixLevel0Error("tokenizer weights changed during prefix-head training")
    if not _params_equal(generator_before, _clone_params(generator)):
        raise PrefixLevel0Error("generator weights changed during prefix-head training")
    head_changed = not _params_equal(head_before, _clone_params(head))
    metrics = {
        "val_ce_T1": val_ce_t1,
        "val_ce_T4": val_ce_t4,
        "val_ce_Tfull": val_ce_full,
        "val_unigram_nll": val_unigram,
        "uniform_nll": float(math.log(codebook_size)),
    }
    save_prefix_head(
        latest,
        head,
        int(max_steps),
        embed_dim=embed_dim,
        codebook_size=codebook_size,
        context_length=context_length,
        config_name=cfg["name"],
        extra={"optimizer": optimizer.state_dict(), **metrics},
    )
    summary = {
        "system": "NCM-iWSP-GPT1",
        "format": PREFIX_LEVEL0_FORMAT,
        "checkpoint": str(latest),
        "log": str(log_path),
        "steps": int(max_steps),
        "resumed_from": start,
        "last_loss": last["loss"],
        "last_mean_T": last["mean_T"],
        "last_lengths": [int(value) for value in last["lengths"].tolist()],
        "tokenizer_unchanged": True,
        "generator_unchanged": True,
        "generator_loaded": generator_path is not None,
        "head_changed": bool(head_changed),
        "head_parameters": head_parameter_count(embed_dim, codebook_size, context_length),
        "predict_level0": False,
        "prefix_len": 1,
        "context_length": context_length,
        "codebook_size": codebook_size,
        "embed_dim": embed_dim,
        "device": str(resolved),
        "seed": seed_value,
        "val_every": int(val_every),
        "packed": {
            "kind": meta.get("kind"),
            "path": str(packed_path),
            "train_windows": int(train.shape[0]),
            "val_windows": int(val.shape[0]),
        },
        "does_not_submit_kaggle": True,
        **metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
