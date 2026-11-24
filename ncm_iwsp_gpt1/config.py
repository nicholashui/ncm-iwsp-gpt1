"""YAML configs. There is no optimizer section on purpose."""

from __future__ import annotations

from pathlib import Path

import yaml

from ncm_iwsp_gpt1.dialogue import normalize_history_placement
from ncm_iwsp_gpt1.levels import square_levels
from ncm_iwsp_gpt1.prefix_protocol import normalize_prefix_source


_RESERVED_NAMES = {"ncm-gpt1", "gpt1", "ncmgpt1"}


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    _validate(cfg, path)
    return cfg


def _folded(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "")


def _validate(cfg: dict, path: Path) -> None:
    for key in ("name", "seed", "pretok", "tokenizer", "generator", "data"):
        if key not in cfg:
            raise ValueError(f"{path} is missing '{key}'")
    if "train" in cfg or "optimizer" in cfg:
        raise ValueError(f"{path} must not contain a train or optimizer section")
    folded = _folded(cfg["name"])
    if folded in _RESERVED_NAMES or folded.startswith("ncm-gpt1"):
        raise ValueError(
            "do not name this config NCM-GPT1. That name is the gradient BookCorpus model. "
            "This system is NCM-iWSP-GPT1."
        )
    if "iwsp" not in folded:
        raise ValueError("config name must identify NCM-iWSP-GPT1, not the gradient NCM-GPT1 run")
    pretok = cfg["pretok"]
    if str(pretok.get("kind", "")) != "gpt2":
        raise ValueError("pretok.kind must be gpt2")
    pad_to = int(pretok["pad_vocab_to"])
    if pad_to < 50258 or pad_to % 64 != 0:
        raise ValueError("pretok.pad_vocab_to must be a multiple of 64 and at least 50258")
    tokenizer = cfg["tokenizer"]
    context = int(tokenizer["context_length"])
    levels = [int(x) for x in tokenizer["levels"]]
    expected = square_levels(context)
    if levels != expected:
        raise ValueError(
            f"tokenizer.levels must be the square schedule for context {context}: {expected}"
        )
    if int(tokenizer.get("enc_layers", 0)) != 0:
        raise ValueError("enc_layers must stay 0 (embedding lookup only)")
    if int(tokenizer.get("dec_layers", 1)) != 1:
        raise ValueError("dec_layers must stay 1")
    codebook = tokenizer["codebook_size"]
    if isinstance(codebook, (list, tuple)):
        raise ValueError("codebook_size must be one integer shared by every level")
    if int(codebook) < 2:
        raise ValueError("codebook_size must be at least 2")
    generator = cfg["generator"]
    if generator.get("predict_level0", False):
        raise ValueError("predict_level0 must stay false, as in NCM-GPT1")
    try:
        normalize_prefix_source(generator.get("prefix_source", "eos"))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    prefix_len = generator.get("prefix_len", 1)
    if isinstance(prefix_len, bool) or not isinstance(prefix_len, (int, float)) or int(prefix_len) < 0:
        raise ValueError("generator.prefix_len must be an integer >= 0")
    if int(prefix_len) != prefix_len:
        raise ValueError("generator.prefix_len must be an integer >= 0")
    if generator.get("attn_pattern", "block_diagonal") != "block_diagonal":
        raise ValueError("attn_pattern must stay block_diagonal")
    if int(generator["n_embd"]) % int(generator["n_head"]) != 0:
        raise ValueError("generator n_embd must be divisible by n_head")
    if int(tokenizer["embed_dim"]) % int(tokenizer["n_heads"]) != 0:
        raise ValueError("tokenizer embed_dim must be divisible by n_heads")
    hidden = int(cfg["data"]["hidden_pct"])
    visible = int(cfg["data"]["visible_pct"])
    if hidden < 0 or visible < 0 or hidden + visible >= 100:
        raise ValueError("hidden_pct and visible_pct must be non-negative and sum to < 100")
    _validate_dialogue(cfg)


def _validate_dialogue(cfg: dict) -> None:
    """Phase 3 block. Absent on phase 1 and phase 2 configs."""
    if "dialogue" not in cfg or cfg["dialogue"] is None:
        return
    dialogue = cfg["dialogue"]
    if not isinstance(dialogue, dict):
        raise ValueError("dialogue must be a mapping")
    if "add_special_role_tokens" in dialogue and dialogue["add_special_role_tokens"] is not False:
        raise ValueError(
            "dialogue.add_special_role_tokens must stay false. "
            "Phase 3 does not add user/assistant ids or expand GPT-2 BPE or encoder.wte."
        )
    try:
        normalize_history_placement(dialogue.get("history_placement"))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    truncation = str(dialogue.get("truncation", "keep_tail")).strip().lower()
    if truncation != "keep_tail":
        raise ValueError(
            "dialogue.truncation must be keep_tail (drop the oldest tokens from the left)"
        )
    pad_side = str(dialogue.get("pad_side", "right")).strip().lower()
    if pad_side != "right":
        raise ValueError("dialogue.pad_side must be right")
    for key, default in (
        ("user_marker", "User"),
        ("assistant_marker", "Assistant"),
        ("system_marker", "System"),
    ):
        marker = str(dialogue.get(key, default)).strip()
        if not marker or marker.startswith("<") or "<|" in marker:
            raise ValueError(f"dialogue.{key} must be a plain-text role marker, not a special token")
