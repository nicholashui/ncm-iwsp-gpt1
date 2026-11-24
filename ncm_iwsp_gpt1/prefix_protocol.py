"""Generator prefix protocol for NCM-iWSP-GPT1.

Phase 1 fills ``prefix_len`` with ``<|endoftext|>``. Phase 2 puts the start of
the window, or the prompt, into that same prefix. The attention mask is
already a prefix-LM; this module only decides which token ids occupy it.

Defaults that the plan left open are named here so a later change is a config
edit, not a silent branch:

- The 12/768 phase-2 config uses ``prefix_len`` 32, not 64.
- A short prefix is right-padded with ``pad_token_id`` (50257). The left side
  stays the start of the prompt. Extra tokens are dropped from the right.
- The checkpoint format string stays ``ncm-iwsp-gpt1-finetune``. New files
  record ``prefix_len`` and ``prefix_source``. A legacy file that omits them
  can warm-start the persistent weights.
"""

from __future__ import annotations

import torch

PREFIX_SOURCE_EOS = "eos"
PREFIX_SOURCE_WINDOW = "window"
PREFIX_SOURCES = (PREFIX_SOURCE_EOS, PREFIX_SOURCE_WINDOW)

# Smaller of the two plan candidates. 64 is shape-checked and not the config.
PHASE2_PREFIX_LEN = 32
PHASE2_PREFIX_CANDIDATES = (32, 64)
# Runnable context is 32, so the small phase-2 file uses a shorter prefix.
# Pad and truncation are then visible inside one window. 32 and 64 are still
# built in tests on that same small module.
PHASE2_SMALL_PREFIX_LEN = 8
PHASE2_PAD_SIDE = "right"

# These buffers are rebuilt from prefix_len in NextConceptGenerator.__init__.
# A v3 checkpoint must not be required to carry them, and a stale copy must
# not be loaded over the rebuilt ones.
_STALE_EXACT = frozenset({"attn_mask", "lvl_1L", "rope_cos", "rope_sin"})
_STALE_PREFIXES = ("rope_cos_gen_", "rope_sin_gen_")

PHASE2_FIXED_PROMPTS = ("How are you?", "Hello world", "The cat sat", "下一概念")

LEGACY_PREFIX_WARNING = (
    "checkpoint does not record prefix_len. "
    "Persistent transformer and lm_heads weights can still load, and mask or RoPE "
    "buffers are rebuilt. These weights may be a phase-1 fine-tune that only saw "
    "an end-of-text prefix. Loading them does not train the longer prefix."
)


class PrefixProtocolError(ValueError):
    """A prefix id or a checkpoint record cannot be used with this config."""


def normalize_prefix_source(value) -> str:
    """Return ``eos`` or ``window``. Missing means the phase-1 end-of-text prefix."""
    if value is None:
        return PREFIX_SOURCE_EOS
    text = str(value).strip().lower().replace("-", "_")
    if text in {"", "eos", "endoftext", "end_of_text"}:
        return PREFIX_SOURCE_EOS
    if text in {"window", "prompt", "tokens"}:
        return PREFIX_SOURCE_WINDOW
    raise ValueError(
        f"generator.prefix_source must be 'eos' or 'window' (found {value!r})"
    )


def prefix_source_of(generator) -> str:
    return normalize_prefix_source(getattr(generator, "prefix_source", PREFIX_SOURCE_EOS))


def window_prefix_ids(token_ids, prefix_len: int, pad_id: int) -> list[int]:
    """Left-align ``token_ids``, right-pad with ``pad_id``, and keep the start.

    ``prefix_len`` 0 returns an empty list. A longer id list is truncated to
    the first ``prefix_len`` tokens. This is the phase-2 inference prefix.
    """
    length = int(prefix_len)
    if length < 0:
        raise PrefixProtocolError("prefix_len must be >= 0")
    kept = [int(token) for token in list(token_ids)[:length]]
    if len(kept) < length:
        kept.extend([int(pad_id)] * (length - len(kept)))
    return kept


def batch_prefix_tokens(
    tokens: torch.Tensor,
    prefix_len: int,
    prefix_source: str,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    """Token ids that occupy the generator prefix for one training or eval batch.

    ``eos`` repeats the end-of-text id. ``window`` copies ``tokens[:, :prefix_len]``
    and right-pads with ``pad_id`` when the row is shorter than the prefix.
    Packed windows are already context length, so the pad branch is only the
    short-row case.
    """
    source = normalize_prefix_source(prefix_source)
    length = int(prefix_len)
    if length < 0:
        raise PrefixProtocolError("prefix_len must be >= 0")
    batch = int(tokens.shape[0])
    if length == 0:
        return torch.empty(batch, 0, dtype=torch.long, device=tokens.device)
    if source == PREFIX_SOURCE_EOS:
        return torch.full((batch, length), int(eos_id), dtype=torch.long, device=tokens.device)
    width = int(tokens.shape[1]) if tokens.ndim == 2 else 0
    if width >= length:
        return tokens[:, :length].to(dtype=torch.long)
    body = tokens.to(dtype=torch.long)
    pad = torch.full((batch, length - width), int(pad_id), dtype=torch.long, device=tokens.device)
    return torch.cat([body, pad], dim=1)


def display_continuation(prompt: str, decoded_window: str) -> tuple[str, bool]:
    """Optionally show the prompt in front of a decoded window.

    The decoded window itself is the evaluation text. The returned string is
    only for display, and only when the window does not already start with the
    prompt. A blank prompt is not prepended.
    """
    prompt_text = str(prompt).strip()
    decoded = decoded_window if isinstance(decoded_window, str) else str(decoded_window)
    if not prompt_text or decoded.startswith(prompt_text):
        return decoded, False
    return prompt_text + "\n" + decoded, True


def is_stale_generator_buffer(key: str) -> bool:
    name = str(key)
    if name.startswith("module."):
        name = name[len("module.") :]
    if name in _STALE_EXACT:
        return True
    return name.startswith(_STALE_PREFIXES)


def drop_stale_generator_buffers(state: dict) -> tuple[dict, list[str]]:
    """Drop mask and RoPE buffers so a longer prefix can load v3 weights."""
    if not isinstance(state, dict):
        raise PrefixProtocolError("generator state must be a dict")
    kept = {}
    dropped = []
    for key, value in state.items():
        name = str(key)
        if is_stale_generator_buffer(name):
            dropped.append(name)
            continue
        kept[name] = value
    return kept, dropped


def load_generator_state(module, state: dict) -> list[str]:
    """Strict-load persistent generator weights, ignoring stale mask and RoPE buffers."""
    cleaned, dropped = drop_stale_generator_buffers(state)
    module.load_state_dict(cleaned, strict=True)
    return dropped


def recorded_prefix(blob: dict | None) -> dict | None:
    """Read ``prefix_len`` / ``prefix_source`` from a checkpoint or its meta.

    Returns None when the file does not record a prefix. That is the v3
    ``latest.pt`` shape: persistent weights only, no protocol field.
    """
    if not isinstance(blob, dict):
        return None
    architecture = blob.get("architecture")
    source = architecture if isinstance(architecture, dict) else None
    if source is None or ("prefix_len" not in source and "prefix_source" not in source):
        source = blob
    if "prefix_len" not in source and "prefix_source" not in source:
        return None
    recorded = {}
    if "prefix_len" in source and source["prefix_len"] is not None:
        recorded["prefix_len"] = int(source["prefix_len"])
    if "prefix_source" in source and source["prefix_source"] is not None:
        recorded["prefix_source"] = normalize_prefix_source(source["prefix_source"])
    return recorded or None


def prefix_metadata_problems(recorded: dict | None, prefix_len: int, prefix_source: str) -> list[str]:
    """Differences between a recorded prefix and the module about to use it.

    A missing record is not a problem. Callers that must refuse a legacy file
    check for None themselves.
    """
    if not recorded:
        return []
    problems = []
    source = normalize_prefix_source(prefix_source)
    if "prefix_len" in recorded and int(recorded["prefix_len"]) != int(prefix_len):
        problems.append(f"prefix_len {recorded['prefix_len']} != {int(prefix_len)}")
    recorded_source = recorded.get("prefix_source", PREFIX_SOURCE_EOS)
    if recorded_source != source:
        problems.append(f"prefix_source {recorded_source} != {source}")
    return problems


def legacy_prefix_warning(blob: dict | None, prefix_source: str) -> str | None:
    """Warn when a phase-2 config loads weights that never recorded a prefix."""
    if normalize_prefix_source(prefix_source) != PREFIX_SOURCE_WINDOW:
        return None
    if recorded_prefix(blob) is None:
        return LEGACY_PREFIX_WARNING
    return None
