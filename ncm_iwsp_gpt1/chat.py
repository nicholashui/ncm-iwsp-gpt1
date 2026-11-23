"""One-window text decode for NCM-iWSP-GPT1.

This is not a causal language model. ``predict_level0`` is false, so level 0
is not a generator target. Shipped configs set ``prefix_len`` to 1, and the
weights were trained with that slot set to ``<|endoftext|>``. A short prompt
is repeated, with end-of-text between copies, until the context is full.
``VQVAE.encode`` then sees prompt tokens rather than a pad-filled window.
The generator samples the finer codes from that level-0 code and the decoder
emits one window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from ncm_iwsp_gpt1.build import build_models, lock_for_eval
from ncm_iwsp_gpt1.convert import FORMAT_NAME, ConvertError, load_converted_checkpoint
from ncm_iwsp_gpt1.device import resolve_device

FINETUNE_FORMAT = "ncm-iwsp-gpt1-finetune"
QUIT_COMMANDS = frozenset({"quit", "exit", "/quit", "/exit"})
_FOREIGN_BITS = ("post_upsample_convs.", ".proj.", "log_inv_T")
_SKIP_META = frozenset({"model", "optimizer", "scaler", "tokenizer", "generator"})

CHAT_DESCRIPTION = """\
Decode one text window with NCM-iWSP-GPT1.

This is not a causal language model and not ChatGPT-style continuation.
predict_level0 is false, so level 0 is not a generator target. The shipped
configs set prefix_len to 1. That one slot is <|endoftext|>, the prefix the
generator was trained with. It is an embedding lookup with no position added.
The prompt is not placed in that slot.

A short line is not padded out to the context. Padding a 4-token question
to 512 makes level 0 the same code as an all-pad window, and greedy decode
of that code collapses. The line is encoded with GPT-2 BPE, an end-of-text
token is appended, and that block is repeated until the context is full
(512 for configs/target.yaml, 32 for configs/gpt1_data.yaml). VQVAE.encode
runs on that window. Chat keeps only level 0, one code on both shipped
schedules. NextConceptGenerator.generate samples every finer level from
that code and the end-of-text prefix. The VQ-VAE decoder then turns the
code grid into one window. Pad and end-of-text ids are stripped before
printing.

The reply is that window, conditioned on the coarse code of the packed
prompt. It is not an answer to the question and not token continuation.
Turns are independent: the previous reply is not fed back.

Type a line and press Enter. Quit with quit, exit, /quit, or /exit.
Ctrl-D also quits. --prompt runs one turn and prints the decoded text on
stdout. temperature 0 is argmax. top_k 0 and top_p 1 disable truncation.

Weights, including v3, are not in this git repository. Layouts loaded as-is:
  --checkpoint FILE
      one file, format ncm-iwsp-gpt1-from-gradient-checkpoint
  --tokenizer tokenizer.pt --generator generator.pt
      each blob is typically {model, step, converted_from, ...}
  --tokenizer tokenizer.pt --finetune latest.pt
      latest.pt format ncm-iwsp-gpt1-finetune; the model state dict is the
      generator and the optimizer is ignored
With no weights, the seeded initialization is used and a warning is printed.
"""


class ChatError(ValueError):
    """The chat command cannot load weights or cannot decode a turn."""


class ChatSession:
    """Frozen tokenizer plus generator, ready to decode one window per turn."""

    def __init__(self, pretok, vqvae, generator, device, temperature, top_k, top_p, weights_label, weights_detail, config_name):
        self.pretok = pretok
        self.vqvae = vqvae
        self.generator = generator
        self.device = device
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.weights_label = str(weights_label)
        self.weights_detail = str(weights_detail)
        self.config_name = str(config_name)
        self.context_length = int(vqvae.context_length)
        self.prefix_len = int(generator.prefix_len)
        self.level0_len = int(vqvae.vq.levels[0])

    def reply(self, text: str) -> dict:
        """Encode ``text``, sample finer codes, and decode one window."""
        if not isinstance(text, str):
            raise ChatError("prompt must be a string")
        packed = pack_prompt_window(self.pretok, text, self.context_length)
        prefix_len = self.prefix_len
        if prefix_len < 0:
            raise ChatError("prefix_len must be >= 0")
        # Training feeds this slot a constant end-of-text id, not a content token.
        prefix_ids = [int(self.pretok.eos_token_id)] * prefix_len
        tokens = torch.tensor([packed["window_ids"]], dtype=torch.long, device=self.device)
        if prefix_len:
            prefix = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        else:
            prefix = torch.empty(1, 0, dtype=torch.long, device=self.device)
        with torch.no_grad():
            indices = self.vqvae.encode(tokens)
            level0 = indices[:, : self.level0_len]
            codes = self.generator.generate(
                level0,
                prefix,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            decoded_ids = self.vqvae.decode_logits(codes).argmax(dim=-1)[0]
        packed["prefix_ids"] = prefix_ids
        packed["prefix_len"] = prefix_len
        packed["text"] = self.pretok.decode(decoded_ids, skip_special_tokens=True)
        packed["level0"] = [int(value) for value in level0[0].tolist()]
        return packed


def is_quit(text: str) -> bool:
    return str(text).strip().lower() in QUIT_COMMANDS


def open_session(
    cfg: dict,
    device: str | torch.device | None = "auto",
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: int = 0,
    checkpoint: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    generator_path: str | Path | None = None,
    finetune_path: str | Path | None = None,
) -> ChatSession:
    """Build the modules, optionally load weights, and move them to ``device``."""
    _check_sampling(temperature, top_k, top_p, seed)
    resolved = resolve_device(device)
    prepared = prepare_weights(checkpoint, tokenizer_path, generator_path, finetune_path)
    pretok, vqvae, generator = build_models(cfg)
    if prepared is not None:
        prepared.apply(vqvae, generator)
        weights_label = prepared.layout
        weights_detail = prepared.detail
        del prepared
    else:
        weights_label = "init"
        weights_detail = "seeded initialization; no checkpoint loaded"
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    vqvae.to(resolved)
    generator.to(resolved)
    torch.manual_seed(int(seed))
    return ChatSession(
        pretok,
        vqvae,
        generator,
        resolved,
        temperature,
        top_k,
        top_p,
        weights_label,
        weights_detail,
        cfg.get("name", ""),
    )


def startup_message(session: ChatSession) -> str:
    lines = [
        (
            f"{session.config_name}: context {session.context_length}, "
            f"prefix_len {session.prefix_len}, device {session.device}. "
            "Not a causal LM. A short prompt is repeated with end-of-text "
            "between copies so level 0 is not a pad code. The generator "
            "prefix is end-of-text. Finer codes are sampled and decoded as "
            "one window. The previous reply is not used."
        ),
        (
            f"temperature={session.temperature:g} top_k={session.top_k} "
            f"top_p={session.top_p:g}"
        ),
    ]
    if session.weights_label == "init":
        lines.append(
            "warning: no checkpoint loaded. Replies use the seeded initialization, "
            "not downloaded v3 weights."
        )
    else:
        lines.append(session.weights_detail)
    return "\n".join(lines)


def format_note(result: dict) -> str:
    level0 = ",".join(str(value) for value in result["level0"])
    prefix = ",".join(str(value) for value in result["prefix_ids"])
    truncated = "yes" if result["truncated"] else "no"
    repeated = "yes" if result["repeated"] else "no"
    return (
        f"level0={level0} prefix={prefix} "
        f"prompt_tokens={result['prompt_tokens']} "
        f"encoded_pad={result['encoded_pad']} "
        f"repeated={repeated} truncated={truncated}"
    )


def run_prompt(session: ChatSession, text: str) -> int:
    print(startup_message(session), file=sys.stderr, flush=True)
    result = session.reply(text)
    print(result["text"], flush=True)
    print(format_note(result), file=sys.stderr, flush=True)
    return 0


def run_interactive(session: ChatSession) -> int:
    print(startup_message(session), file=sys.stderr, flush=True)
    print("Type a prompt. Quit with quit, exit, /quit, or /exit.", file=sys.stderr, flush=True)
    while True:
        try:
            line = input("you> ")
        except EOFError:
            print(file=sys.stderr)
            return 0
        except KeyboardInterrupt:
            print(file=sys.stderr)
            return 130
        if is_quit(line):
            return 0
        if not line.strip():
            continue
        result = session.reply(line)
        print(result["text"], flush=True)
        print(format_note(result), file=sys.stderr, flush=True)


def prepare_weights(checkpoint, tokenizer_path, generator_path, finetune_path):
    """Read checkpoint blobs before the modules are built.

    Returns None when no weights were requested. File, format, and tied-head
    errors are raised here so a wrong path does not allocate the model.
    """
    checkpoint, tokenizer_path, generator_path, finetune_path = _check_weight_flags(
        checkpoint, tokenizer_path, generator_path, finetune_path
    )
    if checkpoint is None and tokenizer_path is None:
        return None
    if checkpoint is not None:
        blob = _read_blob(checkpoint)
        if not isinstance(blob, dict) or blob.get("format") != FORMAT_NAME:
            found = blob.get("format") if isinstance(blob, dict) else type(blob).__name__
            raise ChatError(
                f"{checkpoint} is not an {FORMAT_NAME} checkpoint (found {found!r}). "
                "Run `python -m ncm_iwsp_gpt1 convert` first, or pass "
                "--tokenizer with --generator or --finetune."
            )
        if "tokenizer" not in blob or "generator" not in blob:
            raise ChatError(f"{checkpoint} is missing tokenizer or generator state")
        tok_state, tok_meta = _as_state(blob["tokenizer"], "tokenizer")
        gen_state, gen_meta = _as_state(blob["generator"], "generator")
        _require_tokenizer(tok_state)
        _require_untied(gen_state)
        # load_converted_checkpoint reads these entries. Keep the unwrapped
        # maps so a module. prefix is not required of the caller.
        blob["tokenizer"] = tok_state
        blob["generator"] = gen_state
        return _Prepared(
            layout="converted",
            detail=_detail("converted", None, None, checkpoint, tok_meta, gen_meta),
            tokenizer_state=tok_state,
            generator_state=gen_state,
            converted_blob=blob,
            checkpoint=str(checkpoint),
        )

    tok_blob = _read_blob(tokenizer_path)
    tok_state, tok_meta = _as_state(tok_blob, "tokenizer")
    _require_tokenizer(tok_state)
    del tok_blob
    gen_file = finetune_path if finetune_path is not None else generator_path
    gen_blob = _read_blob(gen_file)
    if finetune_path is not None:
        found = gen_blob.get("format") if isinstance(gen_blob, dict) else None
        if found != FINETUNE_FORMAT:
            raise ChatError(
                f"{finetune_path} format is {found!r}. "
                f"--finetune expects {FINETUNE_FORMAT}. "
                "The model entry is loaded into the generator; optimizer state is ignored."
            )
    gen_state, gen_meta = _as_state(gen_blob, "generator")
    _require_untied(gen_state)
    layout = "finetune" if (isinstance(gen_blob, dict) and gen_blob.get("format") == FINETUNE_FORMAT) else "split"
    del gen_blob
    return _Prepared(
        layout=layout,
        detail=_detail(layout, tokenizer_path, gen_file, None, tok_meta, gen_meta),
        tokenizer_state=tok_state,
        generator_state=gen_state,
    )


class _Prepared:
    def __init__(self, layout, detail, tokenizer_state, generator_state, converted_blob=None, checkpoint=None):
        self.layout = layout
        self.detail = detail
        self.tokenizer_state = tokenizer_state
        self.generator_state = generator_state
        self.converted_blob = converted_blob
        self.checkpoint = checkpoint

    def apply(self, vqvae, generator) -> None:
        if self.layout == "converted":
            try:
                load_converted_checkpoint(
                    self.checkpoint,
                    vqvae,
                    generator,
                    blob=self.converted_blob,
                )
            except ConvertError as exc:
                raise ChatError(str(exc)) from exc
            return
        _load_module(vqvae, self.tokenizer_state, "tokenizer")
        _load_module(generator, self.generator_state, "generator")


def pack_prompt_window(pretok, text: str, context_length: int) -> dict:
    """Fill one context with the prompt, without a pad tail.

    Training windows are slices of real tokens. A short paragraph is
    concatenated with what follows it; only a corpus tail is padded.
    Left-aligning a 4-token line and filling the other 508 positions with
    ``<|pad|>`` makes the level-0 vector match an all-pad window (cosine
    about 1 under the average-pool cascade this quantizer starts from), so
    the generator conditions on the pad code and greedy decode repeats a
    few frequent words.

    A short line is followed by end-of-text and that block is repeated to
    the context length. A line that is already longer than the context is
    truncated to the first ``context_length`` tokens.
    """
    if not isinstance(text, str):
        raise ChatError("prompt must be a string")
    context_length = int(context_length)
    if context_length < 1:
        raise ChatError("context_length must be positive")
    ids = list(pretok.encode(text))
    truncated = len(ids) > context_length
    kept = ids[:context_length]
    eos = int(pretok.eos_token_id)
    pad = int(pretok.pad_token_id)
    repeated = len(kept) < context_length
    if not repeated:
        window = list(kept)
    else:
        unit = (list(kept) + [eos]) if kept else [eos]
        window = []
        while len(window) < context_length:
            window.extend(unit)
        window = window[:context_length]
    return {
        "window_ids": window,
        "prompt_ids": kept,
        "truncated": bool(truncated),
        "repeated": bool(repeated),
        "prompt_tokens": len(kept),
        "encoded_pad": sum(token == pad for token in window),
        "context_length": context_length,
    }


def _check_sampling(temperature, top_k, top_p, seed) -> None:
    if float(temperature) < 0:
        raise ChatError("temperature must be >= 0; 0 is argmax")
    if int(top_k) < 0:
        raise ChatError("top_k must be >= 0; 0 disables top-k")
    top_p = float(top_p)
    if not 0.0 <= top_p <= 1.0:
        raise ChatError("top_p must be between 0 and 1; 1 disables nucleus sampling")
    if int(seed) < 0:
        raise ChatError("seed must be >= 0")


def _check_weight_flags(checkpoint, tokenizer_path, generator_path, finetune_path):
    checkpoint = _path_or_none(checkpoint)
    tokenizer_path = _path_or_none(tokenizer_path)
    generator_path = _path_or_none(generator_path)
    finetune_path = _path_or_none(finetune_path)
    if checkpoint is not None and any(path is not None for path in (tokenizer_path, generator_path, finetune_path)):
        raise ChatError("pass either --checkpoint or the split files, not both")
    if generator_path is not None and finetune_path is not None:
        raise ChatError("pass either --generator or --finetune, not both")
    if finetune_path is not None and tokenizer_path is None:
        raise ChatError("--finetune needs --tokenizer (the frozen tokenizer.pt)")
    if generator_path is not None and tokenizer_path is None:
        raise ChatError("--generator needs --tokenizer")
    if tokenizer_path is not None and generator_path is None and finetune_path is None:
        raise ChatError("--tokenizer needs --generator or --finetune")
    for path in (checkpoint, tokenizer_path, generator_path, finetune_path):
        if path is not None and not path.is_file():
            raise ChatError(f"checkpoint not found: {path}")
    return checkpoint, tokenizer_path, generator_path, finetune_path


def _path_or_none(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _detail(layout, tokenizer_path, generator_path, checkpoint, tok_meta, gen_meta) -> str:
    parts = [f"loaded {layout}"]
    if checkpoint is not None:
        parts.append(f"checkpoint={checkpoint}")
    if tokenizer_path is not None:
        parts.append(f"tokenizer={tokenizer_path}")
    if generator_path is not None:
        parts.append(f"generator={generator_path}")
    tok_step = _as_int(tok_meta.get("step"))
    gen_step = _as_int(gen_meta.get("step"))
    if tok_step is not None:
        parts.append(f"tokenizer_step={tok_step}")
    if gen_step is not None:
        parts.append(f"generator_step={gen_step}")
    return " ".join(parts)


def _read_blob(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _as_state(blob, label: str) -> tuple[dict, dict]:
    """Return a tensor state dict and the non-weight metadata.

    Accepts a raw tensor map, ``{model: ...}`` (split init or fine-tune), or
    a combined file's ``tokenizer`` / ``generator`` entry. Optimizer and
    scaler entries are not returned.
    """
    if not isinstance(blob, dict) or not blob:
        raise ChatError(f"{label} checkpoint is empty or not a dict")
    meta = {key: value for key, value in blob.items() if key not in _SKIP_META}
    state = None
    if isinstance(blob.get("model"), dict):
        state = blob["model"]
    elif label == "tokenizer" and isinstance(blob.get("tokenizer"), dict):
        state = blob["tokenizer"]
        meta = {key: value for key, value in blob.items() if key not in _SKIP_META}
    elif label == "generator" and isinstance(blob.get("generator"), dict):
        state = blob["generator"]
        meta = {key: value for key, value in blob.items() if key not in _SKIP_META}
    elif blob and all(torch.is_tensor(value) for value in blob.values()):
        state = blob
        meta = {}
    if state is None:
        raise ChatError(
            f"{label} checkpoint has no tensor state dict. "
            "Expected {model: ...}, a raw state dict, or the converted "
            f"{FORMAT_NAME} file passed to --checkpoint."
        )
    keys = [str(key) for key in state.keys()]
    if keys and all(key.startswith("module.") for key in keys):
        state = {key[len("module.") :]: value for key, value in state.items()}
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ChatError(f"{label} state dict contains a non-tensor")
    return state, meta


def _require_tokenizer(state: dict) -> None:
    if "encoder.wte.weight" not in state:
        raise ChatError(
            "tokenizer state has no encoder.wte.weight. "
            "Point --tokenizer at the frozen VQ-VAE file (tokenizer.pt)."
        )


def _require_untied(state: dict) -> None:
    foreign = [key for key in state if any(bit in str(key) for bit in _FOREIGN_BITS)]
    if foreign:
        raise ChatError(
            "generator checkpoint still has tied-head or post-upsample tensors "
            f"({foreign[0]}). Chat loads the converted untied generator "
            "(lm_heads.*.weight), not an NCM-GPT1 trainer checkpoint."
        )
    heads = [key for key in state if str(key).startswith("lm_heads.") and str(key).endswith(".weight")]
    if not heads:
        raise ChatError(
            "generator state has no lm_heads.*.weight. "
            "Use the converted generator.pt or a ncm-iwsp-gpt1-finetune latest.pt."
        )


def _load_module(module, state: dict, label: str) -> None:
    try:
        module.load_state_dict(state, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise ChatError(f"{label} checkpoint does not match this config: {exc}") from exc


def _as_int(value):
    if value is None or isinstance(value, bool):
        return None
    if torch.is_tensor(value) and value.numel() == 1:
        return int(value.item())
    if isinstance(value, (int, float)):
        return int(value)
    return None
