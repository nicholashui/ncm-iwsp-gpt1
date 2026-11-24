"""One-window continuation for NCM-iWSP-GPT1.

This is not a causal language model and not a chatbot. ``predict_level0`` is
false, so level 0 is not a generator target.

``configs/target.yaml`` and ``configs/gpt1_data.yaml`` are phase 1:
``prefix_len`` 1, and that slot stays ``<|endoftext|>`` so existing weights
see the prefix they were trained with. ``configs/target_phase2.yaml`` is the
longer prefix. Its slots hold the prompt (right-padded with the pad id,
truncated to ``prefix_len``). The default chat config stays phase 1.

The default path samples the single level-0 code from a prefix head on the
real prompt tokens (``encoder.wte``, not a window filled by repeating the
prompt). The generator then samples the finer codes. Packed-repeat encode
and a train-set unigram are explicit baselines, not the fallback. Missing
``--prefix-head`` still exits on both phases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from ncm_iwsp_gpt1.build import build_models, lock_for_eval
from ncm_iwsp_gpt1.convert import FORMAT_NAME, ConvertError, load_converted_checkpoint
from ncm_iwsp_gpt1.device import resolve_device
from ncm_iwsp_gpt1.prefix_level0 import (
    PREFIX_LEVEL0_FORMAT,
    PrefixLevel0Error,
    embed_prompt_tokens,
    level0_code_counts,
    load_prefix_head,
    load_windows_for_level0,
    prefix_blob_matches,
    read_prefix_blob,
    sample_logits,
)
from ncm_iwsp_gpt1.prefix_protocol import (
    display_continuation,
    legacy_prefix_warning,
    load_generator_state,
    prefix_metadata_problems,
    recorded_prefix,
    window_prefix_ids,
)

FINETUNE_FORMAT = "ncm-iwsp-gpt1-finetune"
QUIT_COMMANDS = frozenset({"quit", "exit", "/quit", "/exit"})
LEVEL0_SOURCES = ("prefix_head", "packed", "unigram")
_FOREIGN_BITS = ("post_upsample_convs.", ".proj.", "log_inv_T")
_SKIP_META = frozenset({"model", "optimizer", "scaler", "tokenizer", "generator"})
MISSING_PREFIX_HEAD = (
    "No prefix-head checkpoint was given, so chat will not decode. "
    f"Pass --prefix-head PATH (format {PREFIX_LEVEL0_FORMAT}) to sample "
    "level 0 from the real prompt tokens. The prompt is not repeated to fill "
    "the context. Packed-repeat encode is not the default. "
    "For that previous path, pass --level0-source packed. "
    "For a prompt-independent level-0 draw, pass --level0-source unigram --packed PATH."
)

CHAT_DESCRIPTION = f"""\
Continue one text window with NCM-iWSP-GPT1.

This is not a causal language model and not ChatGPT. It continues a short
prompt as one reconstructed window. It does not answer questions. predict_level0
is false, so level 0 is not a generator target.

Phase 1 is the default config, configs/target.yaml (and the small
configs/gpt1_data.yaml). Those configs set prefix_len to 1. That slot stays
<|endoftext|>, the prefix the existing generator weights were trained with.
The prompt is not placed in that slot.

Phase 2 is configs/target_phase2.yaml (small file: configs/gpt1_data_phase2.yaml).
prefix_len is 32 on the 12/768 file. The prompt tokens fill that prefix.
A short prompt is right-padded with the pad id. A longer prompt keeps its
first prefix_len tokens. Select phase 2 with --config. This command does not
start training.

The default path needs a prefix level-0 head (--prefix-head, format
{PREFIX_LEVEL0_FORMAT}). The head reads the real prompt through the frozen
tokenizer encoder.wte and samples one level-0 code. The line is not padded
to the context and not repeated until the context is full. NextConceptGenerator.generate
then samples the finer codes. The VQ-VAE decoder turns the code grid into one
window. stderr prints level0_source=prefix_head. On phase 2 it also prints
prefix_source=window. stdout may show the original prompt in front of the
decoded window; the scored text is the window the model decoded.

Without --prefix-head, chat stops. It does not silently encode a packed
repeat of the prompt. Two comparison baselines are explicit:
  --level0-source packed
      previous path: append end-of-text and repeat that block to the context
      (512 for configs/target.yaml, 32 for configs/gpt1_data.yaml), then
      VQVAE.encode and keep level 0
  --level0-source unigram --packed packed_windows.pt
      draw level 0 from an add-one unigram of those windows. The prompt does
      not choose the code. On phase 1 the generator prefix is still
      end-of-text. On phase 2 the prefix is the prompt; the phase-1 config
      remains the unigram-plus-end-of-text control.

Turns are independent: the previous reply is not fed back. temperature 0 is
argmax. top_k 0 and top_p 1 disable truncation. This is continuation, not a
chatbot reply.

Weights, including v3 and the prefix head, are not in this git repository.
Layouts loaded as-is:
  --checkpoint FILE
      one file, format {FORMAT_NAME}
  --tokenizer tokenizer.pt --generator generator.pt
      each blob is typically {{model, step, converted_from, ...}}
  --tokenizer tokenizer.pt --finetune latest.pt
      latest.pt format ncm-iwsp-gpt1-finetune; the model state dict is the
      generator and the optimizer is ignored
Train the head with `python -m ncm_iwsp_gpt1 train-prefix-level0`.
"""


class ChatError(ValueError):
    """The chat command cannot load weights or cannot decode a turn."""


class ChatSession:
    """Frozen tokenizer plus generator, ready to decode one window per turn."""

    def __init__(
        self,
        pretok,
        vqvae,
        generator,
        device,
        temperature,
        top_k,
        top_p,
        weights_label,
        weights_detail,
        config_name,
        level0_source,
        prefix_head=None,
        unigram_probs=None,
        level0_detail="",
    ):
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
        self.prefix_source = str(generator.prefix_source)
        self.level0_len = int(vqvae.vq.levels[0])
        self.level0_source = str(level0_source)
        self.prefix_head = prefix_head
        self.unigram_probs = unigram_probs
        self.level0_detail = str(level0_detail)
        self.prefix_warning = ""

    def reply(self, text: str) -> dict:
        """Sample level 0 for ``text``, then finer codes, and decode one window."""
        if not isinstance(text, str):
            raise ChatError("prompt must be a string")
        with torch.no_grad():
            if self.level0_source == "packed":
                packed = pack_prompt_window(self.pretok, text, self.context_length)
                tokens = torch.tensor([packed["window_ids"]], dtype=torch.long, device=self.device)
                level0 = self.vqvae.encode(tokens)[:, : self.level0_len]
            elif self.level0_source == "prefix_head":
                packed = real_prompt_tokens(self.pretok, text, self.context_length)
                if packed["prompt_tokens"] < 1:
                    raise ChatError(
                        "the prefix head needs at least one prompt token. "
                        "An empty line has no embedding to condition on."
                    )
                level0 = self._level0_from_head(packed["prompt_ids"])
            elif self.level0_source == "unigram":
                packed = real_prompt_tokens(self.pretok, text, self.context_length)
                level0 = self._level0_from_unigram()
            else:
                raise ChatError(f"unknown level0_source {self.level0_source!r}")
            prefix_ids, prefix = self._generator_prefix(packed["prompt_ids"])
            decoded_ids = self._decode_ids(level0, prefix)
        decoded = self.pretok.decode(decoded_ids, skip_special_tokens=True)
        if self.prefix_source == "window":
            display, prepended = display_continuation(text, decoded)
        else:
            display, prepended = decoded, False
        pad = int(self.pretok.pad_token_id)
        packed["prefix_ids"] = prefix_ids
        packed["prefix_len"] = len(prefix_ids)
        packed["prefix_source"] = self.prefix_source
        packed["prefix_pad"] = sum(token == pad for token in prefix_ids)
        packed["decoded_window"] = decoded
        packed["text"] = decoded
        packed["display_text"] = display
        packed["prompt_prepended"] = bool(prepended)
        packed["level0"] = [int(value) for value in level0[0].tolist()]
        packed["level0_source"] = self.level0_source
        return packed

    def _level0_from_head(self, prompt_ids: list[int]) -> torch.Tensor:
        if self.prefix_head is None:
            raise ChatError("prefix head is not loaded")
        if self.level0_len != 1:
            raise ChatError("the prefix head predicts a single level-0 code")
        token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        embeddings = embed_prompt_tokens(self.vqvae.encoder.wte, token_ids)
        lengths = torch.tensor([int(token_ids.shape[1])], dtype=torch.long, device=self.device)
        logits = self.prefix_head(embeddings, lengths)
        draw = sample_logits(logits, self.temperature, self.top_k, self.top_p)
        return draw.reshape(1, self.level0_len)

    def _level0_from_unigram(self) -> torch.Tensor:
        if self.unigram_probs is None:
            raise ChatError("unigram baseline has no level-0 counts")
        if self.level0_len != 1:
            raise ChatError("the unigram baseline draws a single level-0 code")
        logits = torch.log(self.unigram_probs.clamp_min(1e-12)).view(1, -1)
        draw = sample_logits(logits, self.temperature, self.top_k, self.top_p)
        return draw.reshape(1, self.level0_len)

    def _generator_prefix(self, prompt_ids: list[int]):
        """Phase 1 fills end-of-text. Phase 2 places the prompt in the prefix."""
        if self.prefix_source != "window":
            return _eos_prefix(self)
        prefix_ids = window_prefix_ids(prompt_ids, self.prefix_len, int(self.pretok.pad_token_id))
        if self.prefix_len:
            prefix = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        else:
            prefix = torch.empty(1, 0, dtype=torch.long, device=self.device)
        return prefix_ids, prefix

    def _decode_ids(self, level0, prefix) -> torch.Tensor:
        codes = self.generator.generate(
            level0,
            prefix,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        return self.vqvae.decode_logits(codes).argmax(dim=-1)[0]


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
    prefix_head_path: str | Path | None = None,
    level0_source: str = "prefix_head",
    packed_path: str | Path | None = None,
    unigram_max_windows: int = 256,
) -> ChatSession:
    """Build the modules, optionally load weights, and move them to ``device``.

    The default ``level0_source`` is ``prefix_head``. That path refuses to
    run until ``prefix_head_path`` is a real checkpoint. ``packed`` and
    ``unigram`` are comparison baselines and must be requested by name.
    """
    _check_sampling(temperature, top_k, top_p, seed)
    _check_weight_flags(checkpoint, tokenizer_path, generator_path, finetune_path)
    source, prefix_head_path, packed_path, head_blob = _check_level0_request(
        cfg, prefix_head_path, level0_source, packed_path, unigram_max_windows
    )
    resolved = resolve_device(device)
    prepared = prepare_weights(checkpoint, tokenizer_path, generator_path, finetune_path)
    pretok, vqvae, generator = build_models(cfg)
    prefix_warning = ""
    if prepared is not None:
        prepared.apply(vqvae, generator)
        problems = prefix_metadata_problems(
            recorded_prefix(prepared.generator_meta),
            int(generator.prefix_len),
            generator.prefix_source,
        )
        if problems:
            raise ChatError(
                "generator checkpoint prefix does not match this config: "
                + "; ".join(problems)
                + ". A checkpoint that records a different prefix_len stays on the "
                "config it was trained with. A v3 file that omits prefix_len can "
                "load on the phase-2 config only as a negative control."
            )
        prefix_warning = legacy_prefix_warning(prepared.generator_meta, generator.prefix_source) or ""
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
    prefix_head = None
    unigram_probs = None
    if source == "prefix_head":
        try:
            prefix_head = load_prefix_head(
                head_blob,
                int(cfg["tokenizer"]["embed_dim"]),
                int(cfg["tokenizer"]["codebook_size"]),
                int(cfg["tokenizer"]["context_length"]),
            )
        except PrefixLevel0Error as exc:
            raise ChatError(str(exc)) from exc
        lock_for_eval(prefix_head)
        prefix_head.to(resolved)
        step = _as_int(head_blob.get("step"))
        level0_detail = f"loaded prefix head {prefix_head_path}"
        if step is not None:
            level0_detail += f" step={step}"
    elif source == "unigram":
        try:
            windows = load_windows_for_level0(packed_path, int(vqvae.context_length))
            counts = level0_code_counts(
                vqvae,
                windows,
                max_windows=int(unigram_max_windows),
                batch_size=8,
            )
        except PrefixLevel0Error as exc:
            raise ChatError(str(exc)) from exc
        probs = (counts + 1.0) / (counts.sum() + counts.numel())
        unigram_probs = probs.to(device=resolved, dtype=torch.float32)
        used = min(int(unigram_max_windows), int(windows.shape[0]))
        level0_detail = f"unigram counts from {packed_path} windows={used}"
    else:
        level0_detail = (
            "baseline level0_source=packed: end-of-text is appended and the "
            "block is repeated to the context, then encoded. This is not the prefix head."
        )
    torch.manual_seed(int(seed))
    session = ChatSession(
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
        source,
        prefix_head,
        unigram_probs,
        level0_detail,
    )
    session.prefix_warning = prefix_warning
    return session


def startup_message(session: ChatSession) -> str:
    if session.prefix_source == "window":
        first = (
            f"{session.config_name}: context {session.context_length}, "
            f"prefix_len {session.prefix_len}, device {session.device}. "
            f"Not a causal LM. level0_source={session.level0_source}. "
            "prefix_source=window. Prompt tokens fill the generator prefix "
            f"(right-pad with pad id {int(session.pretok.pad_token_id)}, "
            "truncate to prefix_len). Level 0 still comes from the selected source. "
            "Finer codes are sampled and decoded as one window. "
            "The previous reply is not used. "
            "This is continuation, not a chatbot."
        )
    else:
        first = (
            f"{session.config_name}: context {session.context_length}, "
            f"prefix_len {session.prefix_len}, device {session.device}. "
            f"Not a causal LM. level0_source={session.level0_source}. "
            "The generator prefix is end-of-text. Finer codes are sampled and "
            "decoded as one window. The previous reply is not used. "
            "This is continuation, not a chatbot."
        )
    lines = [
        first,
        session.level0_detail,
        (
            f"temperature={session.temperature:g} top_k={session.top_k} "
            f"top_p={session.top_p:g}"
        ),
    ]
    if session.prefix_warning:
        lines.append(session.prefix_warning)
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
    prepended = "yes" if result.get("prompt_prepended") else "no"
    return (
        f"level0_source={result['level0_source']} "
        f"level0={level0} prefix={prefix} "
        f"prefix_source={result.get('prefix_source', 'eos')} "
        f"prompt_tokens={result['prompt_tokens']} "
        f"encoded_pad={result['encoded_pad']} "
        f"repeated={repeated} truncated={truncated} "
        f"prompt_prepended={prepended}"
    )


def run_prompt(session: ChatSession, text: str) -> int:
    print(startup_message(session), file=sys.stderr, flush=True)
    result = session.reply(text)
    print(result["display_text"], flush=True)
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
        print(result["display_text"], flush=True)
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
        for key in ("prefix_len", "prefix_source", "architecture"):
            if key in blob and key not in gen_meta:
                gen_meta[key] = blob[key]
        return _Prepared(
            layout="converted",
            detail=_detail("converted", None, None, checkpoint, tok_meta, gen_meta),
            tokenizer_state=tok_state,
            generator_state=gen_state,
            converted_blob=blob,
            checkpoint=str(checkpoint),
            generator_meta=gen_meta,
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
        generator_meta=gen_meta,
    )


class _Prepared:
    def __init__(
        self,
        layout,
        detail,
        tokenizer_state,
        generator_state,
        converted_blob=None,
        checkpoint=None,
        generator_meta=None,
    ):
        self.layout = layout
        self.detail = detail
        self.tokenizer_state = tokenizer_state
        self.generator_state = generator_state
        self.converted_blob = converted_blob
        self.checkpoint = checkpoint
        self.generator_meta = dict(generator_meta or {})

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


def _eos_prefix(session: ChatSession):
    prefix_len = int(session.prefix_len)
    if prefix_len < 0:
        raise ChatError("prefix_len must be >= 0")
    # Training feeds this slot a constant end-of-text id, not a content token.
    prefix_ids = [int(session.pretok.eos_token_id)] * prefix_len
    if prefix_len:
        prefix = torch.tensor([prefix_ids], dtype=torch.long, device=session.device)
    else:
        prefix = torch.empty(1, 0, dtype=torch.long, device=session.device)
    return prefix_ids, prefix


def real_prompt_tokens(pretok, text: str, context_length: int) -> dict:
    """Token ids of the prompt itself, truncated to the context and not repeated.

    The prefix head and the unigram baseline both report this list. Neither
    fills the rest of the window by tiling the prompt.
    """
    if not isinstance(text, str):
        raise ChatError("prompt must be a string")
    context_length = int(context_length)
    if context_length < 1:
        raise ChatError("context_length must be positive")
    ids = list(pretok.encode(text))
    truncated = len(ids) > context_length
    kept = ids[:context_length]
    pad = int(pretok.pad_token_id)
    return {
        "window_ids": list(kept),
        "prompt_ids": list(kept),
        "truncated": bool(truncated),
        "repeated": False,
        "prompt_tokens": len(kept),
        "encoded_pad": sum(token == pad for token in kept),
        "context_length": context_length,
    }


def _check_level0_request(cfg, prefix_head_path, level0_source, packed_path, unigram_max_windows):
    source = str(level0_source or "prefix_head").strip().lower().replace("-", "_")
    if source not in LEVEL0_SOURCES:
        raise ChatError(
            f"unknown --level0-source {level0_source!r}. Use prefix_head, packed, or unigram."
        )
    prefix_head_path = _path_or_none(prefix_head_path)
    packed_path = _path_or_none(packed_path)
    if source != "prefix_head" and prefix_head_path is not None:
        raise ChatError(
            "pass either --prefix-head or an explicit baseline "
            f"--level0-source {source}, not both"
        )
    if source != "unigram" and packed_path is not None:
        raise ChatError("--packed is only used with --level0-source unigram")
    if source in {"prefix_head", "unigram"} and int(cfg["tokenizer"]["levels"][0]) != 1:
        raise ChatError("this path expects a single level-0 code (levels[0] == 1)")
    blob = None
    if source == "prefix_head":
        if prefix_head_path is None:
            raise ChatError(MISSING_PREFIX_HEAD)
        try:
            blob = read_prefix_blob(prefix_head_path)
            prefix_blob_matches(
                blob,
                int(cfg["tokenizer"]["embed_dim"]),
                int(cfg["tokenizer"]["codebook_size"]),
                int(cfg["tokenizer"]["context_length"]),
            )
        except PrefixLevel0Error as exc:
            raise ChatError(str(exc)) from exc
    elif source == "unigram":
        if packed_path is None:
            raise ChatError(
                "--level0-source unigram needs --packed. "
                "The file is only used to count level-0 codes. The prompt is not encoded."
            )
        if not packed_path.is_file():
            raise ChatError(f"packed windows not found: {packed_path}")
        if int(unigram_max_windows) < 1:
            raise ChatError("--unigram-max-windows must be >= 1")
    return source, prefix_head_path, packed_path, blob


def pack_prompt_window(pretok, text: str, context_length: int) -> dict:
    """Fill one context by repeating the prompt. Packed baseline only.

    The prefix-head path does not call this. ``--level0-source packed`` does.
    The window is filled without a pad tail.

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
        if label == "generator":
            load_generator_state(module, state)
        else:
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
