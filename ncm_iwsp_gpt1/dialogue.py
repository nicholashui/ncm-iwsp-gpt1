"""Phase 3 dialogue packing, history placement, and reply slicing.

This is not a causal language model and not a selected chatbot corpus.
``predict_level0`` stays false. The tokenizer stays the frozen GPT-2 BPE
pretokenizer: role markers are plain text (``User``, ``Assistant``), not new
ids, and ``encoder.wte`` is not expanded.

No dialogue or instruction dataset is chosen here. ``pack-dialogue`` requires
a source, a dataset id, an allow-list that contains that id, and a license or
authorization note. Packed windows stay out of git. Banned names still go
through ``packing._refuse_corpus_name`` / ``refuse_banned_label``.

History placement is ``prefix``, ``window``, or ``both``. Truncation is
``keep_tail``: drop the oldest tokens from the left, then right-pad with the
pad id. That is not the phase-2 prompt rule, which keeps the first
``prefix_len`` tokens. Phase 1 and phase 2 do not call this module.

The decoded object is still one context-length window. ``slice_reply`` is the
user-facing cut. Code cross-entropy is not a chatbot score.
"""

from __future__ import annotations

import json
from pathlib import Path

from ncm_iwsp_gpt1.packing import (
    _refuse_corpus_name,
    document_split,
    nonpad_fraction,
    refuse_banned_label,
)

DIALOGUE_FORMAT = "ncm-iwsp-gpt1-dialogue"
DIALOGUE_PACK_FORMAT = "dialogue_windows"
HISTORY_PLACEMENTS = ("prefix", "window", "both")
TRUNCATION = "keep_tail"
PAD_SIDE = "right"
BOOKCORPUS_DATASET = "lucadiliello/bookcorpusopen"

# Plain text. Not special tokens, and not added to the GPT-2 vocabulary.
DEFAULT_USER_MARKER = "User"
DEFAULT_ASSISTANT_MARKER = "Assistant"
DEFAULT_SYSTEM_MARKER = "System"

TRUNCATION_POLICY = (
    "keep_tail: dialogue token streams are oldest-first. Tokens that do not "
    "fit are dropped from the left so the newest turn stays. A short sequence "
    "is right-padded with pad id 50257. Phase 2 prompt prefixes still keep the "
    "first tokens; this rule is only for dialogue history. "
    "prefix: the generator prefix is the tail of the conditioning turns, and "
    "the reconstructed window is the assistant reply only. "
    "window: the reconstructed window is the whole dialogue, and the generator "
    "prefix is the first prefix_len tokens of that window. At inference the "
    "reply does not exist yet, so that prefix is the start of the truncated "
    "history and can differ from the training prefix. "
    "both: the window is the whole dialogue, and the prefix is the conditioning "
    "tail, so training and inference prefixes match. The same history tokens "
    "can sit in the prefix and in the window."
)

DIALOGUE_QUALITY_NOTE = (
    "ce_first, ce_mid, and ce_last are raw per-level code means. "
    "They stay comparable as code cross-entropy when alpha and mid_aux match. "
    "The optimized total loss is not comparable across different alpha or mid_aux. "
    "Code cross-entropy is not a chatbot or Q&A score."
)

_USER_ROLES = frozenset({"user", "human"})
_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "bot", "model"})
_SYSTEM_ROLES = frozenset({"system"})
_USER_FIELDS = ("prompt", "instruction", "question")
_ASSISTANT_FIELDS = ("response", "output", "answer")


class DialogueError(ValueError):
    """A dialogue pack, history, or reply slice cannot be built."""


def normalize_history_placement(value) -> str:
    """Return ``prefix``, ``window``, or ``both``."""
    text = str(value).strip().lower().replace("-", "_")
    if text in HISTORY_PLACEMENTS:
        return text
    raise DialogueError(
        f"history_placement must be prefix, window, or both (found {value!r})"
    )


def dialogue_enabled(cfg: dict) -> bool:
    block = cfg.get("dialogue") if isinstance(cfg, dict) else None
    return isinstance(block, dict) and bool(block.get("enabled"))


def dialogue_settings(cfg: dict) -> dict:
    """Read a phase-3 dialogue block. Phase 1 and 2 configs have none."""
    if not isinstance(cfg, dict) or not isinstance(cfg.get("dialogue"), dict):
        raise DialogueError(
            "dialogue mode needs a phase-3 config with a dialogue section "
            "(configs/target_phase3.yaml or configs/gpt1_data_phase3.yaml). "
            "Phase 1 and phase 2 configs do not turn history on."
        )
    block = cfg["dialogue"]
    if not block.get("enabled"):
        raise DialogueError(
            "dialogue.enabled is false. Phase 3 stays off until that flag is true."
        )
    if block.get("add_special_role_tokens") is True:
        raise DialogueError(
            "dialogue.add_special_role_tokens must stay false. "
            "Phase 3 does not add user/assistant ids or expand GPT-2 BPE or encoder.wte."
        )
    placement = normalize_history_placement(block.get("history_placement"))
    truncation = str(block.get("truncation", TRUNCATION)).strip().lower()
    if truncation != TRUNCATION:
        raise DialogueError(
            f"dialogue.truncation must be {TRUNCATION!r} (drop oldest tokens from the left)"
        )
    pad_side = str(block.get("pad_side", PAD_SIDE)).strip().lower()
    if pad_side != PAD_SIDE:
        raise DialogueError("dialogue.pad_side must be right (pad id 50257)")
    user_marker = _plain_marker(block.get("user_marker", DEFAULT_USER_MARKER), "user_marker")
    assistant_marker = _plain_marker(
        block.get("assistant_marker", DEFAULT_ASSISTANT_MARKER), "assistant_marker"
    )
    system_marker = _plain_marker(
        block.get("system_marker", DEFAULT_SYSTEM_MARKER), "system_marker"
    )
    return {
        "enabled": True,
        "history_placement": placement,
        "truncation": truncation,
        "pad_side": pad_side,
        "user_marker": user_marker,
        "assistant_marker": assistant_marker,
        "system_marker": system_marker,
        "add_special_role_tokens": False,
        "prefix_len": int(cfg["generator"]["prefix_len"]),
        "context_length": int(cfg["tokenizer"]["context_length"]),
    }


def _plain_marker(value, label: str) -> str:
    marker = str(value).strip()
    if not marker or marker.startswith("<") or "<|" in marker:
        raise DialogueError(
            f"dialogue.{label} must be a plain-text role marker, not a special token"
        )
    return marker


def checkpoint_is_dialogue(meta: dict | None) -> bool:
    """True only for a dialogue-format generator checkpoint.

    A phase-1 or phase-2 fine-tune file is not enough to call chat a chatbot.
    """
    if not isinstance(meta, dict):
        return False
    return meta.get("format") == DIALOGUE_FORMAT


def fit_token_budget(token_ids, budget: int, pad_id: int) -> dict:
    """Keep the newest tokens, then right-pad.

    ``budget`` 0 returns an empty list. Extra tokens are dropped from the
    left. A short list is padded on the right with ``pad_id``.
    """
    budget = int(budget)
    if budget < 0:
        raise DialogueError("token budget must be >= 0")
    ids = [int(token) for token in token_ids]
    dropped = 0
    if len(ids) > budget:
        dropped = len(ids) - budget
        ids = ids[-budget:]
    padded = budget - len(ids)
    if padded:
        ids = ids + [int(pad_id)] * padded
    return {
        "ids": ids,
        "dropped": int(dropped),
        "padded": int(padded),
        "truncated": dropped > 0,
    }


def render_turn(marker: str, content: str) -> str:
    marker = _plain_marker(marker, "marker")
    body = str(content).strip()
    if not body:
        raise DialogueError("a dialogue turn is empty")
    return f"{marker}: {body}"


def marker_for(role: str, user_marker: str, assistant_marker: str, system_marker: str) -> str:
    if role == "user":
        return user_marker
    if role == "assistant":
        return assistant_marker
    if role == "system":
        return system_marker
    raise DialogueError(f"unknown dialogue role {role!r}")


def render_transcript(turns, user_marker, assistant_marker, system_marker=DEFAULT_SYSTEM_MARKER) -> str:
    """Newline-joined role lines. This string is the overlap check, not the token stream."""
    lines = []
    for turn in turns:
        marker = marker_for(turn["role"], user_marker, assistant_marker, system_marker)
        lines.append(render_turn(marker, turn["content"]))
    return "\n".join(lines)


def encode_turn(pretok, turn, user_marker, assistant_marker, system_marker=DEFAULT_SYSTEM_MARKER) -> list[int]:
    """One role line plus EOS. No new vocabulary ids."""
    marker = marker_for(turn["role"], user_marker, assistant_marker, system_marker)
    ids = list(pretok.encode(render_turn(marker, turn["content"])) or [])
    if not ids:
        raise DialogueError("a dialogue turn encoded to no tokens")
    return ids + [int(pretok.eos_token_id)]


def encode_turns(pretok, turns, user_marker, assistant_marker, system_marker=DEFAULT_SYSTEM_MARKER) -> list[int]:
    ids: list[int] = []
    for turn in turns:
        ids.extend(encode_turn(pretok, turn, user_marker, assistant_marker, system_marker))
    return ids


def training_prefix_and_window(
    conditioning_ids,
    reply_ids,
    *,
    context_length: int,
    prefix_len: int,
    placement: str,
    pad_id: int,
) -> dict:
    """Build one training window and the generator prefix for that window.

    The window is always ``context_length`` tokens. The prefix is always
    ``prefix_len`` tokens. Level 0 is not a target; callers still encode the
    whole window and predict levels 1..K-1.
    """
    placement = normalize_history_placement(placement)
    reply_ids = [int(token) for token in reply_ids]
    conditioning_ids = [int(token) for token in conditioning_ids]
    if not reply_ids:
        raise DialogueError("a dialogue example needs an assistant reply")
    if placement == "prefix":
        window = fit_token_budget(reply_ids, int(context_length), int(pad_id))
    else:
        window = fit_token_budget(conditioning_ids + reply_ids, int(context_length), int(pad_id))
    if placement == "window":
        prefix_ids = list(window["ids"][: int(prefix_len)])
        prefix_dropped = window["dropped"]
    else:
        fitted = fit_token_budget(conditioning_ids, int(prefix_len), int(pad_id))
        prefix_ids = fitted["ids"]
        prefix_dropped = fitted["dropped"]
    if len(prefix_ids) != int(prefix_len):
        raise DialogueError("dialogue prefix was not padded to prefix_len")
    return {
        "window_ids": window["ids"],
        "prefix_ids": prefix_ids,
        "window_dropped": window["dropped"],
        "prefix_dropped": int(prefix_dropped),
        "placement": placement,
    }


def inference_prefix_ids(
    conditioning_ids,
    *,
    context_length: int,
    prefix_len: int,
    placement: str,
    pad_id: int,
) -> dict:
    """Generator prefix and prefix-head ids for one dialogue turn.

    The reply is not known yet. ``prefix`` and ``both`` use the tail of the
    conditioning ids, matching training. ``window`` uses the first
    ``prefix_len`` tokens of the left-truncated conditioning span.
    The head sees the unpadded tail, capped at the context length.
    """
    placement = normalize_history_placement(placement)
    fitted = fit_token_budget(conditioning_ids, int(context_length), int(pad_id))
    head_ids = [token for token in fitted["ids"] if int(token) != int(pad_id)]
    if placement == "window":
        prefix_ids = list(fitted["ids"][: int(prefix_len)])
        prefix_dropped = fitted["dropped"]
    else:
        prefix = fit_token_budget(conditioning_ids, int(prefix_len), int(pad_id))
        prefix_ids = prefix["ids"]
        prefix_dropped = prefix["dropped"]
    return {
        "prefix_ids": prefix_ids,
        "head_ids": head_ids,
        "history_dropped": int(fitted["dropped"]),
        "prefix_dropped": int(prefix_dropped),
        "placement": placement,
    }


def slice_reply(
    decoded_window: str,
    conditioning_text: str,
    *,
    current_user_line: str,
    user_marker: str,
    assistant_marker: str,
    pretok=None,
    max_reply_tokens: int = 0,
) -> dict:
    """Cut a user-facing reply out of one decoded window.

    The decoded window stays the evaluation text. This slice is display only.

    1. If the window starts with the rendered conditioning text, drop that prefix.
    2. Otherwise, if it starts with the current user line, drop that line.
       Overlap is only a leading strip. Text in the middle is left alone.
    3. If the remainder starts with the assistant marker, drop that marker.
    4. Cut before the next line that starts with the user marker. A later user
       turn the model wrote is not part of the reply.
    5. ``max_reply_tokens`` 0 leaves the remainder. A positive cap keeps the
       first that many BPE tokens of the remainder.
    """
    window = decoded_window if isinstance(decoded_window, str) else str(decoded_window)
    body = window
    stripped_overlap = False
    conditioning = str(conditioning_text or "").strip()
    if conditioning and body.startswith(conditioning):
        body = body[len(conditioning) :]
        stripped_overlap = True
    else:
        user_line = str(current_user_line or "").strip()
        if user_line and body.startswith(user_line):
            rest = body[len(user_line) :]
            if rest.startswith("\n"):
                rest = rest[1:]
            body = rest
            stripped_overlap = True
    body = body.lstrip("\n")
    body, stripped_marker = _strip_leading_marker(body, assistant_marker)
    body, cut_user = _cut_next_user_turn(body, user_marker)
    body = body.strip()
    truncated_tokens = False
    cap = int(max_reply_tokens or 0)
    if cap < 0:
        raise DialogueError("max_reply_tokens must be >= 0")
    if cap and pretok is not None and body:
        ids = list(pretok.encode(body) or [])
        if len(ids) > cap:
            body = pretok.decode(ids[:cap])
            truncated_tokens = True
    return {
        "reply": body,
        "stripped_overlap": bool(stripped_overlap),
        "stripped_assistant_marker": bool(stripped_marker),
        "cut_at_user_marker": bool(cut_user),
        "truncated_reply_tokens": bool(truncated_tokens),
    }


def _strip_leading_marker(text: str, marker: str) -> tuple[str, bool]:
    prefix = f"{marker}:"
    if not text.startswith(prefix):
        return text, False
    rest = text[len(prefix) :].lstrip()
    return rest, True


def _cut_next_user_turn(text: str, user_marker: str) -> tuple[str, bool]:
    needle = f"{user_marker}:"
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.startswith(needle):
            return "\n".join(lines[:index]).strip(), True
    return text, False


def _role_name(raw) -> str:
    text = str(raw or "").strip().lower()
    if text in _USER_ROLES:
        return "user"
    if text in _ASSISTANT_ROLES:
        return "assistant"
    if text in _SYSTEM_ROLES:
        return "system"
    raise DialogueError(
        f"unknown dialogue role {raw!r}. Use user/assistant/system "
        "(human and gpt are accepted as role aliases, not as a dataset name)."
    )


def _turn(role: str, content) -> dict:
    if not isinstance(content, str) or not content.strip():
        raise DialogueError(f"{role} turn has no text")
    return {"role": role, "content": content.strip()}


def parse_dialogue_turns(record) -> list[dict]:
    """One JSON object to role/content turns.

    Accepted shapes, none of which names a corpus: ``messages`` or
    ``conversations`` lists of ``{role, content}`` (``from`` / ``value``
    aliases are accepted), or a single pair under prompt/instruction/question
    plus response/output/answer. A bare ``text`` row is not a dialogue.
    """
    if not isinstance(record, dict):
        raise DialogueError("dialogue row must be a JSON object")
    messages = record.get("messages", record.get("conversations"))
    if isinstance(messages, list):
        if not messages:
            raise DialogueError("dialogue row has an empty message list")
        turns = []
        for item in messages:
            if isinstance(item, str) and item.strip():
                raise DialogueError(
                    "dialogue messages must be role/content objects, not bare strings"
                )
            if not isinstance(item, dict):
                raise DialogueError("each dialogue message must be an object")
            role = _role_name(item.get("role", item.get("from", item.get("speaker"))))
            content = item.get("content", item.get("value", item.get("text")))
            turns.append(_turn(role, content))
        return turns
    user_key = next((key for key in _USER_FIELDS if isinstance(record.get(key), str)), None)
    reply_key = next((key for key in _ASSISTANT_FIELDS if isinstance(record.get(key), str)), None)
    if user_key and reply_key:
        return [_turn("user", record[user_key]), _turn("assistant", record[reply_key])]
    if "text" in record or "content" in record:
        raise DialogueError(
            "a text/content row is the BookCorpus pack path, not pack-dialogue. "
            "Dialogue rows need messages or a prompt/response pair."
        )
    raise DialogueError(
        "dialogue row needs messages, conversations, or a prompt/response pair. "
        "No dataset schema is selected beyond those fields."
    )


def iter_reply_examples(turns: list[dict]):
    """Yield ``(history, reply)`` for each assistant turn that follows a user turn."""
    history: list[dict] = []
    produced = False
    for turn in turns:
        if turn["role"] == "assistant" and any(item["role"] == "user" for item in history):
            yield list(history), dict(turn)
            produced = True
        history.append(turn)
    if not produced:
        raise DialogueError("dialogue row has no assistant reply after a user turn")


def load_allowlist(path: str | Path) -> list[str]:
    """Read dataset ids, one per line. Blank lines and ``#`` comments are ignored.

    A banned label on the list is still refused. An empty file allows nothing.
    """
    path = Path(path)
    if not path.is_file():
        raise DialogueError(f"allow-list not found: {path}")
    found = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            refuse_banned_label(stripped)
        except ValueError as exc:
            raise DialogueError(
                f"{path}:{line_no} is a banned corpus name and cannot be allow-listed. {exc}"
            ) from exc
        if stripped not in seen:
            seen.add(stripped)
            found.append(stripped)
    return found


def require_dataset_authorization(dataset_id: str, allowlist_path: str | Path, license_note: str) -> str:
    """Require an explicit id, an allow-list hit, and a license note.

    The id is not defaulted. ``lucadiliello/bookcorpusopen`` stays on ``pack``.
    """
    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        raise DialogueError(
            "pack-dialogue and finetune-dialogue need --dataset-id. "
            "No dialogue corpus is selected in this repository."
        )
    try:
        refuse_banned_label(dataset_id)
    except ValueError as exc:
        raise DialogueError(str(exc)) from exc
    if dataset_id.lower() == BOOKCORPUS_DATASET:
        raise DialogueError(
            f"{BOOKCORPUS_DATASET} stays on `pack`, not pack-dialogue. "
            "Dialogue packing needs its own dataset id."
        )
    note = str(license_note or "").strip()
    if len(note) < 12:
        raise DialogueError(
            "pass --license-note with the dataset's license or authorization. "
            "This repository does not supply one."
        )
    allowed = load_allowlist(allowlist_path)
    if dataset_id not in allowed:
        raise DialogueError(
            f"dataset id {dataset_id!r} is not on the allow-list {allowlist_path}. "
            "The shipped configs/dialogue_allowlist.txt names no corpus. "
            "Add the id only after the dataset is chosen outside this repository."
        )
    return dataset_id


def iter_dialogue_jsonl(path: str | Path):
    """Yield JSON objects. Calls ``_refuse_corpus_name`` before reading."""
    path = Path(path)
    if not path.is_file():
        raise DialogueError(
            f"dialogue jsonl not found: {path}. Download is off by default. "
            "Pass a local file, or --hf-dataset with --download."
        )
    _refuse_corpus_name(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DialogueError(f"{path}:{line_no} is not JSON") from exc
            if not isinstance(obj, dict):
                raise DialogueError(f"{path}:{line_no} must be a JSON object")
            yield obj


def iter_dialogue_tables(path: str | Path):
    """Yield dialogue records from an arrow/parquet file or a directory of them.

    A ``text``-only table is refused so the BookCorpus reader stays the path
    for those files. This does not download anything. Each file name is passed
    through ``_refuse_corpus_name``.
    """
    path = Path(path)
    if not path.exists():
        raise DialogueError(f"dialogue table path not found: {path}")
    if path.is_file():
        _refuse_corpus_name(path)
        files = [path]
    else:
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".arrow", ".parquet"}
        )
        if not files:
            raise DialogueError(f"no .arrow or .parquet files under {path}")
        for item in files:
            _refuse_corpus_name(item)
    found = 0
    for file in files:
        for record in _records_from_table_file(file):
            found += 1
            yield record
    if found == 0:
        raise DialogueError(f"no dialogue rows found under {path}")


def _records_from_table_file(path: Path):
    from ncm_iwsp_gpt1.packing import _load_table

    table = _load_table(path)
    names = list(table.column_names)
    present = set(names)
    if "messages" in present or "conversations" in present:
        key = "messages" if "messages" in present else "conversations"
        produced = False
        for value in table.column(key).to_pylist():
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise DialogueError(f"{path} column {key!r} has a non-JSON string") from exc
            if value:
                produced = True
                yield {key: value}
        if not produced:
            raise DialogueError(f"{path} column {key!r} had no dialogue rows")
        return
    user_key = next((key for key in _USER_FIELDS if key in present), None)
    reply_key = next((key for key in _ASSISTANT_FIELDS if key in present), None)
    if user_key and reply_key:
        users = table.column(user_key).to_pylist()
        replies = table.column(reply_key).to_pylist()
        produced = False
        for user, reply in zip(users, replies):
            if isinstance(user, str) and user.strip() and isinstance(reply, str) and reply.strip():
                produced = True
                yield {user_key: user, reply_key: reply}
        if not produced:
            raise DialogueError(f"{path} columns {user_key!r}/{reply_key!r} had no text")
        return
    raise DialogueError(
        f"{path} is not a dialogue table (columns {names}). "
        "A text column belongs on `pack`, not pack-dialogue."
    )


def pack_dialogue_records(
    records,
    pretok,
    *,
    context_length: int,
    prefix_len: int,
    placement: str,
    dataset_id: str,
    license_note: str,
    source: str,
    user_marker: str = DEFAULT_USER_MARKER,
    assistant_marker: str = DEFAULT_ASSISTANT_MARKER,
    system_marker: str = DEFAULT_SYSTEM_MARKER,
    hidden_pct: int = 10,
    visible_pct: int = 10,
    max_windows: int = 0,
) -> dict:
    """Pack dialogue rows to one window per assistant reply.

    Unlike BookCorpus packing, rows are not concatenated into one token
    stream. A row that is longer than the context keeps its tail.
    """
    import torch

    placement = normalize_history_placement(placement)
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise DialogueError("dataset id is required and is not defaulted")
    note = str(license_note or "").strip()
    if len(note) < 12:
        raise DialogueError("a license or authorization note is required")
    try:
        refuse_banned_label(dataset_id)
    except ValueError as exc:
        raise DialogueError(str(exc)) from exc
    if dataset_id.lower() == BOOKCORPUS_DATASET:
        raise DialogueError(f"{BOOKCORPUS_DATASET} cannot be packed as dialogue")
    user_marker = _plain_marker(user_marker, "user_marker")
    assistant_marker = _plain_marker(assistant_marker, "assistant_marker")
    system_marker = _plain_marker(system_marker, "system_marker")
    length = int(context_length)
    prefix = int(prefix_len)
    if length < 2:
        raise DialogueError("context_length must be at least 2")
    if prefix < 0:
        raise DialogueError("prefix_len must be >= 0")
    groups = {"visible_train": [], "visible": [], "hidden": []}
    n_rows = 0
    for record in records:
        n_rows += 1
        turns = parse_dialogue_turns(record)
        key = "\n".join(f"{turn['role']}\t{turn['content']}" for turn in turns)
        split = document_split(key, int(hidden_pct), int(visible_pct))
        for history, reply in iter_reply_examples(turns):
            groups[split].append((history, reply))
    if n_rows == 0:
        raise DialogueError("no dialogue rows were provided")
    missing = [name for name, rows in groups.items() if not rows]
    if missing:
        raise DialogueError(
            f"document split left {missing} empty. Lower the percentages or add dialogues. "
            "Splits are by conversation hash before packing."
        )
    cap = int(max_windows or 0)
    packed = {}
    prefixes = {}
    dropped = {}
    for name, rows in groups.items():
        if cap:
            rows = rows[:cap]
        windows = []
        prefix_rows = []
        drop_count = 0
        for history, reply in rows:
            cond = encode_turns(pretok, history, user_marker, assistant_marker, system_marker)
            reply_ids = encode_turn(pretok, reply, user_marker, assistant_marker, system_marker)
            built = training_prefix_and_window(
                cond,
                reply_ids,
                context_length=length,
                prefix_len=prefix,
                placement=placement,
                pad_id=int(pretok.pad_token_id),
            )
            windows.append(built["window_ids"])
            prefix_rows.append(built["prefix_ids"])
            drop_count += int(built["window_dropped"])
        packed[name] = torch.tensor(windows, dtype=torch.long)
        prefixes[name] = torch.tensor(prefix_rows, dtype=torch.long)
        dropped[name] = int(drop_count)
    meta = {
        "format": DIALOGUE_PACK_FORMAT,
        "dialogue": True,
        "dataset": dataset_id,
        "license_note": note,
        "source": str(source),
        "context_length": length,
        "prefix_len": prefix,
        "history_placement": placement,
        "truncation": TRUNCATION,
        "pad_side": PAD_SIDE,
        "truncation_policy": TRUNCATION_POLICY,
        "pad_id": int(pretok.pad_token_id),
        "eos_id": int(pretok.eos_token_id),
        "user_marker": user_marker,
        "assistant_marker": assistant_marker,
        "system_marker": system_marker,
        "special_role_tokens": False,
        "add_special_role_tokens": False,
        "predict_level0": False,
        "tokenizer": "gpt2",
        "pretok_kind": pretok.kind,
        "vocab_size": int(pretok.vocab_size),
        "n_rows": int(n_rows),
        "n_windows": {name: int(windows.shape[0]) for name, windows in packed.items()},
        "tokens_dropped": dropped,
        "nonpad_fraction": {
            name: nonpad_fraction(windows, pretok.pad_token_id) for name, windows in packed.items()
        },
        "hidden_pct": int(hidden_pct),
        "visible_pct": int(visible_pct),
        "split": "sha256-document",
        "one_window_per_reply": True,
        "quality_note": DIALOGUE_QUALITY_NOTE,
    }
    return {
        "visible_train": packed["visible_train"],
        "visible": packed["visible"],
        "hidden": packed["hidden"],
        "prefix_visible_train": prefixes["visible_train"],
        "prefix_visible": prefixes["visible"],
        "prefix_hidden": prefixes["hidden"],
        "meta": meta,
    }
