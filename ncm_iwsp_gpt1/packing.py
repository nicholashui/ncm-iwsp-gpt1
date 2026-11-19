"""Pack BookCorpus-style documents into fixed windows.

Each input row is one document. A SHA-256 of the raw text assigns the whole
document to ``visible_train``, ``visible``, or ``hidden`` before any tokens
are cut. Inside a split, blank-line paragraphs (or single lines, when the row
has no blank lines) are encoded with GPT-2 BPE, terminated with EOS, and
concatenated. Windows are slices of that stream. Only the tail is padded, so
a short paragraph does not become its own full-context row.

The cache is a torch dict in the spirit of NCM-GPT1's ``packed_windows.pt``:
``visible_train``, ``visible``, ``hidden``, and ``meta``. The corpus itself
is never written here. Download is not performed by this module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from ncm_iwsp_gpt1.pretok import GPT2BPE


def _book_lines(count: int) -> list[str]:
    return [
        (
            f"Book paragraph {index}: a short passage about a traveler, a library, "
            "and a window of tokens that should be packed with its neighbors."
        )
        for index in range(count)
    ]


DEMO_DOCUMENTS = [
    "The small model writes its own weights.\n\nA second paragraph shares the same window.",
    "Pack several short paragraphs into one window instead of padding each one.",
    "Random init is the baseline, not a trained model.",
    "A residual codebook keeps a coarse code and a fine code.",
    "Hidden documents must stay out of the writer.",
    "This sentence is long enough to share a window with the next one when the context is small.",
    "Evaluation ignores pad and does not call backward.",
    "Level zero is a unigram owned by the evaluator.",
    "Coarse to fine generation starts from that unigram.",
    "Do not rename this run NCM-GPT1.",
    "The audit rejects optimizers and weight downloads.",
    "A description has to say what the tensors mean.",
    "Short lines should not each become their own padded window.",
    "Visible validation is what the agent is allowed to see.",
    "Another plain English sentence about windows and codes.",
    "Windows are slices of a token stream, not one row per line.",
    "The generator predicts every level except the coarsest.",
    "Codebook usage is a diagnostic, not an update of the codebook.",
    "Fixed prompts are only a qualitative check.",
    "Success is beating random init and the unigram on the hidden split.",
    "No Kaggle quota from the BookCorpus gradient run is spent here.",
    "Research code only. Read the dataset card before use.",
    "Sinusoidal embeddings are a closed form, not a fit.",
    "Each iteration rebuilds from the same seed, then overwrites.",
    "The scalar is the mean of the per-level cross entropies.",
    *_book_lines(80),
]


_BANNED_NAMES = {
    "pretrain_t2t.jsonl",
    "pretrain_t2t_mini.jsonl",
    "sft_t2t.jsonl",
    "sft_t2t_mini.jsonl",
    "dpo.jsonl",
    "rlaif.jsonl",
    "agent_rl.jsonl",
    "agent_rl_math.jsonl",
}
_BANNED_BITS = ("sft", "dpo", "rlaif", "agent_rl", "lora_", "minimind")


def demo_texts() -> list[str]:
    return list(DEMO_DOCUMENTS)


def split_paragraphs(text: str) -> list[str]:
    """Break one document into short pieces without changing its split.

    Blank lines win when they exist. Otherwise each non-empty line is a
    paragraph. A single line stays one piece and is sliced across windows
    only if its tokens exceed the context.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [text]


def _refuse_corpus_name(path: Path) -> None:
    name = path.name.lower()
    if name in _BANNED_NAMES or any(bit in name for bit in _BANNED_BITS):
        raise ValueError(
            f"refusing {path.name}: NCM-iWSP-GPT1 packs lucadiliello/bookcorpusopen, "
            "not MiniMind pretrain, SFT, DPO, RLAIF, or adapter data"
        )


def iter_jsonl_texts(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"data file not found: {path}. Download is off by default; "
            "point this at a local jsonl of {{\"text\": ...}} rows."
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
                raise ValueError(f"{path}:{line_no} is not JSON") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no} must be a JSON object")
            text = obj.get("text", obj.get("content", ""))
            if isinstance(text, str) and text.strip():
                yield text


def _texts_from_table(table):
    names = set(table.column_names)
    for key in ("text", "content", "paragraph"):
        if key in names:
            produced = False
            for value in table.column(key).to_pylist():
                if isinstance(value, str) and value.strip():
                    produced = True
                    yield value
            if not produced:
                raise ValueError(f"column {key!r} had no text")
            return
    raise ValueError(f"no text column in {list(table.column_names)}")


def _load_table(path: Path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path)
    raw = path.read_bytes()
    last_error = None
    for opener in (ipc.open_file, ipc.open_stream):
        try:
            reader = opener(pa.BufferReader(raw))
            return reader.read_all()
        except Exception as exc:  # pragma: no cover - format fallback
            last_error = exc
    raise ValueError(f"could not read arrow file {path}") from last_error


def iter_table_files(path: str | Path):
    """Yield document texts from an arrow/parquet file or a directory of them.

    A Hugging Face dataset cache is a directory of arrow files. This does not
    download anything.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"data path not found: {path}. Download is off by default; "
            "pass a local arrow file, parquet file, or HF cache directory."
        )
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
            raise FileNotFoundError(f"no .arrow or .parquet files under {path}")
    found = 0
    for file in files:
        try:
            table = _load_table(file)
            texts = list(_texts_from_table(table))
        except Exception:
            continue
        for text in texts:
            found += 1
            yield text
    if found == 0:
        raise ValueError(f"no text rows found under {path}")


def document_split(text: str, hidden_pct: int, visible_pct: int) -> str:
    """Hash split on the raw document, independent of token packing."""
    hidden_pct = int(hidden_pct)
    visible_pct = int(visible_pct)
    if hidden_pct < 0 or visible_pct < 0 or hidden_pct + visible_pct >= 100:
        raise ValueError("hidden_pct and visible_pct must be non-negative and sum to < 100")
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < hidden_pct:
        return "hidden"
    if bucket < hidden_pct + visible_pct:
        return "visible"
    return "visible_train"


def split_documents(texts, hidden_pct: int, visible_pct: int) -> dict[str, list[str]]:
    groups = {"visible_train": [], "visible": [], "hidden": []}
    for text in texts:
        groups[document_split(text, hidden_pct, visible_pct)].append(text)
    return groups


def pack_documents(texts, pretok: GPT2BPE, context_length: int, max_windows: int = 0) -> torch.Tensor:
    """Encode paragraphs, append EOS after each, pack to ``L``, pad only the tail."""
    eos = int(pretok.eos_token_id)
    pad = int(pretok.pad_token_id)
    length = int(context_length)
    if length < 2:
        raise ValueError("context_length must be at least 2")
    stream: list[int] = []
    cap = int(max_windows or 0)
    for text in texts:
        for paragraph in split_paragraphs(text):
            ids = list(pretok.encode(paragraph) or [])
            if not ids:
                continue
            stream.extend(ids)
            stream.append(eos)
            if cap and len(stream) >= cap * length:
                break
        if cap and len(stream) >= cap * length:
            break
    if not stream:
        raise ValueError("no tokens produced from the given documents")
    windows: list[list[int]] = []
    for start in range(0, len(stream), length):
        if cap and len(windows) >= cap:
            break
        chunk = stream[start : start + length]
        if len(chunk) < length:
            chunk = chunk + [pad] * (length - len(chunk))
        windows.append(chunk)
    if not windows:
        raise ValueError("packing produced no windows")
    return torch.tensor(windows, dtype=torch.long)


def nonpad_fraction(windows: torch.Tensor, pad_id: int) -> float:
    if windows.numel() == 0:
        return 0.0
    return float((windows != int(pad_id)).float().mean().item())


def make_cache(
    texts,
    pretok: GPT2BPE,
    context_length: int,
    hidden_pct: int,
    visible_pct: int,
    max_windows: int = 0,
) -> dict:
    groups = split_documents(texts, hidden_pct, visible_pct)
    missing = [name for name, docs in groups.items() if not docs]
    if missing:
        raise ValueError(
            f"document split left {missing} empty. Lower the percentages or add documents. "
            "Splits are by document hash before packing, so a handful of rows can miss a bucket."
        )
    packed = {
        name: pack_documents(docs, pretok, context_length, max_windows=max_windows)
        for name, docs in groups.items()
    }
    meta = {
        "context_length": int(context_length),
        "pad_id": int(pretok.pad_token_id),
        "eos_id": int(pretok.eos_token_id),
        "pretok_kind": pretok.kind,
        "tokenizer": "gpt2",
        "dataset": "lucadiliello/bookcorpusopen",
        "vocab_size": int(pretok.vocab_size),
        "n_docs": {name: len(docs) for name, docs in groups.items()},
        "n_windows": {name: int(windows.shape[0]) for name, windows in packed.items()},
        "nonpad_fraction": {
            name: nonpad_fraction(windows, pretok.pad_token_id) for name, windows in packed.items()
        },
        "token_stats": pretok.token_stats(texts),
        "hidden_pct": int(hidden_pct),
        "visible_pct": int(visible_pct),
        "split": "sha256-document",
        "format": "packed_windows",
    }
    return {**packed, "meta": meta}


def save_cache(cache: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)


def load_cache(path: str | Path) -> dict:
    cache = torch.load(Path(path), map_location="cpu", weights_only=False)
    for key in ("visible_train", "visible", "hidden", "meta"):
        if key not in cache:
            raise ValueError(f"cache {path} is missing '{key}'")
    return cache
