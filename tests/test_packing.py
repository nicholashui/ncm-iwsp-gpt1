import json

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import torch

from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.packing import (
    document_split,
    iter_jsonl_texts,
    iter_table_files,
    make_cache,
    pack_documents,
    split_documents,
    split_paragraphs,
)
from ncm_iwsp_gpt1.pretok import GPT2BPE


def _pretok():
    return GPT2BPE(pad_vocab_to=50304)


def _contains(windows: torch.Tensor, seq: list[int]) -> bool:
    flat = windows.reshape(-1).tolist()
    width = len(seq)
    if width == 0 or len(flat) < width:
        return False
    for start in range(len(flat) - width + 1):
        if flat[start : start + width] == seq:
            return True
    return False


def test_paragraphs_stay_short_pieces():
    assert split_paragraphs("one\n\ntwo\n\nthree") == ["one", "two", "three"]
    assert split_paragraphs("one\ntwo") == ["one", "two"]
    assert split_paragraphs("  only one line  ") == ["only one line"]


def test_short_paragraphs_share_a_window_and_512_is_legal():
    pretok = _pretok()
    windows = pack_documents(["abcd", "efgh"], pretok, context_length=16)
    assert windows.shape == (1, 16)
    # Two GPT-2 pieces each, plus EOS, then pad. Not one padded window per paragraph.
    assert int((windows[0] != pretok.pad_token_id).sum()) == 6
    eos_positions = (windows[0] == pretok.eos_token_id).nonzero().flatten().tolist()
    assert eos_positions == [2, 5]
    wide = pack_documents(["one\n\ntwo", "three"], pretok, context_length=512)
    assert wide.shape == (1, 512)
    assert int((wide != pretok.pad_token_id).sum()) < 32
    assert int(wide[0, -1]) == pretok.pad_token_id


def test_long_stream_is_sliced_not_truncated_to_one_doc():
    pretok = _pretok()
    text = " ".join(f"zxq{i:04d}" for i in range(700))
    assert len(pretok.encode(text)) > 512
    windows = pack_documents([text, text], pretok, context_length=512)
    assert windows.shape[0] >= 2
    assert windows.shape[1] == 512
    assert int(windows[1, 0]) != pretok.pad_token_id


def test_document_split_is_stable_and_keeps_paragraphs_together():
    pretok = _pretok()
    assert document_split("same text", 10, 10) == document_split("same text", 10, 10)
    groups = split_documents(["alpha text", "beta text", "gamma text"], 10, 10)
    seen = []
    for docs in groups.values():
        seen.extend(docs)
    assert sorted(seen) == ["alpha text", "beta text", "gamma text"]
    doc = "ALPHA unique paragraph zed\n\nBETA unique paragraph zed"
    others = [f"plain document number {i} about books and windows" for i in range(80)]
    cache = make_cache([doc, *others], pretok, 32, hidden_pct=10, visible_pct=10)
    alpha = pretok.encode("ALPHA unique paragraph zed")
    beta = pretok.encode("BETA unique paragraph zed")
    alpha_hits = [name for name in ("visible_train", "visible", "hidden") if _contains(cache[name], alpha)]
    beta_hits = [name for name in ("visible_train", "visible", "hidden") if _contains(cache[name], beta)]
    assert alpha_hits == beta_hits
    assert len(alpha_hits) == 1


def test_hidden_marker_cannot_appear_in_a_visible_window():
    pretok = _pretok()
    nonce = "uniquehiddenmarkerstring qqq"
    hidden = pack_documents([nonce], pretok, 32)
    visible = pack_documents(["hello there friend", "another visible paragraph"], pretok, 32)
    marker = pretok.encode(nonce) + [pretok.eos_token_id]
    assert _contains(hidden, marker)
    assert not _contains(visible, marker)


def test_jsonl_arrow_and_cache_roundtrip(tmp_path):
    pretok = _pretok()
    path = tmp_path / "books.jsonl"
    rows = [
        {"text": "one short paragraph"},
        {"title": "ignored", "text": "two short paragraph\n\nstill the same document"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    texts = list(iter_jsonl_texts(path))
    assert texts[0] == "one short paragraph"
    assert "still the same document" in texts[1]
    many = [f"paragraph number {i} is its own document about a book" for i in range(40)]
    cache = make_cache(many, pretok, 32, hidden_pct=10, visible_pct=10, max_windows=4)
    assert set(cache) >= {"visible_train", "visible", "hidden", "meta"}
    for key in ("visible_train", "visible", "hidden"):
        assert cache[key].dtype == torch.long
        assert cache[key].shape[1] == 32
        assert cache[key].shape[0] >= 1
    assert cache["meta"]["format"] == "packed_windows"
    assert cache["meta"]["tokenizer"] == "gpt2"
    assert cache["meta"]["dataset"] == "lucadiliello/bookcorpusopen"
    assert cache["meta"]["split"] == "sha256-document"
    assert cache["meta"]["context_length"] == 32

    arrow_path = tmp_path / "books.arrow"
    table = pa.table({"text": ["arrow paragraph one", "arrow paragraph two"]})
    with pa.OSFile(str(arrow_path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    assert list(iter_table_files(arrow_path)) == ["arrow paragraph one", "arrow paragraph two"]
    cache_dir = tmp_path / "hf-cache" / "snapshot"
    cache_dir.mkdir(parents=True)
    parquet_path = cache_dir / "part.parquet"
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"content": ["from the local cache"]}), parquet_path)
    assert list(iter_table_files(cache_dir)) == ["from the local cache"]


def test_refuses_minimind_training_names(tmp_path):
    for name in ("sft_t2t_mini.jsonl", "dpo.jsonl", "pretrain_t2t.jsonl", "rlaif.jsonl"):
        path = tmp_path / name
        path.write_text('{"text": "no"}\n', encoding="utf-8")
        with pytest.raises(ValueError):
            list(iter_jsonl_texts(path))


def test_target_config_packs_512_without_building_the_model():
    cfg = load_config("configs/target.yaml")
    pretok = GPT2BPE.from_config(cfg)
    docs = [f"A short book sentence number {i}." for i in range(50)]
    cache = make_cache(docs, pretok, cfg["tokenizer"]["context_length"], hidden_pct=10, visible_pct=10, max_windows=2)
    assert cache["visible"].shape[1] == 512
    assert cache["hidden"].shape[1] == 512
