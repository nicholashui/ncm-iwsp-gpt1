from pathlib import Path


def test_docs_use_the_renamed_github_repo():
    root = Path(".")
    skip = {".git", ".venv", "venv", ".pytest_cache", "__pycache__"}
    # The retired repository slug must not appear. Pieces are joined so this
    # file itself does not contain that slug.
    retired = ("ncp", "iwsp", "gpt1")
    banned = ("-".join(retired), "_".join(retired))
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for slug in banned:
            assert slug not in text, path


def test_repo_does_not_contain_corpus_or_other_checkouts():
    root = Path(".")
    skip = {".git", ".venv", "venv", ".pytest_cache", "__pycache__"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        assert path.suffix not in {".pt", ".arrow", ".parquet", ".jsonl"}
        assert "bookcorpus" not in path.name.lower()
        assert path.name not in {
            "pretrain_t2t_mini.jsonl",
            "pretrain_t2t.jsonl",
            "packed_windows.pt",
        }
        assert "ncm-gpt1" not in path.parts
        assert "ncm-iwsp" not in path.parts
