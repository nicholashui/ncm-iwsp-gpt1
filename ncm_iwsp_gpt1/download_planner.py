#!/usr/bin/env python3
"""Download the frozen instruction-tuned planner.

The default model is TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Apache-2.0).
Weights are written only to the directory you pass. This repository does
not commit or redistribute those weights.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

DEFAULT_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_OUT = "planners/tinyllama-1.1b-chat"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing_weights(dest: Path) -> Path | None:
    if not (dest / "config.json").is_file():
        return None
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = dest / name
        if candidate.is_file():
            return candidate
    shards = sorted(dest.glob("model-*.safetensors"))
    if shards:
        return shards[0]
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download the TinyLlama chat planner. Weights stay outside git.")
    parser.add_argument("--model-id", default=os.environ.get("NCM_PLANNER_ID", DEFAULT_ID))
    parser.add_argument(
        "--out",
        default=os.environ.get("NCM_PLANNER_DIR", DEFAULT_OUT),
        help="destination directory (env NCM_PLANNER_DIR). Do not commit it.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    weights = _existing_weights(dest)
    if weights is not None:
        print("PLANNER_ALREADY path=%s sha=%s" % (dest, sha256_file(weights)), flush=True)
        return 0
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print("FAIL transformers is required to download the planner", flush=True)
        raise SystemExit(2) from exc
    print("DOWNLOAD start model_id=%s dest=%s" % (args.model_id, dest), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.save_pretrained(dest)
    model.save_pretrained(dest)
    saved = _existing_weights(dest)
    if saved is None:
        print("FAIL no weights after download in %s" % dest, flush=True)
        return 2
    if saved.name.startswith("model-"):
        shards = sorted(dest.glob("model-*.safetensors"))
        print("DOWNLOAD_OK path=%s shards=%s" % (dest, len(shards)), flush=True)
        for shard in shards:
            print("SHARD %s sha=%s bytes=%s" % (shard.name, sha256_file(shard), shard.stat().st_size), flush=True)
        return 0
    print("DOWNLOAD_OK path=%s sha=%s bytes=%s" % (dest, sha256_file(saved), saved.stat().st_size), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
