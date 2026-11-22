#!/usr/bin/env python3
"""Convert a finished gradient checkpoint from a new Kaggle notebook.

This file is not the v10 training kernel. Do not run it inside
nicholashuihk/ncm-gpt-1-train, and do not write into that kernel's outputs.

Read checkpoints only from a Kaggle input dataset, for example:

    /kaggle/input/<dataset>/tokenizer.pt
    /kaggle/input/<dataset>/generator_step100000.pt
    /kaggle/input/<dataset>/generator.pt
    /kaggle/input/<dataset>/latest.pt

``generator_step100000.pt`` is preferred when more than one generator file
is present. Write the converted NCM-iWSP-GPT1 weights only to
``/kaggle/working`` or to an explicit ``--out`` outside the input tree.

Conversion is a closed-form copy plus the tied-head expansion. It does not
train. ``--smoke-forward`` is one random forward and is off unless you set it.
GPU is optional; the convert itself runs on CPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def repo_root() -> Path:
    """Find this checkout, including when Kaggle mounts it under /kaggle/input."""
    here = Path(__file__).resolve()
    candidates = [here.parents[1], Path.cwd()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path for path in kaggle_input.iterdir() if path.is_dir())
    for root in candidates:
        if (root / "ncm_iwsp_gpt1" / "convert.py").is_file():
            return root
    raise SystemExit(
        "Cannot import ncm_iwsp_gpt1. Add this repository as a Kaggle input "
        "or run the script from a checkout. Do not modify the training kernel "
        "nicholashuihk/ncm-gpt-1-train."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=None,
        help="dataset directory that contains tokenizer.pt and a generator checkpoint",
    )
    parser.add_argument("--tokenizer", default=None, help="overrides <input-dir>/tokenizer.pt")
    parser.add_argument(
        "--generator",
        default=None,
        help="overrides the preferred generator file inside --input-dir",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="destination. Default on Kaggle is /kaggle/working/ncm_iwsp_gpt1_converted.pt",
    )
    parser.add_argument("--config", default=None, help="default: <repo>/configs/target.yaml")
    parser.add_argument(
        "--smoke-forward",
        action="store_true",
        help="one random forward after conversion; not a training run",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto, used only with --smoke-forward")
    return parser


def main(argv=None) -> int:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    args = build_parser().parse_args(argv)

    from ncm_iwsp_gpt1.config import load_config
    from ncm_iwsp_gpt1.convert import (
        TOKENIZER_FILENAME,
        ConvertError,
        convert_files,
        prefer_generator_file,
    )

    if args.input_dir:
        folder = Path(args.input_dir)
        tokenizer = Path(args.tokenizer) if args.tokenizer else folder / TOKENIZER_FILENAME
        generator = Path(args.generator) if args.generator else prefer_generator_file(folder)
    else:
        if not args.tokenizer or not args.generator:
            raise SystemExit("pass --input-dir or both --tokenizer and --generator")
        tokenizer = Path(args.tokenizer)
        generator = Path(args.generator)

    if args.out:
        out = Path(args.out)
    elif Path("/kaggle/working").is_dir():
        out = Path("/kaggle/working/ncm_iwsp_gpt1_converted.pt")
    else:
        raise SystemExit("pass --out. On Kaggle the destination is /kaggle/working/<file>.pt")

    config_path = Path(args.config) if args.config else root / "configs" / "target.yaml"
    cfg = load_config(config_path)
    try:
        report = convert_files(
            tokenizer,
            generator,
            out,
            cfg,
            smoke_forward=bool(args.smoke_forward),
            device=args.device,
        )
    except ConvertError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote {out}")
    print(
        "parameters "
        f"tokenizer={report['tokenizer_parameters']} "
        f"generator={report['generator_parameters']} "
        f"total={report['total_parameters']} "
        f"post_upsample_dropped={report['post_upsample_parameters_dropped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
