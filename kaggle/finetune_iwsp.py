#!/usr/bin/env python3
"""Fine-tune the untied NCM-iWSP-GPT1 generator from a new Kaggle notebook.

This file is not the v10 training kernel. Do not open or overwrite
nicholashuihk/ncm-gpt-1-train, and do not write into its outputs or into
the converted-init dataset.

Read-only inputs:

    /kaggle/input/<converted-init>/tokenizer.pt
    /kaggle/input/<converted-init>/generator.pt
    /kaggle/input/<v10-export>/artifacts/packed_windows.pt

``artifacts/packed_windows.pt`` is the v10 window file. It is not necessarily
next to the raw export's tokenizer. Checkpoints, logs, and reports go to
``/kaggle/working/finetune`` (or ``--out``).

Pass ``--confirm`` only after Nicholas has reviewed the paths. This script
does not submit a Kaggle job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parents[1], Path.cwd()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates.extend(path for path in kaggle_input.iterdir() if path.is_dir())
    for root in candidates:
        if (root / "ncm_iwsp_gpt1" / "finetune.py").is_file():
            return root
    raise SystemExit(
        "Cannot import ncm_iwsp_gpt1. Add this repository as a Kaggle input "
        "or run the script from a checkout. Do not modify the training kernel "
        "nicholashuihk/ncm-gpt-1-train."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=None, help="default: /kaggle/input when that directory exists")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--generator", default=None)
    parser.add_argument("--packed", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dry-plan", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--peak-lr", type=float, default=3e-5)
    parser.add_argument("--floor-lr", type=float, default=3e-6)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--mid-aux", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--val-windows", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--reserve-minutes", type=float, default=15.0)
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--no-chunked-attn", action="store_true")
    parser.add_argument(
        "--no-ddp",
        action="store_true",
        help="stay on one GPU even when more than one is visible",
    )
    return parser


def _discover(args, root: Path) -> None:
    if args.dry_plan and not args.input_root:
        return
    input_root = Path(args.input_root) if args.input_root else Path("/kaggle/input")
    if args.tokenizer and args.generator and args.packed:
        return
    if not input_root.is_dir():
        return
    from ncm_iwsp_gpt1.finetune import find_converted_pair, find_packed_windows

    if not args.tokenizer or not args.generator:
        tokenizer, generator = find_converted_pair(input_root)
        args.tokenizer = args.tokenizer or str(tokenizer)
        args.generator = args.generator or str(generator)
        print(f"converted init tokenizer={args.tokenizer}", flush=True)
        print(f"converted init generator={args.generator}", flush=True)
    if not args.packed:
        args.packed = str(find_packed_windows(input_root))
        print(f"packed windows={args.packed}", flush=True)
    del root


def _maybe_launch_workers(args, argv) -> int | None:
    if args.dry_plan or args.no_ddp or os.environ.get("RANK") is not None:
        return None
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return None
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={torch.cuda.device_count()}",
        "--standalone",
        str(Path(__file__).resolve()),
        *argv,
        "--no-ddp",
    ]
    print("launch:", " ".join(command), flush=True)
    return subprocess.call(command)


def main(argv=None) -> int:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    args = build_parser().parse_args(argv)
    _discover(args, root)
    forwarded = list(argv) if argv is not None else sys.argv[1:]
    launched = _maybe_launch_workers(args, forwarded)
    if launched is not None:
        return int(launched)
    if not args.out and not args.dry_plan:
        working = Path("/kaggle/working")
        if working.is_dir():
            args.out = str(working / "finetune")
        else:
            raise SystemExit("pass --out. On Kaggle the destination is /kaggle/working/finetune")
    config = args.config or str(root / "configs" / "target.yaml")
    command = [
        "finetune",
        "--config",
        config,
        "--max-steps",
        str(args.max_steps),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--peak-lr",
        str(args.peak_lr),
        "--floor-lr",
        str(args.floor_lr),
        "--warmup",
        str(args.warmup),
        "--weight-decay",
        str(args.weight_decay),
        "--alpha",
        str(args.alpha),
        "--mid-aux",
        str(args.mid_aux),
        "--val-every",
        str(args.val_every),
        "--val-windows",
        str(args.val_windows),
        "--save-every",
        str(args.save_every),
        "--log-every",
        str(args.log_every),
        "--reserve-minutes",
        str(args.reserve_minutes),
        "--resume",
        str(args.resume),
        "--device",
        args.device,
    ]
    if args.max_hours is not None:
        command += ["--max-hours", str(args.max_hours)]
    if args.seed is not None:
        command += ["--seed", str(args.seed)]
    if args.dry_plan:
        command.append("--dry-plan")
    if args.confirm:
        command.append("--confirm")
    if args.no_checkpointing:
        command.append("--no-checkpointing")
    if args.no_chunked_attn:
        command.append("--no-chunked-attn")
    if args.tokenizer:
        command += ["--tokenizer", args.tokenizer]
    if args.generator:
        command += ["--generator", args.generator]
    if args.packed:
        command += ["--packed", args.packed]
    if args.out:
        command += ["--out", args.out]
    from ncm_iwsp_gpt1.cli import main as cli_main

    return int(cli_main(command))


if __name__ == "__main__":
    raise SystemExit(main())
