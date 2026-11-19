"""Command line for packing, counting, evaluation, and the local dry loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ncm_iwsp_gpt1.audit import AuditError
from ncm_iwsp_gpt1.build import build_models, parameter_report
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.evaluate import evaluate_windows
from ncm_iwsp_gpt1.loop import load_writer, run_dry_loop
from ncm_iwsp_gpt1.packing import (
    demo_texts,
    iter_jsonl_texts,
    iter_table_files,
    load_cache,
    make_cache,
    save_cache,
)
from ncm_iwsp_gpt1.params import estimate_report
from ncm_iwsp_gpt1.pretok import GPT2BPE


def _take(iterator, max_docs: int) -> list[str]:
    texts = []
    cap = int(max_docs or 0)
    for text in iterator:
        texts.append(text)
        if cap and len(texts) >= cap:
            break
    if not texts:
        raise SystemExit("no documents found")
    return texts


def _iter_hf_download(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "pip install datasets  # only required for --download"
        ) from exc
    dataset = load_dataset(name, split=split, streaming=True)
    for row in dataset:
        text = row.get("text", row.get("content", "")) if isinstance(row, dict) else ""
        if isinstance(text, str) and text.strip():
            yield text


def _cmd_pack(args) -> int:
    cfg = load_config(args.config)
    pretok = GPT2BPE.from_config(cfg)
    sources = [args.jsonl, args.arrow, args.hf_cache, args.hf_dataset]
    if args.demo and any(sources):
        raise SystemExit("pass either --demo or one data source, not both")
    if not args.demo and sum(item is not None for item in sources) != 1:
        raise SystemExit("pack needs --demo or exactly one of --jsonl, --arrow, --hf-cache, --hf-dataset")
    if args.hf_dataset and not args.download:
        raise SystemExit(
            "download is off by default. Pass --download to fetch the dataset, "
            "or pack a local --jsonl, --arrow, or --hf-cache. "
            "The corpus for this repo is lucadiliello/bookcorpusopen."
        )
    if args.hf_dataset and "bookcorpusopen" not in args.hf_dataset.lower():
        print(
            "warning: NCM-iWSP-GPT1 is aimed at lucadiliello/bookcorpusopen",
            file=sys.stderr,
        )
    if args.demo:
        texts = demo_texts()
    elif args.jsonl:
        texts = _take(iter_jsonl_texts(args.jsonl), args.max_docs)
    elif args.arrow:
        texts = _take(iter_table_files(args.arrow), args.max_docs)
    elif args.hf_cache:
        texts = _take(iter_table_files(args.hf_cache), args.max_docs)
    else:
        texts = _take(_iter_hf_download(args.hf_dataset, args.hf_split), args.max_docs)
    data = cfg.get("data", {})
    hidden_pct = int(data.get("hidden_pct", 10))
    visible_pct = int(data.get("visible_pct", 10))
    # The built-in paragraph list is only about a hundred rows. A 1% hash
    # bucket would often be empty, so the demo forces a coarser split.
    if args.demo:
        hidden_pct = max(hidden_pct, 10)
        visible_pct = max(visible_pct, 10)
    max_windows = int(args.max_windows) if args.max_windows is not None else int(data.get("max_windows", 0) or 0)
    cache = make_cache(
        texts,
        pretok,
        context_length=int(cfg["tokenizer"]["context_length"]),
        hidden_pct=hidden_pct,
        visible_pct=visible_pct,
        max_windows=max_windows,
    )
    save_cache(cache, args.out)
    print(json.dumps(cache["meta"], indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


def _cmd_count(args) -> int:
    cfg = load_config(args.config)
    use_estimate = bool(args.estimate) or (bool(cfg.get("heavy")) and not args.instantiate)
    if args.instantiate:
        use_estimate = False
    if use_estimate:
        report = estimate_report(cfg)
        report["counted_by"] = "estimate"
    else:
        _, vqvae, generator = build_models(cfg)
        report = parameter_report(vqvae, generator)
        report["counted_by"] = "instantiate"
    report["name"] = cfg["name"]
    report["context_length"] = int(cfg["tokenizer"]["context_length"])
    report["levels"] = [int(level) for level in cfg["tokenizer"]["levels"]]
    report["grid_cells"] = int(sum(report["levels"]))
    report["codebook_size"] = int(cfg["tokenizer"]["codebook_size"])
    report["vocab_size"] = int(cfg["pretok"]["pad_vocab_to"])
    print(json.dumps(report, indent=2))
    return 0


def _cmd_eval(args) -> int:
    cfg = load_config(args.config)
    cache = load_cache(args.packed)
    writer = load_writer(args.writer)
    pretok, vqvae, generator = build_models(cfg)
    writer.write_weights(vqvae, generator, iteration=int(args.iteration))
    split = cache["visible"] if args.split == "visible" else cache["visible_train"]
    metrics = evaluate_windows(
        pretok,
        vqvae,
        generator,
        split,
        train_windows=cache["visible_train"],
        batch_size=int(cfg.get("eval", {}).get("batch_size", 4)),
        max_windows=int(cfg.get("eval", {}).get("max_windows", 8)),
        n_generate=int(cfg.get("eval", {}).get("n_generate", 1)),
        temperature=0.0,
        device=args.device,
    )
    shown = {
        "split": args.split,
        "device": metrics["device"],
        "iteration": int(args.iteration),
        "scalar": metrics["scalar"],
        "mean_level_ce": metrics["mean_level_ce"],
        "length_weighted_ce": metrics["length_weighted_ce"],
        "recon_acc": metrics["recon_acc"],
        "recon_ce": metrics["recon_ce"],
        "mean_exact_acc": metrics["mean_exact_acc"],
        "recon_after_written_levels_acc": metrics["recon_after_written_levels_acc"],
        "codebook_mean_usage": metrics["codebook_usage"]["mean_usage"],
        "most_frequent_token_ce": metrics["baselines"]["most_frequent_token_ce"],
        "most_frequent_token_acc": metrics["baselines"]["most_frequent_token_acc"],
        "token_unigram_nll": metrics["baselines"]["token_unigram_nll"],
        "always_pad_ce": metrics["baselines"]["always_pad_ce"],
        "level0_unigram_nll": metrics["baselines"]["level0_unigram_nll"],
        "mean_predicted_level_unigram_nll": metrics["baselines"]["mean_predicted_level_unigram_nll"],
        "uniform_finer_ce": metrics["baselines"]["uniform_finer_ce"],
        "qualitative": metrics["baselines"]["qualitative"],
        "description": getattr(writer, "LAST_DESCRIPTION", ""),
    }
    print(json.dumps(shown, indent=2, ensure_ascii=False))
    return 0


def _cmd_loop(args) -> int:
    cfg = load_config(args.config)
    cache = load_cache(args.packed)
    try:
        summary = run_dry_loop(
            cfg, cache, args.writer, args.out, iterations=int(args.iters), device=args.device
        )
    except AuditError as exc:
        print("audit rejected the writer:")
        for problem in exc.problems:
            print(f"  - {problem}")
        return 2
    print(json.dumps(summary, indent=2))
    print(
        "Agent-facing scalar is mean_level_ce in results.csv. "
        f"Hidden numbers are only in {summary['hidden_log']} and must not be shown to the writer."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncm_iwsp_gpt1",
        description="NCM-iWSP-GPT1 harness. Writes weights in code. Does not train. Is not NCM-GPT1.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pack = sub.add_parser("pack", help="pack local BookCorpus rows, or the built-in demo paragraphs")
    pack.add_argument("--config", required=True)
    pack.add_argument("--jsonl", default=None, help="local JSONL of {\"text\": ...}. Not committed.")
    pack.add_argument("--arrow", default=None, help="local .arrow or .parquet file. Not committed.")
    pack.add_argument("--hf-cache", default=None, help="local HF dataset directory of arrow/parquet files")
    pack.add_argument("--hf-dataset", default=None, help="dataset name, used only with --download")
    pack.add_argument("--hf-split", default="train")
    pack.add_argument("--download", action="store_true", help="allow a network fetch; off by default")
    pack.add_argument("--demo", action="store_true", help="pack the synthetic paragraphs shipped with the code")
    pack.add_argument("--max-docs", type=int, default=0, help="stop after this many documents (0 = no cap)")
    pack.add_argument("--max-windows", type=int, default=None, help="override data.max_windows")
    pack.add_argument("--out", required=True)
    pack.set_defaults(func=_cmd_pack)

    count = sub.add_parser("count", help="count parameters, or estimate them for the heavy target")
    count.add_argument("--config", required=True)
    count.add_argument("--estimate", action="store_true", help="sum parameter sizes without allocating")
    count.add_argument("--instantiate", action="store_true", help="build the modules even if the config is heavy")
    count.set_defaults(func=_cmd_count)

    evaluate = sub.add_parser("eval", help="evaluate a writer iteration on the visible split")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--packed", required=True)
    evaluate.add_argument("--writer", default="ncm_iwsp_gpt1/writers/stub.py")
    evaluate.add_argument("--iteration", type=int, default=0)
    evaluate.add_argument("--split", choices=("visible", "visible_train"), default="visible")
    evaluate.add_argument("--device", default="auto", help="auto (CUDA if available), cuda, or cpu")
    evaluate.set_defaults(func=_cmd_eval)

    loop = sub.add_parser("loop", help="local propose/write/eval/audit dry loop")
    loop.add_argument("--config", required=True)
    loop.add_argument("--packed", required=True)
    loop.add_argument("--writer", default="ncm_iwsp_gpt1/writers/stub.py")
    loop.add_argument("--out", required=True)
    loop.add_argument("--iters", type=int, default=3)
    loop.add_argument("--device", default="auto", help="auto (CUDA if available), cuda, or cpu")
    loop.set_defaults(func=_cmd_loop)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
