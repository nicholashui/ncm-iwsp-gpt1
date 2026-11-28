"""Command line for packing, counting, evaluation, chat, and the local dry loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ncm_iwsp_gpt1.audit import AuditError
from ncm_iwsp_gpt1.build import build_models, parameter_report
from ncm_iwsp_gpt1.chat import (
    CHAT_DESCRIPTION,
    LEVEL0_SOURCES,
    ChatError,
    open_session,
    run_interactive,
    run_prompt,
)
from ncm_iwsp_gpt1.chat_lmsplice import DESCRIPTION as OFFICIAL_CHAT_DESCRIPTION
from ncm_iwsp_gpt1.chat_lmsplice import add_parser_arguments, execute as execute_official_chat
from ncm_iwsp_gpt1.prefix_level0 import (
    RECOMMENDED_DIALOGUE_PREFIX_OUT,
    PrefixLevel0Error,
    describe_prefix_dialogue_run,
    describe_prefix_run,
    run_prefix_level0,
    run_prefix_level0_dialogue,
)
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.convert import FORMAT_NAME, ConvertError, convert_files, load_converted_checkpoint
from ncm_iwsp_gpt1.evaluate import evaluate_windows
from ncm_iwsp_gpt1.finetune import (
    FinetuneError,
    describe_dialogue_run,
    describe_run,
    run_dialogue_finetune,
    run_finetune,
)
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
    if args.init_checkpoint:
        try:
            load_converted_checkpoint(args.init_checkpoint, vqvae, generator)
        except ConvertError as exc:
            raise SystemExit(str(exc)) from exc
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
            cfg,
            cache,
            args.writer,
            args.out,
            iterations=int(args.iters),
            device=args.device,
            init_checkpoint=args.init_checkpoint,
        )
    except ConvertError as exc:
        raise SystemExit(str(exc)) from exc
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


def _cmd_convert(args) -> int:
    cfg = load_config(args.config)
    try:
        report = convert_files(
            args.tokenizer,
            args.generator,
            args.out,
            cfg,
            smoke_forward=bool(args.smoke_forward),
            device=args.device,
        )
    except ConvertError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    return 0


def _cmd_chat(args) -> int:
    cfg = load_config(args.config)
    try:
        session = open_session(
            cfg,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            checkpoint=args.checkpoint,
            tokenizer_path=args.tokenizer,
            generator_path=args.generator,
            finetune_path=args.finetune,
            prefix_head_path=args.prefix_head,
            level0_source=args.level0_source,
            packed_path=args.packed,
            unigram_max_windows=args.unigram_max_windows,
            dialogue=bool(args.dialogue),
            max_reply_tokens=int(args.max_reply_tokens),
        )
    except (ChatError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        if args.prompt is not None:
            return run_prompt(session, args.prompt)
        return run_interactive(session)
    except ChatError as exc:
        raise SystemExit(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ncm_iwsp_gpt1",
        description=(
            "NCM-iWSP-GPT1. The writer loop scores closed-form weights. "
            "finetune is a separate gradient run of the untied generator. "
            "chat continues one window from a prefix level-0 head. "
            "The default config keeps a one-token end-of-text prefix. "
            "configs/target_phase2.yaml places the prompt in a longer prefix. "
            "Phase 3 is pack-dialogue and finetune-dialogue on configs/target_phase3.yaml. "
            "train-prefix-level0-dialogue trains only the dialogue prefix head. "
            "No dialogue corpus is selected."
        ),
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
    evaluate.add_argument(
        "--init-checkpoint",
        default=None,
        help="converted checkpoint to load before write_weights (stub iteration 0 leaves it in place)",
    )
    evaluate.set_defaults(func=_cmd_eval)

    loop = sub.add_parser("loop", help="local propose/write/eval/audit dry loop")
    loop.add_argument("--config", required=True)
    loop.add_argument("--packed", required=True)
    loop.add_argument("--writer", default="ncm_iwsp_gpt1/writers/stub.py")
    loop.add_argument("--out", required=True)
    loop.add_argument("--iters", type=int, default=3)
    loop.add_argument("--device", default="auto", help="auto (CUDA if available), cuda, or cpu")
    loop.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "converted checkpoint loaded before each write_weights call. "
            "The random-init baseline stays the seeded initialization. "
            "Stub iteration 0 therefore scores the converted weights."
        ),
    )
    loop.set_defaults(func=_cmd_loop)

    convert = sub.add_parser(
        "convert",
        help="expand a finished gradient checkpoint into untied NCM-iWSP-GPT1 weights",
    )
    convert.add_argument("--config", default="configs/target.yaml")
    convert.add_argument("--tokenizer", required=True, help="tokenizer.pt (read-only)")
    convert.add_argument(
        "--generator",
        required=True,
        help="generator.pt, latest.pt, or generator_step100000.pt (read-only)",
    )
    convert.add_argument("--out", required=True, help="destination outside the source directory")
    convert.add_argument(
        "--smoke-forward",
        action="store_true",
        help="run one random forward after conversion; not a training step",
    )
    convert.add_argument("--device", default="cpu", help="device for --smoke-forward (cpu, cuda, or auto)")
    convert.set_defaults(func=_cmd_convert)

    chat = sub.add_parser(
        "chat",
        help="continue one window from a prefix level-0 head",
        description=CHAT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    chat.add_argument("--config", default="configs/target.yaml")
    chat.add_argument("--device", default="auto", help="auto (CUDA if available), cuda, or cpu")
    chat.add_argument("--temperature", type=float, default=0.0, help="0 is argmax")
    chat.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
    chat.add_argument("--top-p", type=float, default=1.0, help="1 disables nucleus sampling")
    chat.add_argument("--seed", type=int, default=0, help="sampling seed; initialization still uses the config seed")
    chat.add_argument(
        "--prompt",
        default=None,
        help="decode this string and exit; omit it for the stdin loop",
    )
    chat.add_argument(
        "--checkpoint",
        default=None,
        help=f"single converted file (format {FORMAT_NAME})",
    )
    chat.add_argument("--tokenizer", default=None, help="frozen tokenizer.pt ({model, step, converted_from, ...})")
    chat.add_argument(
        "--generator",
        default=None,
        help="converted generator.pt ({model, step, converted_from, ...}); not a tied-head trainer file",
    )
    chat.add_argument(
        "--finetune",
        default=None,
        help="latest.pt with format ncm-iwsp-gpt1-finetune; optimizer state is ignored",
    )
    chat.add_argument(
        "--prefix-head",
        default=None,
        help="prefix level-0 checkpoint (format ncm-iwsp-gpt1-prefix-level0). Required unless a baseline is named.",
    )
    chat.add_argument(
        "--level0-source",
        default="prefix_head",
        choices=LEVEL0_SOURCES,
        help="prefix_head (default), or the packed / unigram comparison baselines",
    )
    chat.add_argument(
        "--packed",
        default=None,
        help="packed windows used only by --level0-source unigram, to count level 0",
    )
    chat.add_argument(
        "--unigram-max-windows",
        type=int,
        default=256,
        help="how many packed rows to count for the unigram baseline",
    )
    chat.add_argument(
        "--dialogue",
        action="store_true",
        help=(
            "phase 3 only: keep session history and slice a reply from the decoded window. "
            "Requires a phase-3 config and a dialogue checkpoint (--finetune format "
            "ncm-iwsp-gpt1-dialogue). Without that checkpoint, chat stops and does not "
            "call itself a chatbot."
        ),
    )
    chat.add_argument(
        "--max-reply-tokens",
        type=int,
        default=0,
        help="phase 3 reply cap after slicing; 0 keeps the remainder of the window",
    )
    chat.set_defaults(func=_cmd_chat)

    official = sub.add_parser(
        "official-chat",
        help="decode-side dialogue chat with a frozen TinyLlama planner splice",
        description=OFFICIAL_CHAT_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_parser_arguments(official)
    official.set_defaults(func=execute_official_chat)

    train_prefix = sub.add_parser(
        "train-prefix-level0",
        help="train only the prefix level-0 head; tokenizer and generator stay frozen",
        description=(
            "Train p(level-0 | first T tokens) on packed windows. "
            "T is drawn from 1 through the context length. "
            "The tokenizer and the generator are frozen. predict_level0 stays false "
            "and prefix_len stays 1. This command does not submit a Kaggle job. "
            "packed_windows.pt is not in the git repository; pass a local file. "
            "A long run is not started unless --max-steps is set."
        ),
    )
    train_prefix.add_argument("--config", default="configs/target.yaml")
    train_prefix.add_argument("--tokenizer", default=None, help="frozen tokenizer.pt (required to train)")
    train_prefix.add_argument(
        "--generator",
        default=None,
        help="optional converted generator.pt or finetune latest.pt; loaded and frozen, not trained",
    )
    train_prefix.add_argument("--packed", default=None, help="packed_windows.pt or an iWSP visible cache")
    train_prefix.add_argument("--out", default=None, help="directory for the prefix-head checkpoint and log")
    train_prefix.add_argument("--resume", default=None, help="prefix-level0 checkpoint to continue; step must be < --max-steps")
    train_prefix.add_argument("--dry-plan", action="store_true", help="print the run plan and exit without building")
    train_prefix.add_argument("--max-steps", type=int, default=None, help="step to reach; required unless --dry-plan")
    train_prefix.add_argument("--batch-size", type=int, default=8)
    train_prefix.add_argument("--lr", type=float, default=1e-3)
    train_prefix.add_argument("--weight-decay", type=float, default=0.01)
    train_prefix.add_argument("--grad-clip", type=float, default=1.0)
    train_prefix.add_argument("--log-every", type=int, default=10)
    train_prefix.add_argument("--val-every", type=int, default=50)
    train_prefix.add_argument("--save-every", type=int, default=100)
    train_prefix.add_argument("--val-windows", type=int, default=32)
    train_prefix.add_argument("--unigram-windows", type=int, default=256)
    train_prefix.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    train_prefix.add_argument("--seed", type=int, default=None)
    train_prefix.set_defaults(func=_cmd_train_prefix_level0)

    train_prefix_dialogue = sub.add_parser(
        "train-prefix-level0-dialogue",
        help="train only the level-0 head on a pack-dialogue cache; generator stays frozen",
        description=(
            "Train p(level-0 | unpadded dialogue prefix) on a pack-dialogue cache. "
            "The prefix rows are the generator prefixes (right-padded). Pad is stripped, "
            "matching chat, and T is drawn from 1 through that unpadded length so a short "
            "User: line stays in distribution. The assistant reply is not part of this prefix. "
            "The tokenizer and the generator stay frozen. predict_level0 stays false. "
            "Requires a phase-3 config, --dataset-id matching the cache, --max-steps, and "
            "--confirm. --dry-plan prints the plan and does not train. "
            "Writes only to --out. The phase-1 directory ncm-iwsp-gpt1-prefix-level0 is refused. "
            f"A later NicNicNic run can use {RECOMMENDED_DIALOGUE_PREFIX_OUT}. "
            "That path is not required. This command does not submit a Kaggle job. "
            "finetune-dialogue still does not retrain the prefix head."
        ),
    )
    train_prefix_dialogue.add_argument("--config", default="configs/target_phase3.yaml")
    train_prefix_dialogue.add_argument("--tokenizer", default=None, help="frozen tokenizer.pt (required to train)")
    train_prefix_dialogue.add_argument(
        "--generator",
        default=None,
        help="frozen phase-2 or phase-3 generator checkpoint; loaded to check prefix_len, not trained",
    )
    train_prefix_dialogue.add_argument("--packed", default=None, help="pack-dialogue cache; not committed")
    train_prefix_dialogue.add_argument(
        "--dataset-id",
        default=None,
        help="must match the cache; no default corpus",
    )
    train_prefix_dialogue.add_argument(
        "--out",
        default=None,
        help=(
            "directory for the prefix-head checkpoint and log. "
            f"Not ncm-iwsp-gpt1-prefix-level0. Suggested: {RECOMMENDED_DIALOGUE_PREFIX_OUT}"
        ),
    )
    train_prefix_dialogue.add_argument(
        "--resume",
        default=None,
        help="optional prefix-level0 checkpoint to warm-start; read-only, copied into --out",
    )
    train_prefix_dialogue.add_argument("--dry-plan", action="store_true", help="print the run plan and exit without building")
    train_prefix_dialogue.add_argument(
        "--confirm",
        action="store_true",
        help="required to train. Does not submit a Kaggle job.",
    )
    train_prefix_dialogue.add_argument("--max-steps", type=int, default=None, help="step to reach; required unless --dry-plan")
    train_prefix_dialogue.add_argument("--batch-size", type=int, default=8)
    train_prefix_dialogue.add_argument("--lr", type=float, default=1e-3)
    train_prefix_dialogue.add_argument("--weight-decay", type=float, default=0.01)
    train_prefix_dialogue.add_argument("--grad-clip", type=float, default=1.0)
    train_prefix_dialogue.add_argument("--log-every", type=int, default=10)
    train_prefix_dialogue.add_argument("--val-every", type=int, default=50)
    train_prefix_dialogue.add_argument("--save-every", type=int, default=100)
    train_prefix_dialogue.add_argument("--unigram-windows", type=int, default=256)
    train_prefix_dialogue.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    train_prefix_dialogue.add_argument("--seed", type=int, default=None)
    train_prefix_dialogue.set_defaults(func=_cmd_train_prefix_level0_dialogue)

    finetune = sub.add_parser(
        "finetune",
        help="gradient fine-tune the untied generator from a converted init",
    )
    finetune.add_argument("--config", default="configs/target.yaml")
    finetune.add_argument("--tokenizer", default=None, help="converted tokenizer.pt (read-only)")
    finetune.add_argument("--generator", default=None, help="converted generator.pt with untied heads (read-only)")
    finetune.add_argument("--packed", default=None, help="artifacts/packed_windows.pt or an iWSP visible cache")
    finetune.add_argument("--out", default=None, help="directory for checkpoints and logs")
    finetune.add_argument("--dry-plan", action="store_true", help="print the run plan and exit without building")
    finetune.add_argument(
        "--confirm",
        action="store_true",
        help="required to train. Nicholas confirms paths before any Kaggle submit.",
    )
    finetune.add_argument("--max-steps", type=int, default=100_000)
    finetune.add_argument("--batch-size", type=int, default=1, help="per GPU. Default 1 is the T4-sized choice.")
    finetune.add_argument("--grad-accum", type=int, default=1)
    finetune.add_argument("--peak-lr", type=float, default=3e-5)
    finetune.add_argument("--floor-lr", type=float, default=3e-6)
    finetune.add_argument("--warmup", type=int, default=100)
    finetune.add_argument("--weight-decay", type=float, default=0.1)
    finetune.add_argument("--grad-clip", type=float, default=1.0)
    finetune.add_argument("--alpha", type=float, default=0.0)
    finetune.add_argument("--mid-aux", type=float, default=0.0)
    finetune.add_argument("--corrupt-p-max", type=float, default=0.8)
    finetune.add_argument("--corrupt-p-min", type=float, default=0.2)
    finetune.add_argument("--val-every", type=int, default=1000)
    finetune.add_argument("--val-windows", type=int, default=256)
    finetune.add_argument("--val-batch-size", type=int, default=1)
    finetune.add_argument("--save-every", type=int, default=1000)
    finetune.add_argument("--log-every", type=int, default=50)
    finetune.add_argument("--max-hours", type=float, default=None)
    finetune.add_argument("--reserve-minutes", type=float, default=15.0)
    finetune.add_argument("--resume", default="auto", help="auto, none, or a checkpoint path")
    finetune.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    finetune.add_argument("--workers", type=int, default=0)
    finetune.add_argument("--seed", type=int, default=None)
    finetune.add_argument("--no-checkpointing", action="store_true")
    finetune.add_argument("--no-chunked-attn", action="store_true")
    finetune.set_defaults(func=_cmd_finetune)

    pack_dialogue = sub.add_parser(
        "pack-dialogue",
        help="pack an explicitly authorized dialogue or instruction file",
        description=(
            "Pack dialogue rows into one window per assistant reply. "
            "This is not the BookCorpus pack command. "
            "No dataset is selected: pass --dataset-id, an allow-list that contains that id, "
            "and --license-note. Banned names (MiniMind, SFT, DPO, RLAIF, adapters) are still refused. "
            "The output is not committed. This command does not train."
        ),
    )
    pack_dialogue.add_argument("--config", required=True)
    pack_dialogue.add_argument("--dataset-id", required=True, help="operator-supplied id; not a default corpus")
    pack_dialogue.add_argument("--allow-list", required=True, help="text file of allowed dataset ids, one per line")
    pack_dialogue.add_argument("--license-note", required=True, help="license or authorization for this dataset")
    pack_dialogue.add_argument("--jsonl", default=None, help="local JSONL of dialogue rows. Not committed.")
    pack_dialogue.add_argument("--arrow", default=None, help="local .arrow or .parquet dialogue file. Not committed.")
    pack_dialogue.add_argument("--hf-cache", default=None, help="local HF dataset directory")
    pack_dialogue.add_argument("--hf-dataset", default=None, help="dataset id, used only with --download; must equal --dataset-id")
    pack_dialogue.add_argument("--hf-split", default="train")
    pack_dialogue.add_argument("--download", action="store_true", help="allow a network fetch; off by default")
    pack_dialogue.add_argument("--max-docs", type=int, default=0, help="stop after this many rows (0 = no cap)")
    pack_dialogue.add_argument("--max-windows", type=int, default=None, help="override data.max_windows")
    pack_dialogue.add_argument("--out", required=True)
    pack_dialogue.set_defaults(func=_cmd_pack_dialogue)

    finetune_dialogue = sub.add_parser(
        "finetune-dialogue",
        help="fine-tune the generator on a pack-dialogue cache; the prefix head stays frozen",
        description=(
            "Warm-start the generator, usually from a phase-2 checkpoint, and train on a "
            "pack-dialogue cache. The tokenizer stays frozen. predict_level0 stays false. "
            "The prefix head is not retrained. --dataset-id is required and must match the cache. "
            "Hyperparameters are the existing finetune defaults, not an approved phase-3 recipe. "
            "--confirm is required. Writes only to --out. This command does not submit a Kaggle job."
        ),
    )
    finetune_dialogue.add_argument("--config", default="configs/target_phase3.yaml")
    finetune_dialogue.add_argument("--tokenizer", default=None, help="frozen tokenizer.pt (read-only)")
    finetune_dialogue.add_argument(
        "--generator",
        default=None,
        help="phase-2 latest.pt or converted generator.pt to warm-start (read-only)",
    )
    finetune_dialogue.add_argument("--packed", default=None, help="pack-dialogue cache; not committed")
    finetune_dialogue.add_argument("--dataset-id", default=None, help="must match the cache; no default corpus")
    finetune_dialogue.add_argument("--out", default=None, help="new directory for checkpoints and logs")
    finetune_dialogue.add_argument("--dry-plan", action="store_true", help="print the run plan and exit without building")
    finetune_dialogue.add_argument(
        "--confirm",
        action="store_true",
        help="required to train. Does not submit a Kaggle job.",
    )
    finetune_dialogue.add_argument(
        "--retrain-prefix-head",
        action="store_true",
        help="refused. The prefix head is not retrained by this command.",
    )
    finetune_dialogue.add_argument("--max-steps", type=int, default=100_000)
    finetune_dialogue.add_argument("--batch-size", type=int, default=1, help="per GPU. Default 1 is the existing finetune default.")
    finetune_dialogue.add_argument("--grad-accum", type=int, default=1)
    finetune_dialogue.add_argument("--peak-lr", type=float, default=3e-5)
    finetune_dialogue.add_argument("--floor-lr", type=float, default=3e-6)
    finetune_dialogue.add_argument("--warmup", type=int, default=100)
    finetune_dialogue.add_argument("--weight-decay", type=float, default=0.1)
    finetune_dialogue.add_argument("--grad-clip", type=float, default=1.0)
    finetune_dialogue.add_argument("--alpha", type=float, default=0.0)
    finetune_dialogue.add_argument("--mid-aux", type=float, default=0.0)
    finetune_dialogue.add_argument("--corrupt-p-max", type=float, default=0.8)
    finetune_dialogue.add_argument("--corrupt-p-min", type=float, default=0.2)
    finetune_dialogue.add_argument("--val-every", type=int, default=1000)
    finetune_dialogue.add_argument("--val-windows", type=int, default=256)
    finetune_dialogue.add_argument("--val-batch-size", type=int, default=1)
    finetune_dialogue.add_argument("--save-every", type=int, default=1000)
    finetune_dialogue.add_argument("--log-every", type=int, default=50)
    finetune_dialogue.add_argument("--max-hours", type=float, default=None)
    finetune_dialogue.add_argument("--reserve-minutes", type=float, default=15.0)
    finetune_dialogue.add_argument("--resume", default="auto", help="auto, none, or a dialogue checkpoint path")
    finetune_dialogue.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    finetune_dialogue.add_argument("--workers", type=int, default=0)
    finetune_dialogue.add_argument("--seed", type=int, default=None)
    finetune_dialogue.add_argument("--no-checkpointing", action="store_true")
    finetune_dialogue.add_argument("--no-chunked-attn", action="store_true")
    finetune_dialogue.set_defaults(func=_cmd_finetune_dialogue)
    return parser


def _cmd_finetune(args) -> int:
    cfg = load_config(args.config)
    if isinstance(cfg.get("dialogue"), dict) and cfg["dialogue"].get("enabled"):
        raise SystemExit(
            "this config enables dialogue. Use finetune-dialogue. "
            "The finetune command stays the book-window path."
        )
    options = {
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "peak_lr": args.peak_lr,
        "floor_lr": args.floor_lr,
        "warmup": args.warmup,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "mid_aux": args.mid_aux,
        "corrupt_p_max": args.corrupt_p_max,
        "corrupt_p_min": args.corrupt_p_min,
        "val_windows": args.val_windows,
        "gradient_checkpointing": not args.no_checkpointing,
        "chunked_block_attn": not args.no_chunked_attn,
    }
    if args.dry_plan:
        print(json.dumps(describe_run(cfg, options), indent=2))
        return 0
    if not args.confirm:
        raise SystemExit(
            "Pass --confirm after reviewing the input paths. "
            "This command trains the untied NCM-iWSP-GPT1 generator and does not submit a Kaggle job. "
            "Nicholas confirms before any Kaggle submit."
        )
    if not args.tokenizer or not args.generator or not args.packed or not args.out:
        raise SystemExit("finetune needs --tokenizer, --generator, --packed, and --out")
    try:
        summary = run_finetune(
            cfg,
            args.tokenizer,
            args.generator,
            args.packed,
            args.out,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            peak_lr=args.peak_lr,
            floor_lr=args.floor_lr,
            warmup=args.warmup,
            max_steps=args.max_steps,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            alpha=args.alpha,
            mid_aux=args.mid_aux,
            corrupt_p_max=args.corrupt_p_max,
            corrupt_p_min=args.corrupt_p_min,
            val_every=args.val_every,
            val_windows=args.val_windows,
            val_batch_size=args.val_batch_size,
            save_every=args.save_every,
            log_every=args.log_every,
            max_hours=args.max_hours,
            reserve_minutes=args.reserve_minutes,
            resume=args.resume,
            device=args.device,
            gradient_checkpointing=not args.no_checkpointing,
            chunked_block_attn=not args.no_chunked_attn,
            workers=args.workers,
            seed=args.seed,
        )
    except FinetuneError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))
    return 0


def _take_records(iterator, max_docs: int) -> list:
    rows = []
    cap = int(max_docs or 0)
    for row in iterator:
        rows.append(row)
        if cap and len(rows) >= cap:
            break
    if not rows:
        raise SystemExit("no dialogue rows found")
    return rows


def _iter_hf_rows(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "pip install datasets  # only required for --download"
        ) from exc
    dataset = load_dataset(name, split=split, streaming=True)
    for row in dataset:
        if isinstance(row, dict):
            yield row


def _cmd_pack_dialogue(args) -> int:
    from ncm_iwsp_gpt1.dialogue import (
        DialogueError,
        dialogue_settings,
        iter_dialogue_jsonl,
        iter_dialogue_tables,
        pack_dialogue_records,
        require_dataset_authorization,
    )

    cfg = load_config(args.config)
    try:
        settings = dialogue_settings(cfg)
        dataset_id = require_dataset_authorization(
            args.dataset_id, args.allow_list, args.license_note
        )
    except DialogueError as exc:
        raise SystemExit(str(exc)) from exc
    sources = [args.jsonl, args.arrow, args.hf_cache, args.hf_dataset]
    if sum(item is not None for item in sources) != 1:
        raise SystemExit(
            "pack-dialogue needs exactly one of --jsonl, --arrow, --hf-cache, --hf-dataset. "
            "There is no demo corpus and no default dataset."
        )
    if args.hf_dataset and not args.download:
        raise SystemExit(
            "download is off by default. Pass --download to fetch the dataset, "
            "or pack a local --jsonl, --arrow, or --hf-cache."
        )
    if args.hf_dataset and str(args.hf_dataset).strip() != dataset_id:
        raise SystemExit(
            "--hf-dataset must equal --dataset-id. The id is not rewritten to another corpus."
        )
    try:
        if args.jsonl:
            records = _take_records(iter_dialogue_jsonl(args.jsonl), args.max_docs)
            source = args.jsonl
        elif args.arrow:
            records = _take_records(iter_dialogue_tables(args.arrow), args.max_docs)
            source = args.arrow
        elif args.hf_cache:
            records = _take_records(iter_dialogue_tables(args.hf_cache), args.max_docs)
            source = args.hf_cache
        else:
            try:
                refuse_source = str(args.hf_dataset)
                from ncm_iwsp_gpt1.packing import refuse_banned_label

                refuse_banned_label(refuse_source)
            except ValueError as exc:
                raise DialogueError(str(exc)) from exc
            records = _take_records(_iter_hf_rows(args.hf_dataset, args.hf_split), args.max_docs)
            source = args.hf_dataset
        pretok = GPT2BPE.from_config(cfg)
        data = cfg.get("data", {})
        max_windows = (
            int(args.max_windows) if args.max_windows is not None else int(data.get("max_windows", 0) or 0)
        )
        cache = pack_dialogue_records(
            records,
            pretok,
            context_length=settings["context_length"],
            prefix_len=settings["prefix_len"],
            placement=settings["history_placement"],
            dataset_id=dataset_id,
            license_note=args.license_note,
            source=source,
            user_marker=settings["user_marker"],
            assistant_marker=settings["assistant_marker"],
            system_marker=settings["system_marker"],
            hidden_pct=int(data.get("hidden_pct", 10)),
            visible_pct=int(data.get("visible_pct", 10)),
            max_windows=max_windows,
        )
    except (DialogueError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    save_cache(cache, args.out)
    print(json.dumps(cache["meta"], indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


def _finetune_options(args) -> dict:
    return {
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "peak_lr": args.peak_lr,
        "floor_lr": args.floor_lr,
        "warmup": args.warmup,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "alpha": args.alpha,
        "mid_aux": args.mid_aux,
        "corrupt_p_max": args.corrupt_p_max,
        "corrupt_p_min": args.corrupt_p_min,
        "val_windows": args.val_windows,
        "gradient_checkpointing": not args.no_checkpointing,
        "chunked_block_attn": not args.no_chunked_attn,
    }


def _cmd_finetune_dialogue(args) -> int:
    cfg = load_config(args.config)
    if args.retrain_prefix_head:
        raise SystemExit(
            "the prefix head is not retrained. "
            "--retrain-prefix-head is an error, not a silent retrain."
        )
    options = _finetune_options(args)
    if args.dry_plan:
        try:
            print(json.dumps(describe_dialogue_run(cfg, options), indent=2))
        except FinetuneError as exc:
            raise SystemExit(str(exc)) from exc
        return 0
    if not args.confirm:
        raise SystemExit(
            "Pass --confirm after reviewing the input paths. "
            "finetune-dialogue trains the generator on a dialogue cache and does not submit a Kaggle job. "
            "No dialogue corpus is selected unless you pass --dataset-id."
        )
    if not args.dataset_id:
        raise SystemExit(
            "finetune-dialogue needs --dataset-id. No dialogue corpus is selected in this repository."
        )
    if not args.tokenizer or not args.generator or not args.packed or not args.out:
        raise SystemExit("finetune-dialogue needs --tokenizer, --generator, --packed, --dataset-id, and --out")
    try:
        summary = run_dialogue_finetune(
            cfg,
            args.tokenizer,
            args.generator,
            args.packed,
            args.out,
            dataset_id=args.dataset_id,
            retrain_prefix_head=False,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            peak_lr=args.peak_lr,
            floor_lr=args.floor_lr,
            warmup=args.warmup,
            max_steps=args.max_steps,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            alpha=args.alpha,
            mid_aux=args.mid_aux,
            corrupt_p_max=args.corrupt_p_max,
            corrupt_p_min=args.corrupt_p_min,
            val_every=args.val_every,
            val_windows=args.val_windows,
            val_batch_size=args.val_batch_size,
            save_every=args.save_every,
            log_every=args.log_every,
            max_hours=args.max_hours,
            reserve_minutes=args.reserve_minutes,
            resume=args.resume,
            device=args.device,
            gradient_checkpointing=not args.no_checkpointing,
            chunked_block_attn=not args.no_chunked_attn,
            workers=args.workers,
            seed=args.seed,
        )
    except FinetuneError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_train_prefix_level0(args) -> int:
    cfg = load_config(args.config)
    if args.dry_plan:
        options = {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_steps": args.max_steps,
        }
        try:
            plan = describe_prefix_run(cfg, options)
        except PrefixLevel0Error as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(plan, indent=2))
        return 0
    if args.max_steps is None:
        raise SystemExit(
            "train-prefix-level0 needs --max-steps. "
            "This command does not start a long run on its own and does not submit a Kaggle job. "
            "packed_windows.pt is not in the repository; pass a local --packed file."
        )
    if not args.tokenizer or not args.packed or not args.out:
        raise SystemExit("train-prefix-level0 needs --tokenizer, --packed, and --out")
    try:
        summary = run_prefix_level0(
            cfg,
            args.tokenizer,
            args.packed,
            args.out,
            generator_path=args.generator,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
            val_every=args.val_every,
            save_every=args.save_every,
            unigram_windows=args.unigram_windows,
            val_windows=args.val_windows,
            grad_clip=args.grad_clip,
            resume=args.resume,
        )
    except PrefixLevel0Error as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_train_prefix_level0_dialogue(args) -> int:
    cfg = load_config(args.config)
    if args.dry_plan:
        options = {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_steps": args.max_steps,
        }
        try:
            plan = describe_prefix_dialogue_run(cfg, options)
        except PrefixLevel0Error as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(plan, indent=2))
        return 0
    if not args.confirm:
        raise SystemExit(
            "Pass --confirm after reviewing the input paths. "
            "train-prefix-level0-dialogue trains only the prefix head. "
            "The tokenizer and the generator stay frozen. "
            "This command does not submit a Kaggle job."
        )
    if not args.dataset_id:
        raise SystemExit(
            "train-prefix-level0-dialogue needs --dataset-id. "
            "No dialogue corpus is selected in this repository."
        )
    if args.max_steps is None:
        raise SystemExit(
            "train-prefix-level0-dialogue needs --max-steps. "
            "This command does not start a long run on its own and does not submit a Kaggle job."
        )
    if not args.tokenizer or not args.generator or not args.packed or not args.out:
        raise SystemExit(
            "train-prefix-level0-dialogue needs --tokenizer, --generator, --packed, --dataset-id, and --out"
        )
    try:
        summary = run_prefix_level0_dialogue(
            cfg,
            args.tokenizer,
            args.generator,
            args.packed,
            args.out,
            dataset_id=args.dataset_id,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            log_every=args.log_every,
            val_every=args.val_every,
            save_every=args.save_every,
            unigram_windows=args.unigram_windows,
            grad_clip=args.grad_clip,
            resume=args.resume,
        )
    except PrefixLevel0Error as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
