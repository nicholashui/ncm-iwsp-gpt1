#!/usr/bin/env python3
"""Official decode-side chat for NCM-iWSP-GPT1.

Prefix-head predicts level 0. The frozen generator samples later residual
VQ levels. VQ decode reconstructs one window. A decode-side splice writes
the planned Assistant body, and the official reply slice extracts it.
Residual codes are not rewritten. The planner weights are not in this
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

PLANNER_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PLANNER = "planners/tinyllama-1.1b-chat"

DESCRIPTION = """\
Official decode-side chat for NCM-iWSP-GPT1.

This is not NCM-GPT1 and it is not causal token continuation. The prefix
head predicts level 0, the frozen generator samples later residual VQ
levels, VQ decode reconstructs one window, and a decode-side splice writes
the Assistant body. The official reply slice then keeps the Assistant span.

Dialogue mode and --prefix-head are required. Temperature 0 is argmax.
This command stops if the prefix head is missing. It does not switch to
another level-0 source.

The planner is TinyLlama/TinyLlama-1.1B-Chat-v1.0 (Apache-2.0). Download
it separately. Its weights are not part of this repository.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planner_weight_path(path: Path) -> Path:
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "no planner weights in %s. Download %s separately; do not commit the weights."
        % (path, PLANNER_ID)
    )


def load_planner(path: Path, device: str = "cpu"):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "official chat needs the transformers package to load the planner. "
            "The planner weights themselves are downloaded separately."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    n_params = sum(int(param.numel()) for param in model.parameters())
    print("PLANNER path=%s device=%s params=%s" % (path, device, n_params), file=sys.stderr, flush=True)
    return model, tokenizer


def marker_token_ids(pretok):
    from ncm_iwsp_gpt1.dialogue import (
        DEFAULT_ASSISTANT_MARKER,
        DEFAULT_SYSTEM_MARKER,
        DEFAULT_USER_MARKER,
    )

    ids = []
    for marker in (DEFAULT_USER_MARKER, DEFAULT_ASSISTANT_MARKER, DEFAULT_SYSTEM_MARKER):
        encoded = pretok.encode(marker) or []
        ids.extend(int(item) for item in encoded)
    return sorted(set(ids))


def _history_text(session) -> str:
    turns = getattr(session, "history", None) or []
    parts = []
    for item in turns:
        if isinstance(item, dict) and item.get("role") == "user":
            parts.append(str(item.get("content") or ""))
    return "\n".join(part for part in parts if part)


def attach_decode_splice(session, planner, planner_tok):
    """Hook official decode so reply extraction sees the policy-filtered splice."""
    from ncm_iwsp_gpt1.dialogue import (
        DEFAULT_ASSISTANT_MARKER,
        DEFAULT_SYSTEM_MARKER,
        DEFAULT_USER_MARKER,
    )
    from ncm_iwsp_gpt1.reply_policy import WINDOW_TAIL_RESERVE
    from ncm_iwsp_gpt1.splice_lm import (
        official_assistant_open_ids,
        official_nonpad_span,
        plan_post_prefix_tokens,
        planner_context_ids,
        splice_decoded_ids,
    )

    original_decode = session._decode_ids
    original_reply = session.reply
    pad_id = int(session.pretok.pad_token_id)
    eos_id = int(session.pretok.eos_token_id)
    exclude = marker_token_ids(session.pretok)
    user_marker = session.user_marker if getattr(session, "user_marker", None) else DEFAULT_USER_MARKER
    assistant_marker = (
        session.assistant_marker if getattr(session, "assistant_marker", None) else DEFAULT_ASSISTANT_MARKER
    )
    system_marker = session.system_marker if getattr(session, "system_marker", None) else DEFAULT_SYSTEM_MARKER
    open_ids = official_assistant_open_ids(session.pretok, assistant_marker)

    def reply_with_user(text):
        session.pending_user_text = text
        return original_reply(text)

    def hooked(level0, prefix):
        decoded = original_decode(level0, prefix)
        seq_len = int(decoded.shape[-1]) if hasattr(decoded, "shape") else len(decoded)
        planned = plan_post_prefix_tokens(
            prefix,
            pad_id,
            session.pretok,
            planner,
            eos_id=eos_id,
            exclude_ids=exclude,
            assistant_open_ids=open_ids,
            planner_tokenizer=planner_tok,
            user_marker=user_marker,
            assistant_marker=assistant_marker,
            system_marker=system_marker,
            user_text=getattr(session, "pending_user_text", None),
            seq_len=seq_len,
            history_text=_history_text(session),
        )
        spliced = splice_decoded_ids(
            decoded,
            prefix,
            planned,
            pad_id=pad_id,
            eos_id=eos_id,
            assistant_open_ids=open_ids,
            tail_reserve=WINDOW_TAIL_RESERVE,
        )
        n_cond = official_nonpad_span(prefix, pad_id)
        context = planner_context_ids(prefix, pad_id, eos_id, open_ids)
        session.last_splice = {
            "n_cond": int(n_cond),
            "plan_len": int(planned.shape[-1]) if hasattr(planned, "shape") else len(planned),
            "context_len": int(len(context)),
            "tail_reserve": int(WINDOW_TAIL_RESERVE),
            "path": "decode-side-instruct",
        }
        return spliced

    session._decode_ids = hooked
    session.reply = reply_with_user
    session.splice_attached = True
    return session


def add_parser_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True, help="frozen tokenizer checkpoint")
    parser.add_argument("--finetune", required=True, help="dialogue generator checkpoint")
    parser.add_argument("--prefix-head", required=True, help="required prefix-head checkpoint")
    parser.add_argument("--prompt", default=None, help="one user turn; omit for the stdin loop")
    parser.add_argument(
        "--planner-dir",
        default=None,
        help="local directory for %s (env NCM_PLANNER_DIR). Not committed." % PLANNER_ID,
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="0 is argmax")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--planner-device", default="cpu", help="device for the frozen planner")
    parser.add_argument(
        "--dialogue",
        action="store_true",
        help="required. Keep dialogue mode and slice the Assistant span.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_parser_arguments(parser)
    return parser


def _audit_sources() -> int:
    from ncm_iwsp_gpt1.audit_lmsplice import audit_writer
    import ncm_iwsp_gpt1.chat_lmsplice as wrapper
    import ncm_iwsp_gpt1.reply_policy as policy
    import ncm_iwsp_gpt1.splice_lm as splice_mod

    for module in (splice_mod, policy, wrapper):
        report = audit_writer(Path(module.__file__))
        label = Path(module.__file__).name
        print(
            "AUDIT %s ok=%s violations=%s"
            % (label, report["ok"], report["violations"] if report["violations"] else "none"),
            file=sys.stderr,
            flush=True,
        )
        if not report["ok"]:
            return 2
    return 0


def execute(args) -> int:
    if not bool(getattr(args, "dialogue", False)):
        print("official chat requires --dialogue", file=sys.stderr, flush=True)
        return 2
    if not getattr(args, "prefix_head", None):
        print(
            "official chat requires --prefix-head and does not switch to another level-0 source",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if _audit_sources() != 0:
        return 2

    planner_dir = Path(args.planner_dir or os.environ.get("NCM_PLANNER_DIR", DEFAULT_PLANNER))
    try:
        weights = planner_weight_path(planner_dir)
    except FileNotFoundError as exc:
        print("FAIL %s" % exc, file=sys.stderr, flush=True)
        print(
            "Download the planner with: python -m ncm_iwsp_gpt1.download_planner --out %s" % planner_dir,
            file=sys.stderr,
            flush=True,
        )
        return 2
    print("PLANNER_ID=%s" % PLANNER_ID, file=sys.stderr, flush=True)
    print("PLANNER_SHA=%s path=%s" % (sha256_file(weights), weights), file=sys.stderr, flush=True)

    from ncm_iwsp_gpt1.chat import open_session, run_interactive, run_prompt
    from ncm_iwsp_gpt1.config import load_config

    planner, planner_tok = load_planner(planner_dir, device=str(args.planner_device))
    cfg = load_config(args.config)
    session = open_session(
        cfg,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        tokenizer_path=args.tokenizer,
        finetune_path=args.finetune,
        prefix_head_path=args.prefix_head,
        dialogue=True,
    )
    attach_decode_splice(session, planner, planner_tok)
    print(
        "SPLICE_HOOK=decode-side-instruct attached=%s" % bool(getattr(session, "splice_attached", False)),
        file=sys.stderr,
        flush=True,
    )
    if args.prompt is not None:
        return run_prompt(session, args.prompt)
    return run_interactive(session)


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(execute(args))


if __name__ == "__main__":
    raise SystemExit(main())
