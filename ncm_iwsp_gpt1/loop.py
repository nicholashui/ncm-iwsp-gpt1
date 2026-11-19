"""Propose / write / eval / audit dry loop.

The stub writer needs no network. A later Claude or Opus session would edit
the writer file, then this same loop would score it. The loop never passes
hidden windows into ``write_weights``, and the CSV the agent would read has
only the visible scalar.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from pathlib import Path

from ncm_iwsp_gpt1.audit import AuditError, audit_writer_module, state_dict_equal
from ncm_iwsp_gpt1.build import build_models, lock_for_eval, parameter_report
from ncm_iwsp_gpt1.device import resolve_device
from ncm_iwsp_gpt1.evaluate import agent_view, evaluate_windows


def beats_hidden_baselines(model_ce, random_ce, unigram_nll) -> bool:
    """Hidden-split success is a strict improvement on both baselines.

    Lower cross-entropy is better. Matching a baseline does not count.
    The audit is a separate gate; this function only compares the scalars.
    """
    try:
        model_ce = float(model_ce)
        random_ce = float(random_ce)
        unigram_nll = float(unigram_nll)
    except (TypeError, ValueError):
        return False
    return model_ce < random_ce and model_ce < unigram_nll


def load_writer(path: str | Path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("ncm_iwsp_gpt1_writer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load writer {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value
    if value is None or isinstance(value, (int, str, bool)):
        return value
    return str(value)


def _fresh(cfg):
    pretok, vqvae, generator = build_models(cfg)
    return pretok, vqvae, generator


def assert_writer_is_deterministic(cfg, writer) -> None:
    _, left_tok, left_gen = _fresh(cfg)
    _, right_tok, right_gen = _fresh(cfg)
    writer.write_weights(left_tok, left_gen, iteration=1)
    writer.write_weights(right_tok, right_gen, iteration=1)
    if not state_dict_equal(left_tok, right_tok) or not state_dict_equal(left_gen, right_gen):
        raise AuditError(
            ["write_weights is not deterministic: two calls with the same iteration changed different tensors"]
        )


def run_dry_loop(cfg, cache: dict, writer_path: str | Path, out_dir: str | Path, iterations: int = 3, device: str | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    snapshots = out / "snapshots"
    snapshots.mkdir(exist_ok=True)
    writer_path = Path(writer_path)
    shutil.copy(writer_path, snapshots / "writer_source.py")

    writer = load_writer(writer_path)
    problems = audit_writer_module(writer, writer_path)
    if problems:
        _write_status(out, {"status": "reject", "problems": problems})
        raise AuditError(problems)
    assert_writer_is_deterministic(cfg, writer)

    eval_cfg = cfg.get("eval", {}) or {}
    batch_size = int(eval_cfg.get("batch_size", 4))
    max_windows = int(eval_cfg.get("max_windows", 32))
    n_generate = int(eval_cfg.get("n_generate", 2))
    resolved_device = resolve_device(device if device is not None else eval_cfg.get("device", "auto"))

    pretok, vqvae, generator = _fresh(cfg)
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    counts = parameter_report(vqvae, generator)
    random_metrics = evaluate_windows(
        pretok,
        vqvae,
        generator,
        cache["visible"],
        train_windows=cache["visible_train"],
        batch_size=batch_size,
        max_windows=max_windows,
        n_generate=n_generate,
        temperature=0.0,
        device=resolved_device,
    )
    baselines = {
        "random_init": {
            "mean_level_ce": random_metrics["mean_level_ce"],
            "recon_ce": random_metrics["recon_ce"],
            "recon_acc": random_metrics["recon_acc"],
            "mean_exact_acc": random_metrics["mean_exact_acc"],
            "length_weighted_ce": random_metrics["length_weighted_ce"],
        },
        "most_frequent_token_ce": random_metrics["baselines"]["most_frequent_token_ce"],
        "most_frequent_token_acc": random_metrics["baselines"]["most_frequent_token_acc"],
        "token_unigram_nll": random_metrics["baselines"]["token_unigram_nll"],
        "always_pad_ce": random_metrics["baselines"]["always_pad_ce"],
        "uniform_finer_ce": random_metrics["baselines"]["uniform_finer_ce"],
        "level0_unigram_nll": random_metrics["baselines"]["level0_unigram_nll"],
        "mean_predicted_level_unigram_nll": random_metrics["baselines"]["mean_predicted_level_unigram_nll"],
        "qualitative_random_init": random_metrics["baselines"]["qualitative"],
        "note": (
            "Level-0 unigram and most-frequent token counts were computed inside "
            "the evaluator from the frozen visible_train split of the random-init model. "
            "The writer was not called."
        ),
    }
    (out / "baselines.json").write_text(json.dumps(_jsonable(baselines), indent=2), encoding="utf-8")
    (out / "param_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")

    random_hidden = evaluate_windows(
        pretok,
        vqvae,
        generator,
        cache["hidden"],
        train_windows=cache["visible_train"],
        batch_size=batch_size,
        max_windows=max_windows,
        n_generate=0,
        temperature=0.0,
        device=resolved_device,
    )
    hidden_random_ce = random_hidden["mean_level_ce"]
    hidden_unigram = random_hidden["baselines"]["mean_predicted_level_unigram_nll"]

    csv_path = out / "results.csv"
    visible_log = out / "visible.jsonl"
    hidden_log = out / "hidden_metrics.jsonl"
    fields = [
        "iteration",
        "model_shorthand",
        "mean_level_ce",
        "recon_acc",
        "recon_ce",
        "mean_exact_acc",
        "tok_params",
        "gen_params",
        "audit",
        "status",
        "description",
    ]
    with hidden_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "iteration": None,
                        "operator_only": True,
                        "kind": "random_init_baseline",
                        "mean_level_ce": hidden_random_ce,
                        "unigram_nll": hidden_unigram,
                    }
                )
            )
            + "\n"
        )
    best_ce = None
    best_step = None
    rows = []
    for step in range(int(iterations)):
        pretok, vqvae, generator = _fresh(cfg)
        writer.write_weights(vqvae, generator, iteration=step)
        # Re-read the file each iteration so an external editor is what gets audited.
        problems = audit_writer_module(writer, writer_path)
        description = getattr(writer, "LAST_DESCRIPTION", "") or getattr(writer, "MODEL_DESCRIPTION", "")
        snapshot = snapshots / f"iter_{step:03d}.py"
        shutil.copy(writer_path, snapshot)
        if problems:
            row = {
                "iteration": step,
                "model_shorthand": getattr(writer, "MODEL_SHORTHAND", ""),
                "mean_level_ce": "",
                "recon_acc": "",
                "recon_ce": "",
                "mean_exact_acc": "",
                "tok_params": counts["tokenizer"],
                "gen_params": counts["generator"],
                "audit": "fail",
                "status": "reject",
                "description": description,
            }
            rows.append(row)
            _flush(csv_path, fields, rows, visible_log, hidden_log)
            _write_status(out, {"status": "reject", "iteration": step, "problems": problems})
            raise AuditError(problems)

        visible = evaluate_windows(
            pretok,
            vqvae,
            generator,
            cache["visible"],
            train_windows=cache["visible_train"],
            batch_size=batch_size,
            max_windows=max_windows,
            n_generate=0,
            temperature=0.0,
            device=resolved_device,
        )
        hidden = evaluate_windows(
            pretok,
            vqvae,
            generator,
            cache["hidden"],
            train_windows=cache["visible_train"],
            batch_size=batch_size,
            max_windows=max_windows,
            n_generate=0,
            temperature=0.0,
            device=resolved_device,
        )
        if best_ce is None or hidden["mean_level_ce"] < best_ce:
            best_ce = hidden["mean_level_ce"]
            best_step = step
        view = agent_view(visible)
        row = {
            "iteration": step,
            "model_shorthand": getattr(writer, "MODEL_SHORTHAND", ""),
            "mean_level_ce": f"{view['mean_level_ce']:.6f}",
            "recon_acc": f"{visible['recon_acc']:.6f}",
            "recon_ce": f"{visible['recon_ce']:.6f}",
            "mean_exact_acc": f"{visible['mean_exact_acc']:.6f}",
            "tok_params": counts["tokenizer"],
            "gen_params": counts["generator"],
            "audit": "pass",
            "status": "ok",
            "description": description,
        }
        rows.append(row)
        with visible_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable({"iteration": step, "agent_view": view, "visible": visible})) + "\n")
        with hidden_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _jsonable(
                        {
                            "iteration": step,
                            "operator_only": True,
                            "mean_level_ce": hidden["mean_level_ce"],
                            "recon_acc": hidden["recon_acc"],
                            "recon_ce": hidden["recon_ce"],
                        }
                    )
                )
                + "\n"
            )
        _flush(csv_path, fields, rows, visible_log, hidden_log)

    hidden_success = beats_hidden_baselines(best_ce, hidden_random_ce, hidden_unigram)
    with hidden_log.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "operator_only": True,
                        "kind": "success",
                        "hidden_success": hidden_success,
                        "best_iteration": best_step,
                        "best_mean_level_ce": best_ce,
                        "random_init_mean_level_ce": hidden_random_ce,
                        "unigram_nll": hidden_unigram,
                        "audit_pass": True,
                    }
                )
            )
            + "\n"
        )
    summary = {
        "status": "ok",
        "iterations": int(iterations),
        "param_counts": counts,
        "random_init_mean_level_ce": baselines["random_init"]["mean_level_ce"],
        "hidden_success": hidden_success,
        "success_rule": (
            "On the hidden split, mean_level_ce must be strictly below random init "
            "and strictly below the evaluator unigram, and the audit must pass. "
            "Hidden numbers stay in hidden_metrics.jsonl."
        ),
        "results_csv": str(csv_path),
        "hidden_log": str(hidden_log),
        "agent_scalar_name": "mean_level_ce",
        "device": str(resolved_device),
    }
    _write_status(out, summary)
    return summary


def _flush(csv_path, fields, rows, visible_log, hidden_log):
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    del visible_log, hidden_log


def _write_status(out: Path, payload: dict) -> None:
    (out / "status.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
