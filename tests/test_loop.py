import csv
import json

import torch

from ncm_iwsp_gpt1.build import build_models
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.loop import beats_hidden_baselines, run_dry_loop
from ncm_iwsp_gpt1.packing import demo_texts, make_cache, save_cache
from ncm_iwsp_gpt1.pretok import GPT2BPE
from ncm_iwsp_gpt1.writers.stub import write_weights


def test_success_rule_is_strict():
    assert beats_hidden_baselines(1.0, 2.0, 1.5) is True
    assert beats_hidden_baselines(2.0, 2.0, 1.0) is False
    assert beats_hidden_baselines(1.5, 2.0, 1.0) is False
    assert beats_hidden_baselines(3.0, 2.0, 4.0) is False
    assert beats_hidden_baselines(None, 2.0, 1.0) is False


def test_stub_noop_matches_seed_and_later_iteration_writes():
    cfg = load_config("configs/gpt1_data.yaml")
    _, left, left_gen = build_models(cfg)
    _, right, right_gen = build_models(cfg)
    write_weights(left, left_gen, iteration=0)
    assert torch.equal(left.encoder.wte.weight, right.encoder.wte.weight)
    write_weights(right, right_gen, iteration=2)
    assert not torch.equal(left.encoder.wte.weight, right.encoder.wte.weight)
    codebook = right.vq.embeddings[0].weight
    assert torch.count_nonzero(codebook) == codebook.shape[0]


def test_dry_loop_writes_visible_csv_and_keeps_hidden_numbers_out(tmp_path):
    cfg = load_config("configs/gpt1_data.yaml")
    pretok = GPT2BPE.from_config(cfg)
    cache = make_cache(demo_texts(), pretok, 32, hidden_pct=10, visible_pct=10, max_windows=2)
    save_cache(cache, tmp_path / "packed.pt")
    out = tmp_path / "run"
    summary = run_dry_loop(cfg, cache, "ncm_iwsp_gpt1/writers/stub.py", out, iterations=2, device="cpu")
    assert summary["status"] == "ok"
    assert summary["device"] == "cpu"
    assert isinstance(summary["hidden_success"], bool)
    with (out / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["iteration"] for row in rows] == ["0", "1"]
    assert all(row["audit"] == "pass" for row in rows)
    header = (out / "results.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "hidden" not in header
    hidden_lines = [
        json.loads(line)
        for line in (out / "hidden_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert hidden_lines[0]["kind"] == "random_init_baseline"
    assert hidden_lines[-1]["kind"] == "success"
    assert hidden_lines[-1]["hidden_success"] == summary["hidden_success"]
    assert hidden_lines[-1]["hidden_success"] == beats_hidden_baselines(
        hidden_lines[-1]["best_mean_level_ce"],
        hidden_lines[-1]["random_init_mean_level_ce"],
        hidden_lines[-1]["unigram_nll"],
    )
    assert all(row["operator_only"] is True for row in hidden_lines)
    baselines = json.loads((out / "baselines.json").read_text(encoding="utf-8"))
    assert "random_init" in baselines
    assert "hidden" not in baselines
    assert (out / "snapshots" / "iter_000.py").exists()
    assert (out / "snapshots" / "iter_001.py").exists()
