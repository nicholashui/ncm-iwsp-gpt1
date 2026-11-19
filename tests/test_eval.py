import math

import torch

from ncm_iwsp_gpt1.build import build_models
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.evaluate import _token_ce_acc, agent_view, evaluate_windows
from ncm_iwsp_gpt1.packing import demo_texts, make_cache
from ncm_iwsp_gpt1.pretok import GPT2BPE
from ncm_iwsp_gpt1.writers.stub import write_weights


def _cache():
    cfg = load_config("configs/gpt1_data.yaml")
    pretok = GPT2BPE.from_config(cfg)
    cache = make_cache(demo_texts(), pretok, 32, hidden_pct=10, visible_pct=10, max_windows=4)
    return cfg, pretok, cache


def test_pad_positions_do_not_change_token_ce():
    logits = torch.zeros(1, 4, 5)
    tokens = torch.tensor([[2, 3, 1, 1]])
    changed = tokens.clone()
    changed[0, 2] = 4
    changed[0, 3] = 0
    left = _token_ce_acc(logits, tokens, pad_id=1)
    right = _token_ce_acc(logits, changed, pad_id=1)
    assert left["ce"] == right["ce"]
    assert left["n"] == 2


def test_evaluator_metrics_and_unigram_stay_inside_eval():
    cfg, pretok, cache = _cache()
    _, vqvae, generator = build_models(cfg)
    write_weights(vqvae, generator, iteration=0)
    metrics = evaluate_windows(
        pretok,
        vqvae,
        generator,
        cache["visible"],
        train_windows=cache["visible_train"],
        batch_size=2,
        max_windows=4,
        n_generate=1,
        temperature=0.0,
        device="cpu",
    )
    assert metrics["recon_tokens"] > 0
    assert 0.0 <= metrics["recon_acc"] <= 1.0
    assert metrics["recon_ce"] > 0
    assert metrics["mean_level_ce"] > 0
    assert metrics["scalar"] == metrics["mean_level_ce"]
    assert len(metrics["generator"]["per_level"]) == len(vqvae.vq.levels) - 1
    for row in metrics["generator"]["per_level"]:
        assert 0.0 <= row["exact_acc"] <= 1.0
        assert row["ce"] > 0
    assert metrics["baselines"]["most_frequent_token_ce"] > 0
    assert 0.0 <= metrics["baselines"]["most_frequent_token_acc"] <= 1.0
    assert metrics["baselines"]["token_unigram_nll"] > 0
    assert metrics["baselines"]["always_pad_ce"] > 0
    assert metrics["baselines"]["level0_unigram_nll"] > 0
    assert metrics["baselines"]["mean_predicted_level_unigram_nll"] > 0
    assert abs(metrics["baselines"]["uniform_finer_ce"] - math.log(64)) < 1e-6
    kinds = {row["kind"] for row in metrics["baselines"]["qualitative"]}
    assert "level0_unigram_plus_random_finer" in kinds
    assert "fixed_prompt" in kinds
    view = agent_view(metrics)
    assert set(view) == {"scalar", "mean_level_ce"}
    assert "hidden" not in view

    before = vqvae.encoder.wte.weight.detach().clone()
    other = evaluate_windows(
        pretok,
        vqvae,
        generator,
        cache["visible"],
        train_windows=cache["hidden"],
        batch_size=2,
        max_windows=4,
        n_generate=0,
        device="cpu",
    )
    assert torch.equal(before, vqvae.encoder.wte.weight)
    assert other["baselines"]["level0_unigram_nll"] is not None
