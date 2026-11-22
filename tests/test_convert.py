"""Closed-form NCM-GPT1 checkpoint -> NCM-iWSP-GPT1 weights.

The source repository is not imported. Tied heads are a small fixture that
follows the published distance formula, and the destination modules are the
ones this package actually builds.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ncm_iwsp_gpt1.build import build_models, parameter_report
from ncm_iwsp_gpt1.cli import main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.convert import (
    ConvertError,
    convert_modules,
    expand_tied_head,
    load_converted_checkpoint,
    refuse_output_path,
)
from ncm_iwsp_gpt1.loop import run_dry_loop
from ncm_iwsp_gpt1.packing import demo_texts, make_cache
from ncm_iwsp_gpt1.pretok import GPT2BPE


def _reference_tied_logits(x, codebook, proj_weight, log_inv_t, bias):
    """Independent copy of the tied-head forward. Not the converter."""
    z = x @ proj_weight.transpose(-1, -2)
    row_sq = (codebook * codebook).sum(dim=-1)
    scale = row_sq.mean().clamp(min=1e-12)
    return torch.exp(log_inv_t) * (2.0 * (z @ codebook.transpose(-1, -2)) - row_sq) / scale + bias


def test_tied_head_logits_match_linear_on_random_x():
    generator = torch.Generator().manual_seed(0)
    vocab, embed, width = 7, 5, 11
    codebook = torch.randn(vocab, embed, generator=generator, dtype=torch.float64)
    proj_weight = torch.randn(embed, width, generator=generator, dtype=torch.float64)
    log_inv_t = torch.tensor(0.37, dtype=torch.float64)
    bias = torch.randn(vocab, generator=generator, dtype=torch.float64)
    hidden = torch.randn(3, 4, width, generator=generator, dtype=torch.float64)

    weight, out_bias = expand_tied_head(codebook, proj_weight, log_inv_t, bias)
    tied = _reference_tied_logits(hidden, codebook, proj_weight, log_inv_t, bias)
    linear = F.linear(hidden, weight, out_bias)
    assert weight.shape == (vocab, width)
    assert out_bias.shape == (vocab,)
    assert torch.allclose(tied, linear, rtol=1e-10, atol=1e-10)

    row_sq = (codebook * codebook).sum(dim=-1)
    scale = row_sq.mean().clamp(min=1e-12)
    alpha = 2.0 * torch.exp(log_inv_t) / scale
    assert torch.allclose(weight, alpha * (codebook @ proj_weight), rtol=1e-10, atol=1e-10)
    assert torch.allclose(out_bias, bias - (alpha / 2.0) * row_sq, rtol=1e-10, atol=1e-10)


def test_zero_codebook_stays_finite_because_scale_is_floored():
    codebook = torch.zeros(4, 3, dtype=torch.float64)
    proj_weight = torch.randn(3, 6, dtype=torch.float64)
    log_inv_t = torch.tensor(0.0, dtype=torch.float64)
    bias = torch.zeros(4, dtype=torch.float64)
    weight, out_bias = expand_tied_head(codebook, proj_weight, log_inv_t, bias)
    assert torch.isfinite(weight).all()
    assert torch.isfinite(out_bias).all()
    hidden = torch.randn(2, 6, dtype=torch.float64)
    tied = _reference_tied_logits(hidden, codebook, proj_weight, log_inv_t, bias)
    assert torch.allclose(tied, F.linear(hidden, weight, out_bias), rtol=1e-10, atol=1e-10)


def _synthetic_sources(vqvae, generator):
    """Build a state dict shaped like a tied-head NCM-GPT1 checkpoint."""
    tokenizer_sd = {key: value.detach().cpu().clone() for key, value in vqvae.state_dict().items()}
    tokenizer_sd["encoder.wte.weight"].add_(0.25)
    generator_sd = {}
    for key, value in generator.state_dict().items():
        if key.startswith("lm_heads."):
            continue
        generator_sd[key] = value.detach().cpu().clone()
    generator_sd["transformer.wte.weight"].add_(0.5)

    dropped = 0
    embed = vqvae.embed_dim
    for index, head in enumerate(generator.lm_heads):
        vocab, width = head.weight.shape
        codebook = tokenizer_sd[f"vq.embeddings.{index + 1}.weight"]
        assert codebook.shape == (vocab, embed)
        proj = torch.randn(embed, width)
        log_inv_t = torch.tensor(0.05 * (index + 1))
        bias = torch.randn(vocab)
        generator_sd[f"lm_heads.{index}.proj.weight"] = proj
        generator_sd[f"lm_heads.{index}.log_inv_T"] = log_inv_t
        generator_sd[f"lm_heads.{index}.bias"] = bias
        generator_sd[f"post_upsample_convs.{index}.weight"] = torch.randn(embed, embed, 3)
        generator_sd[f"post_upsample_convs.{index}.bias"] = torch.randn(embed)
        dropped += embed * embed * 3 + embed
    # Persistent buffers that the source checkpoint keeps and this model does not.
    generator_sd["lvl_1L"] = torch.zeros(sum(generator.levels[1:]), dtype=torch.long)
    generator_sd["attn_mask_for_masking"] = torch.zeros(1, 1, 2, 2)
    n_codes = tokenizer_sd["vq.embeddings.0.weight"].shape[0]
    for index in range(len(vqvae.vq.embeddings)):
        tokenizer_sd[f"vq.ema_cluster_size_{index}"] = torch.zeros(n_codes)
        tokenizer_sd[f"vq.ema_w_{index}"] = torch.zeros(n_codes, embed)
        tokenizer_sd[f"vq.codebook_hit_{index}"] = torch.zeros(n_codes, dtype=torch.long)
    return tokenizer_sd, generator_sd, dropped


def test_tokenizer_and_body_are_identity_and_heads_expand(tmp_path):
    cfg = load_config("configs/gpt1_data.yaml")
    _, source_vq, source_gen = build_models(cfg)
    tokenizer_sd, generator_sd, dropped_puc = _synthetic_sources(source_vq, source_gen)
    _, vqvae, generator = build_models(cfg, seed=1)
    report = convert_modules(tokenizer_sd, generator_sd, vqvae, generator)

    for key, value in vqvae.state_dict().items():
        assert key in tokenizer_sd
        assert value.shape == tokenizer_sd[key].shape
        assert torch.equal(value, tokenizer_sd[key].to(dtype=value.dtype))
    assert not any(key.startswith("vq.ema_") or key.startswith("vq.codebook_hit_") for key in vqvae.state_dict())

    for key, value in generator.state_dict().items():
        if key.startswith("lm_heads."):
            continue
        assert key in generator_sd
        assert value.shape == generator_sd[key].shape
        assert torch.equal(value, generator_sd[key].to(dtype=value.dtype))
    assert not any("post_upsample" in key for key in generator.state_dict())

    embed = vqvae.embed_dim
    for index, head in enumerate(generator.lm_heads):
        codebook = tokenizer_sd[f"vq.embeddings.{index + 1}.weight"]
        proj = generator_sd[f"lm_heads.{index}.proj.weight"]
        log_inv_t = generator_sd[f"lm_heads.{index}.log_inv_T"]
        bias = generator_sd[f"lm_heads.{index}.bias"]
        weight, out_bias = expand_tied_head(codebook, proj, log_inv_t, bias)
        assert torch.allclose(head.weight, weight.to(dtype=head.weight.dtype), rtol=1e-5, atol=1e-5)
        assert torch.allclose(head.bias, out_bias.to(dtype=head.bias.dtype), rtol=1e-5, atol=1e-5)
        hidden = torch.randn(2, 3, head.in_features)
        tied = _reference_tied_logits(hidden, codebook, proj, log_inv_t, bias)
        assert torch.allclose(tied, F.linear(hidden, head.weight, head.bias), rtol=1e-4, atol=1e-4)
        assert not torch.equal(codebook, tokenizer_sd["vq.embeddings.0.weight"]) or index == 0

    counts = parameter_report(vqvae, generator)
    assert counts["tokenizer"] == report["tokenizer_parameters"]
    assert counts["generator"] == report["generator_parameters"]
    assert report["post_upsample_parameters_dropped"] == dropped_puc
    assert report["heads_expanded"] == len(generator.lm_heads)
    assert report["tokenizer_buffers_skipped"] == 3 * len(vqvae.vq.embeddings)
    assert "approximate" in " ".join(report["notes"]).lower()


def test_target_conversion_fills_the_full_parameter_count():
    cfg = load_config("configs/target.yaml")
    _, vqvae, generator = build_models(cfg)
    tokenizer_sd, generator_sd, dropped_puc = _synthetic_sources(vqvae, generator)
    report = convert_modules(tokenizer_sd, generator_sd, vqvae, generator)
    counts = parameter_report(vqvae, generator)
    assert counts["tokenizer"] == 92_276_480
    assert counts["generator"] == 154_666_240
    assert counts["tokenizer"] + counts["generator"] == 246_942_720
    assert report["total_parameters"] == 246_942_720
    assert report["heads_expanded"] == 22
    assert report["post_upsample_parameters_dropped"] == dropped_puc
    assert dropped_puc == 22 * (384 * 384 * 3 + 384)
    # Every destination parameter was overwritten from the source or the expansion.
    assert all(torch.isfinite(param).all() for param in vqvae.parameters())
    assert all(torch.isfinite(param).all() for param in generator.parameters())
    head = generator.lm_heads[0]
    codebook = tokenizer_sd["vq.embeddings.1.weight"]
    assert codebook.shape == (4096, 384)
    weight, out_bias = expand_tied_head(
        codebook,
        generator_sd["lm_heads.0.proj.weight"],
        generator_sd["lm_heads.0.log_inv_T"],
        generator_sd["lm_heads.0.bias"],
    )
    assert head.weight.shape == (4096, 768)
    assert torch.allclose(head.weight, weight.to(dtype=head.weight.dtype), rtol=1e-5, atol=1e-5)
    assert torch.allclose(head.bias, out_bias.to(dtype=head.bias.dtype), rtol=1e-5, atol=1e-5)


def test_unexpected_source_tensor_is_rejected():
    cfg = load_config("configs/gpt1_data.yaml")
    _, vqvae, generator = build_models(cfg)
    tokenizer_sd, generator_sd, _ = _synthetic_sources(vqvae, generator)
    generator_sd["ctx_wte.weight"] = torch.zeros(4, 4)
    try:
        convert_modules(tokenizer_sd, generator_sd, vqvae, generator)
    except ConvertError as exc:
        assert "ctx_wte.weight" in str(exc)
    else:
        raise AssertionError("unexpected source tensor was ignored")


def test_missing_tied_head_is_rejected():
    cfg = load_config("configs/gpt1_data.yaml")
    _, vqvae, generator = build_models(cfg)
    tokenizer_sd, generator_sd, _ = _synthetic_sources(vqvae, generator)
    del generator_sd["lm_heads.0.log_inv_T"]
    try:
        convert_modules(tokenizer_sd, generator_sd, vqvae, generator)
    except ConvertError as exc:
        assert "log_inv_T" in str(exc)
    else:
        raise AssertionError("missing tied-head scale was ignored")


def test_cli_refuses_to_write_inside_the_source_directory(tmp_path):
    src = tmp_path / "export"
    src.mkdir()
    tokenizer = src / "tokenizer.pt"
    generator = src / "generator.pt"
    torch.save({"model": {}}, tokenizer)
    torch.save({"model": {}}, generator)
    inside = src / "converted.pt"
    before = (tokenizer.stat().st_mtime_ns, generator.stat().st_mtime_ns)
    try:
        main(
            [
                "convert",
                "--config",
                "configs/gpt1_data.yaml",
                "--tokenizer",
                str(tokenizer),
                "--generator",
                str(generator),
                "--out",
                str(inside),
            ]
        )
    except SystemExit as exc:
        assert "different --out" in str(exc).lower() or "source" in str(exc).lower()
    else:
        raise AssertionError("convert wrote inside the source directory")
    assert not inside.exists()
    assert (tokenizer.stat().st_mtime_ns, generator.stat().st_mtime_ns) == before
    with_kaggle = Path("/kaggle/input/some-dataset/converted.pt")
    try:
        refuse_output_path(with_kaggle, tokenizer, generator)
    except ConvertError as exc:
        assert "kaggle" in str(exc).lower()
    else:
        raise AssertionError("kaggle input path was accepted")


def test_cli_roundtrip_and_loop_can_start_from_it(tmp_path):
    cfg = load_config("configs/gpt1_data.yaml")
    _, vqvae, generator = build_models(cfg)
    tokenizer_sd, generator_sd, _ = _synthetic_sources(vqvae, generator)
    src = tmp_path / "kaggle-export"
    src.mkdir()
    tokenizer_path = src / "tokenizer.pt"
    generator_path = src / "generator_step100000.pt"
    torch.save({"model": tokenizer_sd, "step": 100000}, tokenizer_path)
    torch.save(
        {
            "model": generator_sd,
            "step": 100000,
            "optimizer": {"state": {}},
            "inference": {"level0_unigram": torch.zeros(8)},
        },
        generator_path,
    )
    out = tmp_path / "artifacts" / "converted.pt"
    tok_mtime = tokenizer_path.stat().st_mtime_ns
    gen_mtime = generator_path.stat().st_mtime_ns
    code = main(
        [
            "convert",
            "--config",
            "configs/gpt1_data.yaml",
            "--tokenizer",
            str(tokenizer_path),
            "--generator",
            str(generator_path),
            "--out",
            str(out),
            "--smoke-forward",
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert out.is_file()
    assert tokenizer_path.stat().st_mtime_ns == tok_mtime
    assert generator_path.stat().st_mtime_ns == gen_mtime
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["report"]["generator_step"] == 100000
    assert payload["report"]["smoke_forward"]["device"] == "cpu"
    assert len(payload["report"]["smoke_forward"]["logits_shape"]) == 3
    counts = parameter_report(vqvae, generator)
    assert payload["report"]["total_parameters"] == counts["tokenizer"] + counts["generator"]

    _, fresh_vq, fresh_gen = build_models(cfg, seed=3)
    load_converted_checkpoint(out, fresh_vq, fresh_gen)
    assert torch.equal(fresh_vq.encoder.wte.weight, tokenizer_sd["encoder.wte.weight"])
    assert torch.equal(fresh_gen.transformer.wte.weight, generator_sd["transformer.wte.weight"])

    pretok = GPT2BPE.from_config(cfg)
    cache = make_cache(demo_texts(), pretok, 32, hidden_pct=10, visible_pct=10, max_windows=2)
    summary = run_dry_loop(
        cfg,
        cache,
        "ncm_iwsp_gpt1/writers/stub.py",
        tmp_path / "run",
        iterations=1,
        device="cpu",
        init_checkpoint=out,
    )
    assert summary["status"] == "ok"
    assert summary["init_checkpoint"] == str(out)
    baselines = json.loads((tmp_path / "run" / "baselines.json").read_text(encoding="utf-8"))
    assert "random_init" in baselines


def test_generator_file_preference_and_kaggle_notes(tmp_path):
    from ncm_iwsp_gpt1.convert import GENERATOR_FILENAMES, prefer_generator_file

    assert GENERATOR_FILENAMES == ("generator_step100000.pt", "generator.pt", "latest.pt")
    for name in GENERATOR_FILENAMES:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "tokenizer.pt").write_bytes(b"t")
    assert prefer_generator_file(tmp_path).name == "generator_step100000.pt"
    (tmp_path / "generator_step100000.pt").unlink()
    assert prefer_generator_file(tmp_path).name == "generator.pt"

    script = Path("kaggle/convert_iwsp.py").read_text(encoding="utf-8")
    notebook = Path("kaggle/convert_iwsp.ipynb").read_text(encoding="utf-8")
    for text in (script, notebook):
        assert "/kaggle/working" in text
        assert "/kaggle/input" in text
        assert "tokenizer.pt" in text
        assert "generator_step100000.pt" in text
    assert "ncm-gpt-1-train" in notebook or "training kernel" in script + notebook
