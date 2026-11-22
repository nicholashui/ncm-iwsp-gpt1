"""Fine-tune tests use the runnable config. They do not build the 247M target."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ncm_iwsp_gpt1.build import build_models
from ncm_iwsp_gpt1.cli import main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.finetune import (
    FinetuneError,
    assert_untied_generator_state,
    corrupt_codes,
    find_converted_pair,
    find_packed_windows,
    generator_ce_loss,
    load_training_windows,
    loss_is_alpha0,
    loss_note,
    refuse_finetune_output,
    run_finetune,
)
from ncm_iwsp_gpt1.packing import pack_documents


def _fake_init(tmp_path: Path):
    cfg = load_config("configs/gpt1_data.yaml")
    pretok, vqvae, generator = build_models(cfg)
    init = tmp_path / "init"
    init.mkdir()
    torch.save(
        {"model": vqvae.state_dict(), "step": 100000, "converted_from": {"kind": "unit-tokenizer"}},
        init / "tokenizer.pt",
    )
    torch.save(
        {"model": generator.state_dict(), "step": 100000, "converted_from": {"kind": "unit-generator"}},
        init / "generator.pt",
    )
    (init / "convert_report.json").write_text("{}\n", encoding="utf-8")
    windows = pack_documents(
        ["a short packed sentence about a library window", "another sentence beside it"],
        pretok,
        cfg["tokenizer"]["context_length"],
    )
    windows = windows.repeat(8, 1)
    packed_dir = tmp_path / "v10" / "artifacts"
    packed_dir.mkdir(parents=True)
    packed = packed_dir / "packed_windows.pt"
    torch.save(
        {
            "windows": windows.to(dtype=torch.int32),
            "n_docs": 2,
            "n_toks": int(windows.shape[0] * windows.shape[1]),
            "meta": {
                "L": int(windows.shape[1]),
                "max_windows": int(windows.shape[0]),
                "respect_sentences": False,
                "pack_repeat": 1,
                "eos": int(pretok.eos_token_id),
                "pad": int(pretok.pad_token_id),
                "hf_dataset": "lucadiliello/bookcorpusopen",
                "hf_split": "train",
                "hf_text_field": "text",
                "data": "",
                "max_docs": 0,
            },
        },
        packed,
    )
    return cfg, init / "tokenizer.pt", init / "generator.pt", packed


def test_alpha0_loss_is_the_position_mean_and_mid_aux_is_separate():
    torch.manual_seed(0)
    levels = [1, 2, 3]
    logits = torch.randn(2, sum(levels), 4, requires_grad=True)
    targets = torch.randint(0, 4, (2, sum(levels)))
    base = generator_ce_loss(logits, targets, levels, [4, 4, 4], alpha=0.0, mid_aux=0.0)
    manual = F.cross_entropy(logits.reshape(-1, 4), targets.reshape(-1))
    assert torch.allclose(base["loss"], manual)
    assert base["length_weighted_ce"] == pytest.approx(float(base["loss"].detach()))
    shifted = generator_ce_loss(logits, targets, levels, [4, 4, 4], alpha=0.5, mid_aux=0.5)
    assert shifted["ce_first"] == base["ce_first"]
    assert shifted["ce_mid"] == base["ce_mid"]
    assert shifted["ce_last"] == base["ce_last"]
    assert shifted["length_weighted_ce"] == pytest.approx(base["length_weighted_ce"])
    assert not torch.allclose(shifted["loss"], base["loss"])
    assert loss_is_alpha0(0.0, 0.0)
    assert not loss_is_alpha0(0.5, 0.5)
    assert "Do not compare this loss to alpha=0 curves" in loss_note(0.5, 0.5)
    base["loss"].backward()
    assert logits.grad is not None


def test_corrupt_codes_leaves_the_label_tensor_unchanged():
    indices = torch.zeros(4, 6, dtype=torch.long)
    changed = corrupt_codes(indices, [2, 4], [8, 8], p_max=1.0, p_min=1.0)
    assert torch.equal(indices, torch.zeros_like(indices))
    assert not torch.equal(changed, indices)


def test_chunked_attention_matches_the_dense_mask():
    cfg = load_config("configs/gpt1_data.yaml")
    _, vqvae, generator = build_models(cfg)
    generator.eval()
    length = int(generator.prefix_len) + int(sum(generator.levels[1:]))
    hidden = torch.randn(2, length, vqvae.embed_dim)
    generator.chunked_block_attn = False
    dense = generator(hidden)
    generator.chunked_block_attn = True
    chunked = generator(hidden)
    assert torch.allclose(dense, chunked, atol=1e-5, rtol=1e-5)
    assert generator.n_embd == 32


def test_load_v10_tail_and_ignore_hidden_cache(tmp_path):
    windows = torch.arange(10, dtype=torch.long).unsqueeze(1).expand(10, 8).contiguous()
    v10 = tmp_path / "artifacts" / "packed_windows.pt"
    v10.parent.mkdir()
    torch.save({"windows": windows, "meta": {"L": 8}}, v10)
    train, val, meta = load_training_windows(v10, 8, val_windows=2)
    assert meta["kind"] == "v10_packed_windows"
    assert train.shape == (8, 8)
    assert torch.equal(val[:, 0], torch.tensor([8, 9]))
    cache = tmp_path / "iwsp.pt"
    torch.save(
        {
            "visible_train": torch.ones(2, 8, dtype=torch.long),
            "visible": torch.full((1, 8), 2),
            "hidden": torch.full((4, 8), 9),
            "meta": {"context_length": 8},
        },
        cache,
    )
    train, val, meta = load_training_windows(cache, 8, val_windows=256)
    assert meta["kind"] == "iwsp_visible_splits"
    assert torch.equal(train, torch.ones_like(train))
    assert torch.equal(val, torch.full_like(val, 2))
    assert int(train.eq(9).sum()) == 0
    with pytest.raises(FinetuneError, match=r"\[N, 32\]"):
        load_training_windows(v10, 32, val_windows=2)


def test_refuse_unsafe_output_paths(tmp_path):
    init = tmp_path / "init"
    init.mkdir()
    tokenizer = init / "tokenizer.pt"
    generator = init / "generator.pt"
    tokenizer.write_bytes(b"tok")
    generator.write_bytes(b"gen")
    packed = tmp_path / "export" / "artifacts" / "packed_windows.pt"
    packed.parent.mkdir(parents=True)
    packed.write_bytes(b"pack")
    sources = [tokenizer, generator, packed]
    refuse_finetune_output(tmp_path / "working" / "finetune", sources)
    with pytest.raises(FinetuneError, match="holds an input"):
        refuse_finetune_output(init / "run", sources)
    with pytest.raises(FinetuneError, match="/kaggle/input"):
        refuse_finetune_output(Path("/kaggle/input/dataset/out"), sources)
    for name in ("ncm-gpt1-kaggle-100k", "kaggle-v10-weights", "ncm-gpt-1-train"):
        with pytest.raises(FinetuneError, match=name):
            refuse_finetune_output(tmp_path / name / "out", sources)


def test_tied_head_state_is_rejected():
    weight = torch.zeros(4, 3)
    state = {"lm_heads.0.weight": weight, "lm_heads.0.proj.weight": weight}
    with pytest.raises(FinetuneError, match="tied-head"):
        assert_untied_generator_state(state)
    state = {"post_upsample_convs.0.weight": weight, "lm_heads.0.weight": weight}
    with pytest.raises(FinetuneError, match="post-upsample"):
        assert_untied_generator_state(state)


def test_finder_prefers_converted_init_and_artifacts_windows(tmp_path):
    raw = tmp_path / "raw-export"
    raw.mkdir()
    (raw / "tokenizer.pt").write_bytes(b"a")
    (raw / "generator.pt").write_bytes(b"b")
    (raw / "packed_windows.pt").write_bytes(b"root")
    init = tmp_path / "iwsp-init"
    init.mkdir()
    (init / "tokenizer.pt").write_bytes(b"c")
    (init / "generator.pt").write_bytes(b"d")
    (init / "convert_report.json").write_text("{}\n", encoding="utf-8")
    artifacts = raw / "artifacts"
    artifacts.mkdir()
    (artifacts / "packed_windows.pt").write_bytes(b"art")
    tokenizer, generator = find_converted_pair(tmp_path)
    assert tokenizer.parent == init
    assert generator.parent == init
    assert find_packed_windows(tmp_path) == artifacts / "packed_windows.pt"


def test_train_save_resume_and_frozen_tokenizer(tmp_path):
    cfg, tokenizer, generator, packed = _fake_init(tmp_path)
    out = tmp_path / "working" / "finetune"
    stamp = tokenizer.stat().st_mtime_ns
    summary = run_finetune(
        cfg,
        tokenizer,
        generator,
        packed,
        out,
        batch_size=2,
        grad_accum=2,
        peak_lr=0.05,
        floor_lr=0.05,
        warmup=0,
        max_steps=2,
        alpha=0.0,
        mid_aux=0.0,
        corrupt_p_max=0.0,
        corrupt_p_min=0.0,
        val_every=1,
        val_windows=2,
        val_batch_size=1,
        save_every=2,
        log_every=1,
        resume="none",
        device="cpu",
        gradient_checkpointing=True,
        chunked_block_attn=True,
        seed=0,
    )
    assert summary["start_step"] == 0
    assert summary["step"] == 2
    assert summary["init_step"] == 100000
    assert summary["converted_from"] == {"kind": "unit-generator"}
    assert summary["tokenizer_checksum_start"] == summary["tokenizer_checksum_end"]
    assert summary["generator_checksum_start"] != summary["generator_checksum_end"]
    assert tokenizer.stat().st_mtime_ns == stamp
    assert generator.stat().st_mtime_ns
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["loss_comparable_to_alpha0"] is True
    assert report["n_embd"] == 32
    assert "ce_first" in report["loss_note"]
    rows = [
        json.loads(line)
        for line in (out / "val_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for row in rows:
        assert {"val_ce_first", "val_ce_mid", "val_ce_last"} <= set(row)
        assert row["loss_comparable_to_alpha0"] is True
        assert isinstance(row["val_ce_first"], float)
        assert isinstance(row["val_ce_mid"], float)
        assert isinstance(row["val_ce_last"], float)
    step2 = torch.load(out / "generator_step2.pt", map_location="cpu", weights_only=False)
    assert step2["format"] == "ncm-iwsp-gpt1-finetune"
    assert step2["model"]["lm_heads.0.weight"].shape[-1] == 32
    assert "post_upsample_convs.0.weight" not in step2["model"]
    summary2 = run_finetune(
        cfg,
        tokenizer,
        generator,
        packed,
        out,
        batch_size=2,
        grad_accum=1,
        peak_lr=0.05,
        floor_lr=0.05,
        warmup=0,
        max_steps=4,
        alpha=0.0,
        mid_aux=0.0,
        corrupt_p_max=0.0,
        corrupt_p_min=0.0,
        val_every=2,
        val_windows=2,
        save_every=2,
        log_every=1,
        resume="auto",
        device="cpu",
        gradient_checkpointing=True,
        chunked_block_attn=True,
        seed=0,
    )
    assert summary2["start_step"] == 2
    assert summary2["step"] == 4
    assert summary2["resumed_from"]
    latest = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["step"] == 4
    assert not torch.equal(step2["model"]["lm_heads.0.weight"], latest["model"]["lm_heads.0.weight"])
    assert tokenizer.stat().st_mtime_ns == stamp
    bad = tmp_path / "ncm-gpt1-kaggle-100k" / "out"
    with pytest.raises(FinetuneError, match="ncm-gpt1-kaggle-100k"):
        run_finetune(cfg, tokenizer, generator, packed, bad, max_steps=1, device="cpu")
    assert not bad.exists()


def test_kaggle_script_dry_plan(capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("finetune_iwsp_kaggle", "kaggle/finetune_iwsp.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--dry-plan", "--config", "configs/target.yaml"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["batch_size_per_gpu"] == 1
    assert plan["n_layer"] == 12
    assert plan["untied_heads"] is True


def test_cli_dry_plan_does_not_build_the_target(capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("dry-plan must not build configs/target.yaml")

    monkeypatch.setattr("ncm_iwsp_gpt1.finetune.build_models", boom)
    assert main(["finetune", "--config", "configs/target.yaml", "--dry-plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["system"] == "NCM-iWSP-GPT1"
    assert plan["n_layer"] == 12
    assert plan["n_embd"] == 768
    assert plan["n_head"] == 12
    assert plan["context_length"] == 512
    assert plan["grid_cells"] == 4307
    assert plan["batch_size_per_gpu"] == 1
    assert plan["gradient_checkpointing"] is True
    assert plan["chunked_block_attn"] is True
    assert plan["untied_heads"] is True
    assert plan["post_upsample_conv"] is False
    assert plan["predict_level0"] is False
    assert plan["tokenizer_frozen"] is True
    assert plan["loss_comparable_to_alpha0"] is True


def test_cli_requires_confirm_and_refuses_v10_output(tmp_path):
    with pytest.raises(SystemExit, match="--confirm"):
        main(
            [
                "finetune",
                "--config",
                "configs/gpt1_data.yaml",
                "--tokenizer",
                "tokenizer.pt",
                "--generator",
                "generator.pt",
                "--packed",
                "packed_windows.pt",
                "--out",
                str(tmp_path / "working"),
            ]
        )
    with pytest.raises(SystemExit, match="kaggle-v10-weights"):
        main(
            [
                "finetune",
                "--config",
                "configs/gpt1_data.yaml",
                "--tokenizer",
                str(tmp_path / "tokenizer.pt"),
                "--generator",
                str(tmp_path / "generator.pt"),
                "--packed",
                str(tmp_path / "packed_windows.pt"),
                "--out",
                str(tmp_path / "kaggle-v10-weights" / "out"),
                "--confirm",
                "--device",
                "cpu",
            ]
        )
    assert not (tmp_path / "kaggle-v10-weights").exists()
