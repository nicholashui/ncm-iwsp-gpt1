"""Prefix level-0 head: shapes, one train step, and chat wiring on the small config."""

import json
import math

import pytest
import torch

from ncm_iwsp_gpt1.build import build_models, lock_for_eval
from ncm_iwsp_gpt1.chat import ChatError, open_session, pack_prompt_window
from ncm_iwsp_gpt1.cli import main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.packing import pack_documents
from ncm_iwsp_gpt1.prefix_level0 import (
    PREFIX_LEVEL0_FORMAT,
    PrefixLevel0Error,
    PrefixLevel0Head,
    describe_prefix_run,
    embed_prompt_tokens,
    head_parameter_count,
    attention_pool,
    level0_code_counts,
    run_prefix_level0,
    sample_logits,
    sample_prefix_lengths,
    save_prefix_head,
    train_step,
)


def _cfg():
    return load_config("configs/gpt1_data.yaml")


def _forbid_build(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("chat built a model")

    monkeypatch.setattr("ncm_iwsp_gpt1.chat.build_models", boom)


def _windows(pretok, context):
    windows = pack_documents(
        [
            "The cat sat on the mat beside a window.",
            "Hello world is a short line in a book.",
            "A traveler walked into the library and stopped.",
        ],
        pretok,
        int(context),
    )
    if int(windows.shape[0]) < 4:
        windows = windows.repeat(8, 1)
    return windows.contiguous()


def _save_inputs(tmp_path):
    cfg = _cfg()
    pretok, vqvae, generator = build_models(cfg)
    state = vqvae.state_dict()
    state["encoder.wte.weight"] = state["encoder.wte.weight"] + 0.1
    init = tmp_path / "init"
    init.mkdir()
    tokenizer = init / "tokenizer.pt"
    generator_path = init / "generator.pt"
    torch.save({"model": state, "step": 3, "converted_from": {"kind": "unit"}}, tokenizer)
    torch.save(
        {"model": generator.state_dict(), "step": 3, "converted_from": {"kind": "unit"}},
        generator_path,
    )
    windows = _windows(pretok, cfg["tokenizer"]["context_length"])
    packed_dir = tmp_path / "corpus"
    packed_dir.mkdir()
    packed = packed_dir / "packed_windows.pt"
    torch.save({"windows": windows, "meta": {"L": int(windows.shape[1])}}, packed)
    return cfg, tokenizer, generator_path, packed


def _forced_head(path, index=7):
    cfg = _cfg()
    embed = int(cfg["tokenizer"]["embed_dim"])
    codebook = int(cfg["tokenizer"]["codebook_size"])
    context = int(cfg["tokenizer"]["context_length"])
    head = PrefixLevel0Head(embed, codebook, context)
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.zero_()
        head.proj.bias[index] = 8.0
    save_prefix_head(
        path,
        head,
        0,
        embed_dim=embed,
        codebook_size=codebook,
        context_length=context,
        config_name=cfg["name"],
    )
    return path


def test_shipped_configs_keep_phase1_generator_flags():
    for name in ("configs/gpt1_data.yaml", "configs/target.yaml"):
        cfg = load_config(name)
        assert cfg["generator"]["predict_level0"] is False
        assert cfg["generator"]["prefix_len"] == 1
        assert int(cfg["tokenizer"]["levels"][0]) == 1


def test_head_shapes_mask_and_order():
    head = PrefixLevel0Head(32, 64, 32)
    logits = head(torch.randn(2, 5, 32), torch.tensor([3, 5]))
    assert logits.shape == (2, 64)
    assert sum(param.numel() for param in head.parameters()) == head_parameter_count(32, 64, 32)
    assert head_parameter_count(384, 4096, 512) == 512 * 384 + 384 + 2 * 384 + 384 * 4096 + 4096

    rows = torch.zeros(2, 4, 32)
    rows[:, 0, :] = 1
    rows[0, 1, :] = 5
    rows[1, 2, :] = -3
    same = head(rows, torch.tensor([1, 1]))
    assert torch.allclose(same[0], same[1])

    ordered = PrefixLevel0Head(4, 4, 4)
    with torch.no_grad():
        ordered.pos.weight.zero_()
        ordered.pos.weight[1, 1] = 5
        ordered.query.zero_()
        ordered.query[1] = 1
        ordered.norm.weight.fill_(1)
        ordered.norm.bias.zero_()
        ordered.proj.weight.zero_()
        ordered.proj.weight[:, 0] = 1
        ordered.proj.bias.zero_()
    left = torch.zeros(1, 2, 4)
    right = torch.zeros(1, 2, 4)
    left[0, 0, 0] = 1
    right[0, 1, 0] = 1
    assert not torch.allclose(ordered(left, torch.tensor([2])), ordered(right, torch.tensor([2])))
    with pytest.raises(ValueError):
        head(torch.zeros(1, 4, 32), torch.tensor([0]))
    with pytest.raises(ValueError):
        head(torch.zeros(1, 4, 32), torch.tensor([5]))

    hidden = torch.zeros(1, 2, 4)
    hidden[0, 1, 1] = 4
    query = torch.zeros(4)
    query[1] = 1
    pooled = attention_pool(hidden, query, torch.tensor([2]))
    # Softmax is not a hard max. The position with the larger score still dominates.
    assert float(pooled[0, 1]) > 3.5
    assert float(pooled[0, 0]) == 0.0


def test_length_draw_covers_short_and_full_windows():
    generator = torch.Generator().manual_seed(0)
    drawn = sample_prefix_lengths(4000, 32, generator=generator)
    assert int(drawn.min()) == 1
    assert int(drawn.max()) == 32


def test_sample_logits_argmax_matches_temperature_zero():
    logits = torch.tensor([[0.1, 3.0, 0.2]])
    assert int(sample_logits(logits, 0.0, 0, 1.0)) == 1
    torch.manual_seed(0)
    draw = int(sample_logits(logits, 0.8, 0, 1.0))
    assert draw in (0, 1, 2)


def test_train_step_updates_only_the_head():
    cfg = _cfg()
    _pretok, vqvae, generator = build_models(cfg)
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    head = PrefixLevel0Head(32, 64, 32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
    before_tok = [param.detach().clone() for param in vqvae.parameters()]
    before_gen = [param.detach().clone() for param in generator.parameters()]
    before_head = [param.detach().clone() for param in head.parameters()]
    windows = torch.randint(0, 1000, (4, 32))
    stats = train_step(head, optimizer, vqvae, windows)
    assert math.isfinite(stats["loss"])
    assert 1 <= int(stats["lengths"].min()) <= int(stats["lengths"].max()) <= 32
    assert any(not torch.equal(old, param.detach()) for old, param in zip(before_head, head.parameters()))
    assert all(torch.equal(old, param.detach()) for old, param in zip(before_tok, vqvae.parameters()))
    assert all(torch.equal(old, param.detach()) for old, param in zip(before_gen, generator.parameters()))
    assert set(id(param) for param in optimizer.param_groups[0]["params"]) == set(
        id(param) for param in head.parameters()
    )


def test_phase1_refuses_generator_changes_before_training():
    cfg = _cfg()
    cfg["generator"]["predict_level0"] = True
    with pytest.raises(PrefixLevel0Error, match="predict_level0"):
        run_prefix_level0(cfg, "missing-tokenizer.pt", "missing-packed.pt", "out", max_steps=1)
    cfg = _cfg()
    cfg["generator"]["prefix_len"] = 32
    with pytest.raises(PrefixLevel0Error, match="prefix_len"):
        run_prefix_level0(cfg, "missing-tokenizer.pt", "missing-packed.pt", "out", max_steps=1)


def test_dry_plan_does_not_train(capsys):
    assert main(["train-prefix-level0", "--config", "configs/gpt1_data.yaml", "--dry-plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["optimized"] == "prefix_level0_head"
    assert plan["tokenizer_frozen"] is True
    assert plan["generator_frozen"] is True
    assert plan["predict_level0"] is False
    assert plan["prefix_len"] == 1
    assert plan["codebook_size"] == 64
    assert plan["context_length"] == 32
    assert plan["head_parameters"] == head_parameter_count(32, 64, 32)
    assert plan["does_not_submit_kaggle"] is True
    target = describe_prefix_run(load_config("configs/target.yaml"))
    assert target["codebook_size"] == 4096
    assert target["context_length"] == 512
    assert target["prefix_len"] == 1
    assert target["predict_level0"] is False
    with pytest.raises(SystemExit, match="max-steps"):
        main(["train-prefix-level0", "--config", "configs/gpt1_data.yaml"])


def test_one_step_trains_a_small_checkpoint_and_can_resume(tmp_path):
    cfg, tokenizer, generator_path, packed = _save_inputs(tmp_path)
    out = tmp_path / "run"
    summary = run_prefix_level0(
        cfg,
        tokenizer,
        packed,
        out,
        generator_path=generator_path,
        max_steps=1,
        batch_size=2,
        lr=1e-2,
        val_windows=2,
        unigram_windows=8,
        device="cpu",
        seed=0,
    )
    assert summary["tokenizer_unchanged"] is True
    assert summary["generator_unchanged"] is True
    assert summary["generator_loaded"] is True
    assert summary["head_changed"] is True
    assert summary["predict_level0"] is False
    assert summary["prefix_len"] == 1
    assert math.isfinite(summary["last_loss"])
    assert all(1 <= value <= 32 for value in summary["last_lengths"])
    assert math.isfinite(summary["val_ce_T1"])
    assert summary["val_unigram_nll"] > 0
    checkpoint = tmp_path / "run" / "prefix_level0.pt"
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size < 200_000
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert blob["format"] == PREFIX_LEVEL0_FORMAT
    assert set(blob["model"]) == {
        "pos.weight",
        "query",
        "norm.weight",
        "norm.bias",
        "proj.weight",
        "proj.bias",
    }
    assert blob["generator_updated"] is False
    assert blob["tokenizer_updated"] is False
    continued = run_prefix_level0(
        cfg,
        tokenizer,
        packed,
        out,
        generator_path=generator_path,
        max_steps=2,
        batch_size=2,
        lr=1e-2,
        val_windows=2,
        unigram_windows=8,
        device="cpu",
        seed=0,
        resume=checkpoint,
    )
    assert continued["resumed_from"] == 1
    assert continued["steps"] == 2
    assert continued["head_changed"] is True
    assert torch.load(checkpoint, map_location="cpu", weights_only=False)["step"] == 2


def test_refuses_to_write_next_to_packed_windows(tmp_path):
    cfg = _cfg()
    packed = tmp_path / "corpus" / "packed_windows.pt"
    packed.parent.mkdir()
    packed.write_bytes(b"not-read")
    with pytest.raises(PrefixLevel0Error, match="refusing"):
        run_prefix_level0(cfg, packed, packed, packed.parent, max_steps=1)


def test_chat_without_prefix_head_or_baseline_does_not_build(monkeypatch):
    _forbid_build(monkeypatch)
    with pytest.raises(SystemExit, match="prefix-head"):
        main(["chat", "--config", "configs/gpt1_data.yaml", "--device", "cpu", "--prompt", "Hello"])
    with pytest.raises(ChatError, match="prefix-head"):
        open_session(_cfg(), device="cpu")


def test_bad_prefix_checkpoint_does_not_build(monkeypatch, tmp_path):
    _forbid_build(monkeypatch)
    path = tmp_path / "nope.pt"
    torch.save({"format": "ncm-iwsp-gpt1-finetune", "model": {"weight": torch.zeros(1)}}, path)
    with pytest.raises(ChatError, match=PREFIX_LEVEL0_FORMAT):
        open_session(_cfg(), device="cpu", prefix_head_path=path)
    _forced_head(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["embed_dim"] = 16
    torch.save(blob, path)
    with pytest.raises(ChatError, match="embed_dim"):
        open_session(_cfg(), device="cpu", prefix_head_path=path)


def test_prefix_head_uses_real_prompt_not_packed_repeat(tmp_path, monkeypatch, capsys):
    path = _forced_head(tmp_path / "head.pt", index=7)
    session = open_session(_cfg(), device="cpu", temperature=0.0, seed=0, prefix_head_path=path)
    text = "How are you?"
    ids = session.pretok.encode(text)
    assert ids == [2437, 389, 345, 30]
    seen = {}
    real_forward = session.prefix_head.forward

    def spy(embeddings, lengths):
        seen["emb"] = embeddings.detach().cpu().clone()
        seen["lengths"] = lengths.detach().cpu().tolist()
        return real_forward(embeddings, lengths)

    session.prefix_head.forward = spy

    def boom_pack(*args, **kwargs):
        raise AssertionError("prefix-head path packed the prompt")

    def boom_encode(*args, **kwargs):
        raise AssertionError("prefix-head path encoded a window")

    monkeypatch.setattr("ncm_iwsp_gpt1.chat.pack_prompt_window", boom_pack)
    session.vqvae.encode = boom_encode
    out = session.reply(text)
    with torch.no_grad():
        expected = embed_prompt_tokens(session.vqvae.encoder.wte, torch.tensor([ids])).cpu()
    assert torch.equal(seen["emb"], expected)
    assert seen["lengths"] == [len(ids)]
    assert seen["emb"].shape[1] == len(ids)
    assert seen["emb"].shape[1] != session.context_length
    assert out["level0_source"] == "prefix_head"
    assert out["level0"] == [7]
    assert out["prefix_ids"] == [session.pretok.eos_token_id]
    assert out["prefix_len"] == 1
    assert out["repeated"] is False
    assert out["truncated"] is False
    assert out["prompt_tokens"] == 4
    assert out["window_ids"] == ids
    assert isinstance(out["text"], str)
    other = session.reply("Hello world")
    assert other["level0"] == [7]
    assert other["window_ids"] != out["window_ids"]
    packed = pack_prompt_window(session.pretok, text, session.context_length)
    assert len(packed["window_ids"]) == session.context_length
    assert out["window_ids"] != packed["window_ids"]
    long = session.reply("hello " * 80)
    assert long["truncated"] is True
    assert long["repeated"] is False
    assert long["level0"] == [7]
    assert len(long["window_ids"]) == session.context_length
    assert (
        main(
            [
                "chat",
                "--config",
                "configs/gpt1_data.yaml",
                "--device",
                "cpu",
                "--prefix-head",
                str(path),
                "--prompt",
                text,
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "level0_source=prefix_head" in captured.err
    assert "repeated=no" in captured.err
    assert "repeated=yes" not in captured.err
    assert "level0_source=packed" not in captured.err


def test_one_step_head_chat_does_not_pack(tmp_path, monkeypatch):
    cfg, tokenizer, generator_path, packed = _save_inputs(tmp_path)
    out = tmp_path / "trained"
    summary = run_prefix_level0(
        cfg,
        tokenizer,
        packed,
        out,
        generator_path=generator_path,
        max_steps=1,
        batch_size=2,
        lr=1e-2,
        val_windows=2,
        device="cpu",
    )
    session = open_session(
        cfg,
        device="cpu",
        temperature=0.0,
        seed=0,
        tokenizer_path=tokenizer,
        generator_path=generator_path,
        prefix_head_path=summary["checkpoint"],
    )
    ids = session.pretok.encode("The cat sat")
    seen = {}
    real_forward = session.prefix_head.forward

    def spy(embeddings, lengths):
        seen["lengths"] = [int(value) for value in lengths.detach().cpu().tolist()]
        return real_forward(embeddings, lengths)

    session.prefix_head.forward = spy
    monkeypatch.setattr(
        "ncm_iwsp_gpt1.chat.pack_prompt_window",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("packed")),
    )
    session.vqvae.encode = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("encoded"))
    result = session.reply("The cat sat")
    assert result["level0_source"] == "prefix_head"
    assert result["repeated"] is False
    assert result["window_ids"] == ids
    assert seen["lengths"] == [len(ids)]
    assert len(ids) < session.context_length
    assert result["prefix_ids"] == [session.pretok.eos_token_id]
    assert session.generator.prefix_len == 1
    assert 0 <= result["level0"][0] < int(cfg["tokenizer"]["codebook_size"])


def test_unigram_baseline_ignores_the_prompt(tmp_path):
    cfg, _tokenizer, _generator, packed = _save_inputs(tmp_path)
    session = open_session(
        cfg,
        device="cpu",
        temperature=0.0,
        seed=0,
        level0_source="unigram",
        packed_path=packed,
        unigram_max_windows=8,
    )
    windows = torch.load(packed, map_location="cpu", weights_only=False)["windows"].long()
    counts = level0_code_counts(session.vqvae, windows, max_windows=8)
    expected = int(torch.argmax(counts + 1.0))
    session.vqvae.encode = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("encoded"))
    first = session.reply("How are you?")
    second = session.reply("Hello world")
    assert first["level0_source"] == "unigram"
    assert first["level0"] == [expected]
    assert second["level0"] == [expected]
    assert first["prefix_ids"] == [session.pretok.eos_token_id]
    assert first["repeated"] is False
    assert len(first["window_ids"]) == 4
    assert first["window_ids"] != second["window_ids"]


def test_baseline_flags_are_explicit(monkeypatch, tmp_path):
    _forbid_build(monkeypatch)
    with pytest.raises(ChatError, match="not both"):
        open_session(_cfg(), device="cpu", prefix_head_path=tmp_path / "h.pt", level0_source="packed")
    with pytest.raises(ChatError, match="unigram"):
        open_session(_cfg(), device="cpu", level0_source="unigram")
    with pytest.raises(ChatError, match="only used with"):
        open_session(_cfg(), device="cpu", level0_source="packed", packed_path=tmp_path / "w.pt")
