"""Phase 2 prefix protocol. Small modules only; the 12/768 target is not built."""

import json
from pathlib import Path

import pytest
import torch

from ncm_iwsp_gpt1.build import build_models
from ncm_iwsp_gpt1.chat import ChatError, open_session, startup_message
from ncm_iwsp_gpt1.cli import build_parser, main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.evaluate import evaluate_windows
from ncm_iwsp_gpt1.finetune import FinetuneError, _prepare_batch, load_module_init, run_finetune
from ncm_iwsp_gpt1.packing import _refuse_corpus_name, pack_documents
from ncm_iwsp_gpt1.params import estimate_report
from ncm_iwsp_gpt1.prefix_level0 import PrefixLevel0Error, PrefixLevel0Head, save_prefix_head
from ncm_iwsp_gpt1.prefix_protocol import (
    PHASE2_FIXED_PROMPTS,
    PHASE2_PREFIX_CANDIDATES,
    PHASE2_PREFIX_LEN,
    PHASE2_SMALL_PREFIX_LEN,
    batch_prefix_tokens,
    display_continuation,
    load_generator_state,
    window_prefix_ids,
)
from ncm_iwsp_gpt1.pretok import GPT2BPE


def _phase1():
    return load_config("configs/gpt1_data.yaml")


def _phase2():
    return load_config("configs/gpt1_data_phase2.yaml")


def _target_phase2():
    return load_config("configs/target_phase2.yaml")


def _head(path: Path) -> Path:
    head = PrefixLevel0Head(32, 64, 32)
    save_prefix_head(
        path,
        head,
        0,
        embed_dim=32,
        codebook_size=64,
        context_length=32,
        config_name="NCM-iWSP-GPT1-runnable",
    )
    return path


def _pair(tmp_path: Path):
    cfg = _phase1()
    pretok, vqvae, generator = build_models(cfg)
    init = tmp_path / "init"
    init.mkdir()
    tokenizer = init / "tokenizer.pt"
    generator_path = init / "generator.pt"
    torch.save({"model": vqvae.state_dict(), "step": 100000, "converted_from": {"kind": "unit"}}, tokenizer)
    torch.save(
        {"model": generator.state_dict(), "step": 100000, "converted_from": {"kind": "unit"}},
        generator_path,
    )
    windows = pack_documents(
        ["a short packed sentence about a library window", "another sentence beside it"],
        pretok,
        cfg["tokenizer"]["context_length"],
    ).repeat(8, 1)
    packed_dir = tmp_path / "v10" / "artifacts"
    packed_dir.mkdir(parents=True)
    packed = packed_dir / "packed_windows.pt"
    torch.save({"windows": windows.to(dtype=torch.int32), "meta": {"L": int(windows.shape[1])}}, packed)
    return cfg, pretok, tokenizer, generator_path, packed, generator.state_dict()


def test_phase2_configs_keep_phase1_files_and_name_the_protocol():
    phase1 = load_config("configs/target.yaml")
    small = load_config("configs/gpt1_data.yaml")
    target = _target_phase2()
    runnable = _phase2()
    for cfg in (phase1, small):
        assert cfg["generator"]["predict_level0"] is False
        assert cfg["generator"]["prefix_len"] == 1
        assert cfg["generator"].get("prefix_source", "eos") == "eos"
    assert target["name"].startswith("NCM-iWSP-GPT1")
    assert "ncm-gpt1" not in target["name"].lower()
    assert target["generator"]["n_layer"] == 12
    assert target["generator"]["n_embd"] == 768
    assert target["generator"]["n_head"] == 12
    assert target["tokenizer"]["context_length"] == 512
    assert target["tokenizer"]["codebook_size"] == 4096
    assert target["tokenizer"]["levels"] == phase1["tokenizer"]["levels"]
    assert target["generator"]["predict_level0"] is False
    assert target["generator"]["prefix_len"] == PHASE2_PREFIX_LEN == 32
    assert target["generator"]["prefix_source"] == "window"
    assert runnable["generator"]["prefix_len"] == PHASE2_SMALL_PREFIX_LEN == 8
    assert runnable["generator"]["prefix_source"] == "window"
    assert runnable["generator"]["predict_level0"] is False
    predicted = sum(target["tokenizer"]["levels"][1:])
    assert predicted == 4306
    assert 32 + predicted == 4338
    assert 64 + predicted == 4370
    assert estimate_report(phase1) == estimate_report(target)
    assert estimate_report(small)["generator"] == estimate_report(runnable)["generator"]


def test_prefix_32_and_64_rebuild_masks_and_load_phase1_weights():
    phase1 = _phase1()
    _, vqvae, source = build_models(phase1)
    source_state = source.state_dict()
    target_len = sum(source.levels[1:])
    assert set(PHASE2_PREFIX_CANDIDATES) == {32, 64}
    for prefix_len in PHASE2_PREFIX_CANDIDATES:
        cfg = _phase1()
        cfg["generator"]["prefix_len"] = prefix_len
        cfg["generator"]["prefix_source"] = "window"
        _, built_vq, generator = build_models(cfg)
        seq = prefix_len + target_len
        assert generator.prefix_len == prefix_len
        assert generator.prefix_source == "window"
        assert generator.attn_mask.shape == (1, 1, seq, seq)
        mask = generator.attn_mask[0, 0]
        assert torch.all(mask[:prefix_len, :prefix_len] == 0)
        assert torch.isinf(mask[:prefix_len, prefix_len:]).all()
        assert torch.all(mask[prefix_len:, :prefix_len] == 0)
        assert "attn_mask" not in source_state
        dropped = load_generator_state(generator, source_state)
        assert dropped == []
        assert torch.equal(generator.transformer.wte.weight, source.transformer.wte.weight)
        assert torch.equal(generator.lm_heads[0].weight, source.lm_heads[0].weight)
        dirty = dict(source_state)
        dirty["attn_mask"] = torch.zeros(1, 1, 3, 3)
        dirty["lvl_1L"] = torch.zeros(1, 4)
        dirty["rope_cos"] = torch.zeros(2, 2)
        dirty["rope_sin"] = torch.zeros(2, 2)
        dirty["rope_cos_gen_0"] = torch.zeros(1)
        dirty["rope_sin_gen_0"] = torch.zeros(1)
        dropped = load_generator_state(generator, dirty)
        assert "attn_mask" in dropped
        assert "rope_cos" in dropped
        assert generator.attn_mask.shape == (1, 1, seq, seq)
        assert torch.equal(generator.transformer.wte.weight, source.transformer.wte.weight)
        latents = torch.randn(2, seq, built_vq.embed_dim)
        generator.eval()
        generator.chunked_block_attn = False
        dense = generator(latents)
        generator.chunked_block_attn = True
        chunked = generator(latents)
        assert dense.shape == (2, target_len, generator.vocab_sizes[0])
        assert torch.allclose(dense, chunked, atol=1e-5, rtol=1e-5)
        level0 = torch.zeros(2, int(built_vq.vq.levels[0]), dtype=torch.long)
        prefix = torch.randint(0, 100, (2, prefix_len))
        generated = generator.generate(level0, prefix, temperature=0.0)
        assert generated.shape == (2, sum(generator.levels))
        del built_vq, generator, latents, dense, chunked, generated


def test_window_prefix_is_right_pad_and_keeps_the_start():
    pretok = GPT2BPE()
    assert window_prefix_ids([7, 8, 9], 5, 50257) == [7, 8, 9, 50257, 50257]
    assert window_prefix_ids([7, 8, 9, 4, 5, 6], 4, 50257) == [7, 8, 9, 4]
    assert window_prefix_ids([], 2, 50257) == [50257, 50257]
    tokens = torch.tensor([[4, 5, 6, 7]])
    eos = batch_prefix_tokens(tokens, 3, "eos", pretok.eos_token_id, pretok.pad_token_id)
    window = batch_prefix_tokens(tokens, 6, "window", pretok.eos_token_id, pretok.pad_token_id)
    assert eos.tolist() == [[pretok.eos_token_id, pretok.eos_token_id, pretok.eos_token_id]]
    assert window.tolist() == [[4, 5, 6, 7, pretok.pad_token_id, pretok.pad_token_id]]
    assert display_continuation("How are you?", "How are you? fine") == ("How are you? fine", False)
    shown, prepended = display_continuation("How are you?", ",,...")
    assert prepended is True
    assert shown == "How are you?\n,,..."
    assert display_continuation("  ", ",,...") == (",,...", False)
    for prompt in PHASE2_FIXED_PROMPTS:
        ids = pretok.encode(prompt)
        assert ids
        prefix = window_prefix_ids(ids, 32, pretok.pad_token_id)
        assert len(prefix) == 32
        assert prefix[: min(len(ids), 32)] == ids[:32]
        assert prefix[0] == ids[0]


def test_prepare_batch_phase1_stays_eos_and_phase2_uses_the_window_start():
    tokens = torch.randint(0, 1000, (2, 32))
    pretok, vqvae, generator = build_models(_phase1())
    teacher, targets, levels, vocabs = _prepare_batch(
        vqvae,
        generator,
        tokens,
        pretok.eos_token_id,
        corrupt=False,
        p_max=0.0,
        p_min=0.0,
        pad_id=pretok.pad_token_id,
    )
    eos = vqvae.encoder.wte(torch.full((2, 1), pretok.eos_token_id))
    assert generator.prefix_source == "eos"
    assert torch.allclose(teacher[:, :1], eos)
    assert targets.shape[1] == sum(levels)

    _, vqvae2, generator2 = build_models(_phase2())
    teacher2, targets2, levels2, _vocabs2 = _prepare_batch(
        vqvae2,
        generator2,
        tokens,
        pretok.eos_token_id,
        corrupt=False,
        p_max=0.0,
        p_min=0.0,
        pad_id=pretok.pad_token_id,
    )
    expected = vqvae2.encoder.wte(tokens[:, : generator2.prefix_len])
    assert generator2.prefix_source == "window"
    assert generator2.prefix_len == 8
    assert torch.allclose(teacher2[:, :8], expected)
    assert teacher2.shape[1] == 8 + sum(levels2)
    assert targets2.shape[1] == sum(levels2)
    with pytest.raises(ValueError, match="predict_level0"):
        from ncm_iwsp_gpt1.models.generator import NextConceptGenerator

        bad = dict(_phase2()["generator"])
        bad["predict_level0"] = True
        NextConceptGenerator(vqvae2, bad)


def test_phase1_eval_prefix_stays_eos_and_phase2_eval_uses_the_window(monkeypatch):
    seen = []
    real = batch_prefix_tokens

    def spy(tokens, prefix_len, prefix_source, eos_id, pad_id):
        seen.append(prefix_source)
        return real(tokens, prefix_len, prefix_source, eos_id, pad_id)

    monkeypatch.setattr("ncm_iwsp_gpt1.evaluate.batch_prefix_tokens", spy)
    pretok, vqvae, generator = build_models(_phase1())
    windows = torch.randint(0, 500, (2, 32))
    evaluate_windows(pretok, vqvae, generator, windows, batch_size=2, max_windows=2, n_generate=0, device="cpu")
    assert seen and set(seen) == {"eos"}
    seen.clear()
    pretok, vqvae, generator = build_models(_phase2())
    metrics = evaluate_windows(
        pretok,
        vqvae,
        generator,
        windows,
        train_windows=windows,
        batch_size=2,
        max_windows=2,
        n_generate=1,
        temperature=0.0,
        device="cpu",
    )
    assert seen and set(seen) == {"window"}
    qualitative = metrics["baselines"]["qualitative"]
    kinds = {row["kind"] for row in qualitative}
    assert "unigram_l0_then_generator" in kinds
    assert "phase2_fixed_prompt" in kinds
    controls = [row for row in qualitative if row["kind"] == "unigram_l0_then_generator"]
    assert controls and all(row["prefix_source"] == "eos" for row in controls)
    prompts = [row for row in qualitative if row["kind"] == "phase2_fixed_prompt"]
    assert {row["prompt"] for row in prompts} == set(PHASE2_FIXED_PROMPTS)
    assert all(row["level0_source"] == "unigram" for row in prompts)
    assert all(len(row["prefix_ids"]) == 8 for row in prompts)
    how = next(row for row in prompts if row["prompt"] == "How are you?")
    ids = pretok.encode("How are you?")
    assert ids == [2437, 389, 345, 30]
    assert how["prefix_ids"] == ids + [pretok.pad_token_id] * (8 - len(ids))


def test_phase1_chat_prefix_is_still_end_of_text():
    session = open_session(_phase1(), device="cpu", temperature=0.0, level0_source="packed")
    out = session.reply("How are you?")
    assert session.prefix_source == "eos"
    assert out["prefix_ids"] == [session.pretok.eos_token_id]
    assert out["prefix_source"] == "eos"
    assert out["text"] == out["decoded_window"] == out["display_text"]
    assert out["prompt_prepended"] is False
    assert "The generator prefix is end-of-text." in startup_message(session)
    with pytest.raises(ChatError, match="No prefix-head"):
        open_session(_phase1(), device="cpu")


def test_phase2_chat_puts_the_prompt_in_the_prefix(tmp_path):
    head = _head(tmp_path / "prefix_level0.pt")
    cfg = _phase2()
    with pytest.raises(ChatError, match="No prefix-head"):
        open_session(cfg, device="cpu")
    session = open_session(cfg, device="cpu", temperature=0.0, prefix_head_path=head)
    text = "How are you?"
    ids = session.pretok.encode(text)
    assert ids == [2437, 389, 345, 30]
    out = session.reply(text)
    assert out["prefix_source"] == "window"
    assert out["prefix_len"] == 8
    assert out["prefix_ids"] == ids + [session.pretok.pad_token_id] * 4
    assert session.pretok.eos_token_id not in out["prefix_ids"]
    assert out["repeated"] is False
    assert out["truncated"] is False
    assert out["prompt_tokens"] == 4
    assert out["text"] == out["decoded_window"]
    display, prepended = display_continuation(text, out["decoded_window"])
    assert out["display_text"] == display
    assert out["prompt_prepended"] is prepended
    message = startup_message(session)
    assert "prefix_source=window" in message
    assert "The generator prefix is end-of-text." not in message
    assert "not a chatbot" in message

    longer = "The cat sat on the mat and then ran all the way home"
    longer_ids = session.pretok.encode(longer)
    assert 8 < len(longer_ids) <= session.context_length
    long_out = session.reply(longer)
    assert long_out["prefix_ids"] == longer_ids[:8]
    assert long_out["prompt_tokens"] == len(longer_ids)
    assert long_out["truncated"] is False
    assert long_out["repeated"] is False

    huge = "hello " * 80
    huge_ids = session.pretok.encode(huge)
    assert len(huge_ids) > session.context_length
    huge_out = session.reply(huge)
    assert huge_out["truncated"] is True
    assert huge_out["prompt_tokens"] == session.context_length
    assert huge_out["prefix_ids"] == huge_ids[:8]
    assert huge_out["repeated"] is False

    packed = open_session(cfg, device="cpu", temperature=0.0, level0_source="packed")
    packed_out = packed.reply(text)
    assert packed_out["repeated"] is True
    assert packed_out["prefix_ids"] == ids + [session.pretok.pad_token_id] * 4
    assert packed.pretok.eos_token_id not in packed_out["prefix_ids"]
    windows = torch.randint(0, 200, (4, session.context_length), dtype=torch.int32)
    packed_file = tmp_path / "windows.pt"
    torch.save({"windows": windows, "meta": {"L": session.context_length}}, packed_file)
    unigram = open_session(
        cfg,
        device="cpu",
        temperature=0.0,
        level0_source="unigram",
        packed_path=packed_file,
        unigram_max_windows=4,
    )
    uni = unigram.reply(text)
    assert uni["level0_source"] == "unigram"
    assert uni["repeated"] is False
    assert uni["prefix_ids"] == ids + [session.pretok.pad_token_id] * 4
    assert unigram.pretok.eos_token_id not in uni["prefix_ids"]


def test_phase2_chat_refuses_a_recorded_phase1_prefix_and_warns_on_legacy(tmp_path):
    cfg = _phase1()
    _, vqvae, generator = build_models(cfg)
    tok_path = tmp_path / "tokenizer.pt"
    recorded = tmp_path / "phase1-finetune.pt"
    legacy = tmp_path / "legacy-v3.pt"
    torch.save({"model": vqvae.state_dict(), "step": 100000}, tok_path)
    state = generator.state_dict()
    torch.save(
        {
            "format": "ncm-iwsp-gpt1-finetune",
            "model": state,
            "step": 3,
            "architecture": {"prefix_len": 1, "prefix_source": "eos", "predict_level0": False},
            "prefix_len": 1,
            "prefix_source": "eos",
        },
        recorded,
    )
    head = _head(tmp_path / "prefix_level0.pt")
    with pytest.raises(ChatError, match="prefix_len"):
        open_session(
            _phase2(),
            device="cpu",
            tokenizer_path=tok_path,
            finetune_path=recorded,
            prefix_head_path=head,
        )
    dirty = dict(state)
    dirty["attn_mask"] = torch.zeros(1, 1, 2, 2)
    dirty["rope_cos"] = torch.zeros(4, 4)
    dirty["module.rope_sin_gen_0"] = torch.zeros(2)
    torch.save({"format": "ncm-iwsp-gpt1-finetune", "model": dirty, "step": 53232}, legacy)
    session = open_session(
        _phase2(),
        device="cpu",
        temperature=0.0,
        tokenizer_path=tok_path,
        finetune_path=legacy,
        prefix_head_path=head,
    )
    assert "does not record prefix_len" in startup_message(session)
    assert torch.equal(session.generator.transformer.wte.weight.cpu(), state["transformer.wte.weight"])
    out = session.reply("How are you?")
    assert out["prefix_source"] == "window"
    assert out["prefix_ids"][0] == 2437
    assert session.pretok.eos_token_id not in out["prefix_ids"]


def test_phase2_cli_prints_the_window_protocol(tmp_path, capsys):
    head = _head(tmp_path / "prefix_level0.pt")
    with pytest.raises(SystemExit, match="No prefix-head"):
        main(["chat", "--config", "configs/gpt1_data_phase2.yaml", "--device", "cpu", "--prompt", "Hi"])
    session = open_session(_phase2(), device="cpu", temperature=0.0, seed=0, prefix_head_path=head)
    expected = session.reply("How are you?")
    assert (
        main(
            [
                "chat",
                "--config",
                "configs/gpt1_data_phase2.yaml",
                "--device",
                "cpu",
                "--prefix-head",
                str(head),
                "--prompt",
                "How are you?",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == expected["display_text"] + "\n"
    assert "prefix_source=window" in captured.err
    assert "repeated=no" in captured.err
    assert "The generator prefix is end-of-text." not in captured.err
    assert "level0_source=packed" not in captured.err
    action = next(
        item
        for item in build_parser()._subparsers._actions
        if item.__class__.__name__ == "_SubParsersAction"
    )
    chat_help = action.choices["chat"].format_help()
    assert "target_phase2.yaml" in chat_help
    assert "not a causal language model" in chat_help.lower()


def test_phase2_dry_plan_does_not_build_or_retrain_the_head(capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("dry-plan must not build the phase-2 target")

    monkeypatch.setattr("ncm_iwsp_gpt1.finetune.build_models", boom)
    assert main(["finetune", "--config", "configs/target_phase2.yaml", "--dry-plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["system"] == "NCM-iWSP-GPT1"
    assert plan["prefix_len"] == 32
    assert plan["prefix_source"] == "window"
    assert plan["prefix_pad"] == "right"
    assert plan["predict_level0"] is False
    assert plan["retrain_prefix_head"] is False
    assert plan["tokenizer_frozen"] is True
    assert plan["n_layer"] == 12
    assert plan["n_embd"] == 768
    assert plan["n_head"] == 12
    assert plan["context_length"] == 512
    assert plan["codebook_size"] == 4096
    assert plan["generator_sequence_length"] == 4338
    assert plan["loss_comparable_to_alpha0"] is True
    assert "does not start training" in plan["hyperparameter_note"]
    assert "not a chosen Phase 2 recipe" in plan["hyperparameter_note"]
    with pytest.raises(SystemExit, match="prefix_len"):
        main(["train-prefix-level0", "--config", "configs/target_phase2.yaml", "--dry-plan"])
    with pytest.raises(PrefixLevel0Error, match="prefix_len"):
        from ncm_iwsp_gpt1.prefix_level0 import describe_prefix_run

        describe_prefix_run(_phase2())


def test_phase2_finetune_warm_starts_and_refuses_a_cross_protocol_resume(tmp_path):
    _cfg, _pretok, tokenizer, generator_path, packed, _state = _pair(tmp_path)
    out = tmp_path / "working" / "phase2"
    summary = run_finetune(
        _phase2(),
        tokenizer,
        generator_path,
        packed,
        out,
        batch_size=2,
        grad_accum=1,
        peak_lr=0.05,
        floor_lr=0.05,
        warmup=0,
        max_steps=1,
        alpha=0.0,
        mid_aux=0.0,
        corrupt_p_max=0.0,
        corrupt_p_min=0.0,
        val_every=0,
        val_windows=2,
        val_batch_size=2,
        save_every=1,
        log_every=1,
        resume="none",
        device="cpu",
        gradient_checkpointing=False,
        chunked_block_attn=True,
        seed=0,
    )
    assert summary["tokenizer_checksum_start"] == summary["tokenizer_checksum_end"]
    assert summary["step"] == 1
    assert summary["loss_comparable_to_alpha0"] is True
    latest = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["format"] == "ncm-iwsp-gpt1-finetune"
    assert latest["prefix_len"] == 8
    assert latest["prefix_source"] == "window"
    assert latest["architecture"]["predict_level0"] is False
    assert latest["architecture"]["prefix_len"] == 8
    assert "attn_mask" not in latest["model"]
    assert "rope_cos" not in latest["model"]
    stamp = tokenizer.stat().st_mtime_ns
    assert tokenizer.stat().st_mtime_ns == stamp
    with pytest.raises(FinetuneError, match="prefix"):
        run_finetune(
            _phase1(),
            tokenizer,
            generator_path,
            packed,
            tmp_path / "working" / "resume-phase1",
            batch_size=2,
            max_steps=2,
            val_windows=2,
            val_every=0,
            resume=out / "latest.pt",
            device="cpu",
            gradient_checkpointing=False,
            seed=0,
        )
    legacy = tmp_path / "legacy.pt"
    phase1_cfg = _phase1()
    _, _, phase1_generator = build_models(phase1_cfg)
    torch.save(
        {"format": "ncm-iwsp-gpt1-finetune", "model": phase1_generator.state_dict(), "step": 1},
        legacy,
    )
    with pytest.raises(FinetuneError, match="does not record prefix_len"):
        run_finetune(
            _phase2(),
            tokenizer,
            generator_path,
            packed,
            tmp_path / "working" / "resume-legacy",
            batch_size=2,
            max_steps=2,
            val_windows=2,
            val_every=0,
            resume=legacy,
            device="cpu",
            gradient_checkpointing=False,
            seed=0,
        )
    resumed = run_finetune(
        phase1_cfg,
        tokenizer,
        generator_path,
        packed,
        tmp_path / "working" / "resume-ok",
        batch_size=2,
        max_steps=1,
        val_windows=2,
        val_every=0,
        resume=legacy,
        device="cpu",
        gradient_checkpointing=False,
        seed=0,
    )
    assert resumed["start_step"] == 1
    assert resumed["finished"] is True


def test_load_module_init_ignores_stale_buffers(tmp_path):
    cfg = _phase1()
    _, _vqvae, generator = build_models(cfg)
    state = dict(generator.state_dict())
    state["attn_mask"] = torch.zeros(1, 1, 5, 5)
    state["rope_sin"] = torch.ones(3, 3)
    state["module.rope_cos"] = torch.ones(2, 2)
    path = tmp_path / "generator.pt"
    torch.save({"model": state, "step": 53232}, path)
    _, _, phase2 = build_models(_phase2())
    meta = load_module_init(path, phase2, "generator")
    assert meta["step"] == 53232
    assert torch.equal(phase2.transformer.wte.weight, generator.transformer.wte.weight)
    assert phase2.attn_mask.shape[-1] == 8 + sum(phase2.levels[1:])
    assert phase2.prefix_source == "window"


def test_phase2_does_not_open_a_dialogue_pack_path():
    with pytest.raises(ValueError, match="SFT"):
        _refuse_corpus_name(Path("minimind_sft.jsonl"))
    with pytest.raises(ValueError, match="DPO"):
        _refuse_corpus_name(Path("train_dpo.jsonl"))
