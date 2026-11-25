"""Dialogue prefix-head command. Phase 1 and finetune-dialogue guards stay put.

These tests do not build the 12/768 model and do not start a corpus run.
"""

import json
from pathlib import Path

import pytest
import torch

from ncm_iwsp_gpt1.build import build_models, lock_for_eval
from ncm_iwsp_gpt1.cli import build_parser, main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.dialogue import encode_turn, encode_turns, training_prefix_and_window
from ncm_iwsp_gpt1.finetune import load_training_windows
from ncm_iwsp_gpt1.prefix_level0 import (
    PREFIX_LEVEL0_FORMAT,
    PrefixLevel0Error,
    PrefixLevel0Head,
    run_prefix_level0,
    save_prefix_head,
)
from ncm_iwsp_gpt1.pretok import GPT2BPE


DATASET_ID = "example.local/not-a-chosen-corpus"
LICENSE = "local test authorization, not a selected dataset"
PHASE1_DIR = "ncm-iwsp-gpt1-prefix-level0"
PHASE3_DIR = "ncm-iwsp-gpt1-prefix-level0-phase3"
RECOMMENDED_OUT = r"E:\Project\ncm-iwsp-gpt1-prefix-level0-phase3"


def _phase1():
    return load_config("configs/gpt1_data.yaml")


def _phase3_small():
    return load_config("configs/gpt1_data_phase3.yaml")


def _pretok():
    return GPT2BPE(pad_vocab_to=50304)


def _subparser(name):
    action = next(
        item
        for item in build_parser()._subparsers._actions
        if item.__class__.__name__ == "_SubParsersAction"
    )
    return action.choices[name]


def _dialogue_api():
    from ncm_iwsp_gpt1 import prefix_level0 as module

    missing = [
        name
        for name in (
            "describe_prefix_dialogue_run",
            "dialogue_head_batch",
            "run_prefix_level0_dialogue",
            "train_dialogue_prefix_step",
        )
        if not hasattr(module, name)
    ]
    if missing:
        pytest.fail("missing dialogue prefix API: " + ", ".join(missing))
    return module


def _rows(count: int = 80):
    rows = []
    for index in range(count):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": f"question number {index} about a window"},
                    {"role": "assistant", "content": f"reply number {index} stays short"},
                ]
            }
        )
    return rows


def _pack(tmp_path: Path, placement: str = "both"):
    from ncm_iwsp_gpt1.dialogue import pack_dialogue_records

    cfg = _phase3_small()
    cfg["dialogue"]["history_placement"] = placement
    cache = pack_dialogue_records(
        _rows(),
        _pretok(),
        context_length=32,
        prefix_len=8,
        placement=placement,
        dataset_id=DATASET_ID,
        license_note=LICENSE,
        source="unit-test-rows",
        hidden_pct=10,
        visible_pct=10,
    )
    path = tmp_path / "cache" / "dialogue_windows.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cfg, path, cache


def _save_pair(tmp_path: Path, cfg, prefix_len=8, prefix_source="window"):
    _pretok_model, vqvae, generator = build_models(cfg)
    init = tmp_path / "init"
    init.mkdir(parents=True, exist_ok=True)
    tokenizer = init / "tokenizer.pt"
    generator_path = init / "generator.pt"
    torch.save({"model": vqvae.state_dict(), "step": 3}, tokenizer)
    payload = {
        "format": "ncm-iwsp-gpt1-finetune",
        "model": generator.state_dict(),
        "step": 4,
        "predict_level0": False,
    }
    if prefix_len is not None:
        payload["prefix_len"] = int(prefix_len)
        payload["prefix_source"] = prefix_source
        payload["architecture"] = {
            "prefix_len": int(prefix_len),
            "prefix_source": prefix_source,
            "predict_level0": False,
        }
    torch.save(payload, generator_path)
    return tokenizer, generator_path


def _how_are_you_pair():
    pretok = _pretok()
    cond = encode_turns(
        pretok,
        [{"role": "user", "content": "How are you?"}],
        "User",
        "Assistant",
        "System",
    )
    reply = encode_turn(
        pretok,
        {
            "role": "assistant",
            "content": "The quartz sphinx vowed to juggle twelve black kites.",
        },
        "User",
        "Assistant",
        "System",
    )
    built = training_prefix_and_window(
        cond,
        reply,
        context_length=32,
        prefix_len=8,
        placement="both",
        pad_id=int(pretok.pad_token_id),
    )
    return pretok, cond, reply, built


def test_phase1_command_still_rejects_phase3_and_dialogue_packs(tmp_path):
    with pytest.raises(SystemExit, match="prefix_len"):
        main(["train-prefix-level0", "--config", "configs/target_phase3.yaml", "--dry-plan"])
    with pytest.raises(SystemExit, match="prefix_len"):
        main(["train-prefix-level0", "--config", "configs/gpt1_data_phase3.yaml", "--dry-plan"])
    phase1_help = _subparser("train-prefix-level0").format_help()
    assert "prefix_len stays 1" in phase1_help

    _cfg, packed, _cache = _pack(tmp_path)
    with pytest.raises(Exception, match="dialogue"):
        load_training_windows(packed, 32, 2)
    tokenizer, _generator = _save_pair(tmp_path, _phase1(), prefix_len=1, prefix_source="eos")
    with pytest.raises(PrefixLevel0Error, match="dialogue"):
        run_prefix_level0(
            _phase1(),
            tokenizer,
            packed,
            tmp_path / "phase1-out",
            max_steps=1,
            batch_size=1,
            val_windows=2,
            device="cpu",
        )


def test_finetune_dialogue_retrain_prefix_head_still_errors():
    with pytest.raises(SystemExit, match="not retrained"):
        main(
            [
                "finetune-dialogue",
                "--config",
                "configs/target_phase3.yaml",
                "--dry-plan",
                "--retrain-prefix-head",
            ]
        )


def test_dialogue_command_dry_plan_does_not_train(capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("dry-plan built a model")

    monkeypatch.setattr("ncm_iwsp_gpt1.prefix_level0.build_models", boom)
    monkeypatch.setattr("ncm_iwsp_gpt1.finetune.build_models", boom)
    assert (
        main(
            [
                "train-prefix-level0-dialogue",
                "--config",
                "configs/target_phase3.yaml",
                "--max-steps",
                "20000",
                "--batch-size",
                "8",
                "--lr",
                "1e-3",
                "--dry-plan",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["command"] == "train-prefix-level0-dialogue"
    assert plan["optimized"] == "prefix_level0_head"
    assert plan["tokenizer_frozen"] is True
    assert plan["generator_frozen"] is True
    assert plan["predict_level0"] is False
    assert plan["prefix_len"] == 32
    assert plan["prefix_source"] == "window"
    assert plan["history_placement"] == "both"
    assert plan["dataset"] is None
    assert plan["max_steps"] == 20000
    assert plan["batch_size"] == 8
    assert plan["lr"] == 1e-3
    assert plan["does_not_submit_kaggle"] is True
    assert plan["recommended_out"] == RECOMMENDED_OUT
    assert PHASE1_DIR != Path(plan["recommended_out"]).name
    assert "may still not yield coherent chat" in plan["note"]
    assert "not a chatbot" in plan["note"]
    assert plan["features"].startswith("unpadded")
    help_text = _subparser("train-prefix-level0-dialogue").format_help()
    assert PHASE3_DIR in help_text
    assert "--confirm" in help_text
    assert "--dataset-id" in help_text
    out_flag = next(item for item in _subparser("train-prefix-level0-dialogue")._actions if item.dest == "out")
    assert out_flag.default is None
    described = _dialogue_api().describe_prefix_dialogue_run(_phase3_small())
    assert described["prefix_len"] == 8
    assert described["dataset"] is None
    with pytest.raises(SystemExit, match="dialogue"):
        main(["train-prefix-level0-dialogue", "--config", "configs/target.yaml", "--dry-plan"])
    with pytest.raises(SystemExit, match="prefix_len"):
        main(["train-prefix-level0-dialogue", "--config", "configs/target_phase2.yaml", "--dry-plan"])


def test_real_run_requires_confirm_dataset_id_and_max_steps():
    with pytest.raises(SystemExit, match="confirm"):
        main(
            [
                "train-prefix-level0-dialogue",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--dataset-id",
                DATASET_ID,
                "--max-steps",
                "1",
            ]
        )
    with pytest.raises(SystemExit, match="dataset-id"):
        main(
            [
                "train-prefix-level0-dialogue",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--confirm",
                "--max-steps",
                "1",
                "--tokenizer",
                "tokenizer.pt",
                "--generator",
                "generator.pt",
                "--packed",
                "packed.pt",
                "--out",
                "out",
            ]
        )
    with pytest.raises(SystemExit, match="max-steps"):
        main(
            [
                "train-prefix-level0-dialogue",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--confirm",
                "--dataset-id",
                DATASET_ID,
                "--tokenizer",
                "tokenizer.pt",
                "--generator",
                "generator.pt",
                "--packed",
                "packed.pt",
                "--out",
                "out",
            ]
        )


def test_assistant_tokens_stay_out_of_the_prefix_and_short_t_is_accepted():
    api = _dialogue_api()
    pretok, cond, reply, built = _how_are_you_pair()
    assert cond == [12982, 25, 1374, 389, 345, 30, 50256]
    assert len(cond) == 7
    assistant_only = set(reply) - set(cond)
    assert assistant_only
    assert built["window_ids"][7] in assistant_only
    prefixes = torch.tensor([built["prefix_ids"]], dtype=torch.long)
    pad_id = int(pretok.pad_token_id)
    full_ids, lengths, content_lengths = api.dialogue_head_batch(
        prefixes, pad_id, lengths=torch.tensor([7])
    )
    assert content_lengths.tolist() == [7]
    assert lengths.tolist() == [7]
    assert full_ids[0, :7].tolist() == cond
    assert pad_id not in full_ids[0, :7].tolist()
    assert set(full_ids[0, :7].tolist()).isdisjoint(assistant_only)
    short_ids, short_lengths, _content = api.dialogue_head_batch(
        prefixes, pad_id, lengths=torch.tensor([1])
    )
    assert short_lengths.tolist() == [1]
    assert int(short_ids.shape[1]) == 1
    assert int(short_ids[0, 0]) == cond[0]
    four, four_lengths, _content = api.dialogue_head_batch(
        prefixes, pad_id, lengths=torch.tensor([4])
    )
    assert four_lengths.tolist() == [4]
    assert four[0, :4].tolist() == cond[:4]
    with pytest.raises(PrefixLevel0Error, match="T"):
        api.dialogue_head_batch(prefixes, pad_id, lengths=torch.tensor([8]))
    drawn = api.sample_dialogue_prefix_lengths(torch.tensor([7, 7, 7, 7]), generator=torch.Generator().manual_seed(0))
    assert int(drawn.min()) >= 1
    assert int(drawn.max()) <= 7
    many = api.sample_dialogue_prefix_lengths(torch.full((200,), 7), generator=torch.Generator().manual_seed(1))
    assert int(many.min()) < 7
    assert int(many.max()) == 7

    cfg = _phase3_small()
    _pretok_model, vqvae, generator = build_models(cfg)
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    head = PrefixLevel0Head(int(cfg["tokenizer"]["embed_dim"]), int(cfg["tokenizer"]["codebook_size"]), 32)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2)
    windows = torch.tensor([built["window_ids"], built["window_ids"]], dtype=torch.long)
    batch_prefix = prefixes.repeat(2, 1)
    before_tok = [param.detach().clone() for param in vqvae.parameters()]
    before_gen = [param.detach().clone() for param in generator.parameters()]
    stats = api.train_dialogue_prefix_step(
        head,
        optimizer,
        vqvae,
        windows,
        batch_prefix,
        pad_id,
        lengths=torch.tensor([1, 7]),
    )
    assert stats["lengths"].tolist() == [1, 7]
    assert stats["content_lengths"].tolist() == [7, 7]
    assert int(stats["head_token_ids"].shape[1]) == 7
    assert stats["head_token_ids"][0, :1].tolist() == cond[:1]
    assert stats["head_token_ids"][1, :7].tolist() == cond
    attended = set(stats["head_token_ids"][0, :1].tolist()) | set(stats["head_token_ids"][1, :7].tolist())
    assert attended.isdisjoint(assistant_only)
    assert all(torch.equal(old, param.detach()) for old, param in zip(before_tok, vqvae.parameters()))
    assert all(torch.equal(old, param.detach()) for old, param in zip(before_gen, generator.parameters()))
    assert math_finite(stats["loss"])


def math_finite(value) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def test_window_placement_is_refused_because_the_reply_is_in_the_prefix():
    api = _dialogue_api()
    cfg = _phase3_small()
    cfg["dialogue"]["history_placement"] = "window"
    with pytest.raises(PrefixLevel0Error, match="assistant"):
        api.describe_prefix_dialogue_run(cfg)


def test_one_step_trains_the_head_on_dialogue_prefixes_only(tmp_path, monkeypatch):
    api = _dialogue_api()

    def boom(*args, **kwargs):
        raise AssertionError("dialogue command called load_training_windows")

    monkeypatch.setattr("ncm_iwsp_gpt1.prefix_level0.load_training_windows", boom)
    cfg, packed, _cache = _pack(tmp_path)
    tokenizer, generator_path = _save_pair(tmp_path, cfg)
    tok_stamp = tokenizer.stat().st_mtime_ns
    gen_stamp = generator_path.stat().st_mtime_ns
    out = tmp_path / PHASE3_DIR
    summary = api.run_prefix_level0_dialogue(
        cfg,
        tokenizer,
        generator_path,
        packed,
        out,
        dataset_id=DATASET_ID,
        max_steps=1,
        batch_size=2,
        lr=1e-2,
        val_windows=0,
        unigram_windows=8,
        device="cpu",
        seed=0,
    )
    assert summary["command"] == "train-prefix-level0-dialogue"
    assert summary["tokenizer_unchanged"] is True
    assert summary["generator_unchanged"] is True
    assert summary["generator_loaded"] is True
    assert summary["head_changed"] is True
    assert summary["predict_level0"] is False
    assert summary["prefix_len"] == 8
    assert summary["prefix_source"] == "window"
    assert summary["dataset"] == DATASET_ID
    assert summary["history_placement"] == "both"
    assert summary["recommended_out"] == RECOMMENDED_OUT
    assert Path(summary["out"]).name == PHASE3_DIR
    assert all(1 <= length <= cap <= 8 for length, cap in zip(summary["last_lengths"], summary["last_content_lengths"]))
    assert max(summary["last_lengths"]) <= 8
    assert math_finite(summary["val_ce_T1"])
    assert math_finite(summary["val_ce_Tfull"])
    assert summary["val_ce_T32"] is None
    assert summary["val_unigram_nll"] > 0
    assert "may still not yield coherent chat" in summary["note"]
    assert tokenizer.stat().st_mtime_ns == tok_stamp
    assert generator_path.stat().st_mtime_ns == gen_stamp
    checkpoint = out / "prefix_level0.pt"
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert blob["format"] == PREFIX_LEVEL0_FORMAT
    assert blob["prefix_len"] == 8
    assert blob["prefix_source"] == "window"
    assert blob["predict_level0"] is False
    assert blob["dataset"] == DATASET_ID
    assert blob["tokenizer_updated"] is False
    assert blob["generator_updated"] is False
    assert blob["command"] == "train-prefix-level0-dialogue"
    assert (out / "summary.json").is_file()
    assert (out / "train_log.jsonl").is_file()
    assert not (out / "latest.pt").exists()

    phase1_head = tmp_path / "warm" / "prefix_level0.pt"
    embed = int(cfg["tokenizer"]["embed_dim"])
    codebook = int(cfg["tokenizer"]["codebook_size"])
    context = int(cfg["tokenizer"]["context_length"])
    head = PrefixLevel0Head(embed, codebook, context)
    save_prefix_head(
        phase1_head,
        head,
        0,
        embed_dim=embed,
        codebook_size=codebook,
        context_length=context,
        config_name="phase1",
    )
    warm_stamp = phase1_head.stat().st_mtime_ns
    continued = api.run_prefix_level0_dialogue(
        cfg,
        tokenizer,
        generator_path,
        packed,
        tmp_path / "resumed-out",
        dataset_id=DATASET_ID,
        max_steps=1,
        batch_size=2,
        lr=1e-2,
        unigram_windows=4,
        device="cpu",
        seed=0,
        resume=phase1_head,
    )
    assert continued["resumed_from"] == 0
    assert continued["prefix_len"] == 8
    assert phase1_head.stat().st_mtime_ns == warm_stamp
    assert torch.load(continued["checkpoint"], map_location="cpu", weights_only=False)["prefix_len"] == 8

    with pytest.raises(PrefixLevel0Error, match=PHASE1_DIR):
        api.run_prefix_level0_dialogue(
            cfg,
            tokenizer,
            generator_path,
            packed,
            tmp_path / PHASE1_DIR,
            dataset_id=DATASET_ID,
            max_steps=1,
            batch_size=2,
            device="cpu",
        )
    with pytest.raises(PrefixLevel0Error, match="dataset"):
        api.run_prefix_level0_dialogue(
            cfg,
            tokenizer,
            generator_path,
            packed,
            tmp_path / "bad-id",
            dataset_id="other.local/id",
            max_steps=1,
            batch_size=2,
            device="cpu",
        )
    bare_tok, bare_gen = _save_pair(tmp_path / "bare", cfg, prefix_len=None)
    with pytest.raises(PrefixLevel0Error, match="prefix_len"):
        api.run_prefix_level0_dialogue(
            cfg,
            bare_tok,
            bare_gen,
            packed,
            tmp_path / "no-prefix-meta",
            dataset_id=DATASET_ID,
            max_steps=1,
            batch_size=2,
            device="cpu",
        )
    short_tok, short_gen = _save_pair(tmp_path / "short", cfg, prefix_len=1, prefix_source="eos")
    with pytest.raises(PrefixLevel0Error, match="prefix_len"):
        api.run_prefix_level0_dialogue(
            cfg,
            short_tok,
            short_gen,
            packed,
            tmp_path / "phase1-generator",
            dataset_id=DATASET_ID,
            max_steps=1,
            batch_size=2,
            device="cpu",
        )


def test_docs_name_the_dialogue_command_and_keep_the_phase3_guard():
    readme = Path("README.md").read_text(encoding="utf-8")
    chat = Path("docs/chat.md").read_text(encoding="utf-8")
    assert "train-prefix-level0-dialogue" in readme
    assert "train-prefix-level0-dialogue" in chat
    assert PHASE3_DIR in readme
    assert PHASE3_DIR in chat
    assert "not retrained" in chat
    assert "--retrain-prefix-head" in chat
