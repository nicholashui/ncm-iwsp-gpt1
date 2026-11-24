"""Phase 3 dialogue packing, history, and reply slicing on the small config.

These tests do not build the 12/768 model and do not start a corpus run.
"""

import json
from pathlib import Path

import pytest
import torch

from ncm_iwsp_gpt1.build import build_models
from ncm_iwsp_gpt1.chat import ChatError, open_session, startup_message
from ncm_iwsp_gpt1.cli import main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.dialogue import (
    DIALOGUE_FORMAT,
    DIALOGUE_PACK_FORMAT,
    BOOKCORPUS_DATASET,
    fit_token_budget,
    inference_prefix_ids,
    iter_dialogue_jsonl,
    pack_dialogue_records,
    slice_reply,
    training_prefix_and_window,
)
from ncm_iwsp_gpt1.finetune import (
    FinetuneError,
    describe_dialogue_run,
    load_training_windows,
    refuse_finetune_output,
    run_dialogue_finetune,
)
from ncm_iwsp_gpt1.packing import _refuse_corpus_name, make_cache
from ncm_iwsp_gpt1.pretok import GPT2BPE


DATASET_ID = "example.local/not-a-chosen-corpus"
LICENSE = "local test authorization, not a selected dataset"


def _phase1():
    return load_config("configs/gpt1_data.yaml")


def _phase2():
    return load_config("configs/gpt1_data_phase2.yaml")


def _phase3():
    return load_config("configs/gpt1_data_phase3.yaml")


def _pretok():
    return GPT2BPE(pad_vocab_to=50304)


def _rows(count: int = 100):
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


def _allow(tmp_path: Path, dataset_id: str = DATASET_ID) -> Path:
    path = tmp_path / "allow.txt"
    path.write_text(f"# comment\n{dataset_id}\n", encoding="utf-8")
    return path


def _pack(tmp_path: Path, placement: str | None = None):
    cfg = _phase3()
    if placement is not None:
        cfg["dialogue"]["history_placement"] = placement
    cache = pack_dialogue_records(
        _rows(),
        _pretok(),
        context_length=32,
        prefix_len=8,
        placement=cfg["dialogue"]["history_placement"],
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


def test_phase_configs_keep_their_boundaries():
    phase1 = load_config("configs/target.yaml")
    small1 = _phase1()
    phase2 = load_config("configs/target_phase2.yaml")
    small2 = _phase2()
    phase3 = load_config("configs/target_phase3.yaml")
    small3 = _phase3()
    for cfg in (phase1, small1, phase2, small2):
        assert "dialogue" not in cfg
        assert cfg["generator"]["predict_level0"] is False
    assert phase1["generator"]["prefix_len"] == 1
    assert small1["generator"]["prefix_len"] == 1
    assert phase1["generator"].get("prefix_source", "eos") == "eos"
    assert phase2["generator"]["prefix_len"] == 32
    assert phase2["generator"]["prefix_source"] == "window"
    assert small2["generator"]["prefix_len"] == 8
    assert phase3["generator"]["prefix_len"] == 32
    assert phase3["generator"]["prefix_source"] == "window"
    assert phase3["generator"]["predict_level0"] is False
    assert small3["generator"]["prefix_len"] == 8
    assert small3["dialogue"]["enabled"] is True
    assert small3["dialogue"]["history_placement"] == "both"
    assert small3["dialogue"]["add_special_role_tokens"] is False
    assert small3["dialogue"]["truncation"] == "keep_tail"
    assert phase3["data"]["dataset"] == "unset"
    assert small3["data"]["dataset"] == "unset"
    assert phase3["data"]["dataset"] != BOOKCORPUS_DATASET
    with pytest.raises(ValueError, match="add_special_role_tokens"):
        load_config_text = dict(small3)
        load_config_text["dialogue"] = dict(small3["dialogue"])
        load_config_text["dialogue"]["add_special_role_tokens"] = True
        from ncm_iwsp_gpt1.config import _validate

        _validate(load_config_text, Path("configs/gpt1_data_phase3.yaml"))


def test_shipped_allowlist_names_no_dataset():
    text = Path("configs/dialogue_allowlist.txt").read_text(encoding="utf-8")
    ids = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            ids.append(stripped)
    assert ids == []
    lowered = text.lower()
    assert "bookcorpus" not in lowered
    assert "minimind" not in lowered


def test_bookcorpus_pack_path_is_unchanged():
    pretok = _pretok()
    docs = [f"A short book sentence number {i} about a library." for i in range(40)]
    cache = make_cache(docs, pretok, 32, hidden_pct=10, visible_pct=10, max_windows=4)
    assert cache["meta"]["format"] == "packed_windows"
    assert cache["meta"]["dataset"] == BOOKCORPUS_DATASET
    assert "dialogue" not in cache["meta"]
    assert "prefix_visible_train" not in cache


def test_dialogue_pack_refuses_banned_names_and_requires_authorization(tmp_path, monkeypatch):
    banned = tmp_path / "minimind_sft.jsonl"
    banned.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SFT"):
        list(iter_dialogue_jsonl(banned))
    with pytest.raises(ValueError, match="SFT"):
        _refuse_corpus_name(Path("minimind_sft.jsonl"))

    called = []
    real = _refuse_corpus_name

    def spy(path):
        called.append(Path(path).name)
        return real(path)

    monkeypatch.setattr("ncm_iwsp_gpt1.packing._refuse_corpus_name", spy)
    monkeypatch.setattr("ncm_iwsp_gpt1.dialogue._refuse_corpus_name", spy)
    notes = tmp_path / "notes.jsonl"
    notes.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "short reply"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert list(iter_dialogue_jsonl(notes))
    assert called == ["notes.jsonl"]

    allow = tmp_path / "banned-allow.txt"
    allow.write_text("sft_t2t.jsonl\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="SFT"):
        main(
            [
                "pack-dialogue",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--dataset-id",
                "sft_t2t.jsonl",
                "--allow-list",
                str(allow),
                "--license-note",
                LICENSE,
                "--jsonl",
                str(notes),
                "--out",
                str(tmp_path / "out.pt"),
            ]
        )
    empty = tmp_path / "empty.txt"
    empty.write_text("# no ids\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="allow-list"):
        main(
            [
                "pack-dialogue",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--dataset-id",
                DATASET_ID,
                "--allow-list",
                str(empty),
                "--license-note",
                LICENSE,
                "--jsonl",
                str(notes),
                "--out",
                str(tmp_path / "out.pt"),
            ]
        )
    with pytest.raises(SystemExit, match="bookcorpusopen"):
        main(
            [
                "pack-dialogue",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--dataset-id",
                BOOKCORPUS_DATASET,
                "--allow-list",
                str(_allow(tmp_path, BOOKCORPUS_DATASET)),
                "--license-note",
                LICENSE,
                "--jsonl",
                str(notes),
                "--out",
                str(tmp_path / "out.pt"),
            ]
        )


def test_pack_dialogue_writes_an_authorized_cache_outside_git(tmp_path):
    notes = tmp_path / "notes.jsonl"
    notes.write_text(
        "\n".join(json.dumps(row) for row in _rows()) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "dialogue_windows.pt"
    assert (
        main(
            [
                "pack-dialogue",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--dataset-id",
                DATASET_ID,
                "--allow-list",
                str(_allow(tmp_path)),
                "--license-note",
                LICENSE,
                "--jsonl",
                str(notes),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    cache = torch.load(out, map_location="cpu", weights_only=False)
    meta = cache["meta"]
    assert meta["format"] == DIALOGUE_PACK_FORMAT
    assert meta["dataset"] == DATASET_ID
    assert meta["license_note"] == LICENSE
    assert meta["history_placement"] == "both"
    assert meta["truncation"] == "keep_tail"
    assert meta["special_role_tokens"] is False
    assert meta["predict_level0"] is False
    assert meta["dataset"] != BOOKCORPUS_DATASET
    assert cache["visible_train"].shape[1] == 32
    assert cache["prefix_visible_train"].shape[1] == 8
    assert cache["prefix_visible_train"].shape[0] == cache["visible_train"].shape[0]
    pretok = _pretok()
    reply0 = pretok.encode("Assistant: reply number 0 stays short")
    reply1 = pretok.encode("Assistant: reply number 1 stays short")
    windows = torch.cat([cache["visible_train"], cache["visible"], cache["hidden"]], dim=0)
    hits = []
    for row in windows:
        ids = row.tolist()
        has0 = _contains(ids, reply0)
        has1 = _contains(ids, reply1)
        if has0 or has1:
            hits.append((has0, has1, int((row != pretok.pad_token_id).sum())))
    assert hits
    assert all(has0 != has1 for has0, has1, _count in hits)
    assert all(count < 32 for _a, _b, count in hits)


def _contains(ids, seq) -> bool:
    width = len(seq)
    if width == 0 or len(ids) < width:
        return False
    for start in range(len(ids) - width + 1):
        if ids[start : start + width] == seq:
            return True
    return False


def test_history_placement_and_keep_tail_differ():
    pretok = _pretok()
    pad = pretok.pad_token_id
    user = pretok.encode("User: hello") + [pretok.eos_token_id]
    reply = pretok.encode("Assistant: fine") + [pretok.eos_token_id]
    long_user = pretok.encode("User: " + " ".join(f"word{i}" for i in range(80))) + [pretok.eos_token_id]
    assert len(long_user) > 32
    for placement in ("prefix", "window", "both"):
        built = training_prefix_and_window(
            user, reply, context_length=32, prefix_len=8, placement=placement, pad_id=pad
        )
        assert len(built["window_ids"]) == 32
        assert len(built["prefix_ids"]) == 8
        if placement == "prefix":
            assert built["prefix_ids"] == fit_token_budget(user, 8, pad)["ids"]
            assert built["window_ids"] == fit_token_budget(reply, 32, pad)["ids"]
        elif placement == "window":
            assert built["prefix_ids"] == built["window_ids"][:8]
            assert _contains(built["window_ids"], reply)
        else:
            assert built["prefix_ids"] == fit_token_budget(user, 8, pad)["ids"]
            assert _contains(built["window_ids"], reply)
            assert built["window_ids"] != fit_token_budget(reply, 32, pad)["ids"]
    truncated = training_prefix_and_window(
        long_user, reply, context_length=32, prefix_len=8, placement="both", pad_id=pad
    )
    assert truncated["window_dropped"] > 0
    assert truncated["window_ids"][-len(reply) :] == reply
    assert truncated["window_ids"][:4] != long_user[:4]
    assert truncated["prefix_ids"] == fit_token_budget(long_user, 8, pad)["ids"]
    assert truncated["prefix_ids"] != long_user[:8]
    inferred = inference_prefix_ids(
        long_user, context_length=32, prefix_len=8, placement="both", pad_id=pad
    )
    assert inferred["prefix_ids"] == truncated["prefix_ids"]
    window_infer = inference_prefix_ids(
        long_user, context_length=32, prefix_len=8, placement="window", pad_id=pad
    )
    assert window_infer["prefix_ids"] == fit_token_budget(long_user, 32, pad)["ids"][:8]
    assert window_infer["prefix_ids"] != inferred["prefix_ids"]


def test_reply_slice_rules():
    pretok = _pretok()
    conditioning = "User: hi"
    window = "User: hi\nAssistant: fine thanks\nUser: next question"
    sliced = slice_reply(
        window,
        conditioning,
        current_user_line="User: hi",
        user_marker="User",
        assistant_marker="Assistant",
        pretok=pretok,
        max_reply_tokens=0,
    )
    assert sliced["stripped_overlap"] is True
    assert sliced["cut_at_user_marker"] is True
    assert sliced["reply"] == "fine thanks"
    bare = slice_reply(
        "fine thanks",
        conditioning,
        current_user_line="User: hi",
        user_marker="User",
        assistant_marker="Assistant",
        pretok=pretok,
    )
    assert bare["stripped_overlap"] is False
    assert bare["reply"] == "fine thanks"
    capped = slice_reply(
        "fine thanks friend",
        "",
        current_user_line="",
        user_marker="User",
        assistant_marker="Assistant",
        pretok=pretok,
        max_reply_tokens=1,
    )
    assert capped["truncated_reply_tokens"] is True
    assert capped["reply"] == pretok.decode(pretok.encode("fine thanks friend")[:1])


def test_phase3_config_without_dialogue_flag_is_not_a_chatbot():
    session = open_session(_phase3(), device="cpu", temperature=0.0, level0_source="packed")
    assert session.dialogue is False
    assert session.history is None
    message = startup_message(session)
    assert "not a chatbot" in message
    assert "chatbot path" not in message
    out = session.reply("Hello")
    assert session.history is None
    assert out["text"] == out["decoded_window"]
    assert "dialogue" not in out


def test_dialogue_chat_needs_a_dialogue_checkpoint_and_keeps_history(tmp_path):
    cfg = _phase3()
    _pretok_model, vqvae, generator = build_models(cfg)
    tokenizer = tmp_path / "tokenizer.pt"
    phase2_ckpt = tmp_path / "phase2.pt"
    dialogue_ckpt = tmp_path / "dialogue.pt"
    torch.save({"model": vqvae.state_dict(), "step": 3}, tokenizer)
    torch.save(
        {
            "format": "ncm-iwsp-gpt1-finetune",
            "model": generator.state_dict(),
            "step": 4,
            "prefix_len": 8,
            "prefix_source": "window",
            "architecture": {"prefix_len": 8, "prefix_source": "window", "predict_level0": False},
        },
        phase2_ckpt,
    )
    torch.save(
        {
            "format": DIALOGUE_FORMAT,
            "dialogue": True,
            "model": generator.state_dict(),
            "step": 5,
            "prefix_len": 8,
            "prefix_source": "window",
            "retrain_prefix_head": False,
            "architecture": {"prefix_len": 8, "prefix_source": "window", "predict_level0": False},
        },
        dialogue_ckpt,
    )
    with pytest.raises(SystemExit, match="dialogue"):
        main(
            [
                "chat",
                "--config",
                "configs/gpt1_data.yaml",
                "--dialogue",
                "--level0-source",
                "packed",
                "--prompt",
                "Hi",
            ]
        )
    with pytest.raises(SystemExit, match="chatbot"):
        main(
            [
                "chat",
                "--config",
                "configs/gpt1_data_phase3.yaml",
                "--dialogue",
                "--tokenizer",
                str(tokenizer),
                "--finetune",
                str(phase2_ckpt),
                "--level0-source",
                "packed",
                "--prompt",
                "Hi",
            ]
        )
    with pytest.raises(ChatError, match="No prefix-head"):
        open_session(
            cfg,
            device="cpu",
            tokenizer_path=tokenizer,
            finetune_path=dialogue_ckpt,
            dialogue=True,
        )
    session = open_session(
        cfg,
        device="cpu",
        temperature=0.0,
        tokenizer_path=tokenizer,
        finetune_path=dialogue_ckpt,
        dialogue=True,
        level0_source="packed",
        max_reply_tokens=0,
    )
    assert session.dialogue is True
    message = startup_message(session)
    assert "chatbot path" in message
    assert "not a causal" in message.lower()
    assert "not a chatbot" not in message
    pretok = session.pretok

    def fake_decode(_level0, _prefix):
        return torch.tensor(pretok.encode("User: hi\nAssistant: fine thanks\nUser: later"))

    session._decode_ids = fake_decode
    first = session.reply("hi")
    assert first["reply_text"] == "fine thanks"
    assert first["decoded_window"].startswith("User: hi")
    assert first["text"] == first["reply_text"]
    assert first["text"] != first["decoded_window"]
    assert first["repeated"] is True
    assert session.history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "fine thanks"},
    ]
    second_window = "sure"
    session._decode_ids = lambda _level0, _prefix: torch.tensor(pretok.encode(second_window))
    second = session.reply("you")
    cond = []
    from ncm_iwsp_gpt1.dialogue import encode_turns

    cond = encode_turns(
        pretok,
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "fine thanks"},
            {"role": "user", "content": "you"},
        ],
        "User",
        "Assistant",
        "System",
    )
    assert second["prefix_ids"] == fit_token_budget(cond, 8, pretok.pad_token_id)["ids"]
    assert second["history_turns"] == 4
    session.clear_history()
    assert session.history == []
    assert second["reply_text"] == "sure"


def test_dialogue_dry_plan_does_not_train_or_pick_a_dataset(capsys, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("dry-plan must not build the phase-3 target")

    monkeypatch.setattr("ncm_iwsp_gpt1.finetune.build_models", boom)
    with pytest.raises(SystemExit, match="finetune-dialogue"):
        main(["finetune", "--config", "configs/target_phase3.yaml", "--dry-plan"])
    assert main(["finetune-dialogue", "--config", "configs/target_phase3.yaml", "--dry-plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["prefix_len"] == 32
    assert plan["predict_level0"] is False
    assert plan["retrain_prefix_head"] is False
    assert plan["tokenizer_frozen"] is True
    assert plan["dataset"] is None
    assert plan["history_placement"] == "both"
    assert plan["code_ce_is_chat_quality"] is False
    assert "ce_first" in plan["quality_note"]
    assert "not a chatbot" in plan["quality_note"] or "not a chatbot or Q&A" in plan["quality_note"]
    assert "not an approved Phase 3 recipe" in plan["hyperparameter_note"]
    assert "does not start training" in plan["hyperparameter_note"]
    assert plan["generator_sequence_length"] == 4338
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
    described = describe_dialogue_run(_phase3())
    assert described["dataset"] is None
    assert described["retrain_prefix_head"] is False


def test_dialogue_finetune_warm_starts_and_refuses_the_wrong_cache(tmp_path):
    cfg, packed, cache = _pack(tmp_path)
    assert cache["meta"]["dataset"] == DATASET_ID
    _pretok_model, vqvae, generator = build_models(cfg)
    init = tmp_path / "phase2-init"
    init.mkdir()
    tokenizer = init / "tokenizer.pt"
    phase2 = init / "generator.pt"
    torch.save({"model": vqvae.state_dict(), "step": 100000}, tokenizer)
    torch.save(
        {
            "format": "ncm-iwsp-gpt1-finetune",
            "model": generator.state_dict(),
            "step": 53232,
            "prefix_len": 8,
            "prefix_source": "window",
            "architecture": {"prefix_len": 8, "prefix_source": "window", "predict_level0": False},
        },
        phase2,
    )
    books = tmp_path / "books" / "packed_windows.pt"
    books.parent.mkdir()
    torch.save({"windows": cache["visible_train"], "meta": {"L": 32, "dataset": BOOKCORPUS_DATASET}}, books)
    with pytest.raises(FinetuneError, match="pack-dialogue"):
        run_dialogue_finetune(
            cfg, tokenizer, phase2, books, tmp_path / "nope", dataset_id=DATASET_ID, max_steps=1, device="cpu"
        )
    with pytest.raises(FinetuneError, match="dataset-id"):
        run_dialogue_finetune(
            cfg,
            tokenizer,
            phase2,
            packed,
            tmp_path / "nope",
            dataset_id="other.local/id",
            max_steps=1,
            device="cpu",
        )
    with pytest.raises(FinetuneError, match="not retrained"):
        run_dialogue_finetune(
            cfg,
            tokenizer,
            phase2,
            packed,
            tmp_path / "nope",
            dataset_id=DATASET_ID,
            retrain_prefix_head=True,
            max_steps=1,
            device="cpu",
        )
    for name in (
        "ncm-iwsp-gpt1-init",
        "ncm-iwsp-gpt1-finetune-v3",
        "ncm-iwsp-gpt1-prefix-level0",
        "ncm-iwsp-gpt1-finetune-phase2",
    ):
        with pytest.raises(FinetuneError, match=name):
            refuse_finetune_output(tmp_path / name / "out", [tokenizer, phase2, packed])
    with pytest.raises(FinetuneError, match="dialogue pack"):
        load_training_windows(packed, 32, 2)
    out = tmp_path / "dialogue-out"
    stamp = tokenizer.stat().st_mtime_ns
    summary = run_dialogue_finetune(
        cfg,
        tokenizer,
        phase2,
        packed,
        out,
        dataset_id=DATASET_ID,
        batch_size=1,
        peak_lr=0.05,
        floor_lr=0.05,
        warmup=0,
        max_steps=1,
        alpha=0.0,
        mid_aux=0.0,
        corrupt_p_max=0.0,
        corrupt_p_min=0.0,
        val_every=1,
        val_windows=2,
        save_every=1,
        log_every=1,
        resume="none",
        device="cpu",
        gradient_checkpointing=False,
        seed=0,
    )
    assert summary["format"] == DIALOGUE_FORMAT
    assert summary["retrain_prefix_head"] is False
    assert summary["tokenizer_checksum_start"] == summary["tokenizer_checksum_end"]
    assert summary["generator_checksum_start"] != summary["generator_checksum_end"]
    assert summary["init_step"] == 53232
    assert tokenizer.stat().st_mtime_ns == stamp
    assert phase2.stat().st_mtime_ns
    assert not (out / "prefix_level0.pt").exists()
    latest = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
    assert latest["format"] == DIALOGUE_FORMAT
    assert latest["dialogue"] is True
    assert latest["retrain_prefix_head"] is False
    assert latest["predict_level0"] is False
    assert latest["architecture"]["predict_level0"] is False
    assert latest["dataset"] == DATASET_ID
    assert latest["code_ce_is_chat_quality"] is False
    rows = [json.loads(line) for line in (out / "val_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    for row in rows:
        assert {"val_ce_first", "val_ce_mid", "val_ce_last"} <= set(row)
        assert row["code_ce_is_chat_quality"] is False
        assert "not a chatbot or Q&A score" in row["quality_note"]
    train_rows = [
        json.loads(line) for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert train_rows
    assert {"ce_first", "ce_mid", "ce_last"} <= set(train_rows[-1])
    assert train_rows[-1]["code_ce_is_chat_quality"] is False
