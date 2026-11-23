"""Chat command: parser, checkpoint layouts, and one decode. No target weights."""

import argparse

import pytest
import torch

from ncm_iwsp_gpt1.chat import ChatError, is_quit, open_session
from ncm_iwsp_gpt1.cli import build_parser, main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.convert import FORMAT_NAME


def _cfg():
    return load_config("configs/gpt1_data.yaml")


def _built():
    from ncm_iwsp_gpt1.build import build_models

    cfg = _cfg()
    _, vqvae, generator = build_models(cfg)
    tok = {key: value.detach().cpu().clone() for key, value in vqvae.state_dict().items()}
    gen = {key: value.detach().cpu().clone() for key, value in generator.state_dict().items()}
    tok["encoder.wte.weight"].add_(0.25)
    gen["transformer.wte.weight"].add_(0.25)
    return tok, gen


def _forbid_build(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("chat built a model")

    monkeypatch.setattr("ncm_iwsp_gpt1.chat.build_models", boom)


def test_parser_defaults_and_help_state_the_limit():
    parser = build_parser()
    args = parser.parse_args(["chat", "--prompt", "Hello"])
    assert args.cmd == "chat"
    assert args.config == "configs/target.yaml"
    assert args.device == "auto"
    assert args.temperature == 0.0
    assert args.top_k == 0
    assert args.top_p == 1.0
    assert args.seed == 0
    assert args.prompt == "Hello"
    assert args.checkpoint is None
    assert args.tokenizer is None
    assert args.generator is None
    assert args.finetune is None
    action = next(
        item
        for item in parser._subparsers._actions
        if isinstance(item, argparse._SubParsersAction)
    )
    help_text = action.choices["chat"].format_help().lower()
    assert "not a causal language model" in help_text
    assert "prefix_len" in help_text
    assert "level 0" in help_text
    assert "predict_level0" in help_text
    assert "chatgpt" in help_text
    assert FORMAT_NAME in help_text
    assert "ncm-iwsp-gpt1-finetune" in help_text


@pytest.mark.parametrize("word", ["quit", "QUIT", "exit", "/quit", "/exit", "  exit  "])
def test_quit_commands(word):
    assert is_quit(word)


@pytest.mark.parametrize("word", ["quit now", "hello", ""])
def test_lines_that_are_not_quit(word):
    assert not is_quit(word)


def test_flag_errors_do_not_build_the_target(monkeypatch):
    _forbid_build(monkeypatch)
    with pytest.raises(SystemExit, match="tokenizer"):
        main(["chat", "--finetune", "latest.pt", "--prompt", "hi"])
    with pytest.raises(SystemExit, match="not both"):
        main(["chat", "--checkpoint", "a.pt", "--tokenizer", "b.pt", "--prompt", "hi"])


def test_load_converted_checkpoint_layout(tmp_path):
    tok, gen = _built()
    path = tmp_path / "converted.pt"
    torch.save(
        {
            "format": FORMAT_NAME,
            "config_name": "NCM-iWSP-GPT1-runnable",
            "tokenizer": tok,
            "generator": gen,
            "report": {},
        },
        path,
    )
    stamp = path.stat().st_mtime_ns
    session = open_session(_cfg(), device="cpu", checkpoint=path)
    assert session.weights_label == "converted"
    assert torch.equal(session.vqvae.encoder.wte.weight.cpu(), tok["encoder.wte.weight"])
    assert torch.equal(session.generator.transformer.wte.weight.cpu(), gen["transformer.wte.weight"])
    assert path.stat().st_mtime_ns == stamp
    assert all(not param.requires_grad for param in session.vqvae.parameters())
    assert all(not param.requires_grad for param in session.generator.parameters())


def test_load_split_init_blobs(tmp_path):
    tok, gen = _built()
    tokenizer = tmp_path / "tokenizer.pt"
    generator = tmp_path / "generator.pt"
    prefixed = {f"module.{key}": value for key, value in gen.items()}
    torch.save({"model": tok, "step": 100000, "converted_from": {"kind": "unit"}}, tokenizer)
    torch.save({"model": prefixed, "step": 7, "converted_from": {"kind": "unit-generator"}}, generator)
    session = open_session(_cfg(), device="cpu", tokenizer_path=tokenizer, generator_path=generator)
    assert session.weights_label == "split"
    assert "tokenizer_step=100000" in session.weights_detail
    assert "generator_step=7" in session.weights_detail
    assert torch.equal(session.vqvae.encoder.wte.weight.cpu(), tok["encoder.wte.weight"])
    assert torch.equal(session.generator.transformer.wte.weight.cpu(), gen["transformer.wte.weight"])


def test_load_finetune_ignores_optimizer(tmp_path):
    tok, gen = _built()
    tokenizer = tmp_path / "tokenizer.pt"
    latest = tmp_path / "latest.pt"
    torch.save({"model": tok, "step": 100000, "converted_from": {"kind": "unit"}}, tokenizer)
    torch.save(
        {
            "format": "ncm-iwsp-gpt1-finetune",
            "model": gen,
            "step": 4,
            "optimizer": {"state": {"0": {"exp_avg": torch.zeros(3, 3)}}},
            "scaler": None,
        },
        latest,
    )
    session = open_session(_cfg(), device="cpu", tokenizer_path=tokenizer, finetune_path=latest)
    assert session.weights_label == "finetune"
    assert "generator_step=4" in session.weights_detail
    assert torch.equal(session.generator.transformer.wte.weight.cpu(), gen["transformer.wte.weight"])
    assert torch.equal(session.vqvae.encoder.wte.weight.cpu(), tok["encoder.wte.weight"])
    # The same latest.pt is a generator blob; --generator must not require a rewrap.
    via_generator = open_session(_cfg(), device="cpu", tokenizer_path=tokenizer, generator_path=latest)
    assert via_generator.weights_label == "finetune"
    assert torch.equal(via_generator.generator.transformer.wte.weight.cpu(), gen["transformer.wte.weight"])


def test_tied_head_and_shape_mismatch_are_refused(tmp_path):
    tok, gen = _built()
    tokenizer = tmp_path / "tokenizer.pt"
    torch.save({"model": tok, "step": 1, "converted_from": {}}, tokenizer)
    tied = dict(gen)
    tied["lm_heads.0.proj.weight"] = torch.zeros(1, 1)
    bad = tmp_path / "tied.pt"
    torch.save({"model": tied, "step": 1}, bad)
    with pytest.raises(ChatError, match="tied-head"):
        open_session(_cfg(), device="cpu", tokenizer_path=tokenizer, generator_path=bad)
    mismatched = dict(gen)
    mismatched["lm_heads.0.weight"] = torch.zeros(2, 2)
    shape = tmp_path / "shape.pt"
    torch.save({"model": mismatched, "step": 1}, shape)
    with pytest.raises(ChatError, match="does not match"):
        open_session(_cfg(), device="cpu", tokenizer_path=tokenizer, generator_path=shape)


def test_finetune_file_is_not_a_converted_checkpoint(tmp_path):
    _tok, gen = _built()
    path = tmp_path / "latest.pt"
    torch.save({"format": "ncm-iwsp-gpt1-finetune", "model": gen, "optimizer": {}, "step": 1}, path)
    with pytest.raises(ChatError, match=FORMAT_NAME):
        open_session(_cfg(), device="cpu", checkpoint=path)


def test_reply_encodes_level0_and_decodes():
    session = open_session(_cfg(), device="cpu", temperature=0.0, top_k=0, top_p=1.0, seed=0)
    text = "The cat sat"
    out = session.reply(text)
    ids = session.pretok.encode(text)
    assert out["prefix_len"] == 1
    assert out["prefix_ids"] == [ids[0]]
    assert out["truncated"] is False
    assert out["prompt_tokens"] == len(ids)
    assert len(ids) < session.context_length
    window = ids + [session.pretok.pad_token_id] * (session.context_length - len(ids))
    tokens = torch.tensor([window], dtype=torch.long)
    with torch.no_grad():
        indices = session.vqvae.encode(tokens)
        prefix = torch.tensor([out["prefix_ids"]], dtype=torch.long)
        codes = session.generator.generate(indices[:, :1], prefix, temperature=0.0, top_k=0, top_p=1.0)
        decoded = session.pretok.decode(
            session.vqvae.decode_logits(codes).argmax(dim=-1)[0],
            skip_special_tokens=True,
        )
    assert out["level0"] == indices[0, :1].tolist()
    assert codes.shape[1] == sum(session.vqvae.vq.levels)
    assert out["text"] == decoded
    assert isinstance(out["text"], str)
    empty = session.reply("")
    assert empty["prefix_ids"] == [session.pretok.eos_token_id]
    assert empty["prompt_tokens"] == 0


def test_long_prompt_is_truncated_to_context():
    session = open_session(_cfg(), device="cpu", temperature=0.0)
    text = "hello " * 80
    ids = session.pretok.encode(text)
    assert len(ids) > session.context_length
    out = session.reply(text)
    assert out["truncated"] is True
    assert out["prompt_tokens"] == session.context_length
    assert out["prefix_ids"] == [ids[0]]
    window = ids[: session.context_length]
    with torch.no_grad():
        level0 = session.vqvae.encode(torch.tensor([window], dtype=torch.long))[0, :1].tolist()
    assert out["level0"] == level0


def test_prompt_cli_prints_decoded_text(capsys):
    session = open_session(_cfg(), device="cpu", temperature=0.0, seed=0)
    expected = session.reply("Hello")["text"]
    assert main(["chat", "--config", "configs/gpt1_data.yaml", "--device", "cpu", "--prompt", "Hello"]) == 0
    captured = capsys.readouterr()
    assert captured.out == expected + "\n"
    assert "no checkpoint loaded" in captured.err
    assert "Not a causal LM" in captured.err


def test_checkpoint_prompt_cli(tmp_path, capsys):
    tok, gen = _built()
    path = tmp_path / "converted.pt"
    torch.save(
        {"format": FORMAT_NAME, "tokenizer": tok, "generator": gen, "report": {}},
        path,
    )
    session = open_session(_cfg(), device="cpu", temperature=0.0, checkpoint=path)
    expected = session.reply("Hello")["text"]
    assert (
        main(
            [
                "chat",
                "--config",
                "configs/gpt1_data.yaml",
                "--device",
                "cpu",
                "--checkpoint",
                str(path),
                "--prompt",
                "Hello",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out == expected + "\n"
    assert "no checkpoint loaded" not in captured.err
    assert "loaded converted" in captured.err


def test_interactive_one_turn_then_quit(monkeypatch, capsys):
    import io

    session = open_session(_cfg(), device="cpu", temperature=0.0, seed=0)
    expected = session.reply("Hello")["text"]
    monkeypatch.setattr("sys.stdin", io.StringIO("Hello\nquit\n"))
    assert main(["chat", "--config", "configs/gpt1_data.yaml", "--device", "cpu"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "you> " + expected + "\nyou> "
    assert "no checkpoint loaded" in captured.err
