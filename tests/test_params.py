from ncm_iwsp_gpt1.build import build_models, parameter_report
from ncm_iwsp_gpt1.cli import main
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.levels import grid_cells
from ncm_iwsp_gpt1.params import estimate_report


def test_runnable_estimate_matches_instantiation():
    cfg = load_config("configs/gpt1_data.yaml")
    _, vqvae, generator = build_models(cfg)
    measured = parameter_report(vqvae, generator)
    assert measured == estimate_report(cfg)
    # Instantiated on CPU. GPT-2 table 50304 x 32 dominates the tokenizer.
    assert measured["tokenizer"] == 3_284_816
    assert measured["tokenizer_non_embedding"] == 1_674_064
    assert measured["generator"] == 37_280
    assert measured["generator_non_embedding"] == 36_064
    assert vqvae.vq.levels[-1] == 32
    assert generator.n_layer == 2
    assert generator.n_embd == 32


def test_target_instantiation_matches_estimate(capsys):
    cfg = load_config("configs/target.yaml")
    _, vqvae, generator = build_models(cfg)
    measured = parameter_report(vqvae, generator)
    assert measured == estimate_report(cfg)
    # Instantiated on CPU. Tokenizer total matches NCM-GPT1's measured 92.28M.
    # The generator is larger than NCM-GPT1's 101.69M because these heads are untied.
    assert measured["tokenizer"] == 92_276_480
    assert measured["tokenizer_non_embedding"] == 72_763_136
    assert measured["generator"] == 154_666_240
    assert measured["generator_non_embedding"] == 154_353_664
    assert len(vqvae.vq.levels) == 23
    assert grid_cells(vqvae.vq.levels) == 4307
    assert generator.n_layer == 12
    assert generator.n_embd == 768
    assert generator.n_head == 12
    assert main(["count", "--config", "configs/target.yaml"]) == 0
    captured = capsys.readouterr().out
    assert '"counted_by": "estimate"' in captured
    assert "92276480" in captured
