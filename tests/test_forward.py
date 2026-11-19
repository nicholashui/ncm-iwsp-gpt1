import torch

from ncm_iwsp_gpt1.build import build_models, lock_for_eval
from ncm_iwsp_gpt1.config import load_config
from ncm_iwsp_gpt1.packing import pack_documents


def test_smoke_forward_on_synthetic_packed_windows():
    cfg = load_config("configs/gpt1_data.yaml")
    pretok, vqvae, generator = build_models(cfg)
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    windows = pack_documents(
        ["hello world", "pack these words", "a third short line"],
        pretok,
        cfg["tokenizer"]["context_length"],
    )
    with torch.no_grad():
        logits, indices = vqvae.reconstruct_logits(windows)
        ncm = vqvae.vq.idx_to_ncm_input(indices)
        prefix = torch.full((windows.shape[0], 1), pretok.eos_token_id, dtype=torch.long)
        teacher = torch.cat([vqvae.encoder.wte(prefix), ncm], dim=1)
        gen_logits = generator(teacher)
    levels = list(vqvae.vq.levels)
    assert levels == [1, 4, 9, 16, 25, 32]
    assert indices.shape[1] == sum(levels)
    assert logits.shape == (windows.shape[0], 32, pretok.vocab_size)
    assert gen_logits.shape == (windows.shape[0], sum(levels[1:]), 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(gen_logits).all()
    for param in list(vqvae.parameters()) + list(generator.parameters()):
        assert param.requires_grad is False
    level0 = indices[:, :1]
    generated = generator.generate(level0, prefix, temperature=0.0)
    assert generated.shape == (windows.shape[0], sum(levels))
