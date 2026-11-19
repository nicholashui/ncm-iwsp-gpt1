"""Parameter counting, matching HazyResearch/ncm ``utils/misc.py:count_params``.

``estimate_report`` sums the same tensors the modules construct, without
allocating the full NCM-iWSP-GPT1 target. It is checked against a real
instantiation on the runnable config.
"""

from __future__ import annotations

from ncm_iwsp_gpt1.models.quant import _plan_cascade_strides


def count_params(model, exclude_params=()):
    """Return ``(total, non_embedding)`` parameter counts.

    A parameter is excluded from the second number when its name contains any
    of ``exclude_params`` as a substring. Buffers are not counted. Parameters
    with ``requires_grad=False`` (the residual codebooks) are counted.
    """
    total = sum(p.numel() for p in model.parameters())
    non_emb = total
    if exclude_params:
        for name, param in model.named_parameters():
            if any(piece in name for piece in exclude_params):
                non_emb -= param.numel()
    return int(total), int(non_emb)


def _ln(width: int, bias: bool) -> int:
    return int(width) + (int(width) if bias else 0)


def _linear(in_features: int, out_features: int, bias: bool) -> int:
    total = int(in_features) * int(out_features)
    if bias:
        total += int(out_features)
    return total


def _conv(embed: int, kernel: int) -> int:
    return int(embed) * int(embed) * int(kernel) + int(embed)


def _block(n_embd: int, n_head: int, bias: bool, use_qk_norm: bool) -> int:
    total = _ln(n_embd, bias) + _ln(n_embd, bias)
    total += _linear(n_embd, 3 * n_embd, bias)
    total += _linear(n_embd, n_embd, bias)
    if use_qk_norm:
        total += 2 * (int(n_embd) // int(n_head))
    total += _linear(n_embd, 4 * n_embd, bias)
    total += _linear(4 * n_embd, n_embd, bias)
    return total


def estimate_tokenizer(tokenizer_cfg: dict, vocab_size: int) -> tuple[int, int]:
    embed = int(tokenizer_cfg["embed_dim"])
    context = int(tokenizer_cfg["context_length"])
    heads = int(tokenizer_cfg["n_heads"])
    levels = [int(level) for level in tokenizer_cfg["levels"]]
    codebook = int(tokenizer_cfg["codebook_size"])
    vocab = int(vocab_size)
    use_qk = bool(tokenizer_cfg.get("use_qk_norm", True))
    wte = vocab * embed
    wpe = context * embed
    codes = len(levels) * codebook * embed
    resi = tokenizer_cfg.get("quant_resi") or {}
    phi = 0
    ratio = float(resi.get("ratio", 0.5))
    if bool(resi.get("enabled", True)) and abs(ratio) > 1e-6:
        phi = len(levels) * _conv(embed, 3)
    cascade = 0
    if bool(tokenizer_cfg.get("learned_l1_sampling", True)):
        stride = levels[-1] // levels[0]
        if bool(tokenizer_cfg.get("learned_l1_cascade", True)):
            strides = _plan_cascade_strides(stride, int(tokenizer_cfg.get("learned_l1_cascade_base", 4)))
        else:
            strides = [stride]
        cascade = 2 * sum(_conv(embed, step) for step in strides)
    # Decoder.__init__ always builds one block, then extra blocks for dec_layers-1.
    # This repo pins dec_layers=1, and that block hardcodes bias=True.
    blocks = _block(embed, heads, bias=True, use_qk_norm=use_qk)
    ln_f = _ln(embed, bias=False)
    output = _linear(embed, vocab, bias=False)
    total = wte + wpe + codes + phi + cascade + blocks + ln_f + output
    return int(total), int(total - wte - wpe)


def estimate_generator(tokenizer_cfg: dict, generator_cfg: dict) -> tuple[int, int]:
    levels = [int(level) for level in tokenizer_cfg["levels"]]
    codebook = int(tokenizer_cfg["codebook_size"])
    in_dim = int(tokenizer_cfg["embed_dim"])
    n_embd = int(generator_cfg["n_embd"])
    n_head = int(generator_cfg["n_head"])
    n_layer = int(generator_cfg["n_layer"])
    bias = bool(generator_cfg.get("bias", True))
    use_qk = bool(generator_cfg.get("use_qk_norm", True))
    wte = _linear(in_dim, n_embd, bias)
    wle = 0
    if bool(generator_cfg.get("use_level_encoding", True)):
        wle = (len(levels) - 1) * n_embd
    blocks = n_layer * _block(n_embd, n_head, bias=bias, use_qk_norm=use_qk)
    ln_f = _ln(n_embd, bias)
    # lm_heads are constructed with bias=True regardless of the generator bias flag.
    heads = (len(levels) - 1) * _linear(n_embd, codebook, bias=True)
    total = wte + wle + blocks + ln_f + heads
    return int(total), int(total - wte - wle)


def estimate_report(cfg: dict) -> dict:
    vocab = int(cfg["pretok"]["pad_vocab_to"])
    tok_total, tok_non = estimate_tokenizer(cfg["tokenizer"], vocab)
    gen_total, gen_non = estimate_generator(cfg["tokenizer"], cfg["generator"])
    return {
        "tokenizer": tok_total,
        "tokenizer_non_embedding": tok_non,
        "generator": gen_total,
        "generator_non_embedding": gen_non,
        "tokenizer_millions": tok_total / 1e6,
        "generator_millions": gen_total / 1e6,
    }
