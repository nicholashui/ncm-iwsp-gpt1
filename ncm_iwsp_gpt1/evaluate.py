"""Fixed evaluator. The writer does not get to edit this module.

Metrics are computed under ``torch.no_grad``. Level-0 unigrams and token
frequencies are counted here from the frozen visible-train windows. They are
not written into the model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ncm_iwsp_gpt1.build import lock_for_eval
from ncm_iwsp_gpt1.device import resolve_device

# Fixed prompts for the qualitative hook. Prefix length in the shipped configs is 1,
# so a prompt contributes at most one token; level 0 still comes from the unigram.
FIXED_PROMPTS = ("Hello world", "下一概念", "The cat sat")


def _batches(windows: torch.Tensor, batch_size: int, max_windows: int):
    data = windows[: int(max_windows)] if max_windows else windows
    size = max(int(batch_size), 1)
    for start in range(0, data.shape[0], size):
        yield data[start : start + size]


def _token_ce_acc(logits, tokens, pad_id):
    mask = tokens != int(pad_id)
    count = int(mask.sum().item())
    if count == 0:
        return {"ce": None, "acc": None, "n": 0}
    ce = float(F.cross_entropy(logits[mask].float(), tokens[mask]).item())
    pred = logits.argmax(dim=-1)
    acc = float((pred[mask] == tokens[mask]).float().mean().item())
    return {"ce": ce, "acc": acc, "n": count}


def _constant_ce(tokens, class_id, vocab, pad_id):
    mask = tokens != int(pad_id)
    if int(mask.sum().item()) == 0:
        return None
    logits = torch.full((*tokens.shape, int(vocab)), -20.0)
    logits[..., int(class_id)] = 20.0
    return float(F.cross_entropy(logits[mask].float(), tokens[mask]).item())


def _most_frequent_id(tokens, pad_id, vocab):
    flat = tokens.reshape(-1)
    flat = flat[flat != int(pad_id)]
    if flat.numel() == 0:
        return 0
    return int(torch.bincount(flat.long(), minlength=int(vocab)).argmax().item())


def _code_slices(indices, levels):
    offset = 0
    chunks = []
    for level in levels:
        chunks.append(indices[:, offset : offset + int(level)])
        offset += int(level)
    return chunks


def _nonpad(tokens, pad_id):
    flat = tokens.reshape(-1)
    return flat[flat != int(pad_id)]


def _smoothed_nll(train_codes, val_codes, vocab):
    """Add-one unigram NLL. Counts stay inside the evaluator."""
    vocab = int(vocab)
    counts = torch.bincount(train_codes.reshape(-1).long(), minlength=vocab).float()
    probs = (counts + 1.0) / (counts.sum() + vocab)
    gathered = probs[val_codes.reshape(-1).long()].clamp(min=1e-12)
    return float((-gathered.log()).mean().item())


def codebook_usage(indices, levels, vocab_sizes) -> dict:
    usage = []
    offset = 0
    for level, vocab in zip(levels, vocab_sizes):
        chunk = indices[:, offset : offset + int(level)].reshape(-1)
        used = int(torch.unique(chunk).numel())
        vocab = int(vocab)
        fraction = used / vocab if vocab else 0.0
        usage.append(
            {
                "level_len": int(level),
                "vocab": vocab,
                "used": used,
                "usage": fraction,
                "dead_fraction": 1.0 - fraction,
            }
        )
        offset += int(level)
    return {
        "per_level": usage,
        "mean_usage": sum(row["usage"] for row in usage) / max(len(usage), 1),
        "mean_dead_fraction": sum(row["dead_fraction"] for row in usage) / max(len(usage), 1),
    }


def _prefix(batch, prefix_len, eos_id, device):
    return torch.full((batch, int(prefix_len)), int(eos_id), dtype=torch.long, device=device)


def _generator_level_stats(logits, targets, levels, vocab_sizes):
    """Per-level mean CE and exact-code accuracy. Scalar is the unweighted mean CE.

    ``length_weighted_ce`` is the alpha=0 ELBO weighting: each code position is equal,
    so longer levels count more. ``predict_level0`` is false, so ``levels`` and
    ``vocab_sizes`` are the predicted levels only.
    """
    per_level = []
    weighted_sum = 0.0
    weight = 0
    offset = 0
    for level, vocab in zip(levels, vocab_sizes):
        level = int(level)
        vocab = int(vocab)
        level_logits = logits[:, offset : offset + level, :vocab]
        level_tgt = targets[:, offset : offset + level]
        ce = float(F.cross_entropy(level_logits.reshape(-1, vocab).float(), level_tgt.reshape(-1)).item())
        acc = float((level_logits.argmax(dim=-1) == level_tgt).float().mean().item())
        per_level.append({"len": level, "vocab": vocab, "ce": ce, "exact_acc": acc})
        weighted_sum += ce * level
        weight += level
        offset += level
    mean_ce = sum(row["ce"] for row in per_level) / max(len(per_level), 1)
    return {
        "per_level": per_level,
        "mean_level_ce": float(mean_ce),
        "length_weighted_ce": float(weighted_sum / weight) if weight else None,
        "mean_exact_acc": float(sum(row["exact_acc"] for row in per_level) / max(len(per_level), 1)),
    }


def evaluate_windows(
    pretok,
    vqvae,
    generator,
    windows: torch.Tensor,
    train_windows: torch.Tensor | None = None,
    batch_size: int = 4,
    max_windows: int = 32,
    n_generate: int = 2,
    temperature: float = 0.0,
    device: str | torch.device | None = "auto",
) -> dict:
    """Score one packed split. ``train_windows`` is the frozen visible train set.

    Pass visible windows here when computing the agent scalar. Pass hidden
    windows only from the operator log, never into ``write_weights``.
    ``device`` is ``auto`` (CUDA when available), ``cuda``, or ``cpu``.
    """
    lock_for_eval(vqvae)
    lock_for_eval(generator)
    device = resolve_device(device)
    vqvae.to(device)
    generator.to(device)
    windows = windows.long().to(device)
    if train_windows is not None:
        train_windows = train_windows.long().to(device)

    pad_id = int(pretok.pad_token_id)
    eos_id = int(pretok.eos_token_id)
    levels = list(vqvae.vq.levels)
    vocab_sizes = list(vqvae.vq.vqvae_vocab_sizes)
    prefix_len = int(generator.prefix_len)

    recon_logits = []
    recon_tokens = []
    code_chunks = []
    gen_logits = []
    gen_targets = []
    replaced_tokens = []
    replaced_gold = []

    with torch.no_grad():
        for batch in _batches(windows, batch_size, max_windows):
            logits, indices = vqvae.reconstruct_logits(batch)
            recon_logits.append(logits.cpu())
            recon_tokens.append(batch.cpu())
            code_chunks.append(indices.cpu())

            ncm_in = vqvae.vq.idx_to_ncm_input(indices)
            prefix = _prefix(batch.shape[0], prefix_len, eos_id, device)
            prefix_latent = vqvae.encoder.wte(prefix)
            teacher = torch.cat([prefix_latent, ncm_in], dim=1)
            pred_logits = generator(teacher)
            targets = indices[:, levels[0] :]
            if pred_logits.shape[1] != targets.shape[1]:
                raise RuntimeError(
                    f"generator length {pred_logits.shape[1]} != target length {targets.shape[1]}"
                )
            gen_logits.append(pred_logits.cpu())
            gen_targets.append(targets.cpu())

            # Gold level-0, argmax of the written levels, then decode.
            offset = 0
            written = [indices[:, : levels[0]]]
            for head_i, level in enumerate(levels[1:]):
                piece = pred_logits[:, offset : offset + level, : vocab_sizes[head_i + 1]].argmax(dim=-1)
                written.append(piece)
                offset += level
            written_idx = torch.cat(written, dim=1)
            decoded = vqvae.decode_logits(written_idx).argmax(dim=-1)
            replaced_tokens.append(decoded.cpu())
            replaced_gold.append(batch.cpu())

    token_logits = torch.cat(recon_logits, dim=0)
    token_gold = torch.cat(recon_tokens, dim=0)
    all_codes = torch.cat(code_chunks, dim=0)
    g_logits = torch.cat(gen_logits, dim=0)
    g_targets = torch.cat(gen_targets, dim=0)
    recon = _token_ce_acc(token_logits, token_gold, pad_id)
    # Pad the head vocab dimension if the levels differ. Shipped configs use one size.
    max_vocab = max(int(v) for v in vocab_sizes[1:])
    if g_logits.shape[-1] != max_vocab:
        # Heads were concatenated without padding because each head emits its own width
        # only when widths match. Mixed widths are not used by the shipped configs.
        raise RuntimeError("mixed codebook sizes need per-level CE only; got unequal head widths")
    level_stats = _generator_level_stats(g_logits, g_targets, levels[1:], vocab_sizes[1:])
    repl_tok = torch.cat(replaced_tokens, dim=0)
    repl_gold = torch.cat(replaced_gold, dim=0)
    repl_mask = repl_gold != pad_id
    repl_acc = (
        float((repl_tok[repl_mask] == repl_gold[repl_mask]).float().mean().item())
        if int(repl_mask.sum())
        else None
    )

    usage = codebook_usage(all_codes, levels, vocab_sizes)
    # Mode and token unigram are counted on the frozen train split when it is
    # provided. The +20/-20 CE is only a finite stand-in for a hard constant
    # predictor; ``most_frequent_token_acc`` and ``token_unigram_nll`` are the
    # numbers to compare.
    freq_source = token_gold
    baselines = {
        "always_pad_ce": _constant_ce(token_gold, pad_id, pretok.vocab_size, pad_id),
        "uniform_finer_ce": float(math.log(vocab_sizes[1])),
        "constant_logit_margin": 20.0,
    }

    if train_windows is not None:
        train_codes = []
        freq_rows = []
        with torch.no_grad():
            for batch in _batches(train_windows, batch_size, max_windows):
                train_codes.append(vqvae.encode(batch).cpu())
                freq_rows.append(batch.cpu())
        train_idx = torch.cat(train_codes, dim=0)
        freq_source = torch.cat(freq_rows, dim=0)
        train_slices = _code_slices(train_idx, levels)
        val_slices = _code_slices(all_codes, levels)
        level_nll = []
        for level_i, (train_chunk, val_chunk, vocab) in enumerate(
            zip(train_slices, val_slices, vocab_sizes)
        ):
            nll = _smoothed_nll(train_chunk, val_chunk, vocab)
            level_nll.append({"level": level_i, "nll": nll, "predicted": level_i > 0})
        predicted = [row["nll"] for row in level_nll if row["predicted"]]
        baselines["per_level_unigram_nll"] = level_nll
        baselines["level0_unigram_nll"] = level_nll[0]["nll"]
        baselines["mean_predicted_level_unigram_nll"] = float(sum(predicted) / max(len(predicted), 1))
        baselines["token_unigram_nll"] = _smoothed_nll(
            _nonpad(freq_source, pad_id), _nonpad(token_gold, pad_id), pretok.vocab_size
        )
        baselines["qualitative"] = _qualitative(
            pretok,
            vqvae,
            generator,
            train_idx,
            windows,
            n_generate,
            temperature,
            eos_id,
            pad_id,
        )
    else:
        baselines["level0_unigram_nll"] = None
        baselines["mean_predicted_level_unigram_nll"] = None
        baselines["token_unigram_nll"] = _smoothed_nll(
            _nonpad(token_gold, pad_id), _nonpad(token_gold, pad_id), pretok.vocab_size
        )
        baselines["qualitative"] = []

    freq_id = _most_frequent_id(freq_source, pad_id, pretok.vocab_size)
    gold_flat = _nonpad(token_gold, pad_id)
    baselines["most_frequent_token_id"] = freq_id
    baselines["most_frequent_token_ce"] = _constant_ce(token_gold, freq_id, pretok.vocab_size, pad_id)
    baselines["most_frequent_token_acc"] = (
        float((gold_flat == freq_id).float().mean().item()) if gold_flat.numel() else None
    )

    return {
        "n_windows": int(token_gold.shape[0]),
        "recon_ce": recon["ce"],
        "recon_acc": recon["acc"],
        "recon_tokens": recon["n"],
        "generator": level_stats,
        "mean_level_ce": level_stats["mean_level_ce"],
        "length_weighted_ce": level_stats["length_weighted_ce"],
        "mean_exact_acc": level_stats["mean_exact_acc"],
        "recon_after_written_levels_acc": repl_acc,
        "codebook_usage": usage,
        "baselines": baselines,
        "scalar": level_stats["mean_level_ce"],
        "device": str(device),
    }


def _qualitative(pretok, vqvae, generator, train_idx, visible, n_generate, temperature, eos_id, pad_id):
    levels = list(vqvae.vq.levels)
    vocab0 = int(vqvae.vq.vqvae_vocab_sizes[0])
    # Sample on CPU so the seed stays stable across devices, then move codes
    # onto the same device as the modules.
    l0_codes = train_idx[:, : levels[0]].reshape(-1).cpu()
    counts = torch.bincount(l0_codes.long(), minlength=vocab0).float()
    probs = (counts + 1.0) / (counts.sum() + vocab0)
    model_device = next(vqvae.parameters()).device
    del visible  # only used by callers to keep the signature stable
    samples = []
    generator_seed = torch.Generator(device="cpu")
    generator_seed.manual_seed(0)
    n = max(int(n_generate), 0)
    prefix_len = generator.prefix_len
    with torch.no_grad():
        if n:
            draw = torch.multinomial(probs, n * levels[0], replacement=True, generator=generator_seed)
            l0 = draw.view(n, levels[0])
            # Random finer levels: a generative baseline, not a parameter update.
            finer = []
            for vocab, level in zip(vqvae.vq.vqvae_vocab_sizes[1:], levels[1:]):
                finer.append(
                    torch.randint(0, int(vocab), (n, int(level)), generator=generator_seed)
                )
            random_codes = torch.cat([l0, *finer], dim=1).to(model_device)
            random_text = vqvae.decode_logits(random_codes).argmax(dim=-1)
            for row in random_text:
                samples.append(
                    {
                        "kind": "level0_unigram_plus_random_finer",
                        "text": pretok.decode(row.tolist(), skip_special_tokens=True)[:180],
                    }
                )
            prefix = torch.full((n, prefix_len), int(eos_id), dtype=torch.long, device=model_device)
            model_codes = generator.generate(l0.to(model_device), prefix, temperature=temperature, top_k=0, top_p=1.0)
            model_text = vqvae.decode_logits(model_codes).argmax(dim=-1)
            for row in model_text:
                samples.append(
                    {
                        "kind": "unigram_l0_then_generator",
                        "text": pretok.decode(row.tolist(), skip_special_tokens=True)[:180],
                    }
                )
            for prompt in FIXED_PROMPTS:
                ids = pretok.encode(prompt) or [int(eos_id)]
                prefix_ids = (ids + [int(pad_id)] * prefix_len)[:prefix_len]
                prefix = torch.tensor([prefix_ids], dtype=torch.long, device=model_device)
                draw = torch.multinomial(probs, levels[0], replacement=True, generator=generator_seed)
                codes = generator.generate(
                    draw.view(1, levels[0]).to(model_device), prefix, temperature=temperature, top_k=0, top_p=1.0
                )
                text = vqvae.decode_logits(codes).argmax(dim=-1)[0]
                samples.append(
                    {
                        "kind": "fixed_prompt",
                        "prompt": prompt,
                        "text": pretok.decode(text.tolist(), skip_special_tokens=True)[:180],
                    }
                )
    return samples


def agent_view(metrics: dict) -> dict:
    """The single scalar a later agent is allowed to see, plus audit bookkeeping.

    Hidden-split numbers must not be added here.
    """
    return {
        "scalar": metrics["scalar"],
        "mean_level_ce": metrics["mean_level_ce"],
    }
