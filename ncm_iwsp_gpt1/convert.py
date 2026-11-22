"""Closed-form copy of a finished gradient checkpoint into NCM-iWSP-GPT1.

The source generator uses tied codebook heads. Those heads are rewritten as
the untied ``Linear`` layers this package actually builds. Post-upsample
convolutions are dropped. Nothing here trains, and nothing writes back to
the source files.

The tied-head algebra matches the source module ``_TiedCodebookHead``:

    z = x W_proj^T
    logits = exp(tau) * (2 z E^T - ||E||^2) / s + b
    s = mean_i ||E_i||^2, floored at 1e-12 (the source ``d_scale``)

    alpha = 2 exp(tau) / s
    W = alpha E W_proj
    b_tilde = b - (alpha / 2) ||E||^2

``W_proj`` is ``lm_heads.{k}.proj.weight`` with shape ``(embed, n_embd)``.
``tau`` is the scalar parameter ``lm_heads.{k}.log_inv_T``. ``E`` is not in
the generator checkpoint (``register_buffer(..., persistent=False)``). It is
recomputed from the frozen tokenizer codebook ``vq.embeddings[k+1]``, because
``query_level0`` is false and the heads predict levels ``1 .. K-1``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ncm_iwsp_gpt1.build import build_models, parameter_report
from ncm_iwsp_gpt1.device import resolve_device

FORMAT_NAME = "ncm-iwsp-gpt1-from-gradient-checkpoint"
SCALE_FLOOR = 1e-12

# Real v10 export names. generator_step100000.pt is the last in-loop snapshot
# at step 100000; generator.pt and latest.pt are the end-of-run copies.
GENERATOR_FILENAMES = ("generator_step100000.pt", "generator.pt", "latest.pt")
TOKENIZER_FILENAME = "tokenizer.pt"

_TOKENIZER_BUFFER_PREFIXES = (
    "vq.ema_cluster_size_",
    "vq.ema_w_",
    "vq.codebook_hit_",
)
_GENERATOR_BUFFER_NAMES = frozenset(
    {
        "lvl_1L",
        "attn_mask_for_masking",
        "wp_prefix",
        "wp_1L",
        "flex_lvl_full",
    }
)

_PUC_NOTE = (
    "Post-upsample convolutions are leftover and unused. They add "
    "Conv1d(chunk) onto each level's latent before the input projection, so "
    "they mix neighboring positions and cannot be absorbed into a position-wise "
    "Linear. They were initialized at zero, which is an identity residual; at "
    "step 100000 they may be nonzero, so dropping them makes the full generator "
    "forward only approximate. The copied body and the expanded heads are exact "
    "on the same hidden vector."
)


class ConvertError(ValueError):
    """The checkpoint cannot be mapped onto this package's modules."""


def expand_tied_head(codebook, proj_weight, log_inv_t, bias):
    """Return ``(weight, bias)`` of the matching untied ``Linear``.

    ``codebook`` is ``E`` with shape ``(vocab, embed)``. ``proj_weight`` is
    ``W_proj`` with shape ``(embed, n_embd)``. ``log_inv_t`` is the scalar
    ``tau``. ``bias`` has shape ``(vocab,)``.
    """
    if not torch.is_tensor(codebook) or codebook.ndim != 2:
        raise ConvertError(f"codebook E must be a rank-2 tensor, got {getattr(codebook, 'shape', None)}")
    if not torch.is_tensor(proj_weight) or proj_weight.ndim != 2:
        raise ConvertError(f"W_proj must be a rank-2 tensor, got {getattr(proj_weight, 'shape', None)}")
    if codebook.shape[1] != proj_weight.shape[0]:
        raise ConvertError(
            f"E is {tuple(codebook.shape)} and W_proj is {tuple(proj_weight.shape)}; "
            "the embed dimensions do not match"
        )
    if not torch.is_tensor(bias) or bias.ndim != 1 or bias.shape[0] != codebook.shape[0]:
        raise ConvertError(
            f"tied-head bias must have shape {(codebook.shape[0],)}, got {getattr(bias, 'shape', None)}"
        )
    if not torch.is_tensor(log_inv_t) or log_inv_t.numel() != 1:
        raise ConvertError("log_inv_T must be a single scalar (tau)")

    compute = torch.float64
    if codebook.dtype != torch.float64 and proj_weight.dtype != torch.float64:
        compute = torch.float32
    embedding = codebook.detach().to(dtype=compute)
    projection = proj_weight.detach().to(dtype=compute)
    tau = log_inv_t.detach().to(dtype=compute).reshape(())
    raw_bias = bias.detach().to(dtype=compute)
    row_sq = (embedding * embedding).sum(dim=-1)
    scale = row_sq.mean().clamp(min=SCALE_FLOOR)
    alpha = 2.0 * torch.exp(tau) / scale
    weight = alpha * (embedding @ projection)
    out_bias = raw_bias - (alpha / 2.0) * row_sq
    return weight, out_bias


def prefer_generator_file(directory: str | Path) -> Path:
    """Pick the finished generator file inside an export directory."""
    folder = Path(directory)
    for name in GENERATOR_FILENAMES:
        path = folder / name
        if path.is_file():
            return path
    names = ", ".join(GENERATOR_FILENAMES)
    raise ConvertError(f"no generator checkpoint in {folder}. Expected one of: {names}")


def refuse_output_path(out: str | Path, tokenizer_path: str | Path, generator_path: str | Path) -> None:
    """Refuse to write into a source directory or into a Kaggle input dataset."""
    destination = Path(out).resolve()
    parts = destination.parts
    if "kaggle" in parts and "input" in parts:
        kaggle_at = parts.index("kaggle")
        if kaggle_at + 1 < len(parts) and parts[kaggle_at + 1] == "input":
            raise ConvertError(
                f"refusing to write {destination} under /kaggle/input. "
                "Write to /kaggle/working or pass a different --out outside the input datasets."
            )
    sources = (Path(tokenizer_path), Path(generator_path))
    for source in sources:
        source_file = source.resolve()
        source_dir = source_file.parent
        if destination == source_file or _is_inside(destination, source_dir):
            raise ConvertError(
                f"refusing to write {destination} inside the source directory {source_dir}. "
                "Pass a different --out outside that directory. "
                "Do not write converted weights back into the checkpoint export."
            )


def convert_modules(tokenizer_sd: dict, generator_sd: dict, vqvae, generator) -> dict:
    """Copy tokenizer and body tensors, expand tied heads, and drop post-upsample convs.

    Both modules are updated in place. Source mappings are not modified.
    """
    tok_sd = _require_tensor_map(tokenizer_sd, "tokenizer")
    gen_sd = _require_tensor_map(generator_sd, "generator")
    notes = [
        _PUC_NOTE,
        "Codebook E is not stored in the generator checkpoint. Each head k "
        "recomputes E from tokenizer vq.embeddings[k+1] (levels 1..K-1). "
        "s_k is mean_i ||E_i||^2 floored at 1e-12, matching d_scale.",
        "Tokenizer EMA buffers ema_cluster_size_*, ema_w_*, and codebook_hit_* "
        "are persistent in the source checkpoint and are not parameters of this "
        "model. They are not copied.",
        "Generator buffers lvl_1L and attn_mask_for_masking are persistent in "
        "the source checkpoint. This model recomputes lvl_1L and names the mask "
        "attn_mask, both non-persistent. RoPE buffers are not in either checkpoint. "
        "None of those buffers are copied.",
        "A level-0 unigram saved beside the generator weights is not a parameter. "
        "This evaluator counts its own unigram on the visible split.",
    ]

    missing: list[str] = []
    shape_bad: list[str] = []
    unexpected: list[str] = []

    dest_tok = vqvae.state_dict()
    used_tok: set[str] = set()
    for key, tensor in dest_tok.items():
        if key not in tok_sd:
            missing.append(key)
            continue
        src = tok_sd[key]
        if tuple(src.shape) != tuple(tensor.shape):
            shape_bad.append(f"{key}: source {tuple(src.shape)} destination {tuple(tensor.shape)}")
            continue
        dest_tok[key] = src.detach().to(dtype=tensor.dtype, device=tensor.device).contiguous().clone()
        used_tok.add(key)
    skipped_tok = []
    for key in tok_sd:
        if key in used_tok:
            continue
        if key.startswith(_TOKENIZER_BUFFER_PREFIXES):
            skipped_tok.append(key)
            continue
        unexpected.append(key)

    n_heads = len(generator.lm_heads)
    n_codebooks = len(vqvae.vq.embeddings)
    if n_codebooks != n_heads + 1:
        raise ConvertError(
            f"this generator has {n_heads} heads so the tokenizer must have "
            f"{n_heads + 1} codebooks (level 0 is not a head); found {n_codebooks}"
        )

    dest_gen = generator.state_dict()
    used_gen: set[str] = set()
    for key, tensor in dest_gen.items():
        if key.startswith("lm_heads."):
            continue
        if key not in gen_sd:
            missing.append("generator." + key)
            continue
        src = gen_sd[key]
        if tuple(src.shape) != tuple(tensor.shape):
            shape_bad.append(
                f"generator.{key}: source {tuple(src.shape)} destination {tuple(tensor.shape)}"
            )
            continue
        dest_gen[key] = src.detach().to(dtype=tensor.dtype, device=tensor.device).contiguous().clone()
        used_gen.add(key)

    consumed: set[str] = set()
    codebook_gaps: list[float] = []
    for index, head in enumerate(generator.lm_heads):
        proj_key = f"lm_heads.{index}.proj.weight"
        tau_key = f"lm_heads.{index}.log_inv_T"
        bias_key = f"lm_heads.{index}.bias"
        code_key = f"vq.embeddings.{index + 1}.weight"
        for key in (proj_key, tau_key, bias_key):
            if key not in gen_sd:
                missing.append("generator." + key)
        if code_key not in tok_sd:
            missing.append(code_key)
        if any(
            name in missing
            for name in ("generator." + proj_key, "generator." + tau_key, "generator." + bias_key, code_key)
        ):
            continue
        proj = gen_sd[proj_key]
        tau = gen_sd[tau_key]
        bias = gen_sd[bias_key]
        codebook = tok_sd[code_key]
        expected_proj = (int(vqvae.embed_dim), int(generator.n_embd))
        expected_code = (int(head.out_features), int(vqvae.embed_dim))
        if tuple(proj.shape) != expected_proj:
            shape_bad.append(f"generator.{proj_key}: source {tuple(proj.shape)} expected {expected_proj}")
            continue
        if tuple(codebook.shape) != expected_code:
            shape_bad.append(f"{code_key}: source {tuple(codebook.shape)} expected {expected_code}")
            continue
        if tuple(bias.shape) != (int(head.out_features),):
            shape_bad.append(
                f"generator.{bias_key}: source {tuple(bias.shape)} expected {(int(head.out_features),)}"
            )
            continue
        weight, out_bias = expand_tied_head(codebook, proj, tau, bias)
        dest_gen[f"lm_heads.{index}.weight"] = weight.to(dtype=head.weight.dtype, device=head.weight.device)
        dest_gen[f"lm_heads.{index}.bias"] = out_bias.to(dtype=head.bias.dtype, device=head.bias.device)
        consumed.update((proj_key, tau_key, bias_key))
        for extra in ("E", "E_sq", "d_scale"):
            extra_key = f"lm_heads.{index}.{extra}"
            if extra_key in gen_sd:
                consumed.add(extra_key)
                if extra == "E":
                    gap = (gen_sd[extra_key].detach().float() - codebook.detach().float()).abs().max().item()
                    codebook_gaps.append(float(gap))

    if codebook_gaps and max(codebook_gaps) > 1e-4:
        notes.append(
            "The generator checkpoint contains a tied-head E that differs from the "
            f"tokenizer codebook (max abs {max(codebook_gaps):.3e}). "
            "The expansion uses the tokenizer codebook."
        )

    puc_params = 0
    puc_tensors = 0
    skipped_gen = []
    for key, value in gen_sd.items():
        if key in used_gen or key in consumed:
            continue
        if key.startswith("post_upsample_convs."):
            puc_tensors += 1
            puc_params += int(value.numel())
            continue
        if key in _GENERATOR_BUFFER_NAMES or key.startswith("rope_"):
            skipped_gen.append(key)
            continue
        if key.startswith("lm_heads."):
            unexpected.append("generator." + key)
            continue
        unexpected.append("generator." + key)

    problems = []
    if missing:
        problems.append("missing tensors: " + ", ".join(missing))
    if shape_bad:
        problems.append("shape mismatch: " + "; ".join(shape_bad))
    if unexpected:
        problems.append("unexpected tensors: " + ", ".join(unexpected))
    if problems:
        raise ConvertError("; ".join(problems))

    vqvae.load_state_dict(dest_tok)
    generator.load_state_dict(dest_gen)

    counts = parameter_report(vqvae, generator)
    tied_source = 0
    for index in range(n_heads):
        tied_source += int(gen_sd[f"lm_heads.{index}.proj.weight"].numel())
        tied_source += int(gen_sd[f"lm_heads.{index}.bias"].numel())
        tied_source += int(gen_sd[f"lm_heads.{index}.log_inv_T"].numel())
    untied = 0
    for head in generator.lm_heads:
        untied += int(head.weight.numel())
        if head.bias is not None:
            untied += int(head.bias.numel())
    if puc_params == 0:
        notes.append(
            "The generator checkpoint had no post_upsample_convs tensors. "
            "A finished gpt1.yaml run is expected to have them; dropping zero tensors "
            "does not change the forward."
        )
    return {
        "tokenizer_parameters": counts["tokenizer"],
        "generator_parameters": counts["generator"],
        "total_parameters": counts["tokenizer"] + counts["generator"],
        "heads_expanded": int(n_heads),
        "tied_head_source_parameters": int(tied_source),
        "untied_head_parameters": int(untied),
        "post_upsample_parameters_dropped": int(puc_params),
        "post_upsample_tensors_dropped": int(puc_tensors),
        "tokenizer_buffers_skipped": len(skipped_tok),
        "generator_buffers_skipped": len(skipped_gen),
        "notes": notes,
    }


def convert_files(
    tokenizer_path: str | Path,
    generator_path: str | Path,
    out_path: str | Path,
    cfg: dict,
    smoke_forward: bool = False,
    device: str = "cpu",
) -> dict:
    """Read two checkpoints and write one NCM-iWSP-GPT1 weight file.

    The source paths are opened read-only. ``out_path`` must sit outside
    each source directory and outside ``/kaggle/input``.
    """
    tokenizer_path = Path(tokenizer_path)
    generator_path = Path(generator_path)
    out_path = Path(out_path)
    refuse_output_path(out_path, tokenizer_path, generator_path)
    tok_blob = _read_checkpoint(tokenizer_path)
    gen_blob = _read_checkpoint(generator_path)
    tok_sd, tok_meta = _state_dict_from_blob(tok_blob, "tokenizer")
    gen_sd, gen_meta = _state_dict_from_blob(gen_blob, "generator")
    del tok_blob, gen_blob

    _, vqvae, generator = build_models(cfg)
    report = convert_modules(tok_sd, gen_sd, vqvae, generator)
    if tok_meta.get("step") is not None:
        report["tokenizer_step"] = tok_meta["step"]
    if gen_meta.get("step") is not None:
        report["generator_step"] = gen_meta["step"]
    if smoke_forward:
        report["smoke_forward"] = _smoke_forward(vqvae, generator, device)
        vqvae.to("cpu")
        generator.to("cpu")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Check again after the parent exists so a symlink cannot redirect the write.
    refuse_output_path(out_path, tokenizer_path, generator_path)
    payload = {
        "format": FORMAT_NAME,
        "config_name": str(cfg.get("name", "")),
        "tokenizer": vqvae.state_dict(),
        "generator": generator.state_dict(),
        "report": report,
    }
    torch.save(payload, out_path)
    return report


def load_converted_checkpoint(path: str | Path, vqvae, generator) -> dict:
    """Load a file written by :func:`convert_files` into an already built pair."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or blob.get("format") != FORMAT_NAME:
        raise ConvertError(
            f"{path} is not an {FORMAT_NAME} checkpoint. "
            "Run `python -m ncm_iwsp_gpt1 convert` first; do not point this at a raw generator export."
        )
    try:
        vqvae.load_state_dict(blob["tokenizer"])
        generator.load_state_dict(blob["generator"])
    except (RuntimeError, KeyError) as exc:
        raise ConvertError(f"converted checkpoint does not match these modules: {exc}") from exc
    report = blob.get("report")
    return report if isinstance(report, dict) else {}


def _smoke_forward(vqvae, generator, device: str) -> dict:
    """One random forward. This is a shape check, not a training step and not a corpus eval."""
    resolved = resolve_device(device)
    vqvae.to(resolved).eval()
    generator.to(resolved).eval()
    length = int(generator.prefix_len) + int(sum(generator.levels[1:]))
    hidden = torch.randn(1, length, int(vqvae.embed_dim), device=resolved)
    with torch.no_grad():
        logits = generator(hidden)
    if not torch.isfinite(logits).all():
        raise ConvertError("smoke forward produced non-finite logits")
    return {"logits_shape": [int(size) for size in logits.shape], "device": str(resolved)}


def _read_checkpoint(path: Path):
    if not path.is_file():
        raise ConvertError(f"checkpoint not found: {path}")
    before = (path.stat().st_mtime_ns, path.stat().st_size)
    with path.open("rb") as handle:
        try:
            blob = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception:
            handle.seek(0)
            blob = torch.load(handle, map_location="cpu", weights_only=False)
    after = (path.stat().st_mtime_ns, path.stat().st_size)
    if after != before:
        raise ConvertError(f"source file changed while reading: {path}")
    return blob


def _state_dict_from_blob(blob, label: str) -> tuple[dict, dict]:
    meta: dict = {}
    if isinstance(blob, dict) and isinstance(blob.get("model"), dict):
        step = _as_int(blob.get("step"))
        if step is not None:
            meta["step"] = step
        state = blob["model"]
    elif isinstance(blob, dict) and blob and all(torch.is_tensor(value) for value in blob.values()):
        state = blob
    else:
        raise ConvertError(
            f"{label} is not a tensor state dict and has no 'model' state dict. "
            "Expected a raw state dict or a checkpoint saved with a 'model' entry."
        )
    keys = [str(key) for key in state.keys()]
    if keys and all(key.startswith("module.") for key in keys):
        state = {key[len("module.") :]: value for key, value in state.items()}
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ConvertError(f"{label} state dict contains a non-tensor")
    return state, meta


def _require_tensor_map(mapping, label: str) -> dict:
    if not isinstance(mapping, dict) or not mapping:
        raise ConvertError(f"{label} state dict is empty")
    if not all(torch.is_tensor(value) for value in mapping.values()):
        raise ConvertError(f"{label} state dict contains a non-tensor")
    return mapping


def _as_int(value):
    if value is None or isinstance(value, bool):
        return None
    if torch.is_tensor(value) and value.numel() == 1:
        return int(value.item())
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _is_inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True
