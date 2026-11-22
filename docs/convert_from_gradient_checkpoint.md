# Closed-form conversion into NCM-iWSP-GPT1

This converts a finished gradient checkpoint into the untied weights this repository builds. It does not train, it does not download weights, and it does not modify the source files. The gradient model stays in its own repository. This file is the math and the run command.

The 100k artifacts are not in this git repo. Tests use synthetic tensors with the same key names. Point the CLI at real files only on a machine that already has them.

## Algebra

NCM-GPT1 tied-head logits, for one level, are

```
z = x W_proj^T
logits = exp(tau) * (2 z E^T - ||E||^2) / s + b
s = mean_i ||E_i||^2
```

`E` is not in the generator checkpoint. The source registers it with `persistent=False`. Recompute it from the frozen tokenizer codebook. Heads predict levels `1 .. K-1` (`predict_level0` and `query_level0` are false), so head `k` uses `vq.embeddings[k+1]`, not level 0.

The source floors `s` at `1e-12` (`d_scale = E_sq.mean().clamp(min=1e-12)`). The converter uses that same floor. On a trained codebook the floor is inactive.

The matching untied `Linear(n_embd, vocab, bias=True)` is

```
alpha = 2 exp(tau) / s
W = alpha E W_proj
b_tilde = b - (alpha / 2) ||E||^2
logits = x W^T + b_tilde
```

Checked shapes for the 12-layer target:

| Symbol | Tensor | Shape |
| --- | --- | --- |
| `E` | `vq.embeddings[k+1].weight` | `(4096, 384)` |
| `W_proj` | `lm_heads.{k}.proj.weight` | `(384, 768)` |
| `tau` | `lm_heads.{k}.log_inv_T` | scalar |
| `b` | `lm_heads.{k}.bias` | `(4096,)` |
| `W` | `lm_heads.{k}.weight` | `(4096, 768)` |
| `b_tilde` | `lm_heads.{k}.bias` | `(4096,)` |

`W_proj` is already the `nn.Linear` weight. There is no extra transpose beyond the formula above. `z = x W_proj^T` is what `nn.Linear(768, 384, bias=False)` computes.

Parameter counts after a successful target conversion:

| Piece | Count |
| --- | ---: |
| Tokenizer, copied | 92,276,480 |
| Generator body (`wte`, `wle`, 12 blocks, `ln_f`), copied | 85,370,112 |
| Source tied heads, `22 * (768*384 + 4096 + 1)` | 6,578,198 |
| Post-upsample convs, dropped, `22 * (384*384*3 + 384)` | 9,740,544 |
| Untied heads written here, `22 * (768*4096 + 4096)` | 69,296,128 |
| Generator in this repo | 154,666,240 |
| Tokenizer + generator | 246,942,720 |

## What is copied

Tokenizer parameters are copied when the name and shape already match this repo: `encoder.wte`, `encoder.wpe`, `vq.embeddings[k]`, `vq.quant_resi[k]`, `vq.learned_down_l1`, `vq.learned_up_l1`, `decoder.middle_block`, `decoder.ln_f`, `decoder.output_layer`. Dropout has no weights. `GroupNorm` here is `affine=False`, so it has no weights either.

Generator body parameters are copied the same way: `transformer.wte`, `transformer.wle`, each block (`ln_1`, `attn.c_attn`, `attn.c_proj`, `attn.q_norm`, `attn.k_norm`, `ln_2`, `mlp.c_fc`, `mlp.c_proj`), and `transformer.ln_f`. RoPE buffers are not in the checkpoint and are not copied. This model rebuilds them.

## Post-upsample caveat

`post_upsample_convs` is a residual `Conv1d` applied to each level's latent **before** `transformer.wte`:

```
chunk <- chunk + Conv1d(chunk)
```

It mixes neighboring positions. A position-wise `Linear` cannot represent that mix, and this architecture does not gain a post-upsample layer. The tensors are dropped.

They were initialized at zero, which leaves the residual equal to the identity. At step 100,000 they may be nonzero. The copied body and the expanded heads match the source on the same hidden vector. The full generator forward matches only where those convs are still zero. Treat a nonzero conv as an approximation, not as an exact transplant of the trained forward.

Those weights are leftover and unused. They stay in the source checkpoint. They are not written into the output.

## Naming differences that the converter accounts for

Checked against this repo's modules and against the gradient model's source (read-only). No extra keys are invented. An unrecognized source tensor aborts the convert.

- `tau` is the parameter `lm_heads.{k}.log_inv_T`, not a tensor named `tau`.
- `W_proj` is `lm_heads.{k}.proj.weight` with shape `(embed, n_embd)`. The projection bias is absent (`bias=False`). A `proj.bias` tensor is rejected.
- `E`, `E_sq`, and `d_scale` are non-persistent and are not required in the file. If a checkpoint does contain `E`, the converter still uses the tokenizer codebook and records the max absolute gap.
- The tokenizer checkpoint also stores EMA buffers `vq.ema_cluster_size_*`, `vq.ema_w_*`, and `vq.codebook_hit_*`. This quantizer has no EMA. They are not parameters and are not copied. The 92,276,480 count is parameters only.
- The generator checkpoint persists `lvl_1L` and `attn_mask_for_masking`. This model recomputes `lvl_1L` and stores the mask as non-persistent `attn_mask`. The values are not copied.
- There is no `lm_head_l0` when `predict_level0` is false. A tensor with that name is rejected.
- `finished` is not a field inside `generator.pt`. The checkpoint's `step` field is copied into the conversion report. The training log is what reported `finished=true`.
- `inference.level0_unigram` sits beside `model` and is ignored. This evaluator counts its own unigram.
- Optimizer state is ignored. Only `model` is read. A raw tensor state dict, without the `model` wrapper, is also accepted.

Body, tokenizer, and head **shapes** for `gpt1.yaml` match this repo's `configs/target.yaml`. The cascade down/up modules use the same stride plan (`512` with base `4` → five convs). No shape remap was required.

## Command on NicNicNic

Do not run this in CI. The files are on that PC. Write the output under this repo's `artifacts` directory, never under `E:\Project\ncm-gpt1-kaggle-100k`.

`generator_step100000.pt` is the step-100000 snapshot. `generator.pt` and `latest.pt` are the end-of-run copies of that same save. Any one of them is a valid `--generator`.

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 convert ^
  --config configs\target.yaml ^
  --tokenizer E:\Project\ncm-gpt1-kaggle-100k\tokenizer.pt ^
  --generator E:\Project\ncm-gpt1-kaggle-100k\generator_step100000.pt ^
  --out E:\Project\ncm-iwsp-gpt1\artifacts\ncm_iwsp_gpt1_from_100k.pt
```

The source paths are opened read-only. If `--out` resolves inside a source directory, or under `/kaggle/input`, the command stops and writes nothing.

Optional one-forward check (still not a training run). On a CPU laptop, skip it; the target sequence is 4307 positions.

```bat
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 convert ^
  --config configs\target.yaml ^
  --tokenizer E:\Project\ncm-gpt1-kaggle-100k\tokenizer.pt ^
  --generator E:\Project\ncm-gpt1-kaggle-100k\generator.pt ^
  --out E:\Project\ncm-iwsp-gpt1\artifacts\ncm_iwsp_gpt1_from_100k.pt ^
  --smoke-forward --device cpu
```

To score those weights with the stub writer (iteration 0 does not overwrite tensors), pack a context-512 cache first and pass the same target config:

```bat
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 eval ^
  --config configs\target.yaml ^
  --packed cache\bookcorpus.pt ^
  --init-checkpoint E:\Project\ncm-iwsp-gpt1\artifacts\ncm_iwsp_gpt1_from_100k.pt ^
  --iteration 0 --device cuda
```

`--init-checkpoint` on `loop` loads the same file before each `write_weights`. The random-init baseline in `baselines.json` stays the seeded initialization. The config must be the one used at conversion time. A target-config loop is a long forward, not a conversion, and it is not run in CI.

## Kaggle

Use a **new** notebook. Do not open or overwrite the v10 training kernel `nicholashuihk/ncm-gpt-1-train`, and do not write into its output.

Add two inputs: this repository, and a dataset whose files are the v10 export (`tokenizer.pt`, and one of `generator_step100000.pt`, `generator.pt`, `latest.pt`). Conversion is a closed-form copy plus the head expansion. CPU is enough. GPU is only for the optional smoke forward.

```bash
python /kaggle/input/<this-repo>/kaggle/convert_iwsp.py \
  --input-dir /kaggle/input/<v10-dataset> \
  --out /kaggle/working/ncm_iwsp_gpt1_from_100k.pt
```

Or call the package entry point with explicit files. `--out` must be `/kaggle/working/...` (or another path outside the input dataset). The script refuses to write under `/kaggle/input`.

`kaggle/convert_iwsp.ipynb` is that same command in notebook cells. It does not submit a job.

`--smoke-forward --device cuda` runs one random forward after the write. It is off by default. It is not a 100k training loop.
