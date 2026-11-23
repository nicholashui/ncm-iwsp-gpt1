# Fine-tune NCM-iWSP-GPT1 from the converted v10 init

This run gradient-updates the **NCM-iWSP-GPT1** generator: 12 layers, width 768, 12 heads, context 512, 23 residual VQ levels, untied `Linear(768, 4096)` heads, `predict_level0` false. The tokenizer is loaded and then frozen. There is no post-upsample convolution. The weights it starts from are the files already converted on NicNicNic.

This is a separate command from the writer loop. The writer loop still does not call an optimizer. It is also a different model from NCM-GPT1 in `nicholashui/ncm-gpt1`, which trains tied codebook heads and post-upsample convolutions. Nicholas decided there is no second conversion that keeps those pieces and feeds weights back into the NCM-GPT1 trainer.

No command in this repository submits a Kaggle job. Nicholas confirms the dataset list before he submits the new notebook himself. Do not open or edit the kernel `nicholashuihk/ncm-gpt-1-train`.

Logs and figures from kernel version 3 are in [`artifacts/kaggle-finetune-v3/`](../artifacts/kaggle-finetune-v3/).

## What the loss is

Training minimizes next-code cross-entropy on levels `1 .. K-1`. Level 0 is encoded and used as conditioning. It is not a generator target.

The default scalar is **alpha = 0** and **mid_aux = 0**: every predicted code cell has the same weight. That is the length-weighted mean this repository already reports as `length_weighted_ce`. With those defaults, `loss` matches that mean.

`ce_first`, `ce_mid`, and `ce_last` are always the raw means of three predicted levels: the first, the middle index `len // 2`, and the last. For the 22 target heads, `ce_mid` is index 11. They are logged on the training step and, separately, as `val_ce_first`, `val_ce_mid`, and `val_ce_last` on the held-out tail. Validation does not corrupt codes.

NCM-GPT1 v10 logged a different scalar. Its later kernels used alpha 0.5 and mid_aux 0.5. `mid_aux` adds the mean CE of the central third of levels (indices `[7, 14)` on 22 heads), which is a different slice from `ce_mid`. **Do not compare a loss that includes mid_aux, or alpha other than 0, to an alpha=0 curve.** Compare `ce_first` / `ce_mid` / `ce_last`. Passing `--alpha 0.5 --mid-aux 0.5` turns that warning on in every log row and in `report.json` (`loss_comparable_to_alpha0: false`).

Input codes are corrupted from `p_max=0.8` on the coarsest level down to `p_min=0.2` on the finest, matching the recipe that produced the init. Labels stay clean. `--corrupt-p-max 0 --corrupt-p-min 0` turns that off.

The fine-tune step counter starts at 0. The converted file's `step` (100000) is stored as `init_step`. Step 1000 here is not v10 step 101000.

## Why the forward is an approximation of v10

The converter copied the body and expanded the tied heads. It dropped `post_upsample_convs` because those kernels were not zero at step 100000, and a position-wise linear cannot absorb them. Full generator forward is therefore an approximation of the v10 forward. The fine-tune does not put those convolutions back.

## Files to upload

On NicNicNic the converted init is already in `E:\Project\ncm-iwsp-gpt1-init\`. Upload that directory as a **new** Kaggle dataset. Do not copy it into `E:\Project\ncm-gpt1-kaggle-100k`, and do not write it into the v10 kernel output.

| File | Required | Role |
| --- | --- | --- |
| `tokenizer.pt` | yes | `{model, step, converted_from, ...}`. `model` is the VQ-VAE state dict, 92,276,480 parameters. |
| `generator.pt` | yes | `{model, step, converted_from, ...}`. `model` is the untied generator, 154,666,240 parameters. `lm_heads.{k}.weight` is `(4096, 768)` and `lm_heads.{k}.bias` is `(4096,)`. |
| `convert_report.json` | no | If several attached datasets contain `tokenizer.pt` and `generator.pt`, the directory that also has this file is preferred. |
| `convert_v10.py` | no | Not executed. |

A raw NCM-GPT1 `generator.pt` still has `lm_heads.*.proj.weight` and `post_upsample_convs.*`. The trainer refuses those keys.

The GPT-2 BPE pretok is tiktoken's built-in `gpt2` encoding. Do not rebuild a tokenizer. Pad id is 50257, eos id is 50256, embedding rows 50304.

## Packed windows

Reuse the v10 BookCorpus Open pack. It was written under the artifacts tree of the v10 export, not necessarily beside the raw `tokenizer.pt`.

Expected path inside the attached dataset:

```text
artifacts/packed_windows.pt
```

A dataset-root `packed_windows.pt` is used only when no `artifacts/packed_windows.pt` is present. The search prefers a path whose directories include `artifacts`.

Schema written by the v10 trainer:

```text
windows    int32 tensor [N, 512]
n_docs     int
n_toks     int
meta.L                    512
meta.max_windows          524288
meta.respect_sentences    true
meta.pack_repeat          1
meta.eos                  50256
meta.pad                  50257
meta.hf_dataset           lucadiliello/bookcorpusopen
meta.hf_split             train
meta.hf_text_field        text
meta.data                 ""
meta.max_docs             0
```

`windows.shape[1]` must equal the config context (512 for `configs/target.yaml`). The trainer holds out the last `--val-windows` rows (default 256) as validation and trains on the prefix. It does not rebuild the pack.

This repository's own cache (`visible_train`, `visible`, `hidden`, `meta`) is also accepted. `hidden` is not read.

Attach the v10 dataset read-only. The script never writes next to `packed_windows.pt`.

## Where outputs go

Checkpoints, `train_log.jsonl`, `val_log.jsonl`, `report.json`, and `status.json` go to `--out`. On Kaggle that is `/kaggle/working/finetune`.

The run refuses to create that directory when the path is:

- under `/kaggle/input`
- inside the directory of `--tokenizer`, `--generator`, or `--packed`
- any directory named `ncm-gpt1-kaggle-100k`, `kaggle-v10-weights`, or `ncm-gpt-1-train`

`latest.pt` and `generator_step{N}.pt` use format `ncm-iwsp-gpt1-finetune` with `model`, `optimizer`, `step`, and the loss note. Two step files are kept, plus `latest.pt`. A generator checkpoint is on the order of 2 GB once optimizer state is included.

## Kaggle memory defaults

T4 is 16 GiB, capability 7.5. Defaults:

| Knob | Default | Why |
| --- | --- | --- |
| `--batch-size` | 1 per GPU | v10's tied-head generator (101.69M) used batch 2 and about 3.3 GiB with checkpointing. This untied generator is 154.67M. Batch 1 is the conservative start. |
| gradient checkpointing | on | Recompute each block on the backward pass. `--no-checkpointing` disables it. |
| chunked block attention | on | One SDPA per level instead of a 4307×4307 mask. The widest block is the 512-cell level plus the prefix. |
| AMP | fp16 + GradScaler on T4 | bf16 only when capability is 8 or higher. CPU stays fp32. |
| `--grad-accum` | 1 | Raise this to grow the effective batch without storing a second activation graph. |
| `torch.compile` | off | Left off, as on the T4 v10 path. |
| `--peak-lr` | 3e-5 | The v10 from-scratch peak was 3e-4. This run starts from a finished init. |
| `--floor-lr` | 3e-6 | Cosine floor after `--warmup` 100. |
| `--max-steps` | 100000 | A 12-hour session stops on `--max-hours` first. The counter resumes. |
| `--max-hours` | unset in the CLI, 11.5 in the notebook | `--reserve-minutes` 15 leaves time to save. |
| validation | 256-window tail, batch 1, every 1000 steps | Logs `val_ce_first`, `val_ce_mid`, `val_ce_last` as separate fields. |
| workers | 0 | The window tensor is already in memory. At 524288×512 the int32 tensor is about 1 GiB. |

Two T4s use one process per GPU (`torch.distributed.run`). Global batch is `batch_size * grad_accum * n_gpu`. Watch the first log line's `mem_gb` before trying `--batch-size 2`.

## Commands

Dry plan, no modules allocated, safe on a laptop:

```bash
python -m ncm_iwsp_gpt1 finetune --config configs/target.yaml --dry-plan
```

Kaggle, after the three inputs are attached (this repo, the new init dataset, the v10 dataset that holds `artifacts/packed_windows.pt`):

```bash
python kaggle/finetune_iwsp.py \
  --confirm \
  --config configs/target.yaml \
  --tokenizer /kaggle/input/<init-dataset>/tokenizer.pt \
  --generator /kaggle/input/<init-dataset>/generator.pt \
  --packed /kaggle/input/<v10-dataset>/artifacts/packed_windows.pt \
  --out /kaggle/working/finetune \
  --device cuda \
  --batch-size 1 \
  --max-hours 11.5 \
  --reserve-minutes 15
```

The same flags work as `python -m ncm_iwsp_gpt1 finetune`. `--confirm` is required. Without it the process exits before reading the weights.

When two GPUs are visible, `kaggle/finetune_iwsp.py` launches `torch.distributed.run` itself. Pass `--no-ddp` to stay on one GPU. Resume on a later session by attaching the previous `/kaggle/working` output as a read-only dataset and pointing `--resume` at its `latest.pt`, with a new `--out` under `/kaggle/working`:

```bash
python kaggle/finetune_iwsp.py \
  --confirm \
  --tokenizer /kaggle/input/<init-dataset>/tokenizer.pt \
  --generator /kaggle/input/<init-dataset>/generator.pt \
  --packed /kaggle/input/<v10-dataset>/artifacts/packed_windows.pt \
  --resume /kaggle/input/<previous-working>/latest.pt \
  --out /kaggle/working/finetune \
  --max-hours 11.5
```

`--resume auto` continues from `<out>/latest.pt` when that file is already in the output directory.

`kaggle/finetune_iwsp.ipynb` is that command in notebook cells. `NICHOLAS_CONFIRMED` defaults to `False`, so Save & Run All stops before any step. Set it to `True` only after the printed paths are the converted init and `artifacts/packed_windows.pt`. Do not reinstall PyTorch on Kaggle. Do not submit the notebook from this repository.
