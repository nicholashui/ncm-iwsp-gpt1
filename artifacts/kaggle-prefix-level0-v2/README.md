# Kaggle NCM-iWSP-GPT1 prefix level-0 v2

Kernel [nicholashuihk/ncm-iwsp-gpt1-prefix-level0](https://www.kaggle.com/code/nicholashuihk/ncm-iwsp-gpt1-prefix-level0) version 2, status COMPLETE. This is not `nicholashuihk/ncm-gpt-1-train` and not `nicholashuihk/ncm-iwsp-gpt1-finetune`. Those kernels were attached read-only (packed windows, and the v3 `latest.pt`).

Internet was on so tiktoken could download the GPT-2 vocabulary. Hardware metadata says one T4 (`NvidiaTeslaT4`). The log reports two Tesla T4s; training used `device` `cuda`. The run started about 07:00:47 HKT on 2026-11-24. The log spans 7.11s to 2996.45s (49.82 minutes). Training `elapsed_seconds` is 2915.05 (48.58 minutes). Approximate finish is 07:50:43 HKT. It ran 20,000 / 20,000 steps with `stopped_early` false.

This is Phase 1 continuation of a BookCorpus window from a short prefix. It is not a chatbot, and it is not a token-level language-model curve. `val_ce_vs_unigram.svg` plots level-0 cross-entropy only: validation CE at T=1 against the unigram, plus the final T=4 and T=full markers. It does not overlay token perplexity or a 2018 GPT-1 number.

## Validation level-0 CE

From `summary.json`, after 20,000 steps:

| Metric | Value |
| --- | ---: |
| `val_ce_T1` | 5.789092779159546 |
| `val_ce_T4` | 5.4302146434783936 |
| `val_ce_Tfull` | 5.307143807411194 |
| `val_unigram_nll` | 7.636285883229613 |
| `uniform_nll` | 8.317766166719343 |

At step 50, `val_ce_T1` was 7.606337070465088, near the unigram. After 20,000 steps, T=1 is about 1.85 nats below the unigram, so short-prefix CE is not near the unigram.

`head_parameters` is 1,774,720. `tokenizer_unchanged` and `generator_unchanged` are true. `head_changed` is true. `predict_level0` stays false. `prefix_len` stays 1.

## Notebook and source dataset

The repository notebook is [`kaggle/train_prefix_level0.ipynb`](../../kaggle/train_prefix_level0.ipynb). `NICHOLAS_CONFIRMED` stays `False` in that copy. Only the kernel actually pushed to Kaggle may set it to `True`. Do not commit that pushed notebook.

Version 1 failed at dry-plan: `python -m ncm_iwsp_gpt1` started a new process that did not inherit the notebook `sys.path`. The notebook now prepends the repo to `PYTHONPATH`, runs dry-plan and training with `cwd` set to the repo root, and exits if a fresh process still cannot import the package.

The source dataset was `nicholashuihk/ncm-iwsp-gpt1-src-f35e352`: main `f35e352` (`SOURCE_COMMIT.txt`) plus [`patches/time-budget-max-hours.patch`](../../patches/time-budget-max-hours.patch). That patch adds `--max-hours` and `--reserve-minutes` so a short GPU quota can still write `prefix_level0.pt`, `train_log.jsonl`, and `summary.json`. It is not applied to the training code in this checkout. The older dataset `nicholashuihk/ncm-iwsp-gpt1-src` (commit `75ddf00`) was not used.

Also attached, read-only:

- `nicholashuihk/ncm-iwsp-gpt1-v10-init` (`tokenizer.pt`)
- kernel output `nicholashuihk/ncm-iwsp-gpt1-finetune` (complete v3 `latest.pt`)
- kernel output `nicholashuihk/ncm-gpt-1-train` (`artifacts/packed_windows.pt`)

`kernel-metadata.json` records that private kernel. Its `code_file` is `ncm-iwsp-gpt1-prefix-level0.ipynb`, the name used on Kaggle, not the path of the notebook in this repository.

Defaults for batch size (8), lr (1e-3), and logging were unchanged. The run used `--max-steps 20000`, `--max-hours 2.6`, and `--reserve-minutes 12`.

On NicNicNic the same output files are at `E:\Project\ncm-iwsp-gpt1-prefix-level0\`. Chat was not run there: the GPU was still full and free RAM was about 0.3 GB of 32 GB. Do not start local training from this README.

## Files

| File | Contents |
| --- | --- |
| `prefix_level0.pt` | Head checkpoint only. 21,305,039 bytes, sha256 `0ebb615fd9ea29827dd03e4cce24bf06de770178ea869bb0b469d83e70239698`. The Kaggle CLI listed size 909; that column is wrong. No `prefix_level0_step*.pt` files are stored |
| `summary.json` | Final step, time budget, and validation CE |
| `train_log.jsonl` | Training-step log, including `val_ce_T1` every 50 steps |
| `val_ce_vs_unigram.svg` | T=1 validation CE versus the unigram, plus final T=4 and T=full markers |
| `METRICS.txt` | The same final numbers in one page |
| `SHA256SUMS.txt` | sha256 and byte size of the checkpoint, summary, log, kernel log, and plot |
| `ncm-iwsp-gpt1-prefix-level0.log` | Kernel log |
| `kernel-metadata.json` | Metadata of the private version-2 kernel |
| `SOURCE_COMMIT.txt` | `f35e3528cf8e91d1e51b70b9caac140a7924e95b`, the commit in the source dataset |
