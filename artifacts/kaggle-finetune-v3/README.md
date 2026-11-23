# Kaggle NCM-iWSP-GPT1 fine-tune v3

Kernel [nicholashuihk/ncm-iwsp-gpt1-finetune](https://www.kaggle.com/code/nicholashuihk/ncm-iwsp-gpt1-finetune) version 3, status COMPLETE. Internet was enabled so tiktoken could download the GPT-2 vocabulary. The run started about 09:06 HKT on 2026-11-23 and ended about 20:23 HKT (about 11 h 17 m). It stopped at step 53232 of 100000 with `finished` false, on the 11.5 h budget and the 15 min reserve. `init_step` is 100000. The init is `converted_from` `ncm-gpt1-kaggle-v10`. Training used `alpha` 0, `mid_aux` 0, and `peak_lr` 3e-5. The tokenizer stayed frozen. Hardware was two T4 GPUs, batch size 1 per GPU.

This folder has the logs, `report.json`, `status.json`, the comparison table, and the two figures. It does not contain `.pt` weights.

## Validation code-cell CE (256 windows)

`val_ce_first` / `val_ce_mid` / `val_ce_last`:

| Point | val_ce_first | val_ce_mid | val_ce_last |
| --- | ---: | ---: | ---: |
| Step 0, converted init | 4.59708811622113 | 3.9538024071405626 | 0.6316340167541057 |
| Step 53232, end of this session | 4.680270896293223 | 3.6284476934805423 | 0.5161383835365996 |
| Best mid and last, step 50000 | | 3.6239332630789494 | 0.513729034923017 |

The best `val_ce_first` in this log is already the step 0 value.

## NCM-GPT1 Kaggle v10

NCM-GPT1 Kaggle v10 at step 100000 (2026-11-23 04:18:33 HKT): 4.594952832907438 / 5.375384805724025 / 1.177376363426447.

v10 used `alpha` 0.5 and `mid_aux` 0.5. Compare only `ce_first` / `ce_mid` / `ce_last`, not the optimized total loss. Mid and last are much lower than v10 largely already at conversion (untied heads, no post-upsample conv). First is slightly worse after the fine-tune.

## Official 2018 GPT-1

Official 2018 GPT-1 (Radford et al., *Improving Language Understanding by Generative Pre-Training*, §4.1): token-level perplexity 18.4 on BooksCorpus. The paper does not say whether that number is train or validation, and it publishes no curve. If PPL = exp(CE), ln(18.4) = 2.912 nats. The stated setup is 12 layers, width 768, 12 heads, context 512, and a 40,000-merge BPE (not GPT-2 BPE). The 2018 paper does not state a parameter count.

This perplexity is not comparable in magnitude to NCM code-cell CE. `finetune_v3_vs_ncm_gpt1_and_gpt1.png` keeps a separate axis and does not overlay 18.4 or 2.912 on the NCM panels. `finetune_v3_vs_ncm_gpt1.png` is the NCM-iWSP-GPT1 fine-tune v3 versus NCM-GPT1 v10 validation CE only.

## Packed windows

`ncm-gpt-1-train` was used read-only for `packed_windows.pt` and was not modified.

## Files

| File | Contents |
| --- | --- |
| `finetune_v3_vs_ncm_gpt1.png` | NCM-iWSP-GPT1 fine-tune v3 vs NCM-GPT1 v10 validation CE |
| `finetune_v3_vs_ncm_gpt1_and_gpt1.png` | The same NCM curves, plus a separate official 2018 GPT-1 token-PPL reference |
| `finetune_v3_vs_ncm_gpt1.csv` | The series behind those figures |
| `report.json` | Run configuration written by the trainer |
| `status.json` | Step 53232, `finished` false |
| `val_log.jsonl` | Validation rows, including step 0, 50000, and 53232 |
| `train_log.jsonl` | Training-step log |
| `plot_v3_vs_ncm_and_gpt1.py` | Script that drew the second figure. Its paths are the original `/workspace/ncm-iwsp-gpt1-finetune-v3/` locations |
| `ncm-iwsp-gpt1-finetune.log` | Kernel log |
