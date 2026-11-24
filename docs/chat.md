# Chat with NCM-iWSP-GPT1

`python -m ncm_iwsp_gpt1 chat` continues a short prompt as one decoded window. It is not a causal language model, not NCM-GPT1, and not a chatbot. A prompt such as `How are you?` is not expected to come back as `I'm fine`. Code cross-entropy and 2018 GPT-1 perplexity are not the quality check for this path.

`predict_level0` stays false. Level 0 is not a target of `NextConceptGenerator`. The phase-1 configs `configs/target.yaml` and `configs/gpt1_data.yaml` keep `prefix_len` at 1. That one slot is `<|endoftext|>` (id 50256), the prefix the existing weights were trained with. The words of the prompt do not go there. Phase 2 is a separate config, described below. The default `--config` is still `configs/target.yaml`.

## What one turn does

The default path needs a prefix-head checkpoint. Without `--prefix-head`, chat stops. It does not silently repeat the prompt out to 512 tokens and encode that window.

1. GPT-2 BPE encodes the line. `"How are you?"` is four tokens: `How`, ` are`, ` you`, `?` (ids 2437, 389, 345, 30). A longer line is truncated to the context (512 on `configs/target.yaml`, 32 on `configs/gpt1_data.yaml`). The tokens are not repeated to fill the context, and they are not padded.
2. The frozen tokenizer `encoder.wte` embeds those tokens. No tokenizer position is added. The prefix head adds its own positions `0 .. T-1`, attention-pools them, and produces logits over the level-0 codebook (4096 on the target config, 64 on the small config).
3. Chat samples one level-0 code from that distribution. `temperature` 0 is argmax. `top_k` 0 and `top_p` 1 leave the distribution untruncated. The same knobs apply to the finer codes.
4. On the phase-1 configs, the generator prefix is `<|endoftext|>`, repeated `prefix_len` times. Both of those configs use `prefix_len` 1. `NextConceptGenerator.generate` samples levels 1..K-1 from the sampled level-0 code and that prefix.
5. `decode_logits(...).argmax` and `pretok.decode` turn the code grid into text. Pad and end-of-text ids are removed.

stderr prints `level0_source=prefix_head`. `repeated=no` on a short line. The previous reply is not fed back.

This is a continuation in the coarse-code sense: the head maps the prompt onto the level-0 code of a BookCorpus window whose beginning looks like that prompt, and the frozen generator fills the finer codes. It is not token-by-token continuation and not an answer.

## Train the head

Only the head is trained. The tokenizer and the 12/768 generator stay frozen. `predict_level0` is not turned on. `prefix_len` stays 1.

Data is `packed_windows.pt`: BookCorpus Open, length 512, GPT-2 BPE. That file is not in this git repository, and this repository does not start the long run. Point `--packed` at a local copy (v10 `windows`, or this repo's cache with `visible_train` and `visible`). For each row the target is the level-0 code of the full window, from the frozen tokenizer. The input is the first T tokens. T is drawn uniformly from 1 through the context length so a short prompt is in the training distribution.

On the small config the same command is a one-step shape check. On `configs/target.yaml` it is the real head: embedding width 384, codebook 4096, context 512.

```bash
python -m ncm_iwsp_gpt1 train-prefix-level0 \
  --config configs/target.yaml \
  --dry-plan

python -m ncm_iwsp_gpt1 train-prefix-level0 \
  --config configs/target.yaml \
  --tokenizer /path/to/tokenizer.pt \
  --generator /path/to/latest.pt \
  --packed /path/to/artifacts/packed_windows.pt \
  --out /path/to/prefix-level0 \
  --max-steps 20000 \
  --batch-size 8 \
  --device auto
```

`--generator` is optional. When it is passed, the file is loaded and frozen. The loss does not use it. `--tokenizer` is required, because the head reads `encoder.wte` and the level-0 targets come from that tokenizer. The checkpoint written to `--out/prefix_level0.pt` holds the head only (`format` `ncm-iwsp-gpt1-prefix-level0`), not the generator and not the tokenizer. Resume with `--resume` and a larger `--max-steps` (the step to reach, not an extra count).

The log compares validation cross-entropy at prefix lengths 1, 4, and the full window with an add-one unigram of level 0. If the short-prefix number stays near the unigram, this head is not enough. That is phase 2 (a longer generator prefix and a generator fine-tune). Do not go back to tuning the packed-repeat encode. This command does not submit a Kaggle job.

## Kaggle version 2

Kernel [nicholashuihk/ncm-iwsp-gpt1-prefix-level0](https://www.kaggle.com/code/nicholashuihk/ncm-iwsp-gpt1-prefix-level0) version 2 finished this head: 20,000 / 20,000 steps, `stopped_early` false. It started about 07:00:47 HKT on 2026-11-24 and finished about 07:50:43 HKT. Training `elapsed_seconds` is 2915.05. The checkpoint, the metrics, and the plot are in [`artifacts/kaggle-prefix-level0-v2/`](../artifacts/kaggle-prefix-level0-v2/).

| Metric | Value |
| --- | ---: |
| `val_ce_T1` | 5.789092779159546 |
| `val_ce_T4` | 5.4302146434783936 |
| `val_ce_Tfull` | 5.307143807411194 |
| `val_unigram_nll` | 7.636285883229613 |
| `uniform_nll` | 8.317766166719343 |

At step 50, `val_ce_T1` was 7.606337070465088, near the unigram. After 20,000 steps, T=1 is about 1.85 nats below that unigram. [`val_ce_vs_unigram.svg`](../artifacts/kaggle-prefix-level0-v2/val_ce_vs_unigram.svg) is level-0 cross-entropy against the unigram, with final T=4 and T=full markers. It does not overlay token perplexity or a 2018 GPT-1 curve. This is Phase 1 continuation, not a chatbot.

The notebook in this repository is [`kaggle/train_prefix_level0.ipynb`](../kaggle/train_prefix_level0.ipynb). `NICHOLAS_CONFIRMED` stays `False`. Version 1 failed because `python -m ncm_iwsp_gpt1` did not inherit the notebook `sys.path`. The notebook puts the repo on `PYTHONPATH`, runs dry-plan and training with cwd set to the repo root, and exits if a fresh process still cannot import the package. The Kaggle source dataset was commit `f35e352` plus [`patches/time-budget-max-hours.patch`](../patches/time-budget-max-hours.patch). That patch is not applied to the training code in this checkout.

## Weights for chat

The v3 generator is not in this git repository. The prefix head from Kaggle version 2 is [`artifacts/kaggle-prefix-level0-v2/prefix_level0.pt`](../artifacts/kaggle-prefix-level0-v2/prefix_level0.pt) (21,305,039 bytes, sha256 `0ebb615fd9ea29827dd03e4cce24bf06de770178ea869bb0b469d83e70239698`).

| Flags | Role |
| --- | --- |
| `--prefix-head artifacts/kaggle-prefix-level0-v2/prefix_level0.pt` | Required for the default path. Format `ncm-iwsp-gpt1-prefix-level0`. |
| `--tokenizer tokenizer.pt --finetune latest.pt` | Frozen tokenizer plus the fine-tuned generator. Optimizer state in `latest.pt` is ignored. |
| `--tokenizer tokenizer.pt --generator generator.pt` | Converted init pair, if you are not loading a fine-tune. |
| `--checkpoint FILE` | One file with format `ncm-iwsp-gpt1-from-gradient-checkpoint`. |
| `--level0-source packed` | Comparison baseline. Repeats the prompt with end-of-text between copies, encodes that window, and keeps level 0. Not the default. |
| `--level0-source unigram --packed packed_windows.pt` | Comparison baseline. Level 0 is an add-one unigram of up to `--unigram-max-windows` rows (default 256). The prompt does not choose the code. The generator prefix is still end-of-text. |

Pass either `--prefix-head` or one baseline, not both. `--packed` is only for the unigram baseline.

A tied-head trainer checkpoint is refused. With no tokenizer or generator file, chat still runs the prefix head on the seeded initialization and says so on stderr. That is not the v3 model.

## Commands

Checkout at `e:\Project\ncm-iwsp-gpt1`. The examples assume the venv from the README. The prefix head below is the file in git. The tokenizer and the v3 generator are example paths, not files in git.

Default path, after the head has been trained:

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --device auto ^
  --tokenizer E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt ^
  --finetune E:\Project\ncm-iwsp-gpt1-finetune-v3\latest.pt ^
  --prefix-head artifacts\kaggle-prefix-level0-v2\prefix_level0.pt ^
  --temperature 0.8 --top-k 40 --top-p 0.9 ^
  --prompt "The cat sat"
```

Packed-repeat baseline, the previous chat path:

```bat
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --tokenizer E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt ^
  --finetune E:\Project\ncm-iwsp-gpt1-finetune-v3\latest.pt ^
  --level0-source packed ^
  --prompt "The cat sat"
```

Unigram baseline:

```bat
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --tokenizer E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt ^
  --finetune E:\Project\ncm-iwsp-gpt1-finetune-v3\latest.pt ^
  --level0-source unigram ^
  --packed E:\path\to\artifacts\packed_windows.pt ^
  --prompt "The cat sat"
```

Small config, seeded modules, explicit packed baseline. This does not load the 12/768 model and does not need a prefix head:

```bat
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\gpt1_data.yaml ^
  --device cpu ^
  --level0-source packed ^
  --prompt "Hello"
```

Interactive mode reads a line, prints the decoded window, and repeats. Quit with `quit`, `exit`, `/quit`, or `/exit`. Ctrl-D quits as well. A blank line is ignored. `--prompt` prints only the decoded text on stdout. The level-0 source, the level-0 id, and the prefix token go to stderr.

POSIX is the same command with `.venv/bin/python` and `\` line continuations.

## Phase 2

Phase 2 is conditional continuation: the prompt occupies the generator prefix, and a later fine-tune can teach the generator to use those slots. This repository change is the source, the configs, and the tests. It does not start training, does not submit a Kaggle job, and does not turn `predict_level0` on.

Select it with `--config`. The phase-1 files stay as they are, so a v3 checkpoint and the prefix head still run on `configs/target.yaml`.

| Config | What it is |
| --- | --- |
| `configs/target.yaml` | Phase 1. 12/768/12, context 512, `prefix_len` 1, end-of-text prefix. Default chat config. |
| `configs/gpt1_data.yaml` | Phase 1, small enough to run on CPU. |
| `configs/target_phase2.yaml` | Phase 2 target. Same 12/768/12 body, `prefix_len` 32, `prefix_source: window`. |
| `configs/gpt1_data_phase2.yaml` | Phase 2, small. `prefix_len` 8 so a short prompt is padded and a longer one is truncated inside context 32. |

The plan allowed `prefix_len` 32 or 64. **32** is the shipped 12/768 value. It is the smaller candidate, and the fixed prompts are all shorter than 32 tokens. 64 still builds: the mask and RoPE buffers follow `prefix_len`, and the transformer weights do not change shape. Sequence length on the target is 4338 with prefix 32 (4306 predicted cells plus the prefix) and would be 4370 with prefix 64. Switching is a one-line edit of `configs/target_phase2.yaml`.

`prefix_source: window` is the phase-2 protocol.

- Training and quantitative eval copy `tokens[:, :prefix_len]` into the prefix. Labels stay levels 1..K-1. Level 0 is not a generator target.
- Chat places the prompt in the prefix. Fewer tokens than `prefix_len` are **right-padded** with the pad id **50257**. More tokens keep the **first** `prefix_len` ids. The prefix head still sees the prompt up to the context length; only the generator prefix is cut to `prefix_len`.
- stdout may print the original prompt in front of the decoded window when the window does not already start with that prompt. `text` and `decoded_window` are the model's window. Score those, not the spliced display.
- `--level0-source packed` and `unigram` stay explicit baselines. On a phase-2 config their generator prefix is also the prompt. The unigram-plus-end-of-text control is the phase-1 config, not a second hidden prefix inside phase 2. Eval still emits an end-of-text unigram sample and labels it as that control.

The prefix head is **not** retrained. `train-prefix-level0` still refuses `prefix_len != 1`. Chat on the phase-2 config uses the existing head: it reads `encoder.wte` and does not depend on generator `prefix_len`. Retraining the head would be a later change with its own trigger. This path does not add a dialogue loader. The packer still rejects MiniMind SFT, DPO, and RLAIF filenames.

Fine-tune, when someone runs it later, is the existing `finetune` command pointed at `configs/target_phase2.yaml`. The tokenizer stays frozen. The generator warm-starts from a v3 `latest.pt`: persistent `transformer` and `lm_heads` load, and stale mask or RoPE buffers are ignored because those buffers are rebuilt for the new `prefix_len`. The checkpoint format string stays `ncm-iwsp-gpt1-finetune`. New checkpoints record `prefix_len` and `prefix_source`.

- Resume refuses a different prefix, and refuses a legacy file that omits `prefix_len` unless the config is phase 1 (`prefix_len` 1, `prefix_source` eos).
- Chat refuses a checkpoint whose recorded prefix disagrees with the config.
- Chat still loads a legacy file that does not record `prefix_len` (the v3 `latest.pt` shape) and says so on stderr. That is a negative control, not a trained phase-2 model.
- Without `--prefix-head`, chat still exits. It does not fall back to packed-repeat.

Learning rate, step count, batch size, corruption, and validation frequency are the existing finetune CLI defaults. They are **not** a chosen phase-2 recipe. `finetune --dry-plan` prints that note and does not train. A real run still needs `--confirm` and a new `--out` directory. Do not write into the v3, init, or prefix-level0 outputs.

```bash
python -m ncm_iwsp_gpt1 finetune --config configs/target_phase2.yaml --dry-plan

python -m ncm_iwsp_gpt1 chat \
  --config configs/target_phase2.yaml \
  --tokenizer /path/to/tokenizer.pt \
  --finetune /path/to/latest.pt \
  --prefix-head artifacts/kaggle-prefix-level0-v2/prefix_level0.pt \
  --prompt "How are you?"
```

The four prompts the plan names are `How are you?`, `Hello world`, `The cat sat`, and `下一概念`. Readable continuation of those prompts is a later check on a fine-tuned checkpoint. It is not a result of this source change, and it is not a chatbot answer.
