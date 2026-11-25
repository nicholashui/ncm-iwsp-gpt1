# Chat with NCM-iWSP-GPT1

`python -m ncm_iwsp_gpt1 chat` continues a short prompt as one decoded window. It is not a causal language model, not NCM-GPT1, and not a chatbot. A prompt such as `How are you?` is not expected to come back as `I'm fine`. Code cross-entropy and 2018 GPT-1 perplexity are not the quality check for this path.

`predict_level0` stays false. Level 0 is not a target of `NextConceptGenerator`. The phase-1 configs `configs/target.yaml` and `configs/gpt1_data.yaml` keep `prefix_len` at 1. That one slot is `<|endoftext|>` (id 50256), the prefix the existing weights were trained with. The words of the prompt do not go there. Phase 2 is a separate config, described below. Phase 3 is also separate and stays off unless its commands and flags are passed. No dialogue corpus is selected. The default `--config` is still `configs/target.yaml`.

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

The prefix head is **not** retrained by phase-2 `finetune`. `train-prefix-level0` still refuses `prefix_len != 1`. Chat on the phase-2 config uses the existing head: it reads `encoder.wte` and does not depend on generator `prefix_len`. Dialogue head training is the separate command `train-prefix-level0-dialogue`, under Phase 3 below. This phase-2 path does not add a dialogue loader. The packer still rejects MiniMind SFT, DPO, and RLAIF filenames.

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

## Phase 3

Phase 3 is dialogue source on top of the phase-2 prefix. It does not start training, does not submit a Kaggle job, and does not turn `predict_level0` on. Phase 1 and phase 2 configs and commands are unchanged when the phase-3 flags are absent. `chat` still does not call itself a chatbot unless `--dialogue` is set and `--finetune` is a dialogue checkpoint (`format` `ncm-iwsp-gpt1-dialogue`). A missing prefix head still exits. It does not fall back to packed-encode.

No dialogue or instruction corpus is selected. `data.dataset` on the phase-3 yaml is `unset` on purpose. The id is required on the command line at pack time and again at train time. Chinese pretraining and Chinese dialogue corpora are not selected. MiniMind SFT, DPO, and RLAIF names stay refused. The BookCorpus Open `pack` path is unchanged.

| Config | What it is |
| --- | --- |
| `configs/target.yaml` | Phase 1. `prefix_len` 1, end-of-text prefix. Default chat config. |
| `configs/gpt1_data.yaml` | Phase 1, small. |
| `configs/target_phase2.yaml` | Phase 2. `prefix_len` 32, `prefix_source: window`. |
| `configs/gpt1_data_phase2.yaml` | Phase 2, small. `prefix_len` 8. |
| `configs/target_phase3.yaml` | Phase 3 target. Same 12/768 body, `prefix_len` 32, `history_placement: both`. |
| `configs/gpt1_data_phase3.yaml` | Phase 3, small. `prefix_len` 8, same placement. |
| `configs/dialogue_allowlist.txt` | Ships with no dataset ids. |

`prefix_len` 32 matches phase 2 so a phase-2 generator checkpoint can warm-start. 64 is still a one-line edit of `prefix_len` in `configs/target_phase3.yaml`. The mask and RoPE buffers follow that length. The transformer weights do not change shape.

### Packing

`pack-dialogue` is a separate command. It does not go through `make_cache`, and it does not skip `packing._refuse_corpus_name`. A local file name, the dataset id, and every allow-list line are checked against the same banned fragments. An allow-list hit does not authorize a banned name.

The command requires all of:

- `--dataset-id` (there is no default)
- `--allow-list` pointing at a file that lists that exact id
- `--license-note` (this repository does not supply one)
- one source: `--jsonl`, `--arrow`, `--hf-cache`, or `--hf-dataset` plus `--download`

`lucadiliello/bookcorpusopen` stays on `pack`. A text-only row is rejected here. Accepted rows are `messages` or `conversations` lists of role and content, or one prompt/response pair. Role aliases such as `human` and `gpt` are field names, not a chosen corpus. Each assistant reply after a user turn becomes one window. Rows are not concatenated the way BookCorpus paragraphs are. The cache stays out of git.

Role markers are the plain strings `User`, `Assistant`, and `System`. They are not new token ids. `dialogue.add_special_role_tokens` must stay false. GPT-2 BPE and `encoder.wte` are not expanded.

### History

`dialogue.history_placement` is `prefix`, `window`, or `both`. The phase-3 files set `both`. That is a switch, not a tuned recipe.

Truncation is `keep_tail`. Turns are oldest-first. Tokens that do not fit are dropped from the left, so the newest turn stays. A short sequence is right-padded with pad id 50257. This is not the phase-2 prompt rule, which keeps the first `prefix_len` tokens.

- `prefix`: the generator prefix is the tail of the conditioning turns. The training window is the assistant reply only.
- `window`: the training window is the whole dialogue, left-truncated. The generator prefix is the first `prefix_len` tokens of that window. At inference the reply does not exist yet, so this prefix is only the start of the truncated history and can differ from training.
- `both`: the window is the whole dialogue, and the prefix is the conditioning tail, so train and inference prefixes match. Some history tokens can appear in both places.

Phase 2 `prefix_len` is 32 on the 12/768 config (8 on the small file). A long history will not fit in the prefix. `keep_tail` is how that overflow is cut.

### Fine-tune

`finetune-dialogue` warm-starts the generator from `--generator` (a phase-2 `latest.pt` or a converted `generator.pt`) and writes only to a new `--out`. `--confirm` is required. The tokenizer stays frozen. `predict_level0` stays false. The prefix head is not retrained. `--retrain-prefix-head` is an error, not a silent second loop. `--dataset-id` must match the cache. The book-window `finetune` command refuses a phase-3 config and refuses a dialogue cache.

Learning rate, steps, batch size, corruption, and validation frequency are the existing finetune CLI defaults. They are not an approved phase-3 recipe. `finetune-dialogue --dry-plan` prints that and does not train.

`ce_first`, `ce_mid`, and `ce_last` stay raw per-level means. Compare them when `alpha` and `mid_aux` match. Do not compare the optimized total loss across different `alpha` or `mid_aux`. Do not read code cross-entropy as chatbot quality. The checkpoint records `code_ce_is_chat_quality: false`.

### Dialogue prefix head

`finetune-dialogue` still does not retrain the prefix head. The prefix head is not retrained by that command. `--retrain-prefix-head` is an error, not a second loop. `train-prefix-level0` still refuses a phase-3 config and still refuses a dialogue pack.

`train-prefix-level0-dialogue` is the head-only command. It keeps `predict_level0` false. The tokenizer stays frozen. The generator checkpoint is loaded and frozen so `prefix_len` and `prefix_source` can be checked; those weights are not updated. The level-0 target is still the code of the full packed window. The head input is the pack's prefix row with pad id 50257 removed, not the first tokens of the window. T is drawn from 1 through that unpadded length, so a 7-token `User: How are you?` stays in distribution. `history_placement: window` is refused because that prefix can contain the assistant reply. `prefix` and `both` keep the reply out of the prefix. Past turns that are already in the conditioning stay.

`--dataset-id` is required and must match the cache, which must already carry a license note. Banned names are refused. No corpus is selected by the config. `--dry-plan` prints the plan and does not train. A real run needs `--confirm` and `--max-steps`. The command writes only to `--out`. It does not default to `ncm-iwsp-gpt1-prefix-level0`, and that directory is refused. The intended NicNicNic directory is `E:\Project\ncm-iwsp-gpt1-prefix-level0-phase3`. Any other new directory is accepted. An existing phase-1 `prefix_level0.pt` can be passed as `--resume`; it is read-only and the new checkpoint is written under `--out`.

Validation logs level-0 cross-entropy at T=1, T=4, T=7, T=32, and each row's full unpadded length, against an add-one unigram of the dialogue windows. That comparison is not a chatbot score. Aligning the head may still not yield coherent chat. Success of a later run is a readable Dolly-format continuation.

```bash
python -m ncm_iwsp_gpt1 train-prefix-level0-dialogue \
  --config configs/target_phase3.yaml \
  --dry-plan

python -m ncm_iwsp_gpt1 train-prefix-level0-dialogue \
  --config configs/target_phase3.yaml \
  --tokenizer E:/Project/ncm-iwsp-gpt1-init/tokenizer.pt \
  --generator E:/Project/ncm-iwsp-gpt1-finetune-phase3/latest.pt \
  --packed E:/Project/ncm-iwsp-gpt1-dialogue-dolly-15k/packed_windows.pt \
  --dataset-id databricks/databricks-dolly-15k \
  --out E:/Project/ncm-iwsp-gpt1-prefix-level0-phase3 \
  --resume E:/Project/ncm-iwsp-gpt1-prefix-level0/prefix_level0.pt \
  --max-steps 20000 \
  --batch-size 8 \
  --lr 1e-3 \
  --device cuda \
  --confirm
```

`--max-steps 20000` is the previous head budget, not a new recipe. Do not start that run from this repository.

```bash
python -m ncm_iwsp_gpt1 pack-dialogue \
  --config configs/target_phase3.yaml \
  --dataset-id <id-on-your-allow-list> \
  --allow-list /path/to/allow-list.txt \
  --license-note "<license or authorization>" \
  --jsonl /path/to/local.jsonl \
  --out /path/to/dialogue_windows.pt

python -m ncm_iwsp_gpt1 finetune-dialogue \
  --config configs/target_phase3.yaml \
  --dry-plan

python -m ncm_iwsp_gpt1 finetune-dialogue \
  --config configs/gpt1_data_phase3.yaml \
  --tokenizer /path/to/tokenizer.pt \
  --generator /path/to/phase2-latest.pt \
  --packed /path/to/dialogue_windows.pt \
  --dataset-id <same-id> \
  --out /path/to/new-dialogue-out \
  --confirm \
  --max-steps 1
```

### Reply slice

Generation still decodes one whole window. It does not stop at a token. With `--dialogue` and a dialogue checkpoint, stdout prints a slice. `decoded_window` remains the model text. Score that, not the slice.

1. If the decoded window starts with the rendered history, including the current user line, that leading copy is removed. Overlap is only a prefix strip.
2. Otherwise, if it starts with the current `User:` line, that line is removed.
3. A leading `Assistant:` marker is removed.
4. The reply ends before the next line that starts with `User:`.
5. `--max-reply-tokens` 0 leaves that remainder. A positive value keeps the first that many BPE tokens.

Session history stores the user line and that sliced reply, not the rest of the window. The next turn can place that history in the prefix, the window, or both. `/clear` drops it. Without `--dialogue`, history is not stored, including on `configs/target_phase3.yaml`.

```bash
python -m ncm_iwsp_gpt1 chat \
  --config configs/target_phase3.yaml \
  --dialogue \
  --tokenizer /path/to/tokenizer.pt \
  --finetune /path/to/new-dialogue-out/latest.pt \
  --prefix-head artifacts/kaggle-prefix-level0-v2/prefix_level0.pt \
  --prompt "How are you?"
```

The phase-1 prefix head was not trained on dialogue and is not retrained here. A readable answer is not a result of this source change. Passing conditions for phase 3 are still unset. Code CE, a 2018 GPT-1 perplexity, and "the window looks like English" are not those conditions.
