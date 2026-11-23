# Chat with NCM-iWSP-GPT1

`python -m ncm_iwsp_gpt1 chat` continues a short prompt as one decoded window. It is not a causal language model, not NCM-GPT1, and not a chatbot. A prompt such as `How are you?` is not expected to come back as `I'm fine`. Code cross-entropy and 2018 GPT-1 perplexity are not the quality check for this path.

`predict_level0` stays false. Level 0 is not a target of `NextConceptGenerator`. Both shipped configs keep `prefix_len` at 1. That one slot is `<|endoftext|>` (id 50256), the prefix the existing weights were trained with. The words of the prompt do not go there.

## What one turn does

The default path needs a prefix-head checkpoint. Without `--prefix-head`, chat stops. It does not silently repeat the prompt out to 512 tokens and encode that window.

1. GPT-2 BPE encodes the line. `"How are you?"` is four tokens: `How`, ` are`, ` you`, `?` (ids 2437, 389, 345, 30). A longer line is truncated to the context (512 on `configs/target.yaml`, 32 on `configs/gpt1_data.yaml`). The tokens are not repeated to fill the context, and they are not padded.
2. The frozen tokenizer `encoder.wte` embeds those tokens. No tokenizer position is added. The prefix head adds its own positions `0 .. T-1`, attention-pools them, and produces logits over the level-0 codebook (4096 on the target config, 64 on the small config).
3. Chat samples one level-0 code from that distribution. `temperature` 0 is argmax. `top_k` 0 and `top_p` 1 leave the distribution untruncated. The same knobs apply to the finer codes.
4. The generator prefix is `<|endoftext|>`, repeated `prefix_len` times. Both shipped configs use `prefix_len` 1. `NextConceptGenerator.generate` samples levels 1..K-1 from the sampled level-0 code and that prefix.
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

The log compares validation cross-entropy at prefix lengths 1, 4, and the full window with an add-one unigram of level 0. If the short-prefix number stays near the unigram, this head is not enough. That is phase 2 (a longer generator prefix and a generator fine-tune). Do not go back to tuning the packed-repeat encode. This command does not submit a Kaggle job. A new kernel, if one is opened later, stays separate from `ncm-gpt-1-train`, and `NICHOLAS_CONFIRMED` stays false until someone sets it.

## Weights for chat

v3 weights and the prefix head are not in this git repository. Download them onto the machine that will run chat.

| Flags | Role |
| --- | --- |
| `--prefix-head prefix_level0.pt` | Required for the default path. Format `ncm-iwsp-gpt1-prefix-level0`. |
| `--tokenizer tokenizer.pt --finetune latest.pt` | Frozen tokenizer plus the fine-tuned generator. Optimizer state in `latest.pt` is ignored. |
| `--tokenizer tokenizer.pt --generator generator.pt` | Converted init pair, if you are not loading a fine-tune. |
| `--checkpoint FILE` | One file with format `ncm-iwsp-gpt1-from-gradient-checkpoint`. |
| `--level0-source packed` | Comparison baseline. Repeats the prompt with end-of-text between copies, encodes that window, and keeps level 0. Not the default. |
| `--level0-source unigram --packed packed_windows.pt` | Comparison baseline. Level 0 is an add-one unigram of up to `--unigram-max-windows` rows (default 256). The prompt does not choose the code. The generator prefix is still end-of-text. |

Pass either `--prefix-head` or one baseline, not both. `--packed` is only for the unigram baseline.

A tied-head trainer checkpoint is refused. With no tokenizer or generator file, chat still runs the prefix head on the seeded initialization and says so on stderr. That is not the v3 model.

## Commands

Checkout at `e:\Project\ncm-iwsp-gpt1`. The examples assume the venv from the README. Replace the weight paths with the files you actually have. The v3 generator used below is an example path, not a file in git.

Default path, after the head has been trained:

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --device auto ^
  --tokenizer E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt ^
  --finetune E:\Project\ncm-iwsp-gpt1-finetune-v3\latest.pt ^
  --prefix-head E:\Project\ncm-iwsp-gpt1-prefix-level0\prefix_level0.pt ^
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
