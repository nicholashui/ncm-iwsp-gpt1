# Chat with NCM-iWSP-GPT1

`python -m ncm_iwsp_gpt1 chat` decodes one window of text. It is a front end on the existing `VQVAE.encode`, `NextConceptGenerator.generate`, and `VQVAE.decode_logits` path. It does not train, does not call the writer, and does not submit a Kaggle job.

This model is not a causal language model and not NCM-GPT1. `predict_level0` is false, so level 0 is not a generator target. Both shipped configs set `prefix_len` to 1. The generator therefore reads at most one raw token, looked up in the tokenizer embedding with no position added.

## What one turn actually does

1. GPT-2 BPE encodes the line. A leading space is part of the text; GPT-2 BPE keeps it.
2. Tokens past the context are dropped. That context is 512 for `configs/target.yaml` and 32 for `configs/gpt1_data.yaml`. The kept tokens are padded with `<|pad|>` (id 50257) out to the full context.
3. `VQVAE.encode` quantizes that padded window. Chat keeps only level 0. On both shipped square schedules level 0 is a single code.
4. The prefix is the first `prefix_len` prompt tokens, or `<|endoftext|>` when the line encodes to nothing. Further tokens in the line are not a prefix.
5. `NextConceptGenerator.generate` samples levels 1..K-1 from that level-0 code and that prefix. `temperature` 0 is argmax. `top_k` 0 and `top_p` 1 leave the distribution untruncated.
6. `decode_logits(...).argmax` and `pretok.decode` turn the whole code grid into text. Pad and end-of-text ids are removed.

The printed reply is that one window. It is not the next sentence after the prompt. A short prompt is mostly padding, so level 0 is one coarse code of a nearly empty window. The previous reply is not fed back. There is no multi-turn token cache and no long-context continuation.

The evaluator's qualitative hook is a different path: it draws level 0 from a train-set unigram and does not encode the prompt. Chat encodes the prompt because this command has no packed cache.

## Weights

v3 weights are not in this git repository. Neither are `tokenizer.pt`, `generator.pt`, or a fine-tune `latest.pt`. Download them onto the machine that will run chat. The commands below use the NicNicNic paths as examples.

Three layouts load directly. You do not wrap them into a new file.

| Flags | File layout |
| --- | --- |
| `--checkpoint FILE` | One file with format `ncm-iwsp-gpt1-from-gradient-checkpoint`, the file `eval --init-checkpoint` already loads. |
| `--tokenizer tokenizer.pt --generator generator.pt` | The converted init pair. Each blob is typically `{model, step, converted_from, ...}`. `model` is the state dict. |
| `--tokenizer tokenizer.pt --finetune latest.pt` | Frozen tokenizer plus a fine-tune checkpoint of format `ncm-iwsp-gpt1-finetune`. The `model` entry loads into the generator. Optimizer state is ignored. The file is still read in full, and with the optimizer included it is on the order of 2 GB. |

A tied-head NCM-GPT1 trainer checkpoint (`lm_heads.*.proj.weight`, `post_upsample_convs`) is refused. Point `--generator` at the converted untied generator.

With no weight flags, chat builds the seeded initialization and says so on stderr. That is not the v3 model. `configs/gpt1_data.yaml` is the small shape check (context 32). `configs/target.yaml` is the 12-layer / 768-wide model (context 512) and is the default.

## Windows

Checkout at `e:\Project\ncm-iwsp-gpt1`. The examples assume the venv from the README install section.

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat --help
```

Converted init pair (tokenizer 92,276,480 parameters, generator 154,666,240):

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --device auto ^
  --tokenizer E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt ^
  --generator E:\Project\ncm-iwsp-gpt1-init\generator.pt
```

One shot, after a fine-tune `latest.pt` has been downloaded. Replace the `--finetune` path with the file you actually have:

```bat
cd /d e:\Project\ncm-iwsp-gpt1
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --device auto ^
  --tokenizer E:\Project\ncm-iwsp-gpt1-init\tokenizer.pt ^
  --finetune E:\Project\ncm-iwsp-gpt1-finetune\latest.pt ^
  --temperature 0.8 --top-k 40 --top-p 0.9 ^
  --prompt "Hello"
```

A single converted file uses `--checkpoint` instead of the pair:

```bat
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\target.yaml ^
  --checkpoint E:\Project\ncm-iwsp-gpt1\artifacts\ncm_iwsp_gpt1_from_100k.pt ^
  --prompt "Hello"
```

Interactive mode reads a line, prints the decoded window, and repeats. Quit with `quit`, `exit`, `/quit`, or `/exit`. Ctrl-D quits as well. A blank line is ignored. `--prompt` prints only the decoded text on stdout so a script can capture it. The level-0 id, the prefix token, and any truncation note go to stderr.

Small config, seeded initialization, no weights. This does not load the 12/768 model:

```bat
.venv\Scripts\python.exe -m ncm_iwsp_gpt1 chat ^
  --config configs\gpt1_data.yaml ^
  --device cpu ^
  --prompt "Hello"
```

POSIX is the same command with `.venv/bin/python` and `\` line continuations.
