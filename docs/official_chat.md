# Official decode-side chat

This path is **NCM-iWSP-GPT1**, not NCM-GPT1, and it is not causal token continuation.

One turn is:

1. The prefix head predicts level 0 from the dialogue prefix. `--prefix-head` is required. If it is missing, official chat stops. It does not switch to another level-0 source.
2. The frozen generator samples the later residual VQ levels. `predict_level0` stays false. This command does not train that generator.
3. VQ decode reconstructs one window (512 tokens on `configs/target_phase3.yaml`).
4. A decode-side splice writes a planned Assistant body into that window. Residual codes are not rewritten.
5. `slice_reply` extracts the Assistant span.

The planner is [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) (Apache-2.0). Download it yourself. Do not commit the weights, a tokenizer checkpoint, a generator checkpoint, or a packed dialogue cache.

```bash
python -m ncm_iwsp_gpt1.download_planner --out "$NCM_PLANNER_DIR"
```

`$NCM_PLANNER_DIR` defaults to `planners/tinyllama-1.1b-chat` when you omit `--out`. That directory is gitignored.

Official chat is dialogue mode at temperature 0 (argmax), with `--top-k 0` and `--top-p 1`. Pass your own checkpoint paths. The shell wrapper reads the same values from the environment:

```bash
export NCM_TOKENIZER=/path/to/tokenizer.pt
export NCM_FINETUNE=/path/to/dialogue-latest.pt
export NCM_PREFIX_HEAD=/path/to/prefix_level0.pt
export NCM_PLANNER_DIR=/path/to/tinyllama-1.1b-chat
export NCM_CONFIG=configs/target_phase3.yaml   # optional

scripts/official_chat.sh --prompt "What is a short way to boil an egg?"
```

The same entry point without the wrapper:

```bash
python -m ncm_iwsp_gpt1.chat_lmsplice \
  --config configs/target_phase3.yaml \
  --tokenizer "$NCM_TOKENIZER" \
  --finetune "$NCM_FINETUNE" \
  --prefix-head "$NCM_PREFIX_HEAD" \
  --planner-dir "$NCM_PLANNER_DIR" \
  --dialogue \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --seed 0 \
  --prompt "What is a short way to boil an egg?"
```

`python -m ncm_iwsp_gpt1 official-chat` takes the same flags. `--finetune` must be a dialogue checkpoint (`ncm-iwsp-gpt1-dialogue`). Omit `--prompt` for the stdin loop. Quit with `quit` or `exit`. `/clear` drops history.

Before the splice, a general reply policy (not a table of questions) does seven things:

1. Answer only the current user question, in at most two sentences, and stop at end-of-turn or the first complete Assistant turn.
2. If the planner emits a story, script, poem, or canned help-desk dump, replace it with a short conversational reply.
3. Strip echoed user wording, `User:` / `Assistant:` labels, and a reciprocal follow-up answered in the same turn.
4. Drop a time, place, or appointment the user did not mention.
5. Keep a reserved tail in the reconstructed window and cut the body on a clause boundary so the reply is not sliced mid-clause.
6. Refuse or give a safe alternative for harmful or high-risk requests (weapons, unauthorized access, phishing, poisons, leftover prescription drugs, reckless all-in financial advice, and similar harm).
7. If the question has no concrete topic, ask one clarifying question. If it asks for a preference, mark the reply as an opinion.

`pytest tests/test_reply_policy.py` covers those helpers and does not load planner or generator weights.
