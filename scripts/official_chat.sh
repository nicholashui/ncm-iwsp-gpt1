#!/bin/sh
# Official decode-side chat. Checkpoints and the TinyLlama planner are local files.
set -eu
: "${NCM_TOKENIZER:?set NCM_TOKENIZER to the frozen tokenizer checkpoint}"
: "${NCM_FINETUNE:?set NCM_FINETUNE to the dialogue generator checkpoint}"
: "${NCM_PREFIX_HEAD:?set NCM_PREFIX_HEAD to the prefix-head checkpoint}"
: "${NCM_PLANNER_DIR:?set NCM_PLANNER_DIR to a local TinyLlama chat directory}"

exec "${PYTHON:-python}" -m ncm_iwsp_gpt1.chat_lmsplice \
  --config "${NCM_CONFIG:-configs/target_phase3.yaml}" \
  --tokenizer "$NCM_TOKENIZER" \
  --finetune "$NCM_FINETUNE" \
  --prefix-head "$NCM_PREFIX_HEAD" \
  --planner-dir "$NCM_PLANNER_DIR" \
  --dialogue \
  --temperature 0 \
  --top-k 0 \
  --top-p 1 \
  --seed 0 \
  "$@"
