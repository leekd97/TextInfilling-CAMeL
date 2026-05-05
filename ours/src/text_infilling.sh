#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd):${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODELS="${MODELS:-LLAMA-3.1-8B Trillion-7B-preview LLAMA-3.1-8B-Instruct}"
PROMPT_CULTURES="${PROMPT_CULTURES:-co}"
LANGUAGE="${LANGUAGE:-ar}"
PROMPT_N="${PROMPT_N:-50}"
ENTITY_N="${ENTITY_N:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
LOAD_FLAGS="${LOAD_FLAGS:---load_in_4bit}"
GPU_IDS="${GPU_IDS:-2 2 3}"
EVALUATE_AFTER="${EVALUATE_AFTER:-1}"

read -r -a GPU_ARRAY <<< "$GPU_IDS"
read -r -a LOAD_FLAG_ARRAY <<< "$LOAD_FLAGS"

if [ "${#GPU_ARRAY[@]}" -eq 0 ]; then
  echo "GPU_IDS is empty. Example: GPU_IDS=\"2 3\""
  exit 1
fi

PARALLEL_JOBS="${PARALLEL_JOBS:-${#GPU_ARRAY[@]}}"

run_one() {
  local gpu_id="$1"
  local model_name="$2"
  local prompt_culture="$3"

  echo "[RUN] gpu=${gpu_id} model=${model_name} culture=${prompt_culture}"
  CUDA_VISIBLE_DEVICES="$gpu_id" python3 text_infilling.py \
    --model_name "$model_name" \
    --prompt_culture "$prompt_culture" \
    --language "$LANGUAGE" \
    --prompt_sample_count "$PROMPT_N" \
    --entity_sample_count "$ENTITY_N" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --check_prev \
    "${LOAD_FLAG_ARRAY[@]}"

  if [ "$EVALUATE_AFTER" = "1" ]; then
    python3 ../results/text_infilling/evaluate.py \
      --model_name "$model_name" \
      --prompt_culture "$prompt_culture" \
      --language "$LANGUAGE" \
      --prompt_sample_count "$PROMPT_N" \
      --entity_sample_count "$ENTITY_N" \
      --draw_figure
  fi
}

pids=()
job_index=0
for prompt_culture in $PROMPT_CULTURES; do
  for model_name in $MODELS; do
    gpu_id="${GPU_ARRAY[$((job_index % ${#GPU_ARRAY[@]}))]}"
    run_one "$gpu_id" "$model_name" "$prompt_culture" &
    pids+=("$!")
    job_index=$((job_index + 1))

    if [ "${#pids[@]}" -ge "$PARALLEL_JOBS" ]; then
      wait "${pids[0]}"
      pids=("${pids[@]:1}")
    fi
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
