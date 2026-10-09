#!/usr/bin/env bash
# A small chat LLM for user-sim experiments.
# Usage: run_small_llm.sh <model dir> <served name> <gpu> <port> <mem fraction> [extra vllm args]
set -euo pipefail
. "$(dirname "$0")/../common.sh"
MODEL=${1:?}; NAME=${2:?}; GPU=${3:?}; PORT=${4:?}; MEM=${5:?}; shift 5
case $MODEL in /*) ;; *) MODEL=$IG_MODELS_DIR/$MODEL ;; esac
export CUDA_VISIBLE_DEVICES=$GPU
use_env "$IG_LLM_ENV"
exec vllm serve "$MODEL" --served-model-name "$NAME" --host "$IG_BIND_HOST" --port "$PORT" --max-model-len 8192 \
  --gpu-memory-utilization "$MEM" "$@"
