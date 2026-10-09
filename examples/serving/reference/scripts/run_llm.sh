#!/usr/bin/env bash
# User-simulator LLM on vLLM: $LLM_MODEL, tensor-parallel $LLM_TP x data-parallel (#GPUs / TP) behind one endpoint.
# Overrides: GPUS, PORT, MAX_NUM_SEQS (concurrent sequences per replica; the GPU tuner's cap knob), DP.
set -euo pipefail
. "$(dirname "$0")/../common.sh"
GPUS=${GPUS:-$LLM_GPUS}
NGPU=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
export CUDA_VISIBLE_DEVICES=$GPUS
use_env "$IG_LLM_ENV"
# shellcheck disable=SC2086
exec vllm serve "$IG_MODELS_DIR/$LLM_MODEL" \
  --served-model-name $LLM_SERVED \
  --host "$IG_BIND_HOST" --port "${PORT:-$LLM_PORT}" \
  --tensor-parallel-size "$LLM_TP" --data-parallel-size "${DP:-$((NGPU / LLM_TP))}" \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --max-model-len 32768 --max-num-seqs "${MAX_NUM_SEQS:-$LLM_MAX_NUM_SEQS}" \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}'
