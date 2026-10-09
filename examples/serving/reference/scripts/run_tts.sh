#!/usr/bin/env bash
# One vLLM-Omni Qwen3-TTS CustomVoice replica. Usage: run_tts.sh <gpu> <port>
set -euo pipefail
. "$(dirname "$0")/../common.sh"
GPU=${1:?gpu}; PORT=${2:?port}
export CUDA_VISIBLE_DEVICES=$GPU
use_env "$IG_TTS_ENV"
exec vllm serve "$IG_MODELS_DIR/$TTS_MODEL" --omni \
  --served-model-name "Qwen/$TTS_MODEL" \
  --host "$IG_BIND_HOST" --port "$PORT"
