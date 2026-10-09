#!/usr/bin/env bash
# Any vLLM-Omni TTS model on a GPU it may share. Usage: run_tts_model.sh <model dir> <served name> <gpu> <port> <deploy config>
# (a relative model dir is taken under $IG_MODELS_DIR). MAX_NUM_SEQS=n: every stage's max_num_seqs = n (a copy of
# the deploy config in the log dir; the GPU tuner's cap knob).
set -euo pipefail
. "$(dirname "$0")/../common.sh"
MODEL=${1:?model}; NAME=${2:?name}; GPU=${3:?gpu}; PORT=${4:?port}; CFG=${5:?deploy config}
case $MODEL in /*) ;; models/*) MODEL=$IG_MODELS_DIR/${MODEL#models/} ;; *) MODEL=$IG_MODELS_DIR/$MODEL ;; esac
export CUDA_VISIBLE_DEVICES=$GPU
if [ -n "${MAX_NUM_SEQS:-}" ]; then
  OUT=$IG_LOG_DIR/tts_deploy_$(basename "$CFG" .yaml)_s${MAX_NUM_SEQS}_${PORT}.yaml
  sed -E "s/^(    max_num_seqs:) [0-9]+/\1 $MAX_NUM_SEQS/" "$CFG" > "$OUT"
  CFG=$OUT
fi
use_env "$IG_TTS_ENV"
exec vllm serve "$MODEL" --omni --served-model-name "$NAME" --deploy-config "$CFG" --host "$IG_BIND_HOST" --port "$PORT"
