#!/usr/bin/env bash
# nvidia/personaplex-7b-v1 full-duplex agent on vLLM-Omni (duplex patches recommended), GPUS (default 2), PORT (default 8022).
# Weights: $IG_MODELS_DIR/personaplex-7b-v1. Deploy config: configs/personaplex_1gpu.yaml (rendered with local paths).
set -euo pipefail
. "$(dirname "$0")/../common.sh"
export CUDA_VISIBLE_DEVICES=${GPUS:-2}
export PYTHONPATH=$IG_SERVING_DIR/pyshim${PYTHONPATH:+:$PYTHONPATH}
use_env "$IG_OMNI_ENV"
exec vllm serve "$IG_MODELS_DIR/personaplex-7b-v1" --omni \
  --served-model-name nvidia/personaplex-7b-v1 \
  --deploy-config "$(render "$IG_SERVING_DIR/configs/personaplex_1gpu.yaml" "$IG_OMNI_ENV")" \
  --trust-remote-code \
  --host "$IG_BIND_HOST" --port "${PORT:-8022}"
