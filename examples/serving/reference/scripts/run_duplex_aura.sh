#!/usr/bin/env bash
# aurateam/AURA full-duplex agent on vLLM-Omni (duplex patches recommended), GPUS (default 2,3), PORT (default 8020).
# Weights: $IG_MODELS_DIR/AURA. Deploy config: configs/aura_omni_2gpu.yaml (rendered with local paths).
set -euo pipefail
. "$(dirname "$0")/../common.sh"
export CUDA_VISIBLE_DEVICES=${GPUS:-2,3}
export PYTHONPATH=$IG_SERVING_DIR/pyshim${PYTHONPATH:+:$PYTHONPATH}
use_env "$IG_OMNI_ENV"
exec vllm serve "$IG_MODELS_DIR/AURA" --omni \
  --served-model-name aurateam/AURA \
  --deploy-config "$(render "$IG_SERVING_DIR/configs/aura_omni_2gpu.yaml" "$IG_OMNI_ENV")" \
  --trust-remote-code \
  --host "$IG_BIND_HOST" --port "${PORT:-8020}"
