#!/usr/bin/env bash
# nvidia/NVIDIA-NemotronLabs-VoiceChat-11B full-duplex agent on vLLM-Omni (duplex patches recommended), GPUS (default 2,3), PORT (default 8021).
# Weights: $IG_MODELS_DIR/NVIDIA-NemotronLabs-VoiceChat-11B. Deploy config: configs/nemotron_voicechat_duplex_2gpu.yaml (rendered with local paths).
set -euo pipefail
. "$(dirname "$0")/../common.sh"
export CUDA_VISIBLE_DEVICES=${GPUS:-2,3}
export PYTHONPATH=$IG_SERVING_DIR/pyshim${PYTHONPATH:+:$PYTHONPATH}
use_env "$IG_OMNI_ENV"
exec vllm serve "$IG_MODELS_DIR/NVIDIA-NemotronLabs-VoiceChat-11B" --omni \
  --served-model-name nvidia/NVIDIA-NemotronLabs-VoiceChat-11B \
  --deploy-config "$(render "$IG_SERVING_DIR/configs/nemotron_voicechat_duplex_2gpu.yaml" "$IG_OMNI_ENV")" \
  --trust-remote-code \
  --host "$IG_BIND_HOST" --port "${PORT:-8021}"
