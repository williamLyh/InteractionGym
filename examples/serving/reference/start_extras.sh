#!/usr/bin/env bash
# Optional user-simulator models for compatibility experiments (README "Extra user-sim models"). They share GPUs
# with the core services; edit the GPU / port / memory arguments below for your layout.
set -euo pipefail
cd "$(dirname "$0")"
. ./common.sh
start() { tmux has-session -t "=$IG_SESSION_PREFIX$1" 2>/dev/null || start_session "$1" "$2"; }
start x_tts_design "scripts/run_tts_model.sh Qwen3-TTS-12Hz-1.7B-VoiceDesign Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign 6 8006 configs/qwen3_tts_shared.yaml"
start x_tts_cv06   "scripts/run_tts_model.sh Qwen3-TTS-12Hz-0.6B-CustomVoice Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice 7 8007 configs/qwen3_tts_shared.yaml"
# VoxCPM2 needs the voxcpm package: install it into the TTS env, or set VOXCPM_PYTHONPATH to a directory holding it
start x_tts_voxcpm "env PYTHONPATH=${VOXCPM_PYTHONPATH:-} scripts/run_tts_model.sh VoxCPM2 openbmb/VoxCPM2 3 8008 configs/voxcpm2_shared.yaml"
start x_llm_qwen4b "scripts/run_small_llm.sh Qwen3-4B Qwen3-4B 2 8020 0.45"
start x_llm_phi    "scripts/run_small_llm.sh Phi-4-mini-instruct Phi-4-mini-instruct 2 8021 0.4"
echo "started extras: 8006 8007 8008 8020 8021 (~2-4 min)"
