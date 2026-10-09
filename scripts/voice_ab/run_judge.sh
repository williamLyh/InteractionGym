#!/bin/bash
# Qwen3-Omni-30B-A3B-Instruct (thinker, text out) on upstream vLLM 0.30 (envs/llm via voice_ab/venv, which adds librosa;
# envs/tts's vllm_omni plugin overrides the model with a version that fails at startup). Usage:
#   run_judge.sh <gpus> <tp> <dp> <port> [extra vllm args]
S=${IG_SERVICES:?set IG_SERVICES}; R=${VOICE_AB_ROOT:?set VOICE_AB_ROOT}
GPUS=$1; TP=$2; DP=$3; PORT=$4; shift 4
export CUDA_VISIBLE_DEVICES=$GPUS HF_HUB_OFFLINE=1
export CUDA_HOME=$S/envs/llm/lib/python3.12/site-packages/nvidia/cu13 PATH=$R/venv/bin:$S/envs/llm/bin:$S/envs/llm/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
C=$R/cache; mkdir -p $C
export VLLM_CACHE_ROOT=$C/vllm TORCHINDUCTOR_CACHE_DIR=$C/inductor TRITON_CACHE_DIR=$C/triton
exec $R/venv/bin/python -m vllm.entrypoints.cli.main serve ${JUDGE_MODEL_DIR:-$R/models/Qwen3-Omni-30B-A3B-Instruct} \
  --served-model-name Qwen3-Omni-30B-A3B-Instruct --host 127.0.0.1 --port $PORT \
  --tensor-parallel-size $TP --data-parallel-size $DP --max-model-len 32768 --max-num-seqs 64 \
  --gpu-memory-utilization 0.88 --limit-mm-per-prompt '{"audio": 2, "image": 0, "video": 0}' "$@"
