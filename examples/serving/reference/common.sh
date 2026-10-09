# Sourced by every script of the reference deployment: settings (serving.env), paths, caches and helpers.
# shellcheck shell=bash
IG_SERVING_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ig_compat: settings under the pre-rename DIG_* names (environment or serving.env) still work, as a deprecated
# fallback: each DIG_X is copied to IG_X when IG_X is unset (exported if DIG_X was), with one warning.
ig_compat() {
  local v n warned=
  for v in $(compgen -v DIG_); do
    n=IG_${v#DIG_}
    [ -n "${!n+x}" ] && continue
    printf -v "$n" '%s' "${!v}"
    case $(declare -p "$v") in "declare -"*x*) export "${n?}" ;; esac
    [ -n "$warned" ] || { echo "warning: DIG_* settings are deprecated, rename them to IG_* (e.g. $v -> $n)" >&2; warned=1; }
  done
}
ig_compat
_env_file=${IG_SERVING_ENV:-$IG_SERVING_DIR/serving.env}
# shellcheck disable=SC1090
[ -f "$_env_file" ] && . "$_env_file"
ig_compat

: "${IG_SERVICES_DIR:=$IG_SERVING_DIR}"
: "${IG_MODELS_DIR:=$IG_SERVICES_DIR/models}"
: "${IG_CACHE_DIR:=$IG_SERVICES_DIR/cache}"
: "${IG_LOG_DIR:=$IG_SERVICES_DIR/logs}"
: "${IG_LLM_ENV:=$IG_SERVICES_DIR/envs/llm}"
: "${IG_TTS_ENV:=$IG_SERVICES_DIR/envs/tts}"
: "${IG_OMNI_ENV:=$IG_SERVICES_DIR/envs/omni}"
: "${IG_SESSION_PREFIX:=dig_}"
: "${IG_BIND_HOST:=127.0.0.1}"

: "${LLM_MODEL:=Qwen3.8-27B-FP8}"
: "${LLM_SERVED:=Qwen/Qwen3.8-27B-FP8 Qwen3.8-27B}"
: "${LLM_GPUS:=0,1}"
: "${LLM_TP:=2}"
: "${LLM_PORT:=8000}"
: "${LLM_MAX_NUM_SEQS:=128}"
: "${TTS_MODEL:=Qwen3-TTS-12Hz-1.7B-CustomVoice}"
: "${TTS_REPLICAS=6:8002 7:8003}"
: "${TTS_PORT:=8001}"
: "${CLONE_MODEL:=Qwen3-TTS-12Hz-1.7B-Base}"
: "${CLONE_REPLICAS=}"
: "${CLONE_PORT:=8005}"
: "${AGENT_MODEL:=MiniCPM-o-4_5}"
: "${AGENT_LAYOUT:=thinker}"
case $AGENT_LAYOUT in
  thinker) : "${AGENT_SERVERS=2:8010 3:8011 4:8012 5:8013}"; : "${AGENT_MAX_SESSIONS:=16}" ;;
  audio) : "${AGENT_SERVERS=4,5:8010}"; : "${AGENT_MAX_SESSIONS:=4}" ;;
  *) echo "AGENT_LAYOUT must be thinker or audio (got $AGENT_LAYOUT)" >&2; return 1 2>/dev/null || exit 1 ;;
esac
: "${AGENT_TOKEN_TRACE:=1}"
: "${IG_MINICPMO_OVERLAY=}"

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export XDG_CACHE_HOME=$IG_CACHE_DIR VLLM_CACHE_ROOT=$IG_CACHE_DIR/vllm HF_HOME=$IG_CACHE_DIR/huggingface
export MODELSCOPE_CACHE=$IG_CACHE_DIR/modelscope TORCHINDUCTOR_CACHE_DIR=$IG_CACHE_DIR/torchinductor
export TRITON_CACHE_DIR=$IG_CACHE_DIR/triton
mkdir -p "$IG_CACHE_DIR" "$IG_LOG_DIR"

# use_env <python env dir>: its bin/ first on PATH. Without a system CUDA toolkit (no CUDA_HOME), use the pip one
# inside the env (nvidia/cu13), which flashinfer's JIT needs.
use_env() {
  export PATH=$1/bin:$PATH
  if [ -z "${CUDA_HOME:-}" ]; then
    local c
    c=$(ls -d "$1"/lib/python3*/site-packages/nvidia/cu13 2>/dev/null | head -n 1 || true)
    if [ -n "$c" ]; then export CUDA_HOME=$c PATH=$c/bin:$PATH; fi
  fi
}

# omni_pkg_dir <python env dir>: the installed vllm_omni package directory
omni_pkg_dir() {
  "$1/bin/python" -c 'import os, vllm_omni; print(os.path.dirname(vllm_omni.__file__))'
}

# render <config> <python env dir>: a copy of a deploy YAML in $IG_LOG_DIR with @MODELS_DIR@ and @VLLM_OMNI_DIR@
# filled in; prints its path
render() {
  local out
  out=$IG_LOG_DIR/$(basename "$1" .yaml).rendered.yaml
  sed -e "s#@MODELS_DIR@#$IG_MODELS_DIR#g" -e "s#@VLLM_OMNI_DIR@#$(omni_pkg_dir "$2")#g" "$1" > "$out"
  echo "$out"
}

# start_session <name> <command>: (re)start a tmux session $IG_SESSION_PREFIX<name>, logging to $IG_LOG_DIR/<name>.log
start_session() {
  local s=$IG_SESSION_PREFIX$1
  tmux kill-session -t "=$s" 2>/dev/null || true
  tmux new-session -d -s "$s" "cd '$IG_SERVING_DIR' && $2 2>&1 | tee '$IG_LOG_DIR/$1.log'"
  echo "started $s"
}

# our_sessions <regex>: running tmux sessions $IG_SESSION_PREFIX<name> whose <name> matches
our_sessions() {
  tmux ls -F '#S' 2>/dev/null | grep -E "^${IG_SESSION_PREFIX}($1)\$" || true
}
