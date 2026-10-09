#!/usr/bin/env bash
# (Re)start the core services in tmux: the user-simulator LLM, TTS replicas + proxy, optional voice-clone TTS
# replicas + proxy, and the full-duplex agent server(s). Layout and paths: serving.env (see serving.env.example).
# Safe to re-run: stops only this deployment's sessions ($IG_SESSION_PREFIX*) first.
set -euo pipefail
cd "$(dirname "$0")"
. ./common.sh
./stop.sh >/dev/null
sleep 5   # let GPU memory free up

start_session llm "GPUS=$LLM_GPUS scripts/run_llm.sh"

backends=() i=0
for r in $TTS_REPLICAS; do
  start_session "tts$i" "scripts/run_tts.sh ${r%%:*} ${r##*:}"
  backends+=("http://127.0.0.1:${r##*:}"); i=$((i + 1))
done
start_session ttsproxy "TTS_BACKENDS=$(IFS=,; echo "${backends[*]}") PORT=$TTS_PORT scripts/run_tts_proxy.sh"

if [ -n "$CLONE_REPLICAS" ]; then
  backends=() i=0
  for r in $CLONE_REPLICAS; do
    start_session "clone$i" "scripts/run_tts_model.sh $CLONE_MODEL Qwen/$CLONE_MODEL ${r%%:*} ${r##*:} configs/qwen3_tts_shared.yaml"
    backends+=("http://127.0.0.1:${r##*:}"); i=$((i + 1))
  done
  start_session cloneproxy "TTS_BACKENDS=$(IFS=,; echo "${backends[*]}") PORT=$CLONE_PORT scripts/run_tts_proxy.sh"
fi

urls=() i=0
for a in $AGENT_SERVERS; do
  start_session "agent$i" "GPUS=${a%%:*} PORT=${a##*:} MAX_SESSIONS=$AGENT_MAX_SESSIONS scripts/run_minicpmo.sh"
  urls+=("ws://127.0.0.1:${a##*:}/v1/realtime?duplex=1"); i=$((i + 1))
done

echo "ready in a few minutes (LLM ~3-5 min, TTS ~2 min, agent ~5 min); check with ./status.sh"
echo "client environment:"
echo "  export IG_LLM_URL=http://127.0.0.1:$LLM_PORT/v1 IG_TTS_URL=http://127.0.0.1:$TTS_PORT/v1"
if [ -n "$CLONE_REPLICAS" ]; then echo "  export IG_CLONE_URL=http://127.0.0.1:$CLONE_PORT/v1"; fi
if [ ${#urls[@]} -gt 0 ]; then
  echo "  export IG_AGENT_URL='${urls[0]}' IG_AGENT_URLS='$(IFS=,; echo "${urls[*]}")' IG_MODEL_DIR=$IG_MODELS_DIR"
fi
