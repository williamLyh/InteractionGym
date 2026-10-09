#!/usr/bin/env bash
# Sessions, endpoint health and GPU memory.
cd "$(dirname "$0")" && . ./common.sh
tmux ls 2>/dev/null | grep "^$IG_SESSION_PREFIX" || echo "no $IG_SESSION_PREFIX sessions"
ports="$LLM_PORT $TTS_PORT"
for r in $TTS_REPLICAS $CLONE_REPLICAS $AGENT_SERVERS; do ports="$ports ${r##*:}"; done
[ -n "$CLONE_REPLICAS" ] && ports="$ports $CLONE_PORT"
for p in $ports ${EXTRA_PORTS:-8006 8007 8008 8020 8021}; do
  if curl -s -m 3 "http://127.0.0.1:$p/v1/models" >/dev/null; then echo "port $p: UP"; else echo "port $p: down"; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
