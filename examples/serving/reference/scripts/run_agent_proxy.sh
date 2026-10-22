#!/usr/bin/env bash
# One WebSocket address in front of the agent servers: each session goes to the server with the fewest open sessions
# and stays there (src/interaction_gym/agent_proxy.py; it needs only `websockets` >= 13, which the omni env has).
# Env: AGENT_BACKENDS (comma-separated ws://host:port), PORT (default $AGENT_PORT), MAX_SESSIONS (per server).
set -euo pipefail
. "$(dirname "$0")/../common.sh"
: "${AGENT_BACKENDS:?comma-separated ws://host:port of the agent servers}"
exec "${IG_PROXY_PYTHON:-$IG_OMNI_ENV/bin/python}" "$IG_SERVING_DIR/../../../src/interaction_gym/agent_proxy.py" \
  --backends "$AGENT_BACKENDS" --host "$IG_BIND_HOST" --port "${PORT:-$AGENT_PORT}" \
  ${MAX_SESSIONS:+--max-sessions "$MAX_SESSIONS"}
