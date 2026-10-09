#!/usr/bin/env bash
# Least-in-flight proxy over TTS replicas. Env: TTS_BACKENDS (comma-separated base URLs), PORT (default $TTS_PORT).
set -euo pipefail
. "$(dirname "$0")/../common.sh"
cd "$(dirname "$0")"
: "${TTS_BACKENDS:?comma-separated backend URLs}"
export TTS_BACKENDS
exec "$IG_TTS_ENV/bin/uvicorn" tts_proxy:app --host "$IG_BIND_HOST" --port "${PORT:-$TTS_PORT}" --log-level warning
