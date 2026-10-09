#!/usr/bin/env bash
# Stop the core services of this deployment, including servers started by the GPU tuner's launch.sh
# (same session names: <prefix>llm, <prefix>tts0, <prefix>agent1, <prefix>cloneproxy, ...).
cd "$(dirname "$0")" && . ./common.sh
for s in $(our_sessions 'llm|tts[0-9]*|ttsproxy|agent[0-9]*|clone[0-9]*|cloneproxy'); do
  tmux kill-session -t "=$s" && echo "stopped $s"
done
