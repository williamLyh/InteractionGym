#!/usr/bin/env bash
cd "$(dirname "$0")" && . ./common.sh
for s in $(our_sessions 'x_[a-z0-9_]+'); do tmux kill-session -t "=$s" && echo "stopped $s"; done
