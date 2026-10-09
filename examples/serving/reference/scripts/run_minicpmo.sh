#!/usr/bin/env bash
# MiniCPM-o 4.5 full duplex (vLLM-Omni DuplexOmni runtime). WebSocket: ws://$IG_BIND_HOST:<port>/v1/realtime?duplex=1
# (alias /v1/duplex). Two layouts (AGENT_LAYOUT, default from serving.env):
#   thinker  one GPU, Stage 0 only, text-only sessions (configs/minicpmo_4_5_thinker_1gpu.yaml; needs
#            the repository's patches/vllm-omni/ patch in the vLLM-Omni install or overlay); GPUS=a
#   audio    Thinker on the first GPU, Talker + Code2Wav on the second (configs/minicpmo_4_5_2gpu.yaml); GPUS=a,b
# Env: GPUS PORT (default: the first entry of $AGENT_SERVERS); MAX_SESSIONS=n (duplex_session.max_sessions and
#   every stage's max_num_seqs; default $AGENT_MAX_SESSIONS); DEPLOY=<yaml> (overrides the layout's config);
#   IG_OMNI_ENV (a vLLM-Omni with the duplex patches for lockstep / token trace; a stock one runs realtime mode,
#   then set AGENT_TOKEN_TRACE=0); IG_MINICPMO_OVERLAY (optional overlay root, scripts/make_minicpmo_overlay.py).
set -euo pipefail
. "$(dirname "$0")/../common.sh"
FIRST=${AGENT_SERVERS%% *}
GPUS=${GPUS:-${FIRST%%:*}}; PORT=${PORT:-${FIRST##*:}}; MAX_SESSIONS=${MAX_SESSIONS:-$AGENT_MAX_SESSIONS}
case $AGENT_LAYOUT in
  thinker) DEPLOY=${DEPLOY:-$IG_SERVING_DIR/configs/minicpmo_4_5_thinker_1gpu.yaml} ;;
  audio) DEPLOY=${DEPLOY:-$IG_SERVING_DIR/configs/minicpmo_4_5_2gpu.yaml} ;;
esac
OUT=$IG_LOG_DIR/minicpmo_deploy_s${MAX_SESSIONS}_${PORT}.yaml
# the session cap and the stage batch sizes move together (as in the shipped yaml)
sed -E "s/^(  max_sessions:) [0-9]+/\1 $MAX_SESSIONS/; s/^(    max_num_seqs:) [0-9]+/\1 $MAX_SESSIONS/" "$DEPLOY" > "$OUT"
if [ "$AGENT_TOKEN_TRACE" = 0 ]; then sed -i.bak '/enable_debug_events/d' "$OUT"; fi
export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONPATH=${IG_MINICPMO_OVERLAY:+$IG_MINICPMO_OVERLAY:}$IG_SERVING_DIR/pyshim${PYTHONPATH:+:$PYTHONPATH}
use_env "$IG_OMNI_ENV"
TEMPLATE=$(omni_pkg_dir "$IG_OMNI_ENV")/transformers_utils/chat_templates/minicpmo45_native.jinja
exec vllm serve "$IG_MODELS_DIR/$AGENT_MODEL" --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 \
  --deploy-config "$OUT" \
  --trust-remote-code \
  --chat-template "$TEMPLATE" --chat-template-content-format openai \
  --host "$IG_BIND_HOST" --port "$PORT"
