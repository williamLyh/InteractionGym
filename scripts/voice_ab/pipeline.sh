#!/bin/bash
# Voice A/B + example render pipeline (one 8-GPU host). Idempotent and resumable: every stage leaves state/<stage>.done;
# re-running the script skips finished stages. Run in tmux:  tmux new-session -d -s vab_pipeline "bash pipeline.sh"
# Servers run in their own tmux sessions (vab_*); a watchdog restarts any that die or stay unhealthy.
R=${VOICE_AB_ROOT:?set VOICE_AB_ROOT to a work directory}; cd $R
S=${IG_SERVICES:?set IG_SERVICES to your serving directory (examples/serving/reference)}
VPY=$R/venv/bin/python               # envs/llm + speechbrain/jiwer/librosa (system site packages)
export PYTHONPATH=$R/code/src HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co} HF_HUB_DISABLE_XET=1 HF_HOME=$R/hub/hf TORCH_HOME=$R/hub/torch
mkdir -p state logs state/servers
log() { echo "- $(date '+%F %T') $*" | tee -a STATUS.md; }
done_() { touch state/$1.done; log "done: $1"; }
is_done() { [ -f state/$1.done ]; }

# ---- servers: name|port|command ; the watchdog keeps every file in state/servers/ alive
reg() { echo "$2|$3" > state/servers/$1; date +%s > state/servers/$1.t; }
unreg() { rm -f state/servers/$1 state/servers/$1.t; tmux kill-session -t $1 2>/dev/null; }
healthy() { curl -sf -m5 http://127.0.0.1:$1/v1/models >/dev/null; }
start_srv() { tmux kill-session -t $1 2>/dev/null; tmux new-session -d -s $1 "$3 > $R/logs/$1.log 2>&1"; date +%s > state/servers/$1.t; }
watchdog() {
  while true; do
    for f in state/servers/*; do
      [ -f "$f" ] || continue; case $f in *.t|*.bad) continue;; esac
      n=$(basename $f); IFS='|' read -r port cmd < $f
      if ! tmux has-session -t $n 2>/dev/null; then log "watchdog: $n gone, restarting"; start_srv $n $port "$cmd"; continue; fi
      if healthy $port; then rm -f $f.bad; continue; fi
      t0=$(cat $f.t); now=$(date +%s)
      if [ -f $f.bad ] && [ $((now - $(cat $f.bad))) -gt 300 ] && [ $((now - t0)) -gt 1500 ]; then
        log "watchdog: $n unhealthy >5 min, restarting"; rm -f $f.bad; start_srv $n $port "$cmd"
      elif [ ! -f $f.bad ]; then echo $now > $f.bad; fi
    done
    sleep 60
  done
}
up() { # name port command
  reg $1 $2 "$3"; healthy $2 || start_srv $1 $2 "$3"
}
wait_up() { for i in $(seq 180); do ok=1; for p in "$@"; do healthy $p || ok=0; done; [ $ok = 1 ] && return 0; sleep 10; done; log "servers $* not up after 30 min"; return 1; }
retry() { # stage command... : rerun until success (max 6)
  local st=$1; shift
  for k in 1 2 3 4 5 6; do "$@" >> logs/$st.log 2>&1 && return 0; log "$st attempt $k failed (see logs/$st.log)"; sleep 60; done
  return 1
}

watchdog & WD=$!
trap "kill $WD 2>/dev/null" EXIT
log "pipeline start (pid $$)"

TTS_CV="cd $S && scripts/run_tts.sh"
TTS_B="cd $S && scripts/run_tts_model.sh models/Qwen3-TTS-12Hz-1.7B-Base Qwen/Qwen3-TTS-12Hz-1.7B-Base"
B_CFG=$S/envs/tts/lib/python3.12/site-packages/vllm_omni/deploy/qwen3_tts.yaml
PROXY="cd $S/scripts && TTS_BACKENDS"

# ---- 1-2. TTS servers (CustomVoice GPUs 0-3, Base clone GPUs 4-7) -> synthesis -> latency bench
if ! is_done synth || ! is_done bench; then
  for i in 0 1 2 3; do up vab_cv$i 820$((i+1)) "$TTS_CV $i 820$((i+1))"; up vab_b$i 821$((i+1)) "$TTS_B $((i+4)) 821$((i+1)) $B_CFG"; done
  up vab_pcv 8200 "$PROXY=http://127.0.0.1:8201,http://127.0.0.1:8202,http://127.0.0.1:8203,http://127.0.0.1:8204 ../envs/tts/bin/uvicorn tts_proxy:app --host 127.0.0.1 --port 8200 --log-level warning"
  up vab_pb 8210 "$PROXY=http://127.0.0.1:8211,http://127.0.0.1:8212,http://127.0.0.1:8213,http://127.0.0.1:8214 ../envs/tts/bin/uvicorn tts_proxy:app --host 127.0.0.1 --port 8210 --log-level warning"
  wait_up 8201 8202 8203 8204 8211 8212 8213 8214 && log "TTS up"
  is_done synth || { log "running: synth"; retry synth $VPY -u synth.py && done_ synth; }
  is_done bench || { log "running: bench"; retry bench $VPY -u bench.py && done_ bench; }
  for n in vab_cv0 vab_cv1 vab_cv2 vab_cv3 vab_b0 vab_b1 vab_b2 vab_b3 vab_pcv vab_pb; do unreg $n; done
fi
is_done synth || { log "STOP: synth failed"; exit 1; }

# ---- 3. objective metrics: Qwen3-ASR on GPU 0, ECAPA + UTMOS on GPU 1
if ! is_done metrics; then
  up vab_asr 8220 "cd $S && CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 CUDA_HOME=$S/envs/tts/lib/python3.12/site-packages/nvidia/cu13 PATH=$S/envs/tts/bin:\$PATH envs/tts/bin/vllm serve models/Qwen3-ASR-1.7B --served-model-name Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port 8220 --gpu-memory-utilization 0.5 --max-model-len 8192"
  wait_up 8220 && log "ASR up; running: metrics"
  CUDA_VISIBLE_DEVICES=1 retry metrics $VPY -u metrics.py && done_ metrics
  unreg vab_asr
fi

# ---- 4. judge: Qwen3-Omni-30B-A3B-Instruct, DP 2 x TP 4 on all 8 GPUs
if ! is_done judge; then
  until grep -q DLDONE logs/dl.log 2>/dev/null; do sleep 60; done
  up vab_judge 8230 "bash $R/run_judge.sh 0,1,2,3,4,5,6,7 4 2 8230 --moe-backend triton"
  wait_up 8230 && log "judge up; running: judge"
  retry judge $VPY -u judge.py && done_ judge
  unreg vab_judge
fi

# ---- 5. report
is_done report || { $VPY loud.py >> logs/report.log 2>&1; $VPY report.py > logs/report.log 2>&1 && done_ report; }

# ---- 6. example render with the new defaults + MiniCPM-o 4.5 full duplex (lockstep) — waits for state/render.go,
# created once the updated env code is synced to $R/code. LLM GPUs 0,1 | TTS 2 | clone 3 | MiniCPM-o 4,5 + 6,7
if ! is_done render; then
  log "waiting for state/render.go (render stage)"
  until [ -f state/render.go ]; do sleep 60; done
  C=$R/code
  up vab_llm 8100 "cd $S && GPUS=0,1 DP=1 PORT=8100 scripts/run_llm.sh"
  up vab_tts 8102 "$TTS_CV 2 8102"
  up vab_clone 8105 "$TTS_B 3 8105 $B_CFG"
  up vab_mo0 8010 "cd $S && GPUS=4,5 PORT=8010 scripts/run_minicpmo.sh"
  up vab_mo1 8011 "cd $S && GPUS=6,7 PORT=8011 scripts/run_minicpmo.sh"
  wait_up 8100 8102 8105 8010 8011 && log "render servers up; running: render"
  rm -rf $R/out/viewer_v2.prev; [ -d $R/out/viewer_v2 ] && mv $R/out/viewer_v2 $R/out/viewer_v2.prev
  PYTHONPATH=$C/src:$C/scripts/host retry render $VPY -u $C/scripts/host/view_examples_v2.py && done_ render
  for n in vab_llm vab_tts vab_clone vab_mo0 vab_mo1; do unreg $n; done
fi
log "ALL DONE"
for f in state/servers/*; do [ -f "$f" ] && case $f in *.t|*.bad) ;; *) unreg $(basename $f);; esac; done
