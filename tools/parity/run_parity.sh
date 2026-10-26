#!/bin/bash
# The old-vs-new parity runs against real model servers (docs/ARCHITECTURE.md §8): for each benchmark, the `main`
# script runs through the record proxy (tools/parity/proxy.py), then the plugin bundle runs through the same store in
# replay mode — every answer the new run gets is one the old run got, and any request the old run never made is a
# miss — and the episodes, agent traces and report files are compared field by field (compare.py).
#
#   run_parity.sh all                 # every stage, in order (resumable: a stage with a verdict file is skipped)
#   run_parity.sh suite_default sop   # named stages
#   LIVE=1 run_parity.sh audiomc      # third leg: the new run live too (its own record store) for the statistics tier
#
# Layout: $PARITY_OUT/<stage>/{old,new,store,verdict.txt,compare.json,proxy.log}. Servers: see stack.sbatch; the
# proxy listens one port above each server's +10000 (8000 -> 18000 ...).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${PARITY_ROOT:=$HOME/ig_parity}"             # repo (plugin-arch) and repo_main checkouts
: "${PARITY_DATA:=/projects/u6va/yinhongliu/ig_parity}"
: "${PARITY_OUT:=/scratch/u6va/$USER/ig_parity/runs}"
: "${SEEDS:=0}"
: "${N:=3}"                                     # tasks per stage (where a limit applies)
NEW=$PARITY_ROOT/repo; OLD=$PARITY_ROOT/repo_main
PY_NEW=$NEW/.venv/bin/python; PY_OLD=$OLD/.venv/bin/python; IG=$NEW/.venv/bin/ig
mkdir -p "$PARITY_OUT"

# ---- services (through the proxy) and data, identical for both implementations
export IG_LLM_URL=http://127.0.0.1:18000/v1 IG_JUDGE_URL=http://127.0.0.1:18000/v1 IG_LLM_MODEL=${IG_LLM_MODEL:-Qwen3.8-27B}
export IG_TTS_URL=http://127.0.0.1:18001/v1 IG_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
export IG_CLONE_URL=http://127.0.0.1:18005/v1 IG_CLONE_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base
export IG_ASR_URL=http://127.0.0.1:18006/v1 IG_ASR_MODEL=Qwen3-ASR-1.7B
export IG_AGENT_URL="ws://127.0.0.1:18010/v1/realtime?duplex=1" IG_AGENT_MODEL=openbmb/MiniCPM-o-4_5
export IG_AGENT_REF_AUDIO=$PARITY_DATA/models/MiniCPM-o-4_5/assets/system_ref_audio.wav
export IG_SOUNDBANK=${IG_SOUNDBANK:-$PARITY_DATA/ig_soundbank} IG_SOUNDBANK_FETCH=0   # the 5090-1 bank (DEMAND ambience): Zenodo is unreachable from the compute nodes
export IG_AUDIOMC_DATA=$PARITY_DATA/data/audiomc/wav IG_FDB2_DATA=$PARITY_DATA/data/prompts_staged_200.json
export IG_FDB3_DATA=$PARITY_DATA/data/fdb3_opus IG_FDB_DATA=$PARITY_DATA/data/full_duplex_bench
export IG_HUMDIAL_DATA=$PARITY_DATA/data/humdial IG_EASY_TURN_DATA=$PARITY_DATA/data/easy_turn
MAPS="--map 18000=http://127.0.0.1:8000 --map 18001=http://127.0.0.1:8001 --map 18005=http://127.0.0.1:8005 --map 18006=http://127.0.0.1:8006 --map 18010=ws://127.0.0.1:8010"

log() { echo "[$(date +%H:%M:%S)] $*"; }
proxy_start() { # mode store log
  "$PY_NEW" "$HERE/proxy.py" --mode "$1" --store "$2" $MAPS --log "$3" & PROXY_PID=$!
  for _ in $(seq 1 50); do curl -s -o /dev/null http://127.0.0.1:18000/health && break; sleep 0.2; done
}
proxy_stop() { kill -TERM "$PROXY_PID" 2>/dev/null; wait "$PROXY_PID" 2>/dev/null; }
old() { (cd "$OLD" && PYTHONPATH=src:. "$PY_OLD" "$@"); }
new() { (cd "$NEW" && "$@"); }

# compare FILE_OLD FILE_NEW NAME [--json]  -> appends to the stage's verdict
EXPECT_ONLY_REF=""   # a stage sets a regex of join keys the script makes and the bundle does not
cmp_files() {
  local a=$1 b=$2 name=$3; shift 3
  [ -n "$EXPECT_ONLY_REF" ] && [ "${1:-}" != "--json" ] && set -- "$@" --expect-only-ref "$EXPECT_ONLY_REF"
  if [ ! -s "$a" ] || [ ! -s "$b" ]; then echo "$name: MISSING ($([ -s "$a" ] || echo old) $([ -s "$b" ] || echo new))" >> "$V"; return; fi
  local line; line=$("$PY_NEW" "$HERE/compare.py" "$a" "$b" "$@" --out "$S/compare_${name//\//_}.json" 2>&1 | head -1)
  echo "$name: $line" >> "$V"
}
# stage NAME  -- runs ${NAME}_old / ${NAME}_new / ${NAME}_cmp functions
stage() {
  local name=$1; S=$PARITY_OUT/$name; V=$S/verdict.txt
  if [ -s "$V" ] && [ -z "${FORCE:-}" ]; then log "$name: done ($(head -1 "$V"))"; return; fi
  mkdir -p "$S"; : > "$V.partial"
  local rc_old=reused
  if [ -z "${REUSE_OLD:-}" ] || [ ! -d "$S/store/http" ]; then
    log "== $name: old (record)"
    rm -rf "$S/store" "$S/old"; mkdir -p "$S/old"
    proxy_start record "$S/store" "$S/proxy_record.log"
    ( ${name}_old "$S/old" ) > "$S/old.log" 2>&1; rc_old=$?
    proxy_stop
  fi
  log "== $name: new (replay)"
  rm -rf "$S/new"; mkdir -p "$S/new"
  proxy_start replay "$S/store" "$S/proxy_replay.log"
  ( ${name}_new "$S/new" ) > "$S/new.log" 2>&1; local rc_new=$?
  proxy_stop
  { echo "old rc=$rc_old new rc=$rc_new"; echo "proxy: $(cat "$S/store/summary.json" 2>/dev/null | tr -d '\n ')"; echo "misses: $(wc -l < "$S/store/misses.jsonl" 2>/dev/null)"; } > "$V.partial"
  V=$V.partial; ${name}_cmp "$S/old" "$S/new"; V=$S/verdict.txt
  if [ -n "${LIVE:-}" ]; then
    log "== $name: new (live, own store)"
    rm -rf "$S/live" "$S/store_live"; mkdir -p "$S/live"
    proxy_start record "$S/store_live" "$S/proxy_live.log"
    ( ${name}_new "$S/live" ) > "$S/live.log" 2>&1
    proxy_stop
    echo "--- live (new, independent sampling) vs old:" >> "$V.partial"; V=$V.partial; ${name}_cmp "$S/old" "$S/live"; V=$S/verdict.txt
  fi
  mv "$V.partial" "$V"; log "== $name: $(sed -n 1,2p "$V" | tr '\n' ' ')"; sed 's/^/   /' "$V"
}

# ---------------------------------------------------------------- stages
# MiniCPM-o suite, default set (the script's realtime / text / offline-replay takes have no bundle: expected only-in-old)
suite_default_old() { old examples/minicpmo_suite.py --out "$1" --seeds $SEEDS; }
suite_default_new() { new "$IG" run minicpmo_suite --out "$1" --seeds "$SEEDS"; }
suite_default_cmp() { EXPECT_ONLY_REF="realtime|offline-replay"; cmp_files "$1/episodes.jsonl" "$2/episodes.jsonl" episodes; cmp_files "$1/agent_traces.jsonl" "$2/agent_traces.jsonl" traces; }

CHAT_ONLY=${CHAT_ONLY:-chat-hiking-sam know-sky-zoe zh-know-moon-chen}
suite_chat_old() { old examples/minicpmo_suite.py --out "$1" --seeds $SEEDS --set chat --only $CHAT_ONLY; }
suite_chat_new() { new "$IG" run minicpmo_suite_chat --out "$1" --seeds "$SEEDS" --ids "$(echo $CHAT_ONLY | tr " " ,)"; }
suite_chat_cmp() { suite_default_cmp "$@"; }

# SOP calls: two scenarios, the judge included (the script judges at the end; the bundle's scorer judges per episode)
SOP_ONLY=${SOP_ONLY:-booking-mina clinic-frank}
sop_old() { old examples/sop_calls.py --out "$1" --seeds $SEEDS --only $SOP_ONLY; }
sop_new() { new "$IG" run sop_calls --out "$1" --seeds "$SEEDS" --ids "$(echo $SOP_ONLY | tr ' ' ',')"; }
sop_cmp() { cmp_files "$1/episodes.jsonl" "$2/episodes.jsonl" episodes; cmp_files "$1/agent_traces.jsonl" "$2/agent_traces.jsonl" traces; }

# Audio MultiChallenge: the same N conversations on both sides, A and B arms, then the report (rubric judge)
audiomc_ids() { new "$PY_NEW" -c "from interaction_gym.plugins import Composition, compose; r = compose(Composition.bundle('audiomc').resolve()); print(','.join(t.id for t in r.tasks[:$N]))"; }
audiomc_old() { local ids; ids=$(audiomc_ids); old examples/audiomc_ab.py run --data "$IG_AUDIOMC_DATA" --out "$1" --agent minicpmo --seeds $SEEDS --ids "$ids" --concurrency 3 && old examples/audiomc_ab.py report --data "$IG_AUDIOMC_DATA" --out "$1" --agent minicpmo; }
audiomc_new() { new "$IG" run audiomc --out "$1" --seeds "$SEEDS" --ids "$(audiomc_ids)" --concurrency 3 && new "$IG" report audiomc --out "$1"; }
audiomc_cmp() { for arm in A B; do cmp_files "$1/minicpmo/$arm/episodes.jsonl" "$2/agent/$arm/episodes.jsonl" "episodes_$arm"; cmp_files "$1/minicpmo/$arm/agent_traces.jsonl" "$2/agent/$arm/agent_traces.jsonl" "traces_$arm"; done; cmp_files "$1/minicpmo/rows.json" "$2/agent/rows.json" rows --json; }

audiomc_cascaded_old() { local ids; ids=$(audiomc_ids); old examples/audiomc_ab.py run --data "$IG_AUDIOMC_DATA" --out "$1" --agent cascaded --seeds $SEEDS --ids "$ids" --concurrency 3 && old examples/audiomc_ab.py report --data "$IG_AUDIOMC_DATA" --out "$1" --agent cascaded; }
audiomc_cascaded_new() { new "$IG" run audiomc_cascaded --out "$1" --seeds "$SEEDS" --ids "$(audiomc_ids)" --concurrency 3 && new "$IG" report audiomc_cascaded --out "$1"; }
audiomc_cascaded_cmp() { for arm in A B; do cmp_files "$1/cascaded/$arm/episodes.jsonl" "$2/agent/$arm/episodes.jsonl" "episodes_$arm"; done; cmp_files "$1/cascaded/rows.json" "$2/agent/rows.json" rows --json; }

# FD-Bench v2: prepare (reference scripts from the LLM) + run (A: 120 s, B: closing rules) + report
fdb2_old() { old examples/fdb2_ab.py prepare --data "$IG_FDB2_DATA" --out "$1" --limit $N --seeds $SEEDS && old examples/fdb2_ab.py run --data "$IG_FDB2_DATA" --out "$1" --agent minicpmo --limit $N --seeds $SEEDS --concurrency 3 && old examples/fdb2_ab.py report --data "$IG_FDB2_DATA" --out "$1" --agent minicpmo; }
fdb2_new() { new "$IG" prepare fdb2 --out "$1" --limit $N --seeds "$SEEDS" && new "$IG" run fdb2 --out "$1" --limit $N --seeds "$SEEDS" --concurrency 3 && new "$IG" report fdb2 --out "$1"; }
fdb2_cmp() { for arm in A B; do cmp_files "$1/minicpmo/$arm/episodes.jsonl" "$2/agent/$arm/episodes.jsonl" "episodes_$arm"; cmp_files "$1/minicpmo/$arm/agent_traces.jsonl" "$2/agent/$arm/agent_traces.jsonl" "traces_$arm"; done; cmp_files "$1/minicpmo/rows.json" "$2/agent/rows.json" rows --json; }

fdb2_cascaded_old() { old examples/fdb2_ab.py prepare --data "$IG_FDB2_DATA" --out "$1" --limit $N --seeds $SEEDS && old examples/fdb2_ab.py run --data "$IG_FDB2_DATA" --out "$1" --agent cascaded --limit $N --seeds $SEEDS --concurrency 3 && old examples/fdb2_ab.py report --data "$IG_FDB2_DATA" --out "$1" --agent cascaded; }
fdb2_cascaded_new() { new "$IG" prepare fdb2_cascaded --out "$1" --limit $N --seeds "$SEEDS" && new "$IG" run fdb2_cascaded --out "$1" --limit $N --seeds "$SEEDS" --concurrency 3 && new "$IG" report fdb2_cascaded --out "$1"; }
fdb2_cascaded_cmp() { for arm in A B; do cmp_files "$1/cascaded/$arm/episodes.jsonl" "$2/agent/$arm/episodes.jsonl" "episodes_$arm"; done; cmp_files "$1/cascaded/rows.json" "$2/agent/rows.json" rows --json; }

# FD-Bench v3: A replay / B live, the official cascaded agent, then the duplex agent
fdb3_old() { old examples/fdb3_ab.py run --data "$IG_FDB3_DATA" --out "$1" --agent cascaded-official --limit $N --seeds $SEEDS --concurrency 3 && old examples/fdb3_ab.py report --out "$1" --agent cascaded-official; }
fdb3_new() { new "$IG" run fdb3 --out "$1" --limit $N --seeds "$SEEDS" --concurrency 3 && new "$IG" report fdb3 --out "$1"; }
fdb3_cmp() { for arm in A B; do cmp_files "$1/cascaded-official/$arm/episodes.jsonl" "$2/agent/$arm/episodes.jsonl" "episodes_$arm"; done; cmp_files "$1/cascaded-official/rows.json" "$2/agent/rows.json" rows --json; }

fdb3_duplex_old() { old examples/fdb3_ab.py run --data "$IG_FDB3_DATA" --out "$1" --agent minicpmo --limit $N --seeds $SEEDS --concurrency 3 && old examples/fdb3_ab.py report --out "$1" --agent minicpmo; }
fdb3_duplex_new() { new "$IG" run fdb3_duplex --out "$1" --limit $N --seeds "$SEEDS" --concurrency 3 && new "$IG" report fdb3_duplex --out "$1"; }
fdb3_duplex_cmp() { for arm in A B; do cmp_files "$1/minicpmo/$arm/episodes.jsonl" "$2/agent/$arm/episodes.jsonl" "episodes_$arm"; cmp_files "$1/minicpmo/$arm/agent_traces.jsonl" "$2/agent/$arm/agent_traces.jsonl" "traces_$arm"; done; cmp_files "$1/minicpmo/rows.json" "$2/agent/rows.json" rows --json; }

# FD-Bench v1.0 / v1.5, Easy Turn, HumDial: a few samples per subset, the judges in the report
FDB_SUBSETS=${FDB_SUBSETS:-v1.0/synthetic_pause_handling,v1.5/user_backchannel}
fdb_baseline_old() { old examples/fdb_baseline.py run --data "$IG_FDB_DATA" --out "$1" --bench fdb --subsets "$FDB_SUBSETS" --limit $N --agent-urls "$IG_AGENT_URL" --sessions 3 && old examples/fdb_baseline.py report --data "$IG_FDB_DATA" --out "$1" --bench fdb --judge-url "$IG_LLM_URL" --judge-model "$IG_LLM_MODEL"; }
fdb_baseline_new() { new "$IG" run fdb_baseline --out "$1" --set "tasks.subsets=[$FDB_SUBSETS]" --set tasks.limit=$N --concurrency 3 && new "$IG" report fdb_baseline --out "$1"; }
fdb_baseline_cmp() { for s in ${FDB_SUBSETS//,/ }; do cmp_files "$1/$s/episodes.jsonl" "$2/$s/episodes.jsonl" "episodes_$s"; cmp_files "$1/$s/agent_traces.jsonl" "$2/$s/agent_traces.jsonl" "traces_$s"; cmp_files "$1/$s/per_sample.json" "$2/$s/per_sample.json" "per_sample_$s" --json; done; cmp_files "$1/summary.json" "$2/summary.json" summary --json; }

easy_turn_old() { old examples/fdb_baseline.py run --data "$IG_EASY_TURN_DATA" --out "$1" --bench easy_turn --limit $N --agent-urls "$IG_AGENT_URL" --sessions 3 && old examples/fdb_baseline.py report --data "$IG_EASY_TURN_DATA" --out "$1" --bench easy_turn; }
easy_turn_new() { new "$IG" run fdb_baseline --out "$1" --set tasks.bench=easy_turn --set "tasks.data=$IG_EASY_TURN_DATA" --set tasks.limit=$N --concurrency 3 && new "$IG" report fdb_baseline --out "$1" --set tasks.bench=easy_turn --set "tasks.data=$IG_EASY_TURN_DATA"; }
easy_turn_cmp() { for s in $(ls "$1" | grep -v summary); do cmp_files "$1/$s/episodes.jsonl" "$2/$s/episodes.jsonl" "episodes_$s"; done; cmp_files "$1/summary.json" "$2/summary.json" summary --json; }

humdial_old() { old examples/fdb_baseline.py run --data "$IG_HUMDIAL_DATA" --out "$1" --bench humdial --subsets humdial/en/ask,humdial/en/pause --limit $N --agent-urls "$IG_AGENT_URL" --sessions 3 && old examples/fdb_baseline.py report --data "$IG_HUMDIAL_DATA" --out "$1" --bench humdial --judge-url "$IG_LLM_URL" --judge-model "$IG_LLM_MODEL"; }
humdial_new() { new "$IG" run fdb_baseline --out "$1" --set tasks.bench=humdial --set "tasks.data=$IG_HUMDIAL_DATA" --set "tasks.subsets=[humdial/en/ask,humdial/en/pause]" --set tasks.limit=$N --concurrency 3 && new "$IG" report fdb_baseline --out "$1" --set tasks.bench=humdial --set "tasks.data=$IG_HUMDIAL_DATA"; }
humdial_cmp() { for s in humdial/en/ask humdial/en/pause; do cmp_files "$1/$s/episodes.jsonl" "$2/$s/episodes.jsonl" "episodes_$s"; done; cmp_files "$1/summary.json" "$2/summary.json" summary --json; }

ALL="suite_default suite_chat sop audiomc audiomc_cascaded fdb2 fdb2_cascaded fdb3 fdb3_duplex fdb_baseline easy_turn humdial"
[ $# -eq 0 ] && { echo "usage: $0 all | <stage>...  (stages: $ALL)"; exit 2; }
[ "$1" = all ] && set -- $ALL
for s in "$@"; do stage "$s"; done
log "all done: $PARITY_OUT"
for s in "$@"; do echo "== $s"; sed 's/^/   /' "$PARITY_OUT/$s/verdict.txt" 2>/dev/null; done
