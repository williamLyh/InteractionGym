# Parity runs: the `main` scripts against the plugin bundles, on real model servers

The test suite proves the bundles equal the scripts frame for frame on fakes. This folder proves it on the real
servers (the user LLM, the two Qwen3-TTS voices, Qwen3-ASR, the MiniCPM-o 4.5 duplex server) without depending on
the servers being deterministic: a **record / replay proxy** sits in front of every server.

1. `run_parity.sh <stage>` runs the `main` script of a benchmark through `proxy.py --mode record`: every HTTP
   request is stored by method, path and canonical body; every WebSocket session as its client messages with the
   server messages that followed each.
2. The bundle runs through the same store in `--mode replay`. An answer only exists for a request the script made
   with exactly the same body (prompt, sampling fields, seed, audio bytes); anything else is a **miss**, logged with
   a diff against the nearest recorded request, and the run fails at once (HTTP 460 / a WebSocket `error`).
   Identical requests repeated get their recorded answers in order; more repetitions than recorded are counted as
   `overflow` (the new run makes calls the old one did not, e.g. a warm-up or a lost cache).
3. `compare.py` joins the two runs' episodes by task, seed and arm (the scripts' and `ig run`'s id schemes unified)
   and compares every field — turns, timing, scores, judge verdicts, duplex metrics, agent traces, report rows —
   exactly. Fields that cannot agree between two runs are ignored (`run_id`, wall times, measured `*_wall_ms`
   latencies, file paths); metadata the bundles add (`meta.config`, `meta.roles`, …) and the agent's self-description
   (`meta.agent`, each script had its own dict) are reported apart, not as a difference.
4. `report.py` makes one table over all stages (`PARITY_OUT/<stage>/verdict.txt`, `compare_*.json`).

`LIVE=1` adds a third leg: the bundle live against the servers too (its own record store), compared with the old
run — what differs there is sampling, not behaviour.

Stages (`run_parity.sh all`): `suite_default`, `suite_chat`, `sop`, `audiomc`, `audiomc_cascaded`, `fdb2`,
`fdb2_cascaded`, `fdb3`, `fdb3_duplex`, `fdb_baseline`, `easy_turn`, `humdial` — every `examples/*.py` runner that
became a bundle, with a few tasks each (`N`), both arms where the script has them, and the report with its judges.
`REUSE_OLD=1` keeps a stage's recording and only replays; `FORCE=1` redoes a stage that has a verdict.

The 2026-10-26 runs on Isambard (one GH200 node: `~/ig_parity/stack.sbatch` starts the servers, `run_stage.sh`
runs stages inside the job) are summarised in `docs/ARCHITECTURE.md` §8 ("found by the parity runs").
