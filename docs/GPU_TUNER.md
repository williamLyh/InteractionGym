# GPU tuner (`interaction_gym.gpu_tuner`)

The GPU tuner decides how many GPUs each serving service gets (and where), and how many concurrent
sessions / sequences each server admits (its concurrency cap).

An env run needs several inference services at once: the duplex agent (MiniCPM-o on vLLM-Omni,
2 GPUs per server, ≤4 sessions), the user-simulator LLM, TTS, voice-clone TTS, and later a
trainer. A bad split leaves GPUs idle while one service is the bottleneck. On an 8-GPU host we saw
this happen: the user LLM held 4 GPUs and sat idle while the agent was the bottleneck. The tuner
**measures** the running layout and moves GPUs toward the measured load.

## The loop (main path)

```
measure (real episodes, ramped concurrency) → analyse → propose a split → [apply + re-measure]*
```

1. **Measure** (`rebalance.measure`). Run real end-to-end episodes on the current layout and ramp
   env concurrency (2, 4, 8, … up to the agent session capacity, which is always tried) until
   episodes/hour stops rising (<5 %) or a service starts to queue. While the episodes run:
   - `Monitor` samples `nvidia-smi` (GPU util + memory per GPU, locally or over ssh) and each
     server's Prometheus `/metrics`: `num_requests_running` / `num_requests_waiting`, KV-cache
     usage. vLLM-Omni pipeline stages are not summed; the busiest stage counts. Queue depth is
     measured relative to a sample taken before the episodes start, because some servers report a
     constant non-zero `waiting` while idle and other users' load is not ours.
   - `Meter` wrapper clients record how long each episode waits on each service: agent step/ack
     time, user-LLM calls (with token usage), TTS and clone calls (seconds synthesized). No core
     code changes are needed: you wrap the clients the same way `examples/live_user.py` does.
   - The level with the best measured episodes/hour is the round's result.
2. **Analyse** (`rebalance.analyze`). For each service: busy GPUs (GPU util × GPUs, with a shared
   GPU's util split by the services' env-side busy time), util, memory, running/queued, KV, wait
   per episode, and wait share. The bottleneck is chosen in this order:
   1. the service that queues;
   2. else the service at ≥90 % GPU util;
   3. else the agent, if every session is taken and episodes/hour was still rising (sessions stay
      held while an episode waits on the LLM or TTS);
   4. else the service with the largest wait share.

   Services under 30 % util with no queue are marked idle.
3. **Propose** (`rebalance.propose`). First the cheaper action: if the bottleneck is at its
   concurrency cap with GPU util below `--cap-util` (0.6), raise its cap one step (see
   [Concurrency caps](#concurrency-caps)): a restart, no GPUs. Otherwise (`allocate`) split the GPUs in proportion to measured busy
   GPU-seconds. The bottleneck gets a 1.25× boost, because its saturation caps its measured load.
   A session-bound agent always gets at least one more replica, if one fits. The split is then
   rounded to whole replicas: each service starts with one replica, and the service furthest
   below its target gets one more replica that still fits. This repeats until nothing fits.
   Placement (`solve.place`) handles GPUs per replica (aligned blocks for TP/2-GPU servers),
   memory, pins, and co-location (fractional TTS replicas spread over free GPUs first, then pack
   by memory). The proposal also recommends an env concurrency, plus a projected episodes/hour
   from linear extrapolation. The projection is always labelled *projected, NOT measured*.
4. **Apply + re-measure** (flag-gated: `--allow-restart --ssh HOST --stop-cmd CMD`). `SSHLayout`
   copies the generated `launch.sh` to the host, runs *your* stop command, starts the new layout,
   waits for every port's `/v1/models`, and runs one warm-up episode per MiniCPM-o server (the
   first session after a start returns no audio). The loop stops when episodes/hour stops
   improving (<3 %), the bottleneck flips back, the proposal equals the current layout, the
   proposal projects no gain over the measured rate (<3 %), or the round limit is reached. It then restores the best *measured* layout.
   After every round it checks that the servers still answer; if they died (e.g. an external process
   killed them, as some shared clusters do), it restarts the layout and measures the round again. A failed start is retried too.

The output separates measured from projected numbers everywhere. The ramp, the per-service table
and the "best" figure are measured. The proposal's episodes/hour is projected. `layout.yaml`
records which one it is (`measured_episodes_per_hour` vs `projected_episodes_per_hour`).

## Running it

Run it on the GPU host itself. Lockstep episodes wait for one server round trip per 200 ms step, so
running through ssh tunnels from a laptop measures the tunnel, not the GPUs. `--ssh local` restarts
services on the same host without ssh (`--gpu-ssh HOST` / `--ssh HOST` still work from elsewhere).

```bash
# one round on the running layout + a proposal; no restart (a few minutes)
PYTHONPATH=src:. python examples/gpu_tuner.py loop \
    --layout "agent=4+5,llm=0+1,tts=6/7,clone=6" --out runs/gpu_tuner

# full loop: apply proposals and re-measure (RESTARTS SERVICES; only when the host is yours)
PYTHONPATH=src:. python examples/gpu_tuner.py loop --layout "agent=4+5,llm=0+1,tts=6/7,clone=6" \
    --levels 2,4,8 --seconds 180 --rounds 3 --allow-restart --ssh local \
    --stop-cmd 'tmux ls -F "#S" | grep "^dig_" | xargs -r -I{} tmux kill-session -t ={}' --out runs/gpu_tuner

# the same from another machine: --ssh / --gpu-ssh name the GPU host (here `gpu-host`)
PYTHONPATH=src:. python examples/gpu_tuner.py loop --layout "agent=4+5,llm=0+1,tts=6/7,clone=6" \
    --allow-restart --ssh gpu-host --gpu-ssh gpu-host \
    --stop-cmd 'tmux ls -F "#S" | grep "^dig_" | xargs -r -I{} tmux kill-session -t ={}' --out runs/gpu_tuner
```

The stop command must stop every server of the current layout: after the first restart the
servers are the tuner's own sessions (`dig_agent0`, `dig_tts1`, `dig_cloneproxy`, ...), which `./stop.sh`
does not know. The prefix is `dig_` by default; a preset can change it with `session_prefix`. After the stop command the tuner waits until the next layout's GPUs are free (killed vLLM
workers release memory only after a while) and fails at once, with the log tail, if a server
exits while loading.

`--layout` names the running layout and its GPUs. Replicas are separated by `/` and a replica's
GPUs are joined by `+`. Endpoints and launch commands come from a host preset, selected with `--host`
(a TOML path or a name in `gpu_tuner/hosts/`; default `example-8gpu`; `examples/gpu_tuner.py` defaults to
`example-8gpu-thinker`, or `example-8gpu` with `--audio-out`). The shipped
[`hosts/example-8gpu.toml`](../src/interaction_gym/gpu_tuner/hosts/example-8gpu.toml) is a generic
8× 32 GB example that mirrors the `scripts/` of the reference deployment in
[`examples/serving/reference/`](../examples/serving/reference/), with two-GPU MiniCPM-o audio servers;
[`hosts/example-8gpu-thinker.toml`](../src/interaction_gym/gpu_tuner/hosts/example-8gpu-thinker.toml) is the
same with one-GPU Thinker-only servers (16 sessions each), for the text-only episodes that are the runners' default
(`examples/gpu_tuner.py` runs them unless `--audio-out`; layouts like `agent=2/3/4/5`). The measurements below are
for audio episodes. Its `workdir` is a placeholder
(`/opt/dig_services`): set it to the directory that holds that deployment on your host. Copy the preset for
another host. Useful flags:

| flag | what it does |
|---|---|
| `--levels` | the concurrency ramp |
| `--seconds` | time per level. Give each worker time to finish at least one episode. |
| `--max-concurrency` | keeps the load light on shared services |
| `--agent canned` | user-side services only, no duplex server needed |
| `--no-clone` | skip the clone-TTS service |
| `--no-gpu` | no GPU sampling. The split then follows wait share × GPUs held. |
| `--reserve trainer=1` | holds a trainer's GPUs out of the split |
| `--cap-util`, `--cap-gain`, `--mem-limit`, `--max-failures` | the cap rule and its rollback guards (0.6, 0.05, 0.92, 0) |
| `--no-caps` | tune GPU counts only |

Outputs go to `--out`:

- `layout.yaml`: replicas, GPU ids, ports, URLs, recommended `concurrent_episodes`, and
  `IG_*` exports. With more than one agent server, `IG_AGENT_URLS` lists them all: MiniCPM-o
  duplex cannot be data-parallel behind one API server, so each 2-GPU replica gets its own port.
- `launch.sh`: one tmux session per server (`dig_<name>`), on the GPUs the plan gives it. TTS
  replicas sit behind `scripts/tts_proxy.py`. It only *starts* servers and never stops anything.
- `report.txt`: the round tables and the proposal.

### Dry run (laptop)

```bash
uv run python -m interaction_gym.gpu_tuner dry-run --out runs/tune_dry
uv run python -m interaction_gym.gpu_tuner dry-run --no-apply   # one round + proposal only
```

This runs the same loop against `MockCluster`. Its fake services have configurable capacity:
per-stream speed vs load, batch slots that queue beyond capacity, and a session cap. They report
fake GPU util and queue metrics through the same `Monitor` interface. The mock's "truth" is
harsher than its inputs (co-located replicas halve each other's speed, the LLM is 10 % slower).
Starting from the example layout (`EXAMPLE_LAYOUT`: agent on 4+5, LLM on 0-3, TTS on 6 and 7,
clone sharing GPU 6), it finds the agent session-bound with the LLM idle. It moves to
`agent=0+1/2+3, llm=4+5, tts=6, clone=7` and measures roughly 2× the episodes/hour (≈420 →
≈860 mock ep/h at N=8). Example output: see the end of this file.

## Concurrency caps

A service's per-replica concurrency limit is a knob like its GPU count: MiniCPM-o's
`duplex_session.max_sessions` (sessions per server; `MAX_SESSIONS=n scripts/run_minicpmo.sh` writes a copy
of the deploy yaml with `max_sessions` and every stage's `max_num_seqs` set to n), the LLM's
`--max-num-seqs` (`MAX_NUM_SEQS=n scripts/run_llm.sh`) and the clone TTS's stage `max_num_seqs`
(`MAX_NUM_SEQS=n scripts/run_tts_model.sh ...`; `run_tts.sh` has no such knob). In the host preset a service
has `cap` (or `max_sessions` for the agent), `cap_steps` (the ladder the tuner may climb, agent 4, 6, 8, 12)
and `{cap}` in its launch template. In `--layout` and in the reports a cap other than the preset's is
written `@n`: `agent=0+1/2+3@8` is two MiniCPM-o servers with 8 sessions each.

Rule (`propose`): the bottleneck is **concurrency-bound** when its cap is reached and work waits. For the
agent that means every session taken (N = servers × sessions) and either stage requests queue or episodes/hour
was still rising. For the others it means requests queue while about cap × replicas run. Then:

- GPU util < `--cap-util` (0.6): raise its cap one step, before adding replicas. The GPUs are mostly idle,
  so more sessions per server cost a restart and some memory, not GPUs. Projected episodes/hour scale with
  the session capacity, limited by its GPUs (busy GPU-seconds per episode as measured).
- GPU util already high, the cap at its last step, or the cap frozen by a failed raise: more replicas, as
  before. Each proposal says which action it took and why (`why:` in the report).

A cap round re-measures from the previous round's best concurrency up to the new session capacity, e.g.
N=8 and 12 after raising 4 → 6 on two servers. The ramp always reaches the session capacity: when a service
queues below it, the ramp jumps straight to it. A raise is **rolled back** (and that cap frozen) when:

- episodes/hour gained less than `--cap-gain` (5 %);
- a GPU's memory peak went above `--mem-limit` (92 %) and higher than in the previous round (the
  Talker/Code2Wav GPU fills up per session; vLLM's static preallocation of ~90 % does not count);
- episodes failed (lockstep ack timeouts, closed sessions; `--max-failures`, default 0);
- the server logs written during the round show a stage eviction or OOM (`no live replica`, `is dead`,
  `CUDA out of memory`);
- the layout fails to start (e.g. out of memory at load).

The loop then restores the best measured setting and goes on from it (a replica move may still help).
`--no-caps` turns the knob off. The mock dry run has a session-bound agent for this:
`python -m interaction_gym.gpu_tuner dry-run --session-bound --rounds 6` (starting from `EXAMPLE_TUNED_LAYOUT`) raises the cap 4 → 6 → 8 → 12,
rejects 12 (its mock agent GPUs pass 92 % memory) and leaves 8.

The episode runner (`examples/gpu_tuner.py`) puts each episode on the agent server with the fewest open
sessions. Round robin by episode index drifts, and with N = servers × cap one server would get one session too
many and refuse it.

vLLM-Omni and MiniCPM-o `max_sessions` (vLLM-Omni upstream main + our duplex patches): no hard limit. The only check is
admission in `engine/duplex/session/manager.py` (`duplex_session_capacity_exhausted` beyond `max_sessions`), and
`supports_multi_session` just means `max_sessions > 1`. It is not checked against the stages' `max_num_seqs`, so
sessions beyond a stage's `max_num_seqs` queue in that stage's scheduler; `run_minicpmo.sh` moves them together.
Memory grows with the cap. Stage 2 (Code2Wav) reserves one Whole-Euler CFM attention cache per micro-batch row,
`min(max_num_seqs, 16)` rows, and the Thinker's KV is preallocated at `gpu_memory_utilization` 0.9. Its sliding
window bounds each session's blocks, so the sessions share it. The Whisper streaming cache and Talker/Code2Wav
state are per session.

## Predictive fallback

Use this when episodes cannot be run, e.g. to rank layouts before the first restart. You need
per-episode demand (`Meter` from a live run, or `DemandProfile.from_episodes` on saved runs) and a
per-replica supply curve per service (`probe.py`: short closed-loop bursts at several concurrency
levels). `solve.py` then enumerates replica counts, places them, and evaluates a lockstep model:
episode wall = agent compute + every awaited LLM/TTS call, Little's law for in-flight requests,
and the session cap. It returns the best allocation and env concurrency.

```bash
python -m interaction_gym.gpu_tuner demand --from runs/suite --out d.json
python -m interaction_gym.gpu_tuner supply --demand d.json --llm http://localhost:8000/v1 --tts http://localhost:8001/v1 --replicas tts=2 --out s.json
python -m interaction_gym.gpu_tuner predict --demand d.json --supply s.json --current agent=1,llm=2,tts=2,clone=1 --out runs/tune
```

On the synthetic profiles it agrees with the loop (agent=2, llm=1, tts=1, clone=1, N=8).

## Caveats

- On a shared host, other users' load shows up in GPU util. Queue depth is relative to a
  pre-run sample, but util is not. Measure when the services are yours.
- When services share a GPU, its util is split by env-side busy time, which approximates each
  service's share of the work. `nvidia-smi` does not give per-process SM utilisation.
- MiniCPM-o's `/metrics` counts stage requests, not duplex sessions. Session use comes from the
  ramp: the session capacity is reached while throughput is still rising.
- A projection assumes per-episode busy GPU-seconds and episode wall stay the same. That is why
  applying the proposal and re-measuring is the real test.
- The generated `launch.sh` reproduces `scripts/run_*.sh` with GPU ids and ports as parameters.
  Check the host preset before using it on another host.

## Real run (8× RTX 5090 32 GB, 2026-10-04)

Measured on one host with 8× RTX 5090 32 GB: MiniCPM-o 4.5 agents on vLLM-Omni (audio out, lockstep), live user LLM (Qwen3.8-27B FP8) + TTS + clone
TTS, the four suite scenarios, 90 s episode cap, `--levels 2,4,8 --seconds 180`, three restarts by
the generated launch scripts (start + 2 proposals, 6–7.5 min each until every port answered).

```
  1. agent=4+5,llm=0+1,tts=6/7,clone=6            447 ep/h measured at N=4   bottleneck agent
  2. agent=0+1/2+3,llm=4+5,tts=6,clone=7          687 ep/h measured at N=8   bottleneck agent
  3. agent=0+1/2+3,llm=4+5,tts=6,clone=7/6        705 ep/h measured at N=8   bottleneck agent
```

| round | agent util / queue / wait s/ep (share) | llm util / wait | tts util / wait | clone util / wait |
|---|---|---|---|---|
| 1 | 40 % / 1.5 / 27.0 (85 %) | 13 % / 1.75 s | 3 % / 0.42 s | 36 % / 1.98 s |
| 2 | 32 % / 2.1 / 30.1 (80 %) | 27 % / 1.84 s | 4 % / 0.41 s | 31 % / 1.74 s |
| 3 | 33 % / 2.1 / 28.3 (77 %) | 25 % / 1.91 s | 6 % / 0.42 s | 21 % / 1.66 s |

The agent was the bottleneck in every round: all its sessions taken and stage requests queued,
while its GPUs stayed at 32–40 % util (the 4-session cap per server and the per-unit pipeline bound
it, not compute). Doubling the agent servers gave +54 % measured (projected +100 %: episodes waited
longer per unit with 4 busy sessions per server). The third proposal only added a clone replica,
projected at the measured rate; the loop now stops instead of restarting for that.

### Session caps (same layout, 240 s per level)

Since the agent stayed session-bound at low util, a second run raised the per-server session cap
(`MAX_SESSIONS`, which also sets every stage's `max_num_seqs`) instead of adding GPUs:

| cap per server | sessions | ep/h measured | agent util | peak GPU mem | queue | failed | result |
|---|---|---|---|---|---|---|---|
| 4 | 8 | 500 (N=4), **703** (N=8) | 35 % | 93 % (LLM preallocation) | 2.4 | 0 | baseline |
| 6 | 12 | 724 (N=8), **788** (N=12) | 33 % | 93 % | 6.0 | 0 | **kept** |
| 8 | 16 | 860 (N=12), 460 (N=16) | – | 98 % on agent GPUs | 9.4 | 24 (N=16) | rejected |

Cap 6 gives +12 % (projected +50 %); the agent stays the bottleneck with stage requests queueing
and its GPUs about a third busy, so the limit is the per-unit pipeline, not session count. At cap 8
with 16 episodes, Code2Wav on one server ran out of memory ("stage 2 has no live replica") and every
later session timed out; the memory and log guards rolled it back. The practical limit on these
32 GB cards is Talker/Code2Wav memory, between 6 and 8 sessions per server (per-session Whisper
streaming cache and Talker/Code2Wav state; Code2Wav reserves an attention cache per batch slot).
vLLM-Omni has no hard cap on `max_sessions` and does not check it against the stages' `max_num_seqs`.
To run the final layout (cap 6, 12 concurrent episodes), use the `launch.sh` the tuner writes, or pass the
layout to the reference `start.sh` through `serving.env` (e.g. `AGENT_SERVERS="0,1:8010 2,3:8011"`,
see [`examples/serving/reference/`](../examples/serving/reference/)).

## Example (dry run, mock cluster)

```
Round 1: agent=4+5,llm=0+1/2+3,tts=6/7,clone=6
  ramp (measured): N=2:226  N=4:413 ep/h  (ramp finished)
  best: 413 episodes/h at N=4 (session capacity 4); bottleneck: agent — all 4 agent sessions in use and episodes/hour still rising (sessions are held while episodes wait on other services); idle: llm, tts
  service  repl GPUs            held  busy  util mem GB   run queue    KV wait s/ep  share calls/ep
  agent       1 4,5             2.00  1.48   74%   57.6   2.2   0.0     -     18.51    53%      0.0
  llm         2 0,1,2,3         4.00  0.10    3%  115.2   0.8   0.0     -      7.30    21%     12.0
  tts         2 6,7             1.50  0.09    6%   24.0   0.1   0.0     -      1.10     3%      6.0
  clone       1 6               0.50  0.26   52%   12.0   0.6   0.0     -      4.78    14%      6.0
Proposal (split proportional to busy GPU-seconds (nvidia-smi)):
  service  current            proposed           target GPUs GPU-eq now GPU-eq new
  agent    4+5                0+1/2+3                   6.43       2.00       4.00
  llm      0+1/2+3            4+5                       0.36       4.00       2.00
  tts      6/7                6                         0.30       1.50       1.00
  clone    6                  7                         0.91       0.50       1.00
  projected (linear extrapolation, NOT measured): 827 ep/h at N=8 vs 413 measured now  [limits: agent sessions 827, agent 1007, clone 1423, tts 4359, llm 7101]

Rounds:
  1. agent=4+5,llm=0+1/2+3,tts=6/7,clone=6        413 ep/h measured at N=4   bottleneck agent
  2. agent=0+1/2+3,llm=4+5,tts=6,clone=7          856 ep/h measured at N=8   bottleneck agent
Recommendation: agent=0+1/2+3,llm=4+5,tts=6,clone=7 at N=8 (measured) — the measured load already matches the layout (proposal = current)
```
