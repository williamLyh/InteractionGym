# Cross-benchmark results: full-audio run (2026-10-08/09, appendix)

> **Appendix, superseded as the release results.** The release results are now [BENCHMARKS.md](BENCHMARKS.md):
> MiniCPM-o 4.5 Thinker-only (text output timed at `speech_cps`), benchmark users without a background floor, and the
> logistic latency curve, on FD-Bench v3 / v2 and Audio MultiChallenge, open vs closed loop. This file keeps the
> earlier **full-audio** run unchanged: MiniCPM-o with Talker audio on the 2-GPU deployment (today's `--audio-out`),
> closed-loop users with a −60 dBFS background floor (absent in the replayed A conditions of FD-Bench v2 / v3), the
> cascaded and `minicpmo-confirm` agents, and the open-loop-only benchmarks (FD-Bench v1.0 / v1.5, Easy Turn,
> HumDial). Its numbers are as published on 2026-10-09 (scored with the log-normal latency curve of that time), except
> [§0](#0-re-scored-with-the-logistic-latency-curve), which re-scores its `eval.scores` means with the current curve.
> Not bit-identical to, and not directly comparable with, the Thinker-only release run.

These are the results of evaluating duplex agents on the benchmarks integrated in InteractionGym
([docs/BENCHMARKS.md](../docs/BENCHMARKS.md)). Four public benchmarks are run open loop with their official protocols as
converted. Three more are run both **open loop** (fixed user input, as a static benchmark plays it) and **closed loop**
(the same opening, then a simulated user that reacts to the agent). Every benchmark was run on 2026-10-08/09 with the
environment of that time ([§1](#1-setup); changes against the run before it in [§9](#9-comparison-with-the-previous-run)).

Results only: no benchmark data is redistributed here (licenses in [§10](#10-limitations)). Aggregate numbers are in
[`data/summary_full_audio.json`](data/summary_full_audio.json) (made by `aggregate.py` at commit `bdc6089`).

Contents: [0 Re-scored](#0-re-scored-with-the-logistic-latency-curve) · [1 Setup](#1-setup) · [2 Summary table](#2-summary-table) · [3 Full-Duplex-Bench v1.0 / v1.5](#3-full-duplex-bench-v10--v15-open-loop)
· [4 Easy Turn](#4-easy-turn-testset-open-loop) · [5 HumDial-FDBench](#5-humdial-fdbench-open-loop) ·
[6 FD-Bench v3](#6-full-duplex-bench-v3-open-vs-closed-loop) · [7 FD-Bench v2](#7-full-duplex-bench-v2-open-vs-closed-loop) ·
[8 Audio MultiChallenge](#8-audio-multichallenge-open-vs-closed-loop) · [9 Comparison with the previous run](#9-comparison-with-the-previous-run) ·
[10 Limitations](#10-limitations)

## 0. Re-scored with the logistic latency curve

`eval.scores` was recomputed from the saved turns of this run's episodes with the current `eval.latency_score`
(logistic in ms, half score at 950 ms: 500 ms 0.99, 1 s 0.38; docs/FORMAT.md §6.3) next to the log-normal curve of
the time (1 at 200 ms; 500 ms 0.66, 1 s 0.27) with `results/rescore_full_audio.py`. Only latency-graded outcomes
(`responded`, `yielded`) change; every outcome rate, official metric and judge score below is unchanged. The "mean
score" columns of §2–§8 are the old curve.

| benchmark · agent · condition | episodes | scored user turns | mean score, old curve | new curve |
|---|---|---|---|---|
| audiomc · minicpmo · A | 100 | 493 | 0.284 | 0.368 |
| audiomc · minicpmo · B | 100 | 490 | 0.216 | 0.244 |
| fdb2 · cascaded · A | 400 | 2241 | 0.203 | 0.222 |
| fdb2 · cascaded · B | 400 | 2185 | 0.105 | 0.002 |
| fdb2 · minicpmo-confirm · A | 400 | 2252 | 0.237 | 0.308 |
| fdb2 · minicpmo-confirm · B | 400 | 2429 | 0.208 | 0.238 |
| fdb2 · minicpmo · A | 400 | 2254 | 0.167 | 0.216 |
| fdb2 · minicpmo · B | 400 | 2036 | 0.209 | 0.232 |
| fdb3 · cascaded-natural · A | 300 | 300 | 0.053 | 0.021 |
| fdb3 · cascaded-natural · B | 300 | 818 | 0.084 | 0.014 |
| fdb3 · cascaded-official-ep3000 · A | 300 | 300 | 0.007 | 0.000 |
| fdb3 · cascaded-official-ep3000 · B | 300 | 767 | 0.010 | 0.000 |
| fdb3 · cascaded-official · A | 300 | 300 | 0.054 | 0.018 |
| fdb3 · cascaded-official · B | 300 | 779 | 0.085 | 0.019 |
| fdb3 · minicpmo · A | 300 | 300 | 0.446 | 0.642 |
| fdb3 · minicpmo · B | 300 | 1346 | 0.228 | 0.288 |

Open-loop-only benchmarks (all subsets pooled, per scored user turn):

| benchmark | scored user turns | old curve | new curve |
|---|---|---|---|
| FD-Bench v1.0 | 1073 | 0.429 | 0.479 |
| FD-Bench v1.5 | 996 | 0.416 | 0.501 |
| Easy Turn | 1000 | 0.319 | 0.407 |
| HumDial (en + zh) | 1051 | 0.242 | 0.255 |

MiniCPM-o's scores rise most where it answers in 0.5–0.9 s (FD-Bench v3 A: 0.45 → 0.64). The cascaded agent's
fall to about 0 on FD-Bench v3 and in FD-Bench v2 B: its replies come 1.6–2.6 s after the user (800 ms endpointing plus
the fixed latency model), where the new curve gives ~0 and the old one still 0.05–0.15.

## 1. Setup

**Environment.**

- **Episodes** were produced with InteractionGym `main` at **`6f6853d`**.
- **Scoring** used **`ad72ce9`**: `eval.scores` now gives one score per user turn (`86cc52c`), and the FD-Bench ports
  live in the optional component `extras/fdbench/` (`e73cfc7`).
- **Re-scoring.** Every episode's `eval.duplex` / `eval.scores` was recomputed from its saved turns at `ad72ce9` (8,023
  episodes) before the official reports and `aggregate.py` ran. `aggregate.py` recomputes `eval.scores` from the turns
  itself as well.

The user-side defaults now on, all of them since the previous run:

- The simulated user's voice is **cloned from its first turn** (Qwen3-TTS Base) for every later turn, with a neutral
  reference style, trimming and loudness leveling, and a per-episode TTS seed (`e090ea7`). The FD-Bench v2 examiner is
  cloned the same way, both live (B) and in the regenerated static scripts (A). The FD-Bench v3 and Audio
  MultiChallenge users clone the recorded first turn.
- The soundscape (`33ad813`): an always-on background track from the persona's surroundings, with sound-bank
  recordings (a local DEMAND / MUSAN bank built with `scripts/fetch_noise_banks.py`) passed to every `UserSim`.
- LLM-decided user behaviours (`c946419`).

The benchmark users are "quiet" personas with no random events. In practice the soundscape is therefore a **−60 dBFS
pink-noise floor** under the closed-loop user (the sound bank is not drawn on). It is **absent in the replayed A
conditions of FD-Bench v2 / v3**; see the caveat in [§6](#6-full-duplex-bench-v3-open-vs-closed-loop).

Other settings: lockstep simulation (`clock="input"`) at 16 kHz. **Gander is not included.**

**Hardware.** Two hosts with 8× RTX 5090 (32 GB) each, with the same layout on both:

- GPUs 0–3: two MiniCPM-o servers.
- GPUs 4–5: user / examiner / judge LLM.
- GPUs 6–7: two Qwen3-TTS CustomVoice replicas, two Qwen3-TTS Base (clone) replicas and Qwen3-ASR.

**Agents.**

| name | what | used on |
|---|---|---|
| `minicpmo` | MiniCPM-o 4.5 full-duplex, audio output, token trace. Thinker greedy, Talker sampled with a fixed seed. System prompt: none on the open-loop sets (as in the official inference scripts); the official examinee prompt on FD-Bench v2; a per-domain voice-assistant prompt asking it to confirm every detail on FD-Bench v3 ([§6](#6-full-duplex-bench-v3-open-vs-closed-loop)); an official-style examinee prompt on Audio MultiChallenge | all |
| `minicpmo-confirm` | the same model with a voice-assistant prompt that confirms details and uses corrected values | FD-Bench v2 |
| `cascaded` / `cascaded-official` | energy-VAD endpointing (800 ms of silence) → Qwen3-ASR-1.7B → Qwen3.8-27B-FP8 → Qwen3-TTS. Fixed latency model (ASR 150 ms + 10 ms/s, LLM 350 ms + 15 ms/token, TTS 250 ms); model calls memoized per (input, seed). On FD-Bench v3: the official cascaded agent's instructions and tool schemas (tools called from text markup). On FD-Bench v2: no tools, official examinee prompt + "keep it short" | FD-Bench v2, v3 |
| `cascaded-official-ep3000` | `cascaded-official` with 3 s endpointing (a "patient" ablation) | FD-Bench v3 |
| `cascaded-natural` | the cascaded agent with a natural prompt ("if something is missing, ask briefly") | FD-Bench v3 |

**MiniCPM-o deployment** (agent server).

- **Software:** vLLM 0.30.0 + vLLM-Omni `0.30.1.dev97+ge7c7dac58`, i.e. upstream `main` plus our duplex patches (input
  clock, `silence_continuation`, token trace; branch `duplex-input-clock`).
- **Overlay:** `minicpmo_omni`, which includes the per-session audio feature-extractor fix (upstream PR #8638;
  `shared_fe` checked in `minicpmo_4_5/duplex/stage0.py` at every server start). Before that fix, every session's
  reference audio was normalised with the log-mel floor that the last processed chunk had left in a shared processor.
  The fix decouples sessions on one server.
- **Layout:** full audio pipeline, deploy config `minicpmo_cfm8.yaml`. Thinker on one GPU; Talker + Code2Wav on the
  other (Talker 0.30, CFM graphs ≤ 8, HiFT lazy graphs 0).
- **Sessions:** 2 GPUs and **6 concurrent sessions** per server (`max_sessions` 6). Four servers ran over the two hosts.
  Our RL rollouts use a thinker-only (text) deployment instead; the numbers here are the full-audio
  agent.
- **Other services:** user / examiner LLM vLLM 0.30.0; TTS / clone vLLM-Omni 0.30.0.

**Simulated users, examiners and judges.** All LLM roles are a local **Qwen3.8-27B-FP8**, at T = 0 for the judges:

- the FD-Bench v3 closed-loop user;
- the FD-Bench v2 examiner (proxy for GPT-Realtime), its reference assistant and stage tracker;
- the Audio MultiChallenge replanning user;
- every judge: FD-Bench v1.0 relevance (proxy for GPT-4-turbo), v1.5 behaviour (proxy for GPT-4o), v2 (proxy for
  Gemini 2.5 Flash), v3 arguments / response quality (proxy for GPT-4o), Audio MultiChallenge rubric (proxy for o4-mini).

**All judge-based numbers are proxy numbers, not comparable with published tables.** Voices: Qwen3-TTS-12Hz-1.7B
CustomVoice for the examiner's first line and the cascaded agent; Base for cloning.

**Runs, failures, cost.**

- **Scale:** 8,023 episodes (counts per benchmark below). Every runner resumed to completion with **0 episodes given
  up**. There were 13 transient episode retries: 12 sessions refused by MiniCPM-o with `resource_exhausted` (a new
  session opened before the previous one was released at the 6-session cap) and one LLM HTTP 500.
- **No service restarts were needed during the runs.** A watchdog was in place to restart services on these
  hosts, but none was needed.
- **Wall time:** 5.0 h for the runs on host 1, 5.5 h on host 2, and 1.8 h for the official reports (judges) on host 1.
- **Cost:** about **94 GPU-hours** reserved (16 GPUs), of which MiniCPM-o servers took 19.4 server-hours (39 GPU-hours).
  The previous run cost about the same (about 94 GPU-hours) but on one host in about 11.7 h.

**Statistics.** 95% CIs are percentile bootstraps (2,000 resamples). They are computed over samples, or over tasks /
recordings when a task has several seeds. A vs B differences are paired on (task, seed). Rates in square brackets are
[lo, hi].

**Scores.**

- **`eval.scores`** ([docs/FORMAT.md §6.3](../docs/FORMAT.md)) gives each user turn **one** rule-based score in [0, 1]:
  did the agent do what that turn expected? The expectations are respond, yield, wait (stay quiet: a mid-thought pause,
  "hold on"), ignore (a backchannel, aside or noise) and interrupt. Where doing it means taking the floor, the score is
  graded by latency: `latency_score`, best at 200 ms, log-normal. "Score" below is its mean over scored user turns, so
  multi-second replies (the cascaded agent) score low by construction. Outcome rates (e.g. "yielded", "waited") are
  shares of the turns of one expectation.
- **Duplex timing** (`eval.timing_counts`, [docs/BENCHMARKS.md §12](../docs/BENCHMARKS.md); unchanged definitions):
  - turn-take rate: share of directed user turns the agent answers within 5 s;
  - latency: end of user turn → agent reply (median);
  - yield rate: share of intended barge-ins on which the agent stops (barge-ins are user turns meant to overlap:
    interruptions, "stop");
  - cut-in rate: share of directed user turns the agent talks into.

  A user turn that overlaps the agent only because a replayed line's fixed start time fell inside the agent's speech
  is a **collision**: reported apart, never counted in yield or cut-in.

## 2. Summary table

Open loop = fixed user input (A); closed loop = reacting user / examiner (B). n = samples (conversations × seeds).
"—" = not run / not applicable. ↑ / ↓ = higher / lower is better.

| benchmark | metric | n | MiniCPM-o 4.5 · open | MiniCPM-o 4.5 · closed | MiniCPM-o-confirm · open | MiniCPM-o-confirm · closed | Cascaded · open | Cascaded · closed |
|---|---|---|---|---|---|---|---|---|
| FD-Bench v1.0 pause (Candor) | TOR ↓ | 216 | 0.31 [0.25, 0.37] | — | — | — | — | — |
| FD-Bench v1.0 pause (synthetic) | TOR ↓ | 137 | 0.15 [0.09, 0.21] | — | — | — | — | — |
| FD-Bench v1.0 turn-taking | TOR ↑ / latency s ↓ | 119 | 0.96 [0.92, 0.99] / 1.28 [1.13, 1.42] | — | — | — | — | — |
| FD-Bench v1.0 backchannel | JSD ↓ (freq) | 55 | 1.00 (0.00) | — | — | — | — | — |
| FD-Bench v1.0 interruption | TOR ↑ / latency s / relevance 0–5 (proxy) | 200 | 0.94 [0.91, 0.97] / 1.53 [1.40, 1.65] / 4.44 [4.22, 4.62] | — | — | — | — | — |
| FD-Bench v1.5 user interruption | C_RESPOND (proxy) / stop s / response s | 200 | 0.77 [0.71, 0.83] / 1.51 [1.39, 1.61] / 1.93 [1.82, 2.03] | — | — | — | — | — |
| FD-Bench v1.5 backchannel | C_RESUME (proxy) / stop s | 98 | 1.00 / 0.61 [0.57, 0.65] | — | — | — | — | — |
| FD-Bench v1.5 talking to other | C_RESPOND (proxy) ↓ | 100 | 0.56 [0.46, 0.66] | — | — | — | — | — |
| FD-Bench v1.5 background speech | C_RESPOND (proxy) ↓ | 100 | 0.50 [0.40, 0.60] | — | — | — | — | — |
| FD-Bench v1.0 + v1.5 interruptions | yield rate ↑ | 400 | 0.60 (v1.0 0.600 [0.515, 0.679]; v1.5 0.602 [0.533, 0.670]) | — | — | — | — | — |
| Easy Turn complete | responded ↑ / latency ms | 300 | 0.990 [0.977, 1.000] / 850 [810, 920] | — | — | — | — | — |
| Easy Turn incomplete | waited (did not take the floor) ↑ | 300 | 0.003 [0.000, 0.010] | — | — | — | — | — |
| Easy Turn wait ("别说了") | yield rate ↑ / then stayed quiet ↑ | 100 | 0.39 [0.30, 0.50] / 0.08 [0.03, 0.14] | — | — | — | — | — |
| Easy Turn backchannel | ignored ↑ | 100 | 1.00 | — | — | — | — | — |
| HumDial ask / repeat / shift (en + zh) | yield rate ↑ | 150 | 0.88 (per category 0.70–1.00, §5) | — | — | — | — | — |
| HumDial deny / wait (en + zh) | yield rate ↑ | 100 | 0.18 (per category 0.08–0.33, §5) | — | — | — | — | — |
| HumDial pause (en + zh) | waited in the pause ↑ | 50 | 0.44 (en 0.44, zh 0.44) | — | — | — | — | — |
| FD-Bench v3 (tool use) | spoken fulfilment ↑ (MiniCPM-o) / Pass@1 ↑ (cascaded, official judge prompt) | 300 | 0.817 [0.740, 0.890] | 0.983 [0.963, 0.997] | — | — | 0.620 [0.527, 0.703] | 0.720 [0.637, 0.797] |
| FD-Bench v3 | B − A | 300 | | +0.167 [+0.093, +0.243] | | | | +0.100 [+0.050, +0.153] |
| FD-Bench v3 | interruption rate (Δt < 0) / first response ms (median) | 300 | 0.110 / 1020 | 0.050 / 1340 † | — | — | 0.213 / 2629 | 0.207 / 2651 |
| FD-Bench v2 (all splits, B 180 s) | IF (official, proxy) ↑ | 400 | 3.93 [3.85, 4.01] | 4.26 [4.19, 4.33] | 3.92 [3.83, 4.00] | 4.28 [4.20, 4.35] | 3.25 [3.12, 3.39] | 4.14 [4.04, 4.23] |
| FD-Bench v2 | TT (official, proxy) ↑ | 400 | 4.10 [4.04, 4.17] | 4.39 [4.34, 4.45] | 4.18 [4.12, 4.23] | 4.43 [4.38, 4.48] | 3.59 [3.50, 3.68] | 4.46 [4.39, 4.52] |
| FD-Bench v2 | task score (Correction / EntityTracking / Safety) ↑ | 300 | 3.81 [3.63, 3.98] | 4.33 [4.20, 4.46] | 3.82 [3.64, 3.98] | 4.28 [4.12, 4.42] | 3.29 [3.06, 3.52] | 4.32 [4.18, 4.47] |
| FD-Bench v2 | IF_r (reached stages) B − A | 400 | | +0.43 [+0.33, +0.52] | | +0.37 [+0.26, +0.46] | | +0.88 [+0.77, +1.00] |
| FD-Bench v2 | stage score ↑ | 400 | 0.83 [0.80, 0.86] | 0.90 [0.88, 0.92] | 0.84 [0.82, 0.88] | 0.91 [0.89, 0.93] | 0.68 [0.63, 0.72] | 0.86 [0.83, 0.89] |
| FD-Bench v2 | examiner reached its end phrase (B: 120 s / 180 s) | 400 | 0.90 | 0.47 / 0.70 | 0.90 | 0.72 / 0.91 | 0.90 | 0.52 / 0.82 |
| FD-Bench v2 | turn-take rate [collisions left out] / latency s | 400 | 0.81 [0.96] / 1.08 | 0.96 / 1.30 † | 0.87 [0.91] / 1.04 | 0.90 / 1.28 † | 0.78 [0.76] / 1.62 | 0.97 / 1.63 |
| FD-Bench v2 | collisions per conversation / replayed lines invalid | 400 | 3.33 / 85% | 0 / — | 2.07 / 75% | 0 / — | 2.03 / 67% | 0 / — |
| Audio MultiChallenge | rubric items passed ↑ | 100 | 0.264 [0.194, 0.332] | 0.311 [0.238, 0.386] | — | — | — | — |
| Audio MultiChallenge | rubric items passed B − A | 100 | | +0.047 [−0.025, +0.121] | | | | |
| Audio MultiChallenge | all items passed ↑ | 100 | 0.090 [0.040, 0.150] | 0.110 [0.050, 0.170] | — | — | — | — |
| Audio MultiChallenge | turn-take rate / latency s / cut-in rate | 100 | 0.84 / 1.03 / 0.11 | 0.91 / 1.30 / 0.05 | — | — | — | — |
| all closed-loop sets | `eval.scores` mean per user turn ↑ (FD-Bench v3 / v2 / AudioMC) | | 0.45 / 0.17 / 0.28 | 0.23 / 0.21 / 0.22 | — / 0.24 / — | — / 0.21 / — | 0.05 / 0.20 / — | 0.09 / 0.11 / — |

† On FD-Bench v2 / v3, condition B has the closed-loop user's −60 dBFS background floor and A (replayed) has none.
MiniCPM-o answers about 0.2–0.3 s later where the floor is present: in B here, and in both conditions of Audio
MultiChallenge, whose A latency rose from 0.80 to 1.03 s against the previous run without the floor. So B − A timing differences for MiniCPM-o there are partly a background effect ([§6](#6-full-duplex-bench-v3-open-vs-closed-loop)).

## 3. Full-Duplex-Bench v1.0 / v1.5 (open loop)

The run covers all 1,723 episodes, MiniCPM-o 4.5, no system prompt. v1.5 includes the 410 clean inputs the behaviour
judge needs. The official metrics are ported from `v1_v1.5/evaluation` (now `extras/fdbench`): words come from the
agent's own text stream and speech spans from an energy VAD; the judges are proxies
([docs/BENCHMARKS.md §5](../docs/BENCHMARKS.md)). Next to them are the `eval.scores` outcomes (one per user turn) and
the per-turn score.

| subset | n | official (ours) [95% CI] | eval.scores outcomes [95% CI] · mean score | duplex timing: turn-take · latency median · cut-in · yield rate |
|---|---|---|---|---|
| v1.0 Candor pause | 216 | TOR 0.31 [0.25, 0.37] ↓ | waited in pauses 0.700 [0.642, 0.758] (194 waited, 73 took the floor, 10 kept talking); resumed turns that overlapped: yielded 33 / 59 · 0.56 | — (last turn censored) · — · 0.060 · 0.41 |
| v1.0 synthetic pause | 137 | TOR 0.15 [0.09, 0.21] ↓ | waited 0.803 [0.737, 0.869] · 0.67 | — · — · 0.016 · — |
| v1.0 Candor turn-taking | 119 | TOR 0.96 [0.92, 0.99] ↑, latency 1.28 s [1.13, 1.42] | responded 0.815 [0.740, 0.882] (20 talked over) · 0.28 | 0.87 · 1.07 s · 0.168 · — |
| v1.0 ICC backchannel | 55 | TOR 0.00, freq 0.00, JSD 1.00 ↓ | waited 1.000 (produces no backchannels) · 0.41 | — · — · 0.286 · — |
| v1.0 synthetic interruption | 200 | TOR 0.94 [0.91, 0.97], latency 1.53 s [1.40, 1.65], relevance 4.44 [4.22, 4.62] (proxy) | yielded 0.548 [0.461, 0.632] (74 yielded, 54 kept talking, 7 talked over) · 0.26 | 0.91 · 1.03 s · 0.038 · 0.600 [0.515, 0.679] |
| v1.5 user interruption | 200 | C_RESPOND 0.77 / C_RESUME 0.20 (proxy); stop 1.51 s, response 1.93 s | yielded 0.569 [0.497, 0.641] (103 / 72 kept talking / 6 talked over) · 0.24 | 0.88 · 1.01 s · 0.033 · 0.602 [0.533, 0.670] |
| v1.5 user backchannel | 98 | C_RESUME 1.00 (proxy); stop 0.61 s | ignored 1.000 · 0.68 | 1.00 · 0.97 s · 0 · — |
| v1.5 talking to other | 100 | C_RESPOND 0.56 / C_RESUME 0.30 (proxy); stop 1.36 s | ignored the aside 0.400 [0.300, 0.500] (60 replied) · 0.41 | 1.00 · 0.93 s · 0 · — |
| v1.5 background speech | 100 | C_RESPOND 0.50 / C_RESUME 0.47 (proxy); stop 0.87 s | ignored 0.610 [0.510, 0.710] (39 replied) · 0.51 | 1.00 · 0.84 s · 0 · — |

Reading:

- **Taking the turn.** MiniCPM-o takes the turn on silence with a latency of about 1 s (it decides once per 1 s unit).
- **Interruptions.** It yields to about 60 % of real overlapping interruptions.
- **Backchannels and asides.** It ignores backchannels, but replies to about half of the asides and third-party speech.
- **Official interruption TOR.** The official TOR (0.94) counts any speech after the interruption, including not
  stopping; the `yield` outcome separates the two.
- **Censored turns.** Under the per-turn scoring, a user turn left unanswered because the replayed clip ends is
  `censored` (kept, not scored): 140 of the Candor pause and 110 of the synthetic pause turns.

## 4. Easy Turn testset (open loop)

All 800 clips (Mandarin), MiniCPM-o 4.5, no system prompt; episodes as in [docs/BENCHMARKS.md §3](../docs/BENCHMARKS.md).

| state | n | key outcome [95% CI] | other outcomes | turn-take · latency · cut-in · mean score |
|---|---|---|---|---|
| complete | 300 | responded 0.990 [0.977, 1.000] | 3 talked over | 0.99 · 850 ms · 0.010 · 0.41 |
| incomplete | 300 | **waited 0.003 [0.000, 0.010]**: took the floor after 299 / 300 fragments | — | — · — · 0.020 · 0.003 |
| backchannel | 100 | ignored "嗯，也是" 1.000 | opening responded 1.00 | 1.00 · 790 ms · 0 · 0.71 |
| wait ("别说了") | 100 | **yield rate 0.39 [0.30, 0.50]** (timing); per-turn outcome: stayed quiet 0.08 [0.03, 0.14] (58 kept talking, 34 took the floor again) | opening responded 0.99 | 1.00 · 830 ms · 0.050 · 0.26 |

The per-turn scoring now scores "别说了" as a `wait` turn: the agent should stop **and** stay quiet until the user
speaks again. It does both in only 8 % of the clips. The yield rate (stopping at all) is the timing metric comparable
with earlier runs.

## 5. HumDial-FDBench (open loop)

The same 500-sample subset as before: 25 evenly spaced samples × 10 categories × 2 languages (small: ±0.1–0.2 on a
rate). MiniCPM-o 4.5, no system prompt.

| category | key outcome | en [95% CI] | zh [95% CI] | turn-take en / zh | latency median en / zh |
|---|---|---|---|---|---|
| ask | yield rate | 0.96 [0.88, 1.00] | 0.70 [0.50, 0.87] | 0.96 / 0.94 | 1.28 / 1.12 s |
| deny ("no, that's wrong") | yield rate | **0.16 [0.04, 0.32]** | **0.33 [0.16, 0.52]** | 0.78 / 0.84 | 1.25 / 1.16 s |
| repeat | yield rate | 0.78 [0.60, 0.95] | 1.00 | 0.84 / 0.98 | 1.36 / 1.19 s |
| shift | yield rate | 0.96 [0.88, 1.00] | 0.87 [0.71, 1.00] | 0.98 / 0.96 | 1.14 / 0.99 s |
| wait ("hold on") | yield rate (stayed quiet afterwards) | **0.08 [0.00, 0.20]** (0.00) | **0.16 [0.04, 0.32]** (0.00) | 1.00 / 1.00 | 1.35 / 1.08 s |
| backchannel | ignored | 1.00 | 1.00 | 0.96 / 1.00 | 1.47 / 0.80 s |
| talk_to_others | ignored the aside | 0.44 [0.24, 0.64] | 0.28 [0.12, 0.48] | 0.92 / 0.92 | 1.45 / 1.30 s |
| others_talk_to_user_after | ignored the third person | 0.62 [0.42, 0.80] | 0.48 [0.28, 0.68] | 1.00 / 1.00 | 1.47 / 1.16 s |
| others_talk_to_user_before | ignored the third person | 0.20 [0.04, 0.40] | 0.00 | 0.92 / 1.00 | 1.15 / 1.54 s |
| pause | waited in the pause | 0.44 [0.24, 0.64] | 0.44 [0.24, 0.64] | 0.52 / 0.56 | 1.18 / 1.09 s |

Content interruptions (ask / repeat / shift) are yielded to. "No, that's wrong" and "hold on" mostly are not, and after
"hold on" the agent never stays quiet. others_talk_to_user_before and talk_to_others are diagnostic only: they carry no
addressee cue ([docs/BENCHMARKS.md §8](../docs/BENCHMARKS.md)).

## 6. Full-Duplex-Bench v3 (open vs closed loop)

100 recordings × 3 seeds, i.e. 300 pairs per agent.

- **A** is the recording only (official, open loop).
- **B** is the same recording, then a simulated user that answers questions, corrects mistakes and confirms
  ([docs/BENCHMARKS.md §11](../docs/BENCHMARKS.md)). Its scenario card comes from the expected calls, and its voice is
  cloned from the recording.

Audio: 16 kHz Opus 32 kbps transcode of the 48 kHz release.

> **Caveat: the A/B difference now includes a background difference.** Since `33ad813`, a closed-loop `UserSim` puts
> its surroundings' background under the episode. For these "quiet" users that is a −60 dBFS pink-noise floor. The
> replayed A condition (`ReplayUser`) has none. So A and B are no longer identical before the user's second turn:
> - **MiniCPM-o.** First response 1020 → 1340 ms, interruption rate 0.110 → 0.050, and spoken fulfilment of the first
>   response 0.817 (A) → 0.867 (B first). First-turn text is identical in 22/300 pairs, against 172/300 for A vs A of
>   two seeds in the previous run. First-response metrics are identical in 168/300.
> - **Cascaded agents.** Pass@1 of the first response is unchanged (0.620 → 0.617). Identical speech 126/300 and
>   identical tool calls 176/300, against 299/300 and 300/300 before.
>
> To restore A = B by construction, either give the replayed condition the same floor, or turn the floor off for
> benchmark users (`Soundscape(background=False)`). Then re-run FD-Bench v2 / v3. Until then, read B − A on these two
> benchmarks as "closed loop + background floor" vs "open loop, silent line".

**Cascaded agents: Pass@1.** Computed with the official `evaluate_pass_rate.py` logic and a proxy judge. The lenient
judge, in brackets, allows extra arguments and normalises IDs.

| agent | A (official) | B first response | B final | B strict (all calls) | B final − A [95% CI] | recovered (A fail → B pass) | interruption rate A | first response ms A |
|---|---|---|---|---|---|---|---|---|
| cascaded-official | 0.620 (0.680) | 0.617 (0.677) | 0.720 (0.807) | 0.663 (0.750) | +0.100 [+0.050, +0.153] | 0.31 (0.45) | 0.213 | 2629 |
| cascaded-official-ep3000 | 0.623 (0.683) | 0.643 (0.683) | 0.730 (0.790) | 0.687 (0.743) | +0.107 [+0.057, +0.160] | 0.32 (0.40) | 0.000 | 4919 |
| cascaded-natural | 0.620 (0.650) | 0.610 (0.647) | 0.720 (0.787) | 0.687 (0.753) | +0.100 [+0.050, +0.157] | 0.30 (0.43) | 0.213 | 3094 |

- **Variance and ranking.** Per-seed sd of Pass@1 is 0.006–0.026. The three agents are within 1 point of each other in
  both conditions: they are tied, and the order among them is noise.
- **Why A failed** (cascaded-official, 114 failures):
  - wrong arguments 44 % (recovered in B 5/50);
  - asked the user instead of finishing 31 % (15/35 recovered);
  - no tool call 11 % (8/13);
  - wrong tools 10 % (3/11);
  - interrupted in a pause 4 % (4/5).
- **User effort in B:** 1.60 turns after the recording; 22 % of episodes include a correction.
- **Response quality (proxy):** A 0.77, B 0.90.

**MiniCPM-o 4.5: spoken fulfilment.** The agent has no tools, so the measure is whether its speech confirmed every
parameter with its final value: 281 slots, scored by rules with a proxy-LLM fallback.

| subset | n | A | B first response | B final | recovered | broken | slot score A / B final |
|---|---|---|---|---|---|---|---|
| all | 300 | 0.817 [0.740, 0.890] | 0.867 † | 0.983 [0.963, 0.997] | 0.96 (53/55) | 0.012 (3/245) | 0.889 / 0.994 |
| easy / medium / hard | 108 / 102 / 90 | 0.806 / 0.833 / 0.811 | 0.898 / 0.863 / 0.833 | 0.991 / 0.971 / 0.989 | | | |
| PAUSE | 54 | 0.759 | 0.870 | 1.000 | | | |
| SELF_CORRECTION | 51 | 0.941 | 0.941 | 0.980 | | | |

- **A's failures:**
  - asked a question instead of confirming 51 % (26/28 recovered);
  - misheard 20 % (11/11);
  - missed a detail 18 % (10/10);
  - interrupted in a pause 5 %;
  - wrong after a self-correction 5 %.
- **User effort in B:** 3.55 turns after the recording.
- **Claimed results:** without tools, the agent claims a completed action or invents a result in 50 % of B
  conversations (A 5 %). The response-quality judge rewards it (A 0.30 → B 0.89), so response quality is not usable
  as an outcome for tool-less agents in closed loop.

**Timing.** Official definitions, first user turn:

- MiniCPM-o: turn-take 1.000, interruption rate 0.110 (A) / 0.050 (B), first response 1020 / 1340 ms (†).
- Cascaded-official: 0.213 / 2629 ms (ep3000: 0 / 4919 ms).

`eval.scores` over whole conversations:

- MiniCPM-o latency median 0.74 s (A) / 1.20 s (B), cut-in 0.13 / 0.14, mean per-turn score 0.45 / 0.23. The per-turn
  score is latency-graded, and B adds many later user turns answered at about 1.2 s.
- Cascaded-official latency 2.35 s / 1.66 s, cut-in 0.21 / 0.09, mean score 0.05 / 0.09.

There are no collisions: A has one user turn, and the B user waits for the agent. The open-loop user-turn failure rate
is 0 by construction (one user turn per open-loop episode).

## 7. Full-Duplex-Bench v2 (open vs closed loop)

200 examiner tasks (Daily, Correction, EntityTracking, Safety × 50) × seeds 0, 1 = 400 pairs per agent
([docs/BENCHMARKS.md §12](../docs/BENCHMARKS.md)).

- **A** is the official examiner prompts turned into a static script: the same examiner LLM plays against a text-only
  reference assistant, and the lines are replayed on a fixed timeline with a 120 s cap. The scripts were regenerated
  for this run, with the examiner's lines cloned from its first line.
- **B** is the live examiner (official protocol, slow mode) with stage-aware closing and a 180 s cap. It is scored on
  all 180 s and on the official 120 s window.

The † background caveat of [§6](#6-full-duplex-bench-v3-open-vs-closed-loop) applies: B has the −60 dBFS floor and A
does not.

**Official judge (proxy), all splits.** B on 180 s; 120 s window in brackets.

| agent | TT A | TT B | IF A | IF B | IF B − A [95% CI] | task A | task B | task B − A [95% CI] |
|---|---|---|---|---|---|---|---|---|
| cascaded | 3.59 | 4.46 (4.44) | 3.25 | 4.14 (4.02) | +0.89 [+0.77, +1.01] (+0.77) | 3.29 | 4.32 (4.18) | +1.03 [+0.83, +1.24] (+0.89) |
| minicpmo | 4.10 | 4.39 (4.37) | 3.93 | 4.26 (4.15) | +0.33 [+0.23, +0.42] (+0.21) | 3.81 | 4.33 (4.22) | +0.52 [+0.35, +0.71] (+0.42) |
| minicpmo-confirm | 4.18 | 4.43 (4.39) | 3.92 | 4.28 (4.20) | +0.36 [+0.26, +0.46] (+0.28) | 3.82 | 4.28 (4.22) | +0.46 [+0.27, +0.65] (+0.40) |

By split, IF B − A at 180 s for cascaded / minicpmo / minicpmo-confirm:

- Daily +1.18 / +0.34 / +0.27;
- Correction +0.75 / +0.28 / +0.22;
- EntityTracking +1.08 / +0.69 / +0.67;
- Safety +0.55 / −0.01 (n.s.) / +0.30.

**Stage-based scores.** The time cap is not a failure. The proxy LLM analyses the stages; IF_r / task_r are the
official judge on the reached stages.

| agent | end phrase reached A / B 120 s / B 180 s | stages completed A / B | stage score A / B (B − A [95% CI]) | IF_r A / B (B − A [95% CI]) | task_r A / B |
|---|---|---|---|---|---|
| cascaded | 0.90 / 0.52 / 0.82 | 2.73 / 3.29 | 0.68 / 0.86 (+0.18 [+0.14, +0.23]) | 3.29 / 4.17 (+0.88 [+0.77, +1.00]) | 3.32 / 4.37 |
| minicpmo | 0.90 / 0.47 / 0.70 | 3.37 / 3.30 | 0.83 / 0.90 (+0.07 [+0.03, +0.10]) | 3.96 / 4.39 (+0.43 [+0.33, +0.52]) | 3.87 / 4.44 |
| minicpmo-confirm | 0.90 / 0.72 / 0.91 | 3.28 / 3.46 | 0.84 / 0.91 (+0.07 [+0.04, +0.10]) | 3.95 / 4.31 (+0.37 [+0.26, +0.46]) | 3.86 / 4.32 |

**Between agents.** Paired on task and seed, B at 180 s.

| difference | open (A) | closed (B) |
|---|---|---|
| minicpmo − cascaded, TT | +0.51 [+0.40, +0.61] | −0.07 [−0.15, +0.02] |
| minicpmo − cascaded, IF_r | +0.67 [+0.53, +0.81] | +0.22 [+0.10, +0.33] |
| minicpmo − cascaded, stage score | +0.16 [+0.11, +0.20] | +0.04 [0.00, +0.08] |
| minicpmo − cascaded, task_r | +0.55 [+0.33, +0.78] | +0.06 [−0.12, +0.24] |
| minicpmo-confirm − minicpmo, TT | +0.07 [−0.01, +0.15] | +0.04 [−0.03, +0.11] |
| minicpmo-confirm − minicpmo, IF_r | −0.01 [−0.11, +0.08] | −0.08 [−0.16, +0.01] |
| minicpmo-confirm − minicpmo, task_r | −0.01 [−0.17, +0.15] | −0.11 [−0.29, +0.06] |

- **The static script inflates the native-duplex vs cascaded gap.** Open loop, MiniCPM-o leads the cascaded agent on
  every judge score. Closed loop, the gap shrinks 3–9×, and on TT and task_r it is not significant.
- **Prompt choice is not resolved by either condition.** The confirm prompt's cost on IF_r / task score now points the
  same way in closed loop but is not significant.

**Duplex timing** (`eval.timing_counts`; collisions apart):

| agent | cond | turn-take rate [collisions left out] | latency median / p90 s | cut-in rate | collisions / conversation (share of user turns) | agent yielded on collisions | eval.scores mean |
|---|---|---|---|---|---|---|---|
| cascaded | A | 0.78 [0.76] | 1.62 / 1.85 | 0.02 | 2.03 (36 %) | 100 % | 0.20 |
| cascaded | B | 0.97 | 1.63 / 1.86 | 0.01 | 0 | — | 0.11 |
| minicpmo | A | 0.81 [0.96] | 1.08 / 2.09 | 0.02 | 3.33 (59 %) | 57 % | 0.17 |
| minicpmo | B | 0.96 | 1.30 / 1.99 † | 0.01 | 0 | — | 0.21 |
| minicpmo-confirm | A | 0.87 [0.91] | 1.04 / 1.80 | 0.05 | 2.07 (37 %) | 70 % | 0.24 |
| minicpmo-confirm | B | 0.90 | 1.28 / 2.09 † | 0.06 | 0 | — | 0.21 |

The examiner never barges in (slow mode), so there are no intended barge-ins and no yield rate. The raw open-loop
turn-take rate is depressed by collisions. The cascaded agent's latency is the same open and closed (fixed latency
model). MiniCPM-o's is about 0.2 s longer in B, the background effect (†).

**Open-loop user-turn failure rate.** Measured with `turn_validity`, per examiner line after the first; B gives the
checker's noise floor.

| agent | A invalid (any) | timing (collision) | content (responds to something never said) A / B | LLM-only A / B | excess LLM-only per conversation [95% CI] |
|---|---|---|---|---|---|
| cascaded | 67 % | 44 % | 21 % / 7 % | 32 % / 23 % | +8 [+5, +12] pts |
| minicpmo | 85 % | 71 % | 22 % / 5 % | 32 % / 18 % | +13 [+10, +16] pts |
| minicpmo-confirm | 75 % | 44 % | 20 % / 4 % | 42 % / 20 % | +22 [+19, +25] pts |

**Consistency** (agent speech before the examiner's second line, A vs B):

- cascaded: identical text 384/400, first onset 381/400;
- minicpmo: 161/400 and 198/400;
- minicpmo-confirm: 52/400 and 246/400.

MiniCPM-o's onsets differ much more often than in the previous run (396/400). This is the background floor in B (†),
on top of batch nondeterminism. 0/2,400 judge replies were unparsed.

## 8. Audio MultiChallenge (open vs closed loop)

100 conversations (25 per axis, the same ids as before), seed 0, `minicpmo` only, official-style examinee prompt
([docs/BENCHMARKS.md §13](../docs/BENCHMARKS.md)).

- **A** plays the recorded user turns of `ScaleAI/audiomc` in order, each after the agent stops. This is open loop:
  each later turn was recorded in reaction to a *fixed* assistant reply.
- **B** plays the same first recording. Each later turn is then rewritten by an LLM user to fit what the agent
  actually said, keeping every fact, request, instruction and self-repair. It is spoken in a voice cloned from the
  first recording.

A and B have the same number of user turns, and here both conditions have the background floor. The official judge
prompt (proxy) scores the agent's reply to the last user turn per rubric item, on the conversation as it happened.
This is not the official protocol: a duplex model cannot be handed someone else's replies as its own history, so it
talks through the whole conversation. The cascaded agent is not run: its long replies and 800 ms endpointing inside
the long, pause-rich recordings run past the time cap.

| axis | n | rubric items passed A | B | B − A [95% CI] | all items passed A | B | B − A [95% CI] | complete A / B |
|---|---|---|---|---|---|---|---|---|
| all | 100 | 0.264 [0.194, 0.332] | 0.311 [0.238, 0.386] | +0.047 [−0.025, +0.121] | 0.090 | 0.110 | +0.020 [−0.050, +0.090] | 1.00 / 0.99 |
| Inference memory | 25 | 0.233 | 0.163 | −0.070 [−0.260, +0.120] | 0.16 | 0.08 | −0.08 [−0.28, +0.12] | 1.00 / 1.00 |
| Instruction retention | 25 | 0.217 | 0.333 | +0.117 [+0.003, +0.263] | 0.08 | 0.16 | +0.08 [−0.08, +0.24] | 1.00 / 1.00 |
| Voice editing | 25 | 0.423 | 0.475 | +0.052 [−0.043, +0.151] | 0.04 | 0.00 | −0.04 [−0.12, 0.00] | 1.00 / 0.96 |
| Self-coherence | 25 | 0.183 | 0.272 | +0.088 [−0.051, +0.247] | 0.08 | 0.20 | +0.12 [0.00, +0.28] | 1.00 / 1.00 |

- **Absolute level.** MiniCPM-o passes about a quarter to a third of the rubric items; 9–11 % of conversations pass
  every item.
- **Open vs closed: no significant difference overall.** Instruction retention is at the edge (+0.117, lower bound
  +0.003, n = 25). The direction overall (closed ≥ open) matches the other benchmarks. But AudioMC's rubrics test memory
  of facts and instructions given by the user, which the closed-loop rewrite keeps, so later turns depend little on the
  reply. The axis with the largest gain changed from run to run (voice editing previously, instruction retention now):
  per-axis differences at n = 25 are not reliable.
- **Duplex timing** (`eval.timing_counts`):

  | | A | B |
  |---|---|---|
  | turn-take rate | 84 % | 91 % |
  | latency median (p90) | 1.03 s (1.80 s) | 1.30 s (2.23 s) |
  | cut-in rate | 11 % | 5 % |
  | `eval.scores` mean | 0.28 | 0.22 |

  There are **no collisions** and no intended barge-ins (the user waits for the agent in both conditions), and no
  non-directed sounds.
- **Open-loop user-turn failure rate** (`turn_validity`, per user turn after the first; B = noise floor of the LLM
  checker): invalid A 25 % vs B 18 %, **+7 points** (per conversation +8 [+2, +13]). By type: content 12 % vs 7 %,
  ignored question 11 % vs 10 %, stale 1 % vs 2 %, timing 0 %. 53 % of A conversations vs 48 % of B have at least one
  flagged turn.
- **Consistency (A vs B before the user's second turn).** The first agent onset is identical in 96/100 pairs; the
  first agent turn text in only 18/100. The first turns are long, MiniCPM-o often speaks inside the ~18 s recordings,
  and batch nondeterminism at 6 concurrent sessions changes the text.

## 9. Comparison with the previous run

The previous run used env `917b988` on one host, 2026-10-07, with the scoring of that time. Three changes since then
affect the numbers.

**(1) Scoring.** `eval.scores` now emits exactly one event per user turn (`86cc52c`). The previous run had two or three
events per user turn, so **its `eval.scores` outcome rates are not directly comparable** with the ones here.

- "yielded" now also fails turns where the agent talked over the user again or never took the floor again.
- "ignore" is a single `ignored` outcome.
- "别说了" / "hold on" turns are `wait` turns (stop *and* stay quiet).
- Turns cut off by the clip's end are `censored`.
- The mean score is new.

The official benchmark metrics (TOR, C_RESPOND, judge scores, Pass@1, spoken fulfilment, rubric rates) and the duplex
timing metrics (`eval.timing_counts`: turn-take, latency, yield rate, cut-in, collisions) have unchanged definitions,
so the comparisons below use those.

**(2) The user side** (`c946419`, `33ad813`, `e090ea7`):

- voices cloned from the first turn everywhere, including the FD-Bench v2 examiner and its static scripts, with a
  neutral style, leveling and a per-episode seed;
- LLM-decided user behaviours;
- the soundscape, i.e. the −60 dBFS floor under every closed-loop user and the Audio MultiChallenge A user.

The open-loop benchmarks (FD-Bench v1/v1.5, Easy Turn, HumDial) replay fixed audio and use none of this.

**(3) Serving.** The MiniCPM-o servers now include the per-session feature-extractor fix (PR #8638). Four servers ran
instead of two (6 sessions each, as before).

Per benchmark, previous → now:

- **FD-Bench v1.0 / v1.5, Easy Turn, HumDial: unchanged within noise.**
  - Official metrics: Candor pause TOR 0.324 → 0.31; synthetic pause 0.153 → 0.15; turn-taking TOR 0.958 → 0.96,
    latency 1.27 → 1.28 s; interruption TOR 0.94 → 0.94, relevance 4.43 → 4.44; v1.5 C_RESPOND 0.745 → 0.77 /
    talking-to-other 0.52 → 0.56 / background 0.49 → 0.50.
  - Yield rate: v1.0 0.613 → 0.600, v1.5 0.583 → 0.602; Easy Turn wait 0.41 → 0.39.
  - Easy Turn: complete responded 0.993 → 0.990; incomplete waited 0.023 → 0.003 (7 → 1 of 300, i.e. it almost never
    waits).
  - HumDial yield rates (timing): ask / repeat / shift 0.86 → 0.88; deny / wait 0.20 → 0.18; pause waited 0.46 → 0.44.
- **FD-Bench v3: unchanged where it can be compared.**
  - Cascaded-official Pass@1: A 0.597 → 0.620, B final 0.707 → 0.720, B − A +0.110 → +0.100. ep3000 and natural are
    likewise within 2 points.
  - MiniCPM-o spoken fulfilment: A 0.820 → 0.817, B final 0.973 → 0.983, recovered 0.93 → 0.96, claims 51 % → 50 %.
  - **New:** "B first response" (0.800 → 0.867) and A-vs-B consistency no longer match A, because of the B-only
    background floor (caveat in §6).
- **FD-Bench v2: the open-vs-closed gap holds; between-agent gaps shrink in closed loop.**
  - IF_r B − A: cascaded +0.92 → +0.88, minicpmo +0.54 → +0.43, confirm +0.42 → +0.37.
  - Stage score B − A: +0.21 / +0.10 / +0.09 → +0.18 / +0.07 / +0.07.
  - The A scores rose for all agents (IF A cascaded 3.05 → 3.25, minicpmo 3.85 → 3.93; TT A cascaded 3.36 → 3.59).
    The static scripts were regenerated, with cloned examiner voices.
  - MiniCPM-o's closed-loop TT advantage over the cascaded agent (previously +0.13 [+0.04, +0.24]) is gone (−0.07
    [−0.15, +0.02]), and its IF_r advantage halved (+0.40 → +0.22). In B, MiniCPM-o answers about 0.2 s later than in
    the previous run (1.00 → 1.30 s; background floor). The cascaded agent's B scores rose (TT 4.27 → 4.46).
  - The confirm prompt's closed-loop cost on IF_r / task_r is no longer significant (−0.12 / −0.22 → −0.08 / −0.11).
  - Collisions and validity are within a few points (collisions per conversation 1.89 / 3.32 / 2.17 → 2.03 / 3.33 /
    2.07; excess LLM-flagged lines +9 / +11 / +21 → +8 / +13 / +22 points).
- **Audio MultiChallenge.** Rubric items passed: A 0.275 → 0.264, B 0.297 → 0.311, B − A +0.022 → +0.047; neither run
  is significant overall. Per axis, the previous voice-editing gain (+0.085) and now instruction retention (+0.117)
  each lead once; at n = 25 neither is reliable. Latency rose (A 0.80 → 1.03 s, B 0.92 → 1.30 s), which is the
  background floor, now in both conditions. Turn-take is 86 / 91 % → 84 / 91 %. The open-loop failure rate is +8 →
  +7 points.

## 10. Limitations

- **Proxy judges and examiner.** Every judge, the FD-Bench v2 examiner and reference assistant, and both closed-loop
  users are the same local Qwen3.8-27B-FP8, not the official GPT-4-turbo / GPT-4o / GPT-Realtime / Gemini 2.5 Flash /
  o4-mini. Absolute levels are not comparable with published tables. A and B share the judge, so differences are more
  reliable than levels. The judge is of the same family as the cascaded agent's LLM.
- **Background floor in B only (FD-Bench v2 / v3).** See §6: B − A differences there mix the closed loop with a −60 dBFS
  noise floor that A does not have. This affects MiniCPM-o's timing most.
- **Ported measurement.** FD-Bench v1/v1.5 words come from the agent's text stream (not ASR of its audio) and speech
  spans from an energy VAD. FD-Bench v2 Channel B is the agent's text with segment times (not ASR word chunks). Only the
  slow examiner mode is ported. The open-loop v2 script is one plausible static construction (heuristic timeline), not
  an official one. The closed-loop v2 cap is 180 s; the official 120 s window is reported too.
- **FD-Bench v3 audio** is a 16 kHz Opus 32 kbps transcode of the 48 kHz release, with the room-tone tail after the
  request removed in both conditions. The cascaded agent's timing comes from a fixed latency model.
- **Simulated users.** Closed-loop users are LLMs with TTS; on FD-Bench v3 and Audio MultiChallenge their voices are
  cloned from the first recording. They may be more cooperative than real people. On FD-Bench v3 they know the target
  values and repeat them, which makes closed-loop success partly a measure of the user's cooperation. No human
  calibration was done.
- **Spoken fulfilment** (FD-Bench v3, MiniCPM-o) measures confirmation of parameters, not task completion. Audio
  MultiChallenge is not run with its official fixed-history protocol (the duplex model talks through the conversation).
- **Nondeterminism.** MiniCPM-o ran with 6 concurrent sessions per server. Its decoding is not batch-invariant, so
  per-episode text differs between runs and between A and B even where the input is identical; onsets and aggregates
  are stable. No batch-1 control was run.
- **Sizes.** HumDial uses 25 samples per category and language; Audio MultiChallenge 100 conversations, one seed.
- **Dataset licenses** (results only; no data is redistributed):
  - Full-Duplex-Bench v1.0: CC BY-NC 4.0 for the Candor / ICC parts (plus upstream terms), MIT for the synthetic parts.
  - Full-Duplex-Bench v1.5: MIT.
  - **Full-Duplex-Bench v2 and v3: CC BY-NC 4.0** (non-commercial research only). This covers the ported prompts, mock
    APIs and judge prompts in the optional component `extras/fdbench/` too.
  - Easy Turn testset: Apache-2.0. HumDial-FDBench: Apache-2.0. Audio MultiChallenge: MIT.
  - Models: MiniCPM-o 4.5, Qwen3.8-27B, Qwen3-TTS and Qwen3-ASR are under their own terms.
