# Benchmark results: open vs closed loop

These are the **release results** of InteractionGym: **MiniCPM-o 4.5** evaluated **open loop** (fixed user
input, as a static benchmark plays it) and **closed loop** (the same opening, then a simulated user that reacts to
the agent) on three public benchmarks integrated in the env ([docs/BENCHMARKS.md](../docs/BENCHMARKS.md)):
Full-Duplex-Bench v3 (tool-use requests under disfluency), Full-Duplex-Bench v2 (an automated examiner) and Audio
MultiChallenge (multi-turn memory and instruction following). Run on 2026-10-09 with env `main` at **`b7d64a5`**.

The earlier full-audio run (2026-10-08/09: Talker audio, cascaded and `minicpmo-confirm` agents, and the open-loop-only
benchmarks FD-Bench v1.0 / v1.5, Easy Turn and HumDial) is kept as an appendix in
[BENCHMARKS_full_audio.md](BENCHMARKS_full_audio.md); it is not directly comparable with this run (§8).

Results only: no benchmark data is redistributed here (licenses in [§9](#9-limitations)). Aggregate numbers are in
[`data/summary.json`](data/summary.json), made by [`aggregate.py`](aggregate.py) from the run folders, with the
turn-indexed metrics of [`turn_metrics.py`](turn_metrics.py).

Contents: [1 Setup](#1-setup) · [2 Summary](#2-summary-table) · [3 Consistency](#3-consistency-a-and-b-are-the-same-until-the-users-second-turn) ·
[4 FD-Bench v3](#4-full-duplex-bench-v3) · [5 FD-Bench v2](#5-full-duplex-bench-v2) ·
[6 Audio MultiChallenge](#6-audio-multichallenge) · [7 What only closed loop measures](#7-what-only-closed-loop-measures) ·
[8 Comparison with the full-audio run](#8-comparison-with-the-full-audio-run) · [9 Limitations](#9-limitations)

## 1. Setup

**Conditions.**

- **A (open loop)** replays the benchmark's fixed user input: the FD-Bench v3 recording; the FD-Bench v2 examiner
  prompts turned into a static script (the same examiner LLM played against a text-only reference assistant, lines
  replayed on a fixed timeline, 120 s cap); the recorded Audio MultiChallenge user turns, each played after the agent
  stops.
- **B (closed loop)** starts identically and then reacts: an LLM user that answers questions, corrects mistakes and
  confirms (FD-Bench v3, scenario card from the expected calls); the live examiner with stage-aware closing (FD-Bench
  v2, official prompts, slow mode, 180 s cap, also scored on the official 120 s window); each later Audio
  MultiChallenge turn rewritten to fit what the agent said, keeping every fact, request, instruction and self-repair.

**What is new against the full-audio run** (all three are the env defaults at `b7d64a5`):

- **Thinker-only MiniCPM-o.** `VllmOmniDuplexAgent(audio_out=False)`: the model's text, timed at `speech_cps`
  (11.3 characters per second), from a one-GPU Thinker-only server (docs/BENCHMARKS.md, "Agent output").
- **No background in evaluations.** Benchmark users are built with `benchmarks.benchmark_user`: no background
  track, no −60 dBFS floor, no noise events or asides (docs/BENCHMARKS.md, "No background in evaluations"). A and B
  give the agent the same microphone until the user's second turn ([§3](#3-consistency-a-and-b-are-the-same-until-the-users-second-turn)).
- **Logistic latency curve.** `eval.latency_score(Δ) = 1 / (1 + exp((Δ − 950 ms) / 100 ms))` for `respond` latency
  and `yield` stop latency: 200 ms 0.999, 500 ms 0.989, 700 ms 0.924, 800 ms 0.818, 1 s 0.378, 1.2 s 0.076
  (docs/FORMAT.md §6.3).

Unchanged: the simulated user's voice is cloned from its first turn (the recording on FD-Bench v3 / Audio
MultiChallenge, the examiner's first line on FD-Bench v2, both live in B and in the static A scripts); LLM-decided
user behaviours; lockstep simulation (`clock="input"`) at 16 kHz.

**Agent.** `minicpmo`: MiniCPM-o 4.5 full-duplex, Thinker greedy (T = 0, repetition penalty 1.1), token trace on.
System prompt: a per-domain voice-assistant prompt asking it to confirm every detail (FD-Bench v3); the official
examinee prompt (FD-Bench v2); an official-style examinee prompt (Audio MultiChallenge). No cascaded agent and no
prompt variants in this run.

**Serving.** vLLM 0.30.0 + vLLM-Omni `0.30.1.dev97+ge7c7dac58` (upstream `main` plus the duplex patches: input clock,
`silence_continuation`, token trace) with a clean overlay `eval_thinker`: that install's `vllm_omni` plus
`examples/serving/reference/patches/minicpmo_thinker_only.patch` (8 files). The install already carries the
per-session feature-extractor fix (`shared_fe`, upstream PR #8638), checked at every server start. Deploy config
`configs/minicpmo_4_5_thinker_1gpu.yaml` (one GPU, `max_model_len` 16384), 16 sessions per server. Eight servers,
one per GPU on GPUs 0–3 of both hosts. Concurrency: 16 sessions per server on FD-Bench v2 / v3, 8 on Audio
MultiChallenge (its long episodes share the KV cache). Not the RL overlay (`rl_omni`, which has sampler changes).

**Simulated users, examiners and judges.** All LLM roles are a local **Qwen3.8-27B-FP8**, at T = 0 for the judges:
the FD-Bench v3 user, the FD-Bench v2 examiner / reference assistant / stage tracker, the Audio MultiChallenge
replanning user, and every judge (FD-Bench v3 response quality and claimed results, proxy for GPT-4o; FD-Bench v2,
proxy for Gemini 2.5 Flash; Audio MultiChallenge rubric, proxy for o4-mini; the spoken-fulfilment fallback; the
stage analysis; `turn_validity`). **All judge-based numbers are proxy numbers, not comparable with published
tables.** Voices: Qwen3-TTS-12Hz-1.7B CustomVoice for the examiner's first line, Base for cloning.

**Sizes and seeds.** With the cascaded agents dropped, seeds were raised where cheap:

| benchmark | conversations | seeds | pairs (A, B) | episodes |
|---|---|---|---|---|
| FD-Bench v3 | 100 recordings | 0–4 (was 0–2) | 500 | 1,000 |
| FD-Bench v2 | 200 tasks (Daily, Correction, EntityTracking, Safety × 50) | 0–2 (was 0–1) | 600 | 1,200 |
| Audio MultiChallenge | 100 (25 per axis, the same ids as before) | 0–1 (was 0) | 200 | 400 |

The FD-Bench v2 static scripts of seeds 0 and 1 are the full-audio run's (agent-independent, the same A input); seed
2's were generated for this run. In A the seed changes only server-side numerics (the input is fixed); in B it also
seeds the user.

**Hardware, runs, cost.** Two hosts with 8× RTX 5090 (32 GB): GPUs 0–3 MiniCPM-o servers; 4–5 the user / examiner /
judge LLM; 6–7 Qwen3-TTS CustomVoice and Base replicas. One self-running pipeline per host (tmux; job queue,
watchdog that restarts services, merge, re-scoring, reports, aggregation), resumable.

- **2,600 episodes**, 0 given up; 133 transient episode retries, all `resource_exhausted` (a new session opened
  before the previous one was released at the 16-session cap). No service restarts were needed.
- **Wall time:** 1.5 h for the runs (both hosts in parallel; about 10 minutes of it lost to a launch fix and a queue
  rebalance), then 1.6 h for the reports, judges and turn metrics on host 1: **3.2 h** end to end.
- **Cost:** about **31 GPU-hours** reserved: 8 GPUs × 1.5 h on host 2, 4 GPUs × 1.5 h + 4 GPUs × 3.2 h on host 1.
  The MiniCPM-o servers took 12 GPU-hours. The full-audio run took about 94 GPU-hours for 8,023 episodes.

**Statistics.** 95% CIs are percentile bootstraps (2,000 resamples) over recordings / tasks / conversations, so all
seeds of one item resample together. A vs B differences are paired on (item, seed). Rates in square brackets are
[lo, hi].

**Scores.** `eval.scores` ([docs/FORMAT.md §6.3](../docs/FORMAT.md)) gives each user turn one rule-based score in
[0, 1] (respond, yield, wait, ignore, interrupt), graded by the logistic latency curve where the agent takes the
floor. Duplex timing (`eval.timing_counts`): turn-take rate (directed user turns answered within 5 s), latency (end
of user turn → agent reply, median), cut-in rate (directed user turns the agent talks into), collisions (a replayed
line whose fixed start fell inside the agent's speech; reported apart).

## 2. Summary table

MiniCPM-o 4.5, open loop (A) vs closed loop (B). n = pairs (conversations × seeds). ↑ = higher is better.

| benchmark | metric | n | open (A) | closed (B) | B − A [95% CI] |
|---|---|---|---|---|---|
| FD-Bench v3 | spoken fulfilment ↑, pass@1turn (A) / B final | 500 | 0.822 [0.748, 0.888] | 0.984 [0.970, 0.996] | +0.162 [+0.100, +0.230] |
| FD-Bench v3 | B pass@1turn / pass@2turn / pass@3turn | 500 | (A = 0.822) | 0.824 / 0.944 / 0.964 | +0.002 / +0.122 / +0.142 |
| FD-Bench v3 | recovered (A fail → B pass) / broken (A pass → B fail) | 500 | | 0.933 (83/89) / 0.005 (2/411) | |
| FD-Bench v3 | claims a result it cannot know (no tools) | 500 | 0.052 | 0.534 | |
| FD-Bench v3 | user turns after the recording (B) | 500 | — | 3.60 [3.36, 3.87] | |
| FD-Bench v2 (B 180 s) | IF (official judge, proxy) ↑ | 600 | 3.90 [3.83, 3.98] | 4.33 [4.26, 4.39] | +0.43 [+0.34, +0.51] |
| FD-Bench v2 | TT (official, proxy) ↑ | 600 | 4.09 [4.03, 4.14] | 4.39 [4.34, 4.45] | +0.31 [+0.24, +0.37] |
| FD-Bench v2 | task score (Correction / EntityTracking / Safety) ↑ | 450 | 3.82 [3.66, 3.98] | 4.37 [4.27, 4.47] | +0.54 [+0.38, +0.72] |
| FD-Bench v2 | IF_r (judge on the reached stages) ↑ | 600 | 3.93 [3.86, 4.00] | 4.45 [4.40, 4.50] | +0.52 [+0.44, +0.60] |
| FD-Bench v2 | stage score ↑ / goals completed (of 4) | 600 | 0.84 / 3.42 | 0.89 / 3.36 | +0.05 [+0.02, +0.07] / −0.06 [−0.17, +0.05] |
| FD-Bench v2 | examiner reached its end phrase (B: 120 s / 180 s) | 600 | 0.90 | 0.43 / 0.68 | |
| FD-Bench v2 | turn-take rate [collisions left out] / latency s | 600 | 0.76 [0.96] / 1.06 | 0.97 / 1.01 | |
| FD-Bench v2 | collisions per conversation / replayed lines invalid | 600 | 3.58 / 87 % | 0 / (18 %: checker floor) | |
| Audio MultiChallenge | rubric items passed ↑ | 200 | 0.241 [0.186, 0.301] | 0.285 [0.229, 0.342] | +0.044 [−0.002, +0.092] |
| Audio MultiChallenge | all items passed ↑ | 200 | 0.065 [0.030, 0.105] | 0.075 [0.035, 0.115] | +0.010 [−0.025, +0.045] |
| Audio MultiChallenge | turn-take rate / latency s / cut-in rate | 200 | 0.85 / 0.75 / 0.15 | 0.92 / 0.84 / 0.08 | |
| Audio MultiChallenge | user turns flagged invalid (A) vs checker floor (B) | 200 | 25 % | 18 % | +7 [+3, +11] pts |
| all three | `eval.scores` mean per user turn ↑ (FD-Bench v3 / v2 / AudioMC) | | 0.63 / 0.22 / 0.55 | 0.51 / 0.43 / 0.54 | |

The `eval.scores` means are not comparable across conditions as quality measures: they average over different user
turns (B adds the follow-ups; A's FD-Bench v2 collisions score 0 for turn-taking reasons of the script's making).

## 3. Consistency: A and B are the same until the user's second turn

Until the user's second turn, A and B now feed the agent the same microphone, sample for sample
(`tests/test_benchmark_users.py`; in the full-audio run B alone had the −60 dBFS floor). What is left between A and B
is the server's own nondeterminism: MiniCPM-o's decoding is not batch-invariant at 16 (8) concurrent sessions. The
noise floor is A vs A of two seeds (identical input, different batch neighbours).

| benchmark | what is compared (agent, before the user's 2nd turn) | A vs B (this run) | A vs A, two seeds (noise floor) | A vs B, full-audio run (with the floor in B) |
|---|---|---|---|---|
| FD-Bench v3 | first agent turn, same start and text | 0.556 (278/500) | 0.581 (581/1000) | 0.073 (22/300) |
| FD-Bench v3 | first-response timing metrics identical | 0.980 (490/500) | 0.974 (974/1000) | 0.56 (168/300) |
| FD-Bench v3 | same fulfilment, A vs B first response | 0.934 (467/500) | 0.930 (930/1000) | — |
| FD-Bench v2 | agent speech identical | 0.828 (497/600) | — | 0.40 (161/400) |
| FD-Bench v2 | first agent onset identical | 0.990 (594/600) | — | 0.50 (198/400) |
| Audio MultiChallenge | agent speech identical | 0.195 (39/200) | 0.18 (18/100) | 0.18 (18/100) |
| Audio MultiChallenge | first agent onset identical | 0.990 (198/200) | 0.99 (99/100) | 0.96 (96/100) |

**A vs B is now at the A vs A noise floor on every measure.** Onsets and timing agree in 98–99 % of pairs; texts
agree as often as two runs of the same input do. FD-Bench v3's B first response scores 0.824 against A's 0.822
(pass@1turn, +0.002 [−0.018, +0.022]). So **every B − A difference below is the closed loop itself**, not a
difference in the input. Audio MultiChallenge's first turns are long (~18 s recordings that MiniCPM-o often talks
inside), so its texts diverge early under batch nondeterminism in both comparisons alike.

## 4. Full-Duplex-Bench v3

100 recordings × 5 seeds = 500 pairs; 16 kHz Opus 32 kbps transcode of the 48 kHz release. MiniCPM-o has no tools, so
the outcome is **spoken fulfilment**: did its speech confirm every parameter of the request with its final value
(281 slots, rules with a proxy-LLM fallback; `benchmarks.fdb3_spoken`).

**Pass by user turn.** pass@N-turn = fulfilment over the agent's speech before user turn N + 1 (the conversation up
to the agent's response after the N-th user turn; carried forward once a conversation has ended; definitions in
[docs/BENCHMARKS.md](../docs/BENCHMARKS.md), "Pass by user turn"). The recording is user turn 1, so **A measures
pass@1turn** and is the N = 1 reference. The closed-loop user stops after at most 8 turns (median 4).

| N (user turns) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 (all) |
|---|---|---|---|---|---|---|---|---|
| A (open loop) | **0.822** [0.748, 0.888] | — | — | — | — | — | — | — |
| B pass@N-turn | 0.824 [0.744, 0.892] | 0.944 [0.914, 0.970] | 0.964 [0.940, 0.984] | 0.976 [0.956, 0.992] | 0.978 | 0.980 | 0.982 | **0.984** [0.970, 0.996] |
| B − A | +0.002 [−0.018, +0.022] | +0.122 [+0.070, +0.178] | +0.142 [+0.084, +0.206] | +0.154 [+0.092, +0.222] | +0.156 | +0.158 | +0.160 | +0.162 [+0.100, +0.230] |
| conversations still going | 100 % | 100 % | 96 % | 74 % | 40 % | 23 % | 16 % | 11 % |

Three quarters of the closed-loop gain arrives with the user's first follow-up (N = 2): the open loop scores the
agent's opening move, the closed loop whether the exchange gets there.

**By subset** (A / B final): easy 0.83 / 0.99, medium 0.84 / 0.99, hard 0.79 / 0.97; PAUSE 0.74 / 0.98, FILLER 0.81 /
0.98, HESITATION 0.80 / 1.00, FALSE_START 0.87 / 1.00, SELF_CORRECTION 0.95 / 0.99. Slot score (share of parameters
confirmed correctly) A 0.890 → B final 0.991. Per-seed sd: A 0.02, B final 0.01.

**Why A fell short** (89 failures; first match): asked a question instead of confirming 45 % (37/40 recovered in B),
misheard 22 % (18/20), missed a detail 18 % (16/16), interrupted in a pause 10 % (8/9), wrong after a self-correction
4 % (4/4). 93 % of them were recovered in B.

**User effort in B:** 3.60 user turns after the recording; 15 % of conversations include a correction, 34 % an answer
to the agent's question, 14 % a repetition. Failed openings cost the user more: 4.29 turns and 0.47 corrections when
recovered, against 3.48 and 0.13 when the opening was right. 61 % of recoveries came after the user itself restated
every missing value.

**Claimed results.** Without tools, the agent claims a completed action or reports a result it cannot know in 53 % of
B conversations (A 5 %). The response-quality judge rewards it (A 0.29 → B 0.85; in B 0.93 when it claims, 0.77 when it
does not), so response quality is not a usable outcome for a tool-less agent in closed loop.

**Timing** (official definitions, first user turn): turn-take 1.000 (A) / 0.998 (B), interruption rate 0.126 /
0.110, first response 1020 / 1020 ms (median, non-interrupted). `eval.timing_counts` over whole conversations:
latency median 0.76 s (A) / 0.84 s (B), cut-in rate 0.15 / 0.15. No collisions (A has one user turn; the B user waits
for the agent), so the open-loop user-turn failure rate is 0 by construction.

## 5. Full-Duplex-Bench v2

200 tasks × 3 seeds = 600 pairs. A: the static script (120 s cap); B: the live examiner (180 s cap, also scored on
the official 120 s window).

**Official judge (proxy).** B at 180 s; the 120 s window in brackets.

| split | n | TT A | TT B | IF A | IF B | IF B − A [95% CI] | task A | task B | task B − A [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| all | 600 | 4.09 | 4.39 (4.38) | 3.90 | 4.33 (4.18) | +0.43 [+0.34, +0.51] (+0.28) | 3.82 | 4.37 (4.23) | +0.54 [+0.38, +0.72] (+0.41) |
| Daily | 150 | 4.06 | 4.49 | 3.93 | 4.40 | +0.48 [+0.35, +0.60] | — | — | — |
| Correction | 150 | 4.21 | 4.53 | 4.09 | 4.42 | +0.33 [+0.19, +0.48] | 4.24 | 4.49 | +0.25 [−0.02, +0.54] |
| EntityTracking | 150 | 3.85 | 4.30 | 3.47 | 4.28 | +0.81 [+0.64, +0.96] | 2.93 | 4.26 | +1.33 [+1.10, +1.55] |
| Safety | 150 | 4.22 | 4.25 | 4.11 | 4.20 | +0.09 [−0.04, +0.22] | 4.30 | 4.35 | +0.05 [−0.11, +0.19] |

**Stages** (the time cap is not a failure; a proxy-LLM stage analysis marks each goal T1–T4 reached / completed / how
the agent did; IF_r / task_r = the official judge on the reached goals):

| cond | ended (end phrase) | goals reached | goals completed | all 4 completed | stage score | IF_r | task_r |
|---|---|---|---|---|---|---|---|
| A | 90 % | 3.79 | 3.42 | 67 % | 0.84 | 3.93 | 3.87 |
| B (180 s) | 68 % (31 % forced by the closing rule) | 3.72 | 3.36 | 62 % | 0.89 | 4.45 | 4.47 |
| B − A [95% CI] | | | −0.06 [−0.17, +0.05] | | +0.05 [+0.02, +0.07] | +0.52 [+0.44, +0.60] | +0.60 [+0.44, +0.78] |

**Pass by examiner line** (stages@N-turn: the stage analysis on the conversation before examiner line N + 1; A's
lines are the script's, which never react to the agent):

| N (examiner lines) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | all |
|---|---|---|---|---|---|---|---|---|---|
| goals completed, A | 0.22 | 0.94 | 1.75 | 2.62 | 3.16 | 3.32 | 3.37 | 3.40 | 3.42 |
| goals completed, B | 0.40 | 1.14 | 1.93 | 2.68 | 2.95 | 3.22 | 3.30 | 3.33 | 3.36 |
| stage score, A | 0.55 | 0.74 | 0.80 | 0.81 | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 |
| stage score, B | 0.69 | 0.80 | 0.83 | 0.85 | 0.86 | 0.89 | 0.88 | 0.89 | 0.89 |
| still going A / B | 100 / 100 % | 100 / 100 % | 100 / 97 % | 99 / 88 % | 95 / 74 % | 41 / 35 % | 16 / 16 % | 7 / 8 % | |

(CIs in `data/summary.json`, `turns.fdb2`; ±0.1 on goals completed, ±0.03 on the stage score.) Through the first four
lines B completes goals at least as fast as A and does them better (stage score +0.03 to +0.14). After that the static
script, which was written against a reference assistant that always answered as expected, keeps marching through
its goals regardless of what MiniCPM-o said, while the live examiner retries goals the agent did not complete and
runs into the cap (32 % of B conversations time out against 10 % of A). The goal count is therefore a property of the
script's pacing as much as of the agent; the stage score (how the agent did on what was reached) is the
condition-fair outcome.

**Duplex timing** (`eval.timing_counts`):

| cond | turn-take rate [collisions left out] | latency median / p90 s | cut-in rate | collisions / conversation (share of user turns) | agent yielded on collisions |
|---|---|---|---|---|---|
| A | 0.76 [0.96] | 1.06 / 2.52 | 0.02 | 3.58 (64 %) | 44 % |
| B | 0.97 | 1.01 / 1.57 | 0.02 | 0 | — |

The examiner never barges in (slow mode), so there are no intended barge-ins. MiniCPM-o's latency is now the same in
A and B (1.06 / 1.01 s; with the floor in B it was 1.08 / 1.30 s).

**Open-loop user-turn failure rate** (`turn_validity`, per examiner line after the first; B = the checker's noise
floor): **87 % of A's replayed lines are invalid** (timing / collision 77 %, content: responds to something the agent
never said 23 %, LLM-flagged 31 %) against 18 % flagged in B (0 % timing, 7 % content). Excess LLM-flagged lines +13
[+11, +15] points per conversation; 100 % of A conversations contain at least one invalid line.

## 6. Audio MultiChallenge

100 conversations × 2 seeds = 200 pairs, official-style examinee prompt. The official judge prompt (proxy) scores the
agent's reply to the last user turn per rubric item, on the conversation as it happened (a duplex model cannot be
handed someone else's replies as its history, so this is not the official fixed-history protocol). A and B have the
same number of user turns; every conversation completed in both.

| axis | n | rubric items passed A | B | B − A [95% CI] | all items passed A | B | B − A [95% CI] |
|---|---|---|---|---|---|---|---|
| all | 200 | 0.241 [0.186, 0.301] | 0.285 [0.229, 0.342] | +0.044 [−0.002, +0.092] | 0.065 | 0.075 | +0.010 [−0.025, +0.045] |
| Inference memory | 50 | 0.117 | 0.158 | +0.042 [−0.048, +0.137] | 0.06 | 0.12 | +0.06 [−0.02, +0.14] |
| Instruction retention | 50 | 0.288 | 0.297 | +0.008 [−0.100, +0.115] | 0.12 | 0.10 | −0.02 [−0.10, +0.08] |
| Voice editing | 50 | 0.419 | 0.509 | +0.090 [−0.018, +0.204] | 0.06 | 0.04 | −0.02 [−0.08, +0.04] |
| Self-coherence | 50 | 0.140 | 0.176 | +0.035 [−0.021, +0.094] | 0.02 | 0.04 | +0.02 [0.00, +0.06] |

- **Level.** MiniCPM-o passes a quarter of the rubric items; 7 % of conversations pass every item.
- **Open vs closed: closed ≥ open on every axis, at the edge of significance overall** (+0.044, lower bound −0.002,
  with two seeds; the full-audio run had +0.047 [−0.025, +0.121] with one). The rubrics test memory of facts and
  instructions that the closed-loop rewrite keeps, so later turns depend little on the reply; no axis is significant
  at n = 50.
- **Not turn-indexed.** The rubric grades only the reply to the last user turn, and both conditions have the same
  turns, so there is no pass@N curve here.
- **Duplex timing:** turn-take rate 85 % (A) / 92 % (B), latency median 0.75 / 0.84 s (p90 1.25 / 1.35 s), cut-in
  rate 15 % / 8 %; no collisions or barge-ins (the user waits for the agent in both conditions). The recorded A turns
  are long and pause-rich, and MiniCPM-o cuts into them twice as often as into the rewritten B turns.
- **Open-loop user-turn failure rate:** invalid A 25 % vs B 18 %, **+7 [+3, +11] points** per conversation; content
  11 % vs 6 %, ignored question 13 % vs 12 %; 56 % of A conversations vs 46 % of B have at least one flagged turn.

## 7. What only closed loop measures

The open loop fixes the user's side, so anything that depends on the user reacting to the agent cannot be measured
there. In these results:

**Measured here, closed loop only.**

- **FD-Bench v3 recovered / broken rates and B-final fulfilment** (0.933 / 0.005; 0.984), and the whole pass@N-turn
  curve beyond N = 1. The open loop has one user turn: it cannot tell an agent that asks a sensible question
  (45 % of A's failures) from one that fails.
- **User effort**: follow-up turns (3.60), corrections, answers and repetitions per conversation, and how much more a
  failed opening costs the user (4.29 vs 3.48 turns).
- **Claimed-results rate** (53 % of B conversations): an agent without tools pretending to have acted only shows once
  a user goes on talking to it.
- **FD-Bench v2 stage completion with the stage-aware examiner**: goals reached / completed against an examiner that
  retries what the agent did not do, IF_r / task_r on the reached stages, stages@N-turn, and the cap diagnosis (why
  a conversation timed out).
- **Zero-collision turn-taking**: B's turn-take rate (0.97), latency and cut-in rate on lines that start when the user
  would actually speak. In A, 64 % of FD-Bench v2 lines collide with the agent's speech because their start times
  were fixed in advance.
- **The open loop's own error, the open-loop user-turn failure rate**: 87 % of FD-Bench v2 script lines and 25 % of
  Audio MultiChallenge recorded turns are invalid for the conversation that actually happened (+13 / +7 points above
  the checker's closed-loop floor). Only a closed-loop run supplies that floor, so this error of the open-loop
  protocol is itself a closed-loop measurement.

**Not measured yet** (the env supports the behaviours; these benchmarks or these users do not exercise them):

- **Consequences of the agent talking over the user**: whether the user yields, repeats itself or gets annoyed after
  a cut-in. The users here wait for the agent (FD-Bench v3 / Audio MultiChallenge `yield_after_ms=None`, FD-Bench v2
  slow mode), so cut-ins are counted but have no consequence.
- **Closed-loop barge-ins with content-conditioned intent**: users who interrupt because of what the agent just said
  (a correction, a question, "stop"), with the intent recorded (`LLMListener`). No benchmark user here barges in;
  the only barge-ins are the replayed ones of the open-loop-only benchmarks.
- **Directed vs non-directed sounds placed on the agent's actual speech**: backchannels, asides and noises timed to
  the agent's phrase boundaries as it speaks (the persona-driven user's random events and listener decisions). The
  benchmark users here are deliberately silent apart from their turns (no background, no events), so A and B hear
  the same line.

## 8. Comparison with the full-audio run

The full-audio run ([BENCHMARKS_full_audio.md](BENCHMARKS_full_audio.md), env `6f6853d` episodes, 2026-10-08/09) used
MiniCPM-o with Talker audio on the 2-GPU deployment, the −60 dBFS floor under closed-loop users, 6 sessions per
server, fewer seeds and the log-normal latency curve. The Thinker-only deployment is not bit-identical (logprobs
differ at bf16 level from the first speak unit), and its timing comes from the text at `speech_cps`, so per-episode
comparisons are not meaningful; aggregates are:

| metric | full-audio run | this run |
|---|---|---|
| FD-Bench v3 fulfilment A / B final / B − A | 0.817 / 0.983 / +0.167 | 0.822 / 0.984 / +0.162 |
| FD-Bench v3 B first response (should equal A) | 0.867 (floor in B) | 0.824 |
| FD-Bench v3 claims in B / user turns in B | 50 % / 3.55 | 53 % / 3.60 |
| FD-Bench v2 IF A / B / B − A | 3.93 / 4.26 / +0.33 | 3.90 / 4.33 / +0.43 |
| FD-Bench v2 IF_r B − A / stage score B − A | +0.43 / +0.07 | +0.52 / +0.05 |
| FD-Bench v2 invalid script lines / collisions per conversation | 85 % / 3.33 | 87 % / 3.58 |
| FD-Bench v2 latency A / B (s) | 1.08 / 1.30 | 1.06 / 1.01 |
| Audio MultiChallenge rubric A / B / B − A | 0.264 / 0.311 / +0.047 | 0.241 / 0.285 / +0.044 |
| Audio MultiChallenge latency A / B (s) | 1.03 / 1.30 (floor in both) | 0.75 / 0.84 |
| Audio MultiChallenge open-loop failure excess | +7 pts | +7 pts |

The benchmark outcomes and the open- vs closed-loop differences replicate within their CIs. What changed is what the
floor caused: B's first response now equals A, and MiniCPM-o's latency where the floor was present dropped by
0.2–0.5 s (FD-Bench v2 B 1.30 → 1.01 s; Audio MultiChallenge 1.03 / 1.30 → 0.75 / 0.84 s). FD-Bench v2's IF / IF_r
gains grew slightly (+0.33 → +0.43, +0.43 → +0.52): in the full-audio run the floor delayed MiniCPM-o's answers in B
only, working against B.

## 9. Limitations

- **Proxy judges and examiner.** Every judge, the FD-Bench v2 examiner and reference assistant, and both closed-loop
  users are the same local Qwen3.8-27B-FP8, not the official GPT-4o / GPT-Realtime / Gemini 2.5 Flash / o4-mini.
  Absolute levels are not comparable with published tables; A and B share the judge, so differences are more
  reliable than levels. The judge is also not perfectly deterministic under batching: two report passes over the
  same FD-Bench v3 episodes gave A fulfilment 0.814 and 0.822 (4 of 500 verdicts differ); the numbers here are from
  one consistent pass.
- **Thinker-only timing is estimated.** When MiniCPM-o starts speaking is its own decision at a unit boundary; how
  long each utterance lasts comes from its character count (off by a few tenths of a second per utterance), which
  shapes overlaps and when the user may reply.
- **Simulated users.** LLM users with cloned TTS voices; they may be more cooperative than real people. On FD-Bench v3
  they know the target values and repeat them (61 % of recoveries came after the user restated every missing value),
  so closed-loop success is partly a measure of the user's cooperation. No human calibration was done.
- **Spoken fulfilment** measures confirmation of parameters, not task completion (the agent has no tools). Audio
  MultiChallenge is not run with its official fixed-history protocol.
- **Nondeterminism.** MiniCPM-o's decoding is not batch-invariant at 16 concurrent sessions; per-episode texts differ
  between runs and between A and B even where the input is identical (§3); onsets and aggregates are stable. No
  batch-1 control was run.
- **Sizes.** Audio MultiChallenge has 100 conversations (25 per axis); per-axis differences are not reliable.
- **Ported measurement.** FD-Bench v2 Channel B is the agent's text with segment times (not ASR word chunks); only the
  slow examiner mode is ported; the open-loop v2 script is one plausible static construction, not an official one.
  FD-Bench v3 audio is a 16 kHz Opus transcode with the room-tone tail after the request removed in both conditions.
- **Dataset licenses** (results only; no data is redistributed): **Full-Duplex-Bench v2 and v3: CC BY-NC 4.0**
  (non-commercial research only; this covers the ported prompts, mock APIs and judge prompts in the optional
  component `extras/fdbench/` too). Audio MultiChallenge: MIT. Models: MiniCPM-o 4.5, Qwen3.8-27B and Qwen3-TTS are
  under their own terms.
