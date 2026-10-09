# Existing duplex benchmarks as test sets

Code: `interaction_gym.benchmarks` (`full_duplex_bench`, `easy_turn`, `humdial`, `fdb3`, `fdb3_spoken`, `fdb2`, `audiomc`,
`clip_bank`, `wav`). Runners: `examples/fdb_baseline.py --bench fdb|easy_turn|humdial` (run, then report);
`examples/fdb3_ab.py` (FD-Bench v3, open vs closed loop); `examples/fdb2_ab.py` (FD-Bench v2, §12);
`examples/audiomc_ab.py` (Audio MultiChallenge, §13).
Tests: `tests/test_benchmarks.py` (tiny synthetic fixtures; no benchmark data in the repo).

**Licenses.** This repository never redistributes benchmark data: the modules here are loaders and converters
only, and you download each dataset yourself under that dataset's own license. Data licenses (details in §1):
Full-Duplex-Bench v1.0 CC BY-NC 4.0 for the Candor/ICC parts (plus upstream terms), MIT for the synthetic parts;
v1.5 MIT; v2 and v3 CC BY-NC 4.0 (non-commercial research only); Audio MultiChallenge MIT; Easy Turn and HumDial-FDBench
Apache-2.0; FastTurn treated as CC BY-NC (drawn from SmoothConv); MTR-DuplexBench CC BY 4.0. Code and prompts
ported verbatim from the official repositories live apart from our own code:
the separate optional component `extras/fdbench/` (`interaction-gym-fdbench`, extra `fdbench`; CC BY-NC 4.0, non-commercial, not part of the Apache-2.0 package: the v1/v1.5
metric logic and judge prompt, the v2 judge rubric prompts, the v3 mock APIs, tool schemas, agent instructions
and judge prompts) and `src/interaction_gym/benchmarks/third_party/audiomc/` (MIT: the Audio MultiChallenge
judge prompt). The public module APIs (`full_duplex_bench.sample_metrics`, `fdb2.judge_prompt`, `fdb3.pass_at_1`,
`audiomc.JUDGE_PROMPT`, ...) are unchanged. See [THIRD_PARTY.md](../THIRD_PARTY.md) for the provenance of each file.

## Agent output: text-only by default (MiniCPM-o runners)

Since 2026-10-09 every runner here that drives MiniCPM-o 4.5 (`fdb_baseline.py`, `fdb3_ab.py`, `fdb2_ab.py`,
`audiomc_ab.py`, `minicpmo_suite.py`, `minicpmo_agent.py`, `gpu_tuner.py`, and `gpu_tuner.probe_agent`) runs it
**Thinker-only with text output**, as our RL rollouts do: `VllmOmniDuplexAgent(audio_out=False)` against
the reference deployment's Thinker-only servers (`AGENT_LAYOUT=thinker`, one GPU and 16 sessions each,
`patches/vllm-omni/`, part 06 `minicpmo-thinker-only`). The agent's text is timed at `speech_cps` (MiniCPM-o 4.5: 11.3 characters
per second, calibrated on lockstep audio episodes). `--audio-out` selects the full Thinker + Talker + Code2Wav
deployment (`AGENT_LAYOUT=audio`) and the agent's real speech. `meta.agent.output` records which (`"audio"` or
`"text @ 11.3 chars/s"`), and a runner refuses to resume an output folder that holds the other mode.

Caveats:

- **Not bit-identical to the two-GPU audio deployment.** The Thinker sees the same inputs, but from the first
  speak unit its logprobs differ at bf16 level (mean |diff| 0.013 over the first 24 tokens, max 0.14, measured
  2026-10-04: one stage without `async_chunk` takes another scheduling / kernel path), so sampled trajectories
  diverge after a few units. Behaviour statistics match (speak fraction, text tokens per episode, RL reward).
- **Timing is estimated, not taken from real audio.** When the agent starts speaking is still the model's own
  decision at a unit boundary; how long each utterance lasts (and so overlap, when it stops, when the user may
  reply) comes from its character count. The estimate is off by a few tenths of a second per utterance.
- **Not directly comparable with full-audio runs.** `results/BENCHMARKS.md` (the release results) is a text-only
  run; `results/BENCHMARKS_full_audio.md` (2026-10-08/09) is a full-audio run (`--audio-out` semantics); compare
  text-only runs with text-only runs.

Metrics that read the agent's audio, and what they use instead. No port here runs ASR on the agent's audio: the
agent's words always come from its own transcript. The only agent-audio input is FD-Bench v1.0 / v1.5's voiced
spans (`full_duplex_bench.speech_spans` / `agent_words`: an energy VAD over each agent turn's audio, §5); without
audio they fall back to the whole turn span, i.e. the estimated duration. This affects pause-handling and
backchannel TOR (word spans), smooth-turn-taking / interruption latency (the first word sits at the turn start, not at
the first voiced frame), backchannel frequency / JSD (segment durations) and v1.5 stop / response latencies and overlaps. We
compute these from the text timing too rather than keep audio for FD-Bench v1 alone: an audio-output FD-Bench run
would come from the other deployment (not bit-identical, above), so one cross-benchmark table would mix two
policies, while the official metrics are already approximations (§5: transcript words spread over the turn,
energy VAD instead of ASR + Silero). Use `--audio-out` when a number must come from the agent's real speech, and
for recordings, listening and demos (text-only episodes have no agent audio in the media store or the viewer).
Easy Turn, HumDial, FD-Bench v2 / v3 and Audio MultiChallenge read only turn times and text (judges included).

## No background in evaluations

Since 2026-10-09 every benchmark user is **silent apart from what it says**: no background track and no random
noise events or asides. The closed-loop users and examiners of `fdb2_ab.py`, `fdb3_ab.py` and `audiomc_ab.py` (both
conditions of Audio MultiChallenge) are built with `benchmarks.benchmark_user`, i.e. `UserSim` with
`Soundscape(background=False)` (no surroundings track, no −60 dBFS floor, no sound bank) and
`Behaviors(aside_per_min=0, noise_per_min=0)` (explicit; the benchmark personas had no random events anyway).
`minicpmo_suite.py` uses the same user, and keeps a background only for a scenario that sets one itself
(`booking-mina-cafe`'s cafe noise, which that scenario tests). The open-loop runners (`fdb_baseline.py --bench
fdb|easy_turn|humdial`, and the replayed A conditions of FD-Bench v2 / v3) play `ReplayUser` turns; a recording's
own residual (room tone between the turns of `input.wav`) stays, since it is part of the benchmark input.

Why: the replayed A condition (`ReplayUser`) has no background, so a closed-loop user with the env's general
default (a background from the persona's surroundings: for these "quiet" personas a −60 dBFS pink-noise floor)
made A and B differ before the user's second turn. MiniCPM-o heard the floor and answered later
(2026-10-08/09 full-audio run, `results/BENCHMARKS_full_audio.md`; without the floor, FD-Bench v2 B latency 1.30 →
1.01 s and Audio MultiChallenge 1.03 / 1.30 → 0.75 / 0.84 s, `results/BENCHMARKS.md` §8). Without it, A and B give the agent the same
microphone, sample for sample, until the user's second turn (`tests/test_benchmark_users.py`, FD-Bench v2 and v3
with a fake agent), so B − A is the closed loop alone.

The env's general default for other simulations is unchanged: `UserSim()` lays the persona's surroundings track
under the episode (`soundscape.py`). Pass `soundscape=Soundscape(background=False)` (or use `benchmark_user`)
wherever two conditions must hear the same line.

## 1. What exists and what is usable (checked 2026-10-03)

| Benchmark | What | Data / size | License | Access | Status here |
|---|---|---|---|---|---|
| **Full-Duplex-Bench v1.0** (arXiv 2503.04721) | pause handling (Candor 216, synthetic 137), backchannel (ICC 55), smooth turn-taking (Candor 119), user interruption (synthetic 200); static `input.wav` + annotation JSON | ~480 MB | CC BY-NC 4.0 (Candor/ICC, plus upstream terms); MIT (synthetic) | authors' Google Drive; **HF mirror `Ssshangfu/Full-Duplex-Bench-Data`** | **integrated** |
| **Full-Duplex-Bench v1.5** (arXiv 2507.23159) | overlap handling: user_interruption 200, user_backchannel 98 (README says 99), talking_to_other 100, background_speech 100; `input.wav` + `clean_input.wav` + `metadata.json` | ~230 MB | MIT | same HF mirror | **integrated** |
| **Full-Duplex-Bench v2** (arXiv 2510.07838) | automated examiner (GPT-Realtime) runs a multi-turn task with staged goals; 200 tasks: Daily (5 classes), Correction, EntityTracking, Safety × 50; LLM judge (turn-taking fluency, instruction following, task score) | no audio: `v2/prompts_staged_200.json` (650 KB) in the official repo | **CC BY-NC 4.0** | official GitHub repo | **integrated** (`fdb2`, §12): examiner as a `UserSim` source (closed loop) and agent-free static scripts (open loop) |
| **Audio MultiChallenge** (arXiv 2512.14865, `ScaleAI/audiomc`) | 452 multi-turn conversations, 47 speakers (real speech, 3–8 user turns), axes inference memory / instruction retention / self-coherence / voice editing; 1,712 rubric items on the final reply | 5.3 GB parquet (15 h of user audio) | MIT | HF | **integrated** (`audiomc`, §13) |
| **Full-Duplex-Bench v3** (arXiv 2604.04847) | tool use under real disfluency: 100 human recordings (79 scenarios, 12 speakers), 4 domains, 12 mock APIs, expected tool calls | 100 × `input.wav` 48 kHz (zip from Google Drive) | **CC BY-NC 4.0** (code and data): non-commercial research only | authors' Google Drive (if a GPU host cannot reach Drive, download elsewhere and copy; 16 kHz Opus 32 kbps is enough) | **integrated** (`fdb3`, §11) |
| **Easy Turn testset** (arXiv 2509.23938, `ASLP-lab/Easy-Turn-Testset`) | **turn-state classification** clips: complete 300 / incomplete 300 / backchannel 100 / wait 100 ("wait" = "stop talking / be quiet"), all Mandarin, real:synthetic 1:1 | 800 wavs, 135 MB | Apache-2.0 | HF | **integrated** (`easy_turn`, §3): a classifier test, turned into short episodes (§3) |
| FastTurn testset (arXiv 2604.01897, `ASLP-lab/FastTurn-Testset`) | same 4 states, 22,432 clips / 12.9 h, Mandarin | 3.8 GB tar | card says Apache-2.0 but drawn from SmoothConv (CC BY-NC 4.0): treat as NC | HF | usable for research; same mapping as Easy Turn |
| **HumDial-FDBench** (ICASSP 2026, arXiv 2604.21406, `ASLP-lab/HumDial-FDBench`) | test set only (`Humdial-Track2-Test.zip`): human-recorded **user channel** (mono 16 kHz; the paper's dual-channel conversations and the train/dev sets are not released), en 2,500 + zh 2,499 samples in 10 categories: interruptions ask / deny / repeat / shift / wait (300 each per language), backchannel 300, pause 300, talk_to_others 100, third party talking to the user before 150 / after 150; segment timestamps + word timestamps; `clean_*` without the event for most | 1.8 GB zip, 6.9 GB unpacked | Apache-2.0 | HF | **integrated** (`humdial`, §4) + real clips in the clip bank |
| MTR-DuplexBench (arXiv 2511.10262, `Jeff0918/MTR-DuplexBench`) | multi-round (≈10 rounds) synthetic dialogues: turn-taking, interruption, pause, background, backchannel; + quality/IF/safety | 3.8 GB | CC BY 4.0 | HF | usable; multi-round replay fits `ReplayUser`; not integrated yet |
| Talking Turns (Arora et al., ICLR 2025) | turn-taking judge on Switchboard | not released | Switchboard is LDC | — | not usable |

## 2. Full-Duplex-Bench: data → tasks

`full_duplex_bench.load(root, subset)` → `Sample(task, turns, background, duration_ms)`; `make_env(sample, spec)`.

- The user's turns are **slices of `input.wav`** at their original times; everything outside the turns (room tone,
  breaths, the Candor/ICC channel noise) is a per-episode background track, so the agent's microphone carries
  `input.wav` sample for sample (tested). Audio is resampled once, band-limited, to the agent's rate (16 kHz).
- The episode lasts exactly as long as `input.wav` (the official `output.wav` has the input's length) and never
  ends early on silence; `tail_ms` adds time after it (not official).
- `task.scenario["benchmark"]` keeps the version, subset, sample id, source path, license and the raw annotation
  (`pause.json`, `turn_taking.json`, `interrupt.json`, `metadata.json`).

| Subset | Turns (`kind` / `expects`) |
|---|---|
| v1.0 candor / synthetic pause handling | speech between `[PAUSE]` spans; each turn before a pause `expects: "wait"`, the last one normal |
| v1.0 candor turn taking | one normal turn ending at the `[TURN-TAKING]` time (agent should take the turn → `respond`) |
| v1.0 ICC backchannel | the speaker's monologue split at word gaps ≥ 1 s; all but the last `expects: "wait"` |
| v1.0 synthetic user interruption | context question (normal) + interruption `expects: "yield"` |
| v1.5 user_interruption | context + interruption `expects: "yield"` |
| v1.5 user_backchannel | context + `kind: "backchannel"` |
| v1.5 talking_to_other | context + `kind: "aside"` |
| v1.5 background_speech | context + `kind: "noise"` (third-party speech; FD-Bench mixes it into the user channel, so it is a turn, not a `background` track) |

v1.5 clean variants (`clean=True`) play `clean_input.wav` (no overlap event); the v1.5 behaviour judge compares
the two outputs.

## 3. Easy Turn: clips → episodes

`easy_turn.load(root, "easy_turn/<state>")` (`root` = the dataset checkout with `testset/`). Each clip is cut to its
voice and starts at 0.5 s; the episode has a fixed length, so replies are never cut by an early end.

| State | Episode | Expectation scored (one per user turn) |
|---|---|---|
| complete (300) | the clip, then 5 s | `respond` (not talked over, answered) |
| incomplete (300) | the clip ("因为小时候…"), then 3 s | `wait` (the agent should not take the floor) |
| backchannel (100) | composed: an opening (a `complete` clip of the same speaker, else a stable pick of the same real/synthetic kind), then the clip 4 s after the opening ends, while the agent answers, `kind: "backchannel"` | opening: `respond`; clip: `ignore` |
| wait (100) | composed the same way; the clip ("别说了", "立即静音") `expects: "wait"` | `wait` (stop if talking, then do not take the floor) |

## 4. HumDial-FDBench: data → episodes

`humdial.load(root, "humdial/<en|zh>/<category>")` (`root` = the unpacked zip). Like v1.5: request, then 5 s after it
ends (fixed offset, ±0.7 s) the event; the episode lasts the wav (≈10 s after the last segment); everything outside the
turns is a background track, so the agent hears the wav sample for sample. `clean=True` plays `clean_<id>.wav`.

| Category | Turns |
|---|---|
| ask / deny / repeat / shift | request + event `expects: "yield"` (stop, then answer) |
| wait ("I'm busy, talk later", "hold on") | request + event `expects: "wait"` (stop, then stay quiet) |
| backchannel ("Oh, I see.", "说得不错") | request + `kind: "backchannel"` |
| talk_to_others | request, a follow-up (normal turn: answered, yielded to if the agent talks), a remark to someone else `kind: "aside"` |
| others_talk_to_user_after / _before | request + third person talking to the user `kind: "noise"` (after it, or before the request) |
| pause | one request with `[break]`, cut at the longest gap between its word timestamps; the first half `expects: "wait"` |

All 4,999 samples load (checked). Splits: `humdial.split_of(lang, id, material_frac)` hashes the
**speaker** (`<speaker>_<item>`), so the clip bank and an evaluation never share a voice;
`humdial.ids(root, subset, split="eval", material_frac=0.5)` gives the held-out side.

## 5. Official metrics: parity notes (Full-Duplex-Bench)

`full_duplex_bench.sample_metrics(ep, media)` / `official_metrics(episodes, media, gt_distribution, ratings, behaviours)`
port `v1_v1.5/evaluation` (functions cited in the code):

| Task | Metrics | Port |
|---|---|---|
| pause handling | TOR (↓) | `eval_pause_handling.py`: any output with words spanning ≥ 1 s or > 3 words = take turn |
| smooth turn-taking | TOR (↑), latency (s, ↓) | `eval_smooth_turn_taking.py`: latency = first output word − annotated turn end, negative → 0, over TOR = 1 samples |
| backchannel | TOR (↓), freq (↑), JSD (↓) | `eval_backchannel.py` step for step, incl. its quirk that a later short segment resets TOR; JSD against `icc_gt_distribution.json` (pass the file from a checkout of the official repo) |
| user interruption (v1.0) | TOR (↑), latency (↓), relevance 0–5 (↑) | `asr.py --task user_interruption` crop + `eval_user_interruption.py`; judge prompt verbatim (`judge_interruption`) |
| v1.5 | behaviour C_RESPOND / C_RESUME / C_UNCERTAIN / C_UNKNOWN, stop / response latency | `get_timing.py` (`overlaps`, `response_gaps`, merge gaps 0.6 s / 0.5 s) + `eval_behavior.py` with `instruction/behavior.txt` (`judge_behaviour`) |

Differences that cannot be removed:

1. **Words** come from the agent's own text stream spread linearly over the voiced parts of each turn, not from
   ASR (parakeet-tdt-0.6b-v2) of `output.wav`. The model's transcript is more accurate than ASR, but its word
   timing is approximate (± a few hundred ms inside an utterance). TOR uses word count and span, latency the first
   word, so they shift slightly.
2. **Speech spans** come from an energy VAD on the agent's (clean, synthetic) audio with Silero's default
   min-speech / min-silence / padding; the user side uses the same VAD on the turn audio. Text-only agents (the
   runners' default, "Agent output" above) have no audio: each agent turn's span is its estimated duration.
3. **Judges**: official numbers use GPT-4-turbo (interruption relevance) and GPT-4o-2024-08-06 (behaviour);
   any `TextGen` can be passed. Our baseline used local Qwen3.8-27B-FP8: those columns are *proxy* numbers.
4. **v1.5 latency aggregation** is not specified in the code (it writes per-sample interval lists). `stop` / `resp`
   take the intervals of the overlap event (paper definitions); `stop_all` / `resp_all` average every interval.
5. The official interruption TOR / latency count *any* speech after the interruption ends — an agent that never
   stops and keeps talking past the interruption gets TOR = 1 with a small latency. `eval.scores` `yield` catches
   this (`kept_talking`).
6. Lockstep (`clock="input"`) excludes the model's compute time; MiniCPM-o decides once per 1 s unit, so its onsets
   fall on 1 s boundaries and latencies are quantized to its unit.

## 6. Reusing the data for our own scenarios

`clip_bank.extract(root, out, material_frac=0.5)` → `clips/<kind>/<id>.wav` + `manifest.jsonl` (format in the module
docstring: `id, kind, text, dur_ms, sr, audio, use (clip|opening|interruption), split (material|eval), source{…, license}`);
`clip_bank.human_stats(root)` → pause / backchannel / interruption statistics. Command:
`python -m interaction_gym.benchmarks.clip_bank <fdb data> <out> --material-frac 0.5 --humdial <humdial data>`
(`extract_humdial` adds HumDial's real clips; records carry `lang`, interruptions an `intent`).

**Example bank** (built with the command above: 454 MB, 3,552 material clips at `material_frac=0.5`):

| Source (license) | backchannel | aside | noise (3rd party) | interruption | opening |
|---|---|---|---|---|---|
| FD-Bench v1.5 (MIT, TTS) | 52 | 54 | 35 | 90 | 177 |
| HumDial en (Apache-2.0, human) | 110 | 37 | 130 | 592 (ask 117 / deny 125 / repeat 107 / shift 126 / wait 117) | 867 |
| HumDial zh (Apache-2.0, human) | 99 | 22 | 99 | 485 (ask 97 / deny 98 / repeat 101 / shift 96 / wait 93) | 703 |

**Seeding online scenarios:** `clip_bank.behaviors(manifest, human_stats, lang="en", backchannel_per_min=…, aside_per_min=…)` →
a `Behaviors` whose backchannel / aside wordings come from the material clips ("Of course.", "I feel the same way.",
"Go on."; zh "说得不错", "听着呢") and whose mid-thought pause length is the Candor p10–p90 (0.68–1.28 s) instead of the
default 1.2–2.5 s. Only the wording is reused (the user's TTS renders it). Playing the recorded clip audio itself in
an online episode needs a clip-playing voice path in `UserSim` (not built); the manifest has what it needs
(audio, duration, kind, lang, speaker split).

- **Clips (MIT, v1.5 only):** backchannels (98, 0.4–1.2 s, "yeah right", "uh-huh yeah" — TTS), asides to a named
  third person (100, "Dad, microwave just quit on us."), background speech (100, short TTS remarks). All TTS
  (synthetic) — they add variety of wording, not acoustic realism. Candor/ICC audio is real but CC BY-NC +
  upstream terms: statistics only by default.
- **Openings:** v1.5 `context_text` are 410 distinct single-turn, task-agnostic requests ("What's the population of
  Toronto?", "Mute the TV.") with TTS audio — usable as openings for open-domain episodes, not for task scenarios.
- **Calibration statistics** (`clip_bank.human_stats`):
  - Candor pauses: 277 mid-utterance pauses, median 0.87 s (p10 0.68, p90 1.28; the set selects pauses ≥ ~0.6 s),
    after a median 4.2 s / 13 words of speech; the word before the pause is most often a filler or connective
    ("like" 42, "um" 24, "and", "so", "but", "uh") — i.e. pauses mid-clause, not at clause ends.
  - Synthetic pauses are longer and more uniform (median 1.42 s) and sit after "or" / "and" / "but".
  - ICC human backchannels (pooled over the ICC crowd annotators, so counts are not per-listener rates): median
    0.47 s long; 94 % overlap the speaker's words; median 0.17 s after the end of the speaker's last word.
  - Interruptions are synthetic and placed at a fixed offset of the input (7.0 s after the context ends in v1.0,
    4.0 s in v1.5), not relative to the agent's speech: no information about where real users cut in.
- **Splits:** `full_duplex_bench.split_of(subset, id, material_frac)` (stable hash; v1.0 synthetic interruption k and
  v1.5 user_interruption k are the same content and share a split). Material openings/interruptions whose words
  occur in any eval-split sample are dropped (v1.5 reuses request texts across subsets).
- **Recommendation:** keep Full-Duplex-Bench **for evaluation only** (published numbers are over all samples, and
  its subsets are small). Take clips from HumDial (real, Apache-2.0, split by speaker) and, for volume, Easy Turn's
  trainset (Apache-2.0, 1,145 h with backchannel / wait classes; not downloaded). If FD-Bench clips are used anyway,
  report FD-Bench on `split="eval"` with the same `material_frac`; likewise HumDial on `humdial.ids(…, split="eval")`.

## 7. Baseline: MiniCPM-o 4.5

Measured 2026-10-03 on one host with 8× RTX 5090 32 GB: MiniCPM-o 4.5 on vLLM-Omni duplex (one server on 2 GPUs,
**audio output**: the runners' `--audio-out` today, not their text-only default), lockstep (`clock="input"`), 16 kHz,
no system prompt, token trace on. Each run writes a browsable directory
(`index.html` + `summary.json` + per-subset `episodes.jsonl` / `agent_traces.jsonl` / `per_sample.json`). Runs: all
1,723 FD-Bench episodes (incl. 410 v1.5 clean), all 800 Easy Turn clips, and 500 HumDial samples (25 evenly spaced
samples × 10 categories × 2 languages). Reports:
`python examples/fdb_baseline.py report --bench … --data … --out …` (FD-Bench judges: local Qwen3.8-27B-FP8, so
the rating / behaviour columns are proxies; verdicts are cached in `per_sample.json`, `--rejudge` to redo).
`eval.scores` is recomputed by the report with the episode end (censoring, below).

### 7.1 Full-Duplex-Bench: official metrics (ported) and eval.scores

**Scoring note.** The `eval.scores` columns in §7 were computed with the earlier scoring, which gave a user turn
two or three events (`no_interrupt` + `respond`, `yield` + `no_interrupt` + `respond`, ...) and averaged the
per-expectation means. `eval.scores` now gives exactly one score per user turn (docs/FORMAT.md §6.3) and `total` is
the mean over turns; recompute from the saved turns (`examples/fdb_baseline.py report`) for current numbers. The
duplex timing metrics (`eval.timing_counts`) are unchanged.

| Subset | n | Official (ours) | eval.scores total | by expectation |
|---|---|---|---|---|
| v1.0 Candor pause | 216 | TOR 0.306 ↓ | 0.792 | wait 0.760 · no_interrupt 0.943 · yield 0.035 (74) · respond 0.0 (n=1; 215 censored) |
| v1.0 synthetic pause | 137 | TOR 0.153 ↓ | 0.867 | wait 0.810 · no_interrupt 0.985 · yield 0.007 (26) · respond: all censored |
| v1.0 Candor turn-taking | 119 | TOR 0.958 ↑, latency 1.24 s | 0.559 | respond 0.287 (105/119 answered, median 1.08 s) · no_interrupt 0.832 |
| v1.0 ICC backchannel | 55 | TOR 0.018 ↓, freq 0.0006 ↑, JSD 0.997 ↓ | 0.676 | wait 1.0 · no_interrupt 0.673 · respond 0.0 (n=2, at 0 ms; 53 censored) |
| v1.0 synthetic interruption | 200 | TOR 0.935 ↑, latency 1.55 s, relevance 4.38 (proxy judge) | 0.512 | yield 0.103 (77 yielded / 57 kept talking; 66 not an overlap) · respond 0.287 · no_interrupt 0.970 |
| v1.5 user interruption | 200 | stop 1.55 s, resp 1.91 s; C_RESPOND 0.72 / C_RESUME 0.25 | 0.467 | yield 0.087 (101 / 78 kept talking) · respond 0.287 · no_interrupt 0.975 |
| v1.5 user backchannel | 98 | stop 0.61 s (93 of 98), resp 1.48 s; C_RESUME 1.0 | 0.766 | ignore 0.939 (6 replied) · respond 0.360 |
| v1.5 talking to other | 100 | stop 1.31 s, resp 1.56 s; C_RESPOND 0.58 / C_RESUME 0.30 | 0.583 | ignore 0.33 (67 replied to the aside) · respond 0.418 |
| v1.5 background speech | 100 | stop 0.89 s, resp 1.52 s; C_RESPOND 0.46 / C_RESUME 0.50 | 0.666 | ignore 0.60 (40 replied) · respond 0.398 |

(Not comparable to published tables: our words / VAD / judges differ, §5.) `respond` ≈ 0.29–0.42 everywhere is the latency curve of that time (1.0 at 200 ms, 0.27 at 1 s; replaced on
2026-10-09 by a logistic, docs/FORMAT.md §6.3): MiniCPM-o answers after a median 0.8–1.1 s because it decides
once per 1 s unit.

### 7.2 Easy Turn (all 800)

| State | n | total | by expectation | outcomes |
|---|---|---|---|---|
| complete | 300 | 0.699 | respond 0.412 · no_interrupt 0.987 | 297 answered (median 0.85 s), 3 never, 4 cut in |
| incomplete | 300 | 0.502 | **wait 0.023** · no_interrupt 0.980 | **took the floor in 293/300**, median 0.73 s after the fragment ended |
| backchannel | 100 | 0.699 | ignore 0.68 · respond 0.427 | 67 carried on, 32 replied to the "嗯，也是" |
| wait ("别说了") | 100 | 0.516 | yield 0.104 · wait 0.53 · respond 0.455 · no_interrupt 0.94 | 60 kept talking over "stop", 47 spoke again after it |

### 7.3 HumDial-FDBench (25 per category per language; small: ±0.1 on a rate)

| Category | en total | zh total | key outcome (en / zh) |
|---|---|---|---|
| ask | 0.413 | 0.452 | yielded 23/25 / 15/23 |
| deny | 0.388 | 0.445 | **kept talking 21/25 / 15/24**; never answered 12/25 / 10/25 |
| repeat | 0.401 | 0.440 | yielded 17/23 / 23/25 |
| shift | 0.452 | 0.470 | yielded 24/25 / 22/23 |
| wait | 0.460 | 0.455 | **kept talking 19/24 / 22/25**; then quiet 17/25 / 12/25 |
| backchannel | 0.590 | 0.732 | replied to it 10/25 / 6/25 |
| talk_to_others | 0.349 | 0.359 | replied to the aside 19/25 / 21/25 |
| others_talk_to_user_after | 0.543 | 0.592 | replied to the third person 15/26 / 12/25 |
| others_talk_to_user_before | 0.403 | 0.349 | replied to the third person 21/25 / 25/25 (see §8) |
| pause | 0.480 | 0.467 | took the floor in the pause 13/25 / 14/25 |

### 7.4 What the baseline says about MiniCPM-o 4.5

- Turn-end detection is purely "silence → talk": it answers 98–99 % of complete turns, but also **97.7 % of
  incomplete ones** (Easy Turn) and takes the floor in 23 % of Candor pauses (TOR 0.31 per sample) and in about half of HumDial `[break]` pauses.
- It yields to content interruptions (ask / shift / repeat: 70–95 %) but **not to "stop" or "no, that's wrong"**:
  HumDial wait / deny and Easy Turn wait → it keeps talking in 60–90 % of cases. Yield latency (median 1.5–2.5 s from
  the user's onset) is long because of the 1 s decision unit.
- Non-directed speech: it ignores backchannels (FD-Bench v1.5: 94 %), but replies to asides and third-party speech
  in 40–85 % of cases.
- Language: with no system prompt it often answers Mandarin in English (Easy Turn: 38 % of complete, 66 % of incomplete
  episodes; HumDial zh: 16–52 % per category).


## 8. Test-set quality problems found (by the baseline run)

Fixed in the converters / `eval.scores` (all FD-Bench, Easy Turn and HumDial numbers above use the fixed scoring):

1. **Episodes end at the user's last word.** FD-Bench pause handling and ICC inputs stop 0–150 ms after the final
   speech, so `respond` on the last turn was a guaranteed 0 (`respond 0.0` over 216 / 137 / 55 samples). `eval.scores`
   now takes `end_ms` and marks a turn whose answer window the episode cut `censored` (no score; schema, viewer and
   FORMAT.md updated; `traj.episode` passes the episode end).
2. **A turn cut by the episode end counted as a yield.** An agent still talking at the end of the input carries
   `unsaid`, which read as "stopped": 12 s "yield latencies" in HumDial. Now `kept_talking`.
3. **Replies to asides after the agent's own turn were missed.** `ignore` only checked whether the turn playing at the
   aside was cut; MiniCPM-o finishes its sentence and then says "Oh no, are you okay?" to "Dad, the microwave quit".
   `eval.reply_after` now catches a new agent turn within 2 s with no user turn in between; agreement with the FD-Bench
   v1.5 behaviour judge on C_RESPOND: 85 % (talking to other), 80 % (background), 94 % (backchannel).
4. `wait` turns said over the agent ("别说了", a pause resumed while the agent talks) now also require the agent to stop.

Problems in the data itself (not fixable by conversion; keep in mind when reading the numbers):

5. **Overlap events are placed at a fixed offset, not relative to the agent's speech** (FD-Bench v1.0 +7 s, v1.5
   +4 s, HumDial +5 s after the request). For MiniCPM-o, 66/200 v1.0 interruptions (33 %), 21/200 v1.5
   interruptions, 8–15 % of v1.5 asides / background speech and 4–7 % of composed Easy Turn events arrive when the
   agent has already finished: they are not interruptions at all. The official interruption TOR (0.935) still counts
   them; `eval.scores` scores them as normal turns (`respond`), not as `yield` (only 134 / 179 scored as `yield`).
6. **Official interruption TOR rewards not stopping**: any speech after the interruption ends counts, so an agent
   that talks straight through scores TOR = 1 with a small latency (57 of the 134 overlapping v1.0 cases here).
7. **No transcripts for Candor turn-taking** (only a `[TURN-TAKING]` time): the user turn has empty text, so text
   judges and the viewer show nothing. ASR it if needed.
8. **ICC backchannel** is about *producing* backchannels; `eval.scores` has no "backchannel expected" notion, so the
   converted episodes only score `wait` (official freq / JSD still computed). MiniCPM-o produces
   none (freq 0.0006, JSD 0.997).
9. **HumDial asides lack an addressee cue**: zh talk_to_others are remarks like "今晚的星空真美…", en "i'm thinking of
   going hiking this weekend…" — plausible to address the assistant; FD-Bench's asides name the person ("Dad, …").
   Expect `ignore` here only from models that use voice / prosody cues. Likewise **others_talk_to_user_before**: a
   different voice asks "Did you remember to ice your ankle?" before the user has said anything — answering it is
   reasonable; 46/50 episodes "fail". Treat this category as diagnostic, not as a pass/fail test.
10. **HumDial pauses**: 11 / 300 en `[break]` samples have no gap ≥ 0.3 s in the word timestamps (0–0.24 s);
    `annotation.segments[0].pause_s` records it, filter on it.
11. **Easy Turn "complete" clips are often mid-conversation lines** ("你们老师也是绝了", "适当的吃还是可以的"): a turn end,
    but not a self-contained request. Fine for timing (`respond`), meaningless for answer content. The composed
    backchannel / wait episodes reuse complete clips as openings, so an opening also appears as its own episode.
12. **FD-Bench v1.5 counts**: user_backchannel has 98 samples (README says 99); v1.0 synthetic interruption k and v1.5
    user_interruption k are the same content (shared split).

## 9. What conversion loses

- **Reactivity**: replay is open-loop. The user never reacts to the agent (doesn't stop when talked over, doesn't
  wait for an answer, events fire at fixed times) — exactly what our online users add. Benchmarks stay comparable;
  RL on replayed episodes would reward fitting fixed timings.
- **Dual-channel context**: Candor/ICC/HumDial were two-sided conversations; only the user side is released or used,
  so the agent is dropped into the middle of someone else's dialogue (Candor turn-taking, Easy Turn complete).
- **Official measurement**: words come from the agent's text stream, not ASR of its audio; VAD is energy-based;
  judges are whatever LLM is passed (§5). Official numbers are approximated, not reproduced.
- **Labels → expectations**: classification labels (Easy Turn) become episode expectations with chosen tails (5 s
  respond / 3 s wait) and composed openings; the score depends on those choices.
- **Content quality**: none of these sets has reference answers usable by `eval.scores`; only FD-Bench's judge
  prompts cover relevance. Our task-level reward (tools, state) has no counterpart in any of them (FD-Bench v3 would).

## 10. Recommendations

1. **Evaluation**: keep FD-Bench v1.0/v1.5 (all samples, as published) + Easy Turn (all) + HumDial (eval split by
   speaker) as the offline regression suite; report `eval.scores` per expectation with outcome counts next to the
   official FD-Bench metrics. ~3,000 episodes ≈ 2.5 h on one MiniCPM-o server at 1–2 sessions.
2. **Read results per expectation, not the total**: totals mix censoring, latency curves and category validity.
   Exclude or separately report HumDial others_talk_to_user_before and talk_to_others (§8.9).
3. **Clip bank / online seeding**: use the combined bank (HumDial real en+zh clips, Apache-2.0, speaker-split) via
   `clip_bank.behaviors(…)` now; next, add a clip-playing path to `UserSim` so recorded backchannels / interruptions
   / third-party speech are mixed in as audio rather than re-synthesised.
4. **Calibration**: set `Behaviors.pause_ms` from Candor (0.68–1.28 s, not 1.2–2.5 s); place online interruptions
   relative to the agent's speech (the benchmarks' fixed offsets carry no information about real barge-in timing).
5. **Next data**: Easy Turn trainset (Apache-2.0, 1,145 h) for clip volume and turn-state training data; FD-Bench v3
   (real disfluency + tool use, data on Google Drive) as the first benchmark matching our task episodes;
   MTR-DuplexBench (CC BY 4.0) for multi-round replay. FastTurn: research only (SmoothConv is CC BY-NC).
6. **Agent side**: give MiniCPM-o a language-matching system prompt for zh runs (or report language mismatch as its
   own metric); it answers Mandarin in English up to two thirds of the time.

## Pass by user turn (open- vs closed-loop benchmarks)

`results/turn_metrics.py` scores a closed-loop conversation after each of its user turns, so the open-loop number is
one point of a curve. User turns are numbered from 1 in start order; **cut(N)** is the start of user turn N + 1, or
the end of the conversation if it has at most N user turns. "The conversation up to the agent's response after the
N-th user turn" is everything that starts before cut(N).

- **FD-Bench v3, pass@N-turn**: spoken fulfilment (`fdb3_spoken.outcome`: every parameter of the request confirmed
  in the agent's speech with its final value) over the agent turns that start before cut(N). **pass@1turn** judges
  the agent's first response, which is what the open loop measures: A has one user turn (the recording), so A's score
  is A's pass@1turn, and B's pass@1turn equals it when A and B are identical until the user's second turn. A
  conversation that ended before turn N + 1 keeps its final value (carried forward), so pass@N-turn for large N is
  the closed-loop final outcome. (For a tool-calling agent the same cut applies to Pass@1 over the tool calls; the
  release run has no tool-calling agent.)
- **FD-Bench v2, stages@N-turn**: the report's stage analysis (one proxy-LLM call at T = 0 marking each staged goal
  T1–T4 reached / completed / the agent's behaviour on it) on the transcript up to cut(N), N counting the examiner's
  lines: goals completed (0–4), all four completed, and the stage score (mean over reached goals; yes 1, partial 0.5,
  no 0). For both A (the replayed script's lines, which do not react to the agent) and B (the live examiner).
- **Audio MultiChallenge**: not turn-indexed. Its rubric grades the reply to the last user turn, and A and B have the
  same number of user turns, so there is nothing to grade before the last turn.

## 11. Full-Duplex-Bench v3 (tool use): data → tasks, agent, open vs closed loop

Code: `benchmarks/fdb3.py`, `agents/cascaded.py`, runner `examples/fdb3_ab.py`, tests `tests/test_fdb3.py`. License
**CC BY-NC 4.0** (the mock APIs, tool descriptions, agent instructions and judge prompts are ported from the
official repository and live in the optional component `extras/fdbench/`): non-commercial research only.

**Data → tasks.** One `Task` per recording (`{scenario_id}_{speaker}/`). The user's turn is the recording up to the
end of the request + 300 ms: the end is the first gap of more than 2 s between voiced stretches, else the last voiced
frame — the official rule (`run_tool_benchmark.py`, on parakeet word timestamps) on an energy VAD. The released
recordings are 36–59 s long but the request takes 10–26 s; the rest is room tone with occasional faint background
talk. The official pipeline streams all of it; here it is dropped in both conditions (so that a simulated user can
take the next turn, and so that background talk in the tail cannot trigger the agent). `scenario["turns"]` (for
`ReplayUser`) and `scenario["first_turn"]` (for a closed-loop user) are the recording; `criteria` holds
`expected_tool_calls`; `scenario["benchmark"]` the annotation.

**Tools.** `MockAPIBackend`: `mock_apis.py`'s 12 functions verbatim (same return values; deterministic), behind the
official cascaded agent's tool descriptions and parameter schemas (`cascaded_agent.py`). Arguments are coerced to
the declared types and defaults filled in, as the LiveKit function wrapper does; that is also what the scorer
compares. Tool latency: the scenario's `latency_profile` midpoint (fast 125, normal 500, slow 2000 ms; the official
runner's default is `instant`).

**Scoring** (ports, cited in the code): `pass_at_1` = `evaluate_pass_rate.py` (strict: exactly the expected tools as
a multiset, then every argument correct by an LLM judge with the official prompt — GPT-4o officially; any `TextGen`
here; exact-match fallback). Note that this file's Pass@1 has no response-quality condition (despite its docstring);
`judge_response` is `evaluate_tool_calls.py`'s response judge, reported separately. `timing`: turn-take, Δt = agent
start − end of request (< 0 = interruption), first-response latency (non-interrupted), first-tool-call latency.

**Closed-loop user.** `user_card(meta)`: goal = the request (the script) and the expected calls as steps, with
`$RESULT_k` references described as "whatever step k returns"; facts = every literal argument; for a
self-correction (`state_rollback_details`) only the corrected value. Persona from the acting notes. `UserSource`:
first turn = the recording, then an LLM with `USER_SYSTEM` (answer questions truthfully, correct mistakes and
omissions, confirm, decline anything extra, end when all steps are done); it is told when the assistant has stayed
silent since its last turn (a nudge after 10 s). Voice: Qwen3-TTS Base cloned from the recording (reference = the
trimmed recording + script). Reply gaps: `ResponseDelay`. The user hears only speech, never tool calls.

**Cascaded agent** (`CascadedAgent`): energy VAD (20 ms frames, RMS ≥ 300 of int16; recordings' noise floor is
≤ 80) → a user turn opens after 100 ms of speech and **ends after 800 ms of silence** (LiveKit's cascaded setup in
the official repo: Silero `min_silence_duration` 0.55 s + `min_endpointing_delay` 0.5 s ≈ 1.05 s) → ASR of the turn
→ LLM with the session's tools, called again after each tool result (≤ 6 rounds) → TTS of each text reply.
300 ms of user speech while the agent talks cuts its speech; before a reply goes out it cancels the reply (calls
already sent stay sent). Every model call runs while simulated time stands still; its output is scheduled with a
fixed `LatencyModel` (ASR 150 ms + 10 ms per audio second; LLM 350 ms + 15 ms per generated token — up to the first
sentence for speech, the whole call for a tool call; TTS 250 ms to first audio), so timing is deterministic and does
not depend on server load; measured wall times are logged in `meta.agent_events`. Model calls are memoized by
request (LLM requests carry the seed), so runs that hear the same input produce the same output. Tool calling on
a vLLM server without `--enable-auto-tool-choice`: `ToolChat(native=False)` sends the tools with
`tool_choice="none"` (the chat template still renders them) and parses `<tool_call>` markup (Qwen3-Coder XML or
Hermes JSON) from the text.

**Open vs closed loop** (`examples/fdb3_ab.py`): condition A plays the recording with `ReplayUser`; condition B uses the recording as
the first turn of a `UserSim`. Per (recording, seed) A and B share the model-call cache, and the env is lockstep, so
B equals A up to the user's second turn — the report checks it (agent speech, tool calls with arguments and times,
first-response metrics).

**Agents without tools: spoken fulfilment** (`benchmarks/fdb3_spoken.py`; `examples/fdb3_ab.py --agent minicpmo`).
A native duplex model such as MiniCPM-o 4.5 cannot call tools, so Pass@1 does not apply. Instead: did the agent's
speech confirm every parameter of the request with its FINAL value? Slots = the literal arguments of
`expected_tool_calls` (`$RESULT_k` references excluded; one slot per distinct value), restricted to the values the
caller actually says in the script (rule match, plus three hand-checked quantities the rules cannot see; five
annotated values are never said or contradict the recording — listed in the module); for a self-correction the
target is the corrected value and the original is *superseded*. 281 slots over the 100 recordings (1–6 each), 23 of
them self-corrected. Per slot, rules first (normalised text,
number words → digits, split numbers joined — streaming transcripts write "1 5 0" —, spelled-out codes, per-type
rules for dates, amounts, counts, codes, names, cities, enums with synonyms, phrases): the last agent turn stating the
target or a superseded value decides (target → correct, superseded only → wrong). Slots no rule settles go to an LLM
(T = 0): confirmed / wrong / not_confirmed. `outcome` = share correct, `fulfilled` = all correct; computed on the first
response (A, or B before the user's 2nd turn) and on the whole conversation (B final). The duplex agent runs in
lockstep with the token trace, text-only by default ("Agent output" above; the runs reported here used audio output), with a per-domain voice-assistant system prompt that asks it to let
the caller finish and confirm every detail back (`DUPLEX_PROMPT`); A ends after 8 s of quiet (`--a-idle-ms`). The
report adds a failure taxonomy (silent / wrong after self-correction / misheard / interrupted in a pause / asked a
question / missed a detail), the official timing and response-quality judges, and the A = B check on agent speech
(start, text) before the user's second turn. With the env server's deploy config the Thinker decodes greedily, so
seeds vary only the simulated user; A runs of different seeds are reproducibility replicates.


## 12. Full-Duplex-Bench v2 (automated examiner): open vs closed loop

Code: `benchmarks/fdb2.py`, runner `examples/fdb2_ab.py` (`prepare` → `run` → `report`), tests `tests/test_fdb2.py`.
License **CC BY-NC 4.0** (examiner prompts, staged goals, judge prompts ported from the official repository; the
ported judge rubric prompts live in the optional component `extras/fdbench/`).

**Data → tasks.** `fdb2.load(prompts_staged_200.json)`: one `Task` per examiner task; `scenario["instructions"]` = the
examiner system prompt + " Do not talk over the other speaker." (what `run_dataset.sh` appends) + the task prompt;
`criteria` = staged goals T1–T4 and the split. The examinee's official system prompt is `fdb2.EXAMINEE_PROMPT`
("You are a helpful AI assistant. Always speak in English.").

**Closed loop (the official protocol).** `fdb2.ExaminerSource(llm)` inside `UserSim`: an `LLMSource` with
`EXAMINER_SYSTEM` (the official prompts plus a two-line note that the words are spoken and the other side arrives
as a transcript); the line containing "The conversation is over" is final; at most 12 lines; nudge after 10 s of
silence. Rendered with a stock Qwen3-TTS voice (`serena`); reply gaps from `ResponseDelay`; it yields after 1 s
when talked over and never barges in — the official *slow* examiner mode (GPT-Realtime server VAD,
`interrupt_response: true`); *fast* mode is not ported. Episode ≤ 120 s (the official recording length and ASR
window).

**Open loop (static script).** `fdb2.make_script(task, examiner_llm, assistant_llm)` writes one agent-free reference
dialogue per (task, seed): the same examiner against a text-only reference assistant (`REFERENCE_ASSISTANT`, the
same LLM) until the end phrase. `prepare` renders the examiner lines once (`<out>/scripts/<task>/s<seed>/e*.wav` +
`script.json`) and places them with `fdb2.schedule`: each after the previous one ends + 1 s agent gap + the reference
reply's speaking time at 2.7 words/s + 0.4 s. `ReplayUser` plays them whatever the agent does — the form of a static
multi-turn benchmark (MTR-DuplexBench style). In the closed loop the examiner's first line is the script's first
line (same audio), so both conditions are identical until the examiner's second line (`report` checks the agent's
speech before it).

**Scoring.** `fdb2.judge_prompt(task, ep)` = `eval_single_item.build_full_prompt` with Channel A = the examiner's
words and Channel B segments = the agent's speech segments `[start, end]: text` (the official pipeline uses NeMo
ASR word chunks of the recorded channels); `fdb2.parse_judgement` reads both the official non-JSON format
`[s, e]: tt, if` and variants; `episode_scores` → per-conversation TT / IF means and the task score (dropped for
Daily, which has no task rubric). The official judge is Gemini 2.5 Flash; any `TextGen` can be passed (our runs:
local Qwen3.8-27B-FP8 — proxy numbers). `report` also gives: whether the examiner reached its end phrase (B), examiner
lines, overlaps (examiner line starting while the agent talks), and an LLM check of every examiner line after the
first: does it fit what the agent actually said (`COHERENCE_PROMPT`) — the open-loop artefact.

**Stage-aware examiner closing (`--closing`, `fdb2.StageClosing`).** With a text LLM as examiner (proxy for
GPT-Realtime) many closed-loop runs hit the cap without the end phrase. The diagnosis (`report`'s cap table)
found mostly slow pacing (the examinee talks ≥ 60 % of the
window), with the examiner itself failing to close (all goals done, no end phrase) in 8–23 %. `StageClosing(llm)` adds rules that only
enforce what the official prompt already asks (move on once a goal is covered, do not repeat, finish in ≈ 5 turns, end
with the end phrase): before each examiner line a tracker LLM (T = 0, `TRACK_PROMPT`) marks goals T1–T4 covered
(covered stays covered); then (1) all covered → the examiner is told, as a note "not said aloud", to confirm briefly and
end with the end phrase (appended if it does not say it); (2) more than `max_per_stage` = 3 lines on the same uncovered
goal → told to move on (on T4: close); (3) `max_stall` = 5 lines without any new goal covered → close; (4) two closing
remarks (thanks / goodbye) in a row → close. Every decision is logged in the episode (`meta.examiner_log`). `--max-ms`
sets the cap of both conditions (official 120 s; our rerun used 180 s for B and is scored both on the official 120 s
window, `report --window-ms 120000`, and on all 180 s).

**Rescoring: the cap is not a failure.** `fdb2.stage_prompt` / `parse_stages`: one LLM call per conversation marks
each goal reached / completed, the agent's behaviour on it (yes / partial / no) and the examiner's retries, plus
pleasantry loops and examiner drift; stage score = mean agent behaviour over the reached goals. `judge_prompt(...,
stages=reached)` runs the official judge with only the reached goals listed and the note that the call was cut by the
time limit (IF_r / TT_r / task_r; equal to the official scores when the examiner reached its end phrase). Reported
separately: stages reached / completed, end phrase reached, timeout share. `fdb2.pacing` + `cap_cause` classify each
timeout (agent_failed_stage, pleasantry_loop, examiner_drift, all_done_no_end, slow_pacing, slow_other).

**Duplex timing and open-loop user-turn validity (both runners' reports).** `eval.timing_counts` / `timing_summary`
from the turns (the same rules as `eval.scores`): turn-take rate after a user turn (FD-Bench-style take-over), response
latency (median, p90), yield rate / latency on *intended* barge-ins (user turns marked `expects` "yield" / "interrupt"),
agent cut-in rate, non-directed sounds ignored. A user turn that overlaps the agent only because a replayed line's
fixed start time collided (`eval.collisions`: not intended) is a **collision**: the agent's behaviour on it is reported
apart and not counted in its yield / cut-in scores. `turn_validity` flags each user turn after the first as invalid
for timing (a collision; rule), content (responds to something the agent never said), ignored question (the agent
asked — rule — and the turn does not address it — LLM) or stale (re-gives / re-asks something settled); the LLM part is
calibrated on closed-loop turns (its noise floor) and reported as the excess over it.

## 13. Audio MultiChallenge: open vs closed loop

Code: `benchmarks/audiomc.py`, runner `examples/audiomc_ab.py`, tests `tests/test_audiomc.py`. Data: `ScaleAI/audiomc`
(MIT), unpacked once with `audiomc.extract(parquet, out)` → `<out>/<id>/u<k>.wav`
(16 kHz) + `index.json` (1.7 GB).

Official protocol: the model receives the dialogue with the dataset's *fixed* assistant turns as history and writes
only the reply to the last user turn; o4-mini judges it per rubric item (`audiomc.JUDGE_PROMPT`, the official MIT
prompt, kept in `benchmarks/third_party/audiomc/`). A duplex model
cannot be handed someone else's replies as its own history, so it talks through the whole conversation:

- **A, open loop** — `audiomc.ScriptSource` in `UserSim`: the recorded user turns in order, each after the agent has
  stopped (`ResponseDelay`), never yielding mid-turn. The content is fixed: later turns still react to the dataset's
  assistant.
- **B, closed loop** — `audiomc.ReplanSource(llm)`: the same first recording; each later turn is the scripted turn
  rewritten by an LLM user to fit what the agent actually said, keeping every fact, request, instruction and
  self-repair (so the final challenge and its rubric stay valid); voice cloned from the first recording.
- Same number of user turns in both; the reply to the last user turn (`final_reply`) is judged per rubric item on the
  conversation as it happened (`history(ep)`, the README's `build_grading_conversation_history` with the real turns).
  Reported: rubric pass rate and all-items pass, per axis; A = B check up to the user's second turn.

`report` also gives the duplex timing table and the open-loop user-turn validity table (§12) for both conditions.
