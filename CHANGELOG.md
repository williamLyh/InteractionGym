# Changelog

## [0.1.0] - 2026-10-09

First public version.

### Added
- **Simulation core.** Event-driven core (`Sim`, `Node`, `Frame`, `Segment` / `Chunk`) and the RL-facing `Env` / `VecEnv`, with `AgentSpec`-driven agent steps on a virtual clock.
- **Simulated users.** `ReplayUser` (open loop) and `UserSim` (closed loop) with `ScriptSource` / `LLMSource`.
  - **Subjective sounds are decided by a model in context.** At the agent's phrase boundaries, `LLMListener` returns LISTEN, BACKCHANNEL or INTERRUPT. When the user interrupts, the user LLM writing the barge-in line also picks its intent (correction / question / stop / other).
  - **Pauses inside a turn are placed by the user LLM.** Their frequency follows the persona's `hesitancy`.
  - **Non-subjective events come from a seeded compound Poisson process.** These are noises by surroundings, remarks to someone nearby (written by the user LLM) and being called away.
  - **A background track is part of the environment context.** It can be silence, white, pink or brown noise, or ambience from a sound bank.
  - **Every behaviour is a user turn.** Each carries the agent behaviour it expects (`respond`, `yield`, `ignore`, `wait`, …).
- **One voice per call.** `Voice` synthesizes the first turn with a neutral delivery and clones every later turn from it (Qwen3-TTS Base). Each turn is trimmed of silence and normalised in level, with one TTS seed per episode.
- **Agents.**
  - `VllmOmniDuplexAgent`, the vLLM-Omni duplex adapter: realtime and input-clocked lockstep, audio or text-only output, token trace.
  - `CascadedAgent`.
  - `CannedAgent`.
  - `meta.agent` records the agent each episode ran.
- **Tools.** `ToolWorld` with stateful backends, simulated latency and discoverable schemas. Adapters for τ²-bench and AutomationBench, with their official evaluators.
- **Trajectory format v1.** It has a JSON Schema (docs/FORMAT.md).
- **Evaluation.**
  - `eval.scores` gives one score per user turn: whether the agent did the expected behaviour, graded by a latency curve when it took the conversation over.
  - Timing metrics, user-turn validity checks, and HTML viewer and agent-trace pages.
- **Benchmarks.** Loaders and metric ports for Full-Duplex-Bench v1.0 / v1.5 / v2 / v3, Easy Turn, HumDial-FDBench and Audio MultiChallenge. FD-Bench v2 / v3 and Audio MultiChallenge run open loop and closed loop, with turn-indexed metrics (pass@N-turn).
- **Release results.** `results/BENCHMARKS.md`: MiniCPM-o 4.5, open vs closed loop.
- **Serving and tooling.**
  - A GPU tuner for serving layouts.
  - A parameterised reference deployment (examples/serving/reference). It includes patches for vLLM-Omni's MiniCPM-o:
    - `minicpmo_fe_per_session.patch`: a per-session audio feature extractor;
    - `minicpmo_thinker_only.patch`: Thinker-only text sessions.
  - `scripts/fetch_noise_banks.py`, which builds a local DEMAND / MUSAN sound bank.

### Defaults worth knowing
- **Benchmark users have no background.** Up to the user's second turn, the closed-loop condition's microphone equals the replayed open-loop condition's (`benchmarks.benchmark_user`).
- **Latency curve.** `eval.latency_score` is `1 / (1 + exp((Δ − 950) / 100))` with Δ in ms: about 1 up to 500 ms, 0.92 at 700 ms, then a fast fall after 800 ms.
- **MiniCPM-o runners.** They default to Thinker-only text output timed at `speech_cps`. `--audio-out` selects the full Thinker + Talker + Code2Wav deployment. The two are not bit-identical, and their results are not directly comparable.

### Notes
- **License.** Apache-2.0, except the Audio MultiChallenge judge prompt (MIT) under `benchmarks/third_party/`; see THIRD_PARTY.md.
- **Full-Duplex-Bench material.** It is CC BY-NC 4.0 (non-commercial) and ships as a separate optional distribution, `interaction-gym-fdbench` in `extras/fdbench/`, which is excluded from the main wheel and sdist.
- **Noise banks.** Only the download / build script is shipped, no audio. DEMAND is labelled CC BY-SA 3.0.
- **Former name.** The project was developed as DuplexInteractionGym. Environment variables use the `IG_` prefix; the former `DIG_` names are still read, with a one-time warning.
