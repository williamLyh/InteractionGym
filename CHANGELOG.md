# Changelog

## [Unreleased]

### Added
- **One vLLM-Omni patch for every server feature InteractionGym uses.** `patches/vllm-omni/` has one combined patch
  per supported base, plus its six parts and a README that maps each part to its upstream PR and says when to remove it:
  - `vllm_omni-0.31.0rc1-interactiongym.patch`, for `pip install vllm-omni==0.31.0rc1` with `vllm==0.31.0`;
  - `vllm_omni-main-61cae20d-interactiongym.patch`, for a source checkout of upstream main at `61cae20d`, with the
    upstream tests and docs.

  It contains:
  - input-clocked sessions and `silence_continuation` (vllm-omni#8485, final revision);
  - the token trace, and the MiniCPM-o 4.5 and Qwen3-Omni input-clock opt-ins (follow-ups of #8485);
  - the per-session MiniCPM-o feature extractor (#8638);
  - Thinker-only MiniCPM-o text sessions, ported to the final PR revision.

  vLLM-Omni 0.30.0 is not supported (see the README).
- `scripts/apply_vllm_omni_patch.py`:
  - locates the `vllm_omni` of an environment (`--python`) or a directory (`--target`);
  - checks the version or commit against the supported bases;
  - dry-runs the patch, backs up the touched files and applies it;
  - has `--status` and `--revert` (byte for byte).

### Changed
- The former `examples/serving/reference/patches/{minicpmo_fe_per_session,minicpmo_thinker_only}.patch` are folded
  into the new patch, as parts 05 and 06, and removed. The serving reference, docs/agent_server.md,
  docs/GETTING_STARTED.md and the README now point at `patches/vllm-omni/`.
- `VllmOmniDuplexAgent(trace_tokens=True)` requests the token trace only with `clock="input"`. The patched server refuses
  a traced session without the input clock (`token_trace_requires_input_clock`). A realtime session warns and is not
  traced. `minicpmo_suite.py` traces only its lockstep runs.
- With the patch, lockstep runs on MiniCPM-o 4.5 and Qwen3-Omni only. Nemotron VoiceChat, PersonaPlex and AURA have
  the unit hooks but do not opt in yet: run them in realtime mode. The earlier prototype build enabled them.

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
