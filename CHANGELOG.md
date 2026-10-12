# Changelog

## [Unreleased]

### Added
- **Procedures: SOPBench.** `integrations.sopbench` runs SOPBench (Li et al. 2025; code MIT, data CC BY 4.0) as a spoken
  task: the user holds the case's goal and information and gives each item only when asked; the agent gets SOPBench's
  assistant prompt with the SOP in one of `FORMATS` (`structured`, `list`, `structured+order`) plus a phone-call note,
  and the domain's functions as tools (one backend per case: the SOP depends on it). `evaluate` is SOPBench's
  directed-graph verifier on the episode's calls and final database. Function responses keep the `(success, value)`
  tuple, as in SOPBench's released trajectories, which its evaluator needs; replaying those GPT-4o runs gives the same
  verdict on 586 of 593 cases (the rest changed in SOPBench's code since). Read from a checkout
  (`third_party/sopbench` or `SOPBENCH_PATH`), not shipped. `examples/sopbench_demo.py`.
- **Computer use: BrowserGym.** `integrations.browsergym` wraps a BrowserGym task (e.g. MiniWoB++) as a tool backend:
  `read_page`, `click`, `fill`, `select_option`, `press`, `scroll` act on a real Chromium page and return its
  accessibility tree; the page's instruction is what the simulated user says; `evaluate` returns the benchmark's
  reward. Browsers cannot be copied, so these episodes do not fork. `examples/browsergym_demo.py`.
- **Images in tool results.** `ToolResult.images` (PNG bytes, e.g. a screenshot) are stored by the media store
  (`MediaStore.put_image`) and referenced from the call (`tool_calls[].images`, docs/FORMAT.md §4.5); the viewer's state
  panel shows the newest screenshot at the cursor, and `scripts/build_site.py` copies them.
- **Collaborative coding: ColBench.** `integrations.colbench` runs ColBench Backend Programming (SWEET-RL) as a spoken
  task: the user says the description and answers from the hidden reference code with ColBench's simulator prompt;
  the agent sees the signature, asks, and hands in the function with `submit_code`; the 10 hidden tests run after the
  episode in a separate process. Original setting: the agent cannot run code. ColBench's prompts and data
  (CC BY-NC) are read from a checkout and a download, not shipped. `examples/colbench_demo.py`.
- **`agents.TranscriptToolAgent`.** An LLM with tools that hears the user's transcript (no ASR) and speaks through a
  TTS: it waits while the user talks, stops speaking when talked over, and sends a round's tool calls as its words start.
- **Recorded background ambience by default.** `Soundscape()`, the `UserSim` default, now uses a sound bank in a
  user cache: `$IG_SOUNDBANK`, else `~/.cache/interaction_gym/soundbank`. If the bank has no ambience yet, the env
  fetches the DEMAND part once (about 80 MB, CC BY-SA 3.0), printing one log line. The new `interaction_gym.noisebank`
  holds this code, shared with `scripts/fetch_noise_banks.py`; `python -m interaction_gym.noisebank` pre-builds the
  bank. If the fetch fails, the env warns once and uses the synthetic stand-ins.
  - To opt out: `IG_SOUNDBANK=synthetic`, or `Soundscape(bank=None)` / `Soundscape(bank="synthetic")`.
  - `IG_SOUNDBANK_FETCH=0` uses an existing bank but never downloads.
  - Noise events are never fetched: MUSAN is 11 GB. They come from the bank if it is a full
    `fetch_noise_banks.py` build, else they are synthetic.
  - Benchmark users (`benchmark_soundscape()`) still use no bank.
  - Clip choice and offset stay seeded by the episode.
  - The tests force `IG_SOUNDBANK=synthetic` (tests/conftest.py).
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
