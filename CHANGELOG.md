# Changelog

## [Unreleased]

### Added
- **Plugin layer and `ig run`** ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). Everything above the kernel is a
  plugin of one kind (`role`, `node`, `behaviour`, `backend`, `tool`, `agent`, `task`, `scorer`, `sink`, `driver`,
  `latency`, `mixer`): a factory registered with `@plugin("kind", "name", requires=..., config=Dataclass)`; third
  parties add theirs through the entry point group `interaction_gym.plugins`. A run is a list of rows
  `{id, plugin, config}` layered from bundles (`interaction_gym/bundles/*.yml`, `extends:`) and patches that replace a
  row's config by id, plus `--set row.key=value`; resolving validates every row (plugin exists, config fields known
  and typed, requirements and config `refs` to other rows satisfied) and orders rows by their requirements. The
  resolved configuration goes into each episode's `meta.config` with a digest; `ig run <bundle> --dump-config` prints
  it. Built-in wrappers: `kernel/env`, `role/controlled` / `replay` / `user`, `agent/canned` / `vllm_omni` /
  `openai_realtime` / `cascaded`, `backend/chat` / `tool_chat` / `tts` / `asr` / `fake_chat` / `tts_fake`,
  `tool/functions` + `node/tool_world`, `task/inline` / `jsonl`, `scorer/interaction` / `timing`, `sink/jsonl` /
  `html` / `stdout`, `driver/eval` (concurrency, resume from the jsonl sink). Bundles `core` and `minimal`. The
  existing classes are wrapped, not changed; `tests/test_plugins.py` checks the composed episode equals the one built
  by hand. New dependency: PyYAML.
- **Audio MultiChallenge as a bundle** (`ig run audiomc`, `audiomc_cascaded`): `examples/audiomc_ab.py run` as a
  composition, checked against the script frame for frame (`tests/test_plugins_audiomc.py`; the parity table is in
  docs/ARCHITECTURE.md §8). For it: the `source` plugin kind (`script`, `llm`, `audiomc_script`, `audiomc_replan`; a
  `role/user` names its source row), `task/audiomc`, `backend/clone_fallback` (`clients.CloneWithFallback`),
  `backend/chat` `defaults` and `seed: episode` (every call carries the running episode's seed, what the runners did
  by building one client per episode), `role/controlled` `preset` (a duplex model's chunk), `agent/vllm_omni`
  `preset` / `instructions`, `agent/cascaded` with dict `endpointing` / `latency` and the shared per-pair cache,
  `sink/agent_traces`, `role/user` `benchmark: true` and `soundscape`. `driver/eval` gained what the A/B scripts
  shared: `arms` (each a patch layer; one (task, seed) runs through the arms in order sharing an
  `EpisodeContext.cache`; output `out/<tag>/<arm>/`, episode ids `<tag>/<task>/s<seed>/<arm>`, the layout the
  scripts' `report` commands read), `retries` / `retry_wait_s` / `retry_jitter_s`, `warmup` (one episode per server
  behind the agent URL), the output-mode check on resume, `run_id`, and `meta.condition` / `task_id` / `agent_name`.
  Config strings may reference the environment (`${IG_LLM_URL:-http://127.0.0.1:8000/v1}`); a patch that swaps a
  row's `plugin` drops the old config.
- **Recorded sounds for online users.** Third-party speech: the noise label `speech` (someone else talking nearby, a
  TV), at `Behaviors.third_party_per_min` onsets per minute on top of the noise rates, recordings only (a sound bank's
  `events/speech/`; no synthetic stand-in, so a missing bank is an error). Voice packs (`task.scenario["voice_pack"]`,
  `user.VoicePack`): a recorded speaker for the simulated user — its voice is cloned from the speaker's own reference
  recording from the first turn on, and its backchannels and asides are that speaker's recordings instead of
  synthesized ones (levelled like the synthesized turns; `meta.user.voice_pack`, `random_events.pack_*`).
  `benchmarks.clip_bank.voice_packs` builds them from a clip bank's real speakers (HumDial, material split, one
  speaker per id prefix) and `clip_bank.speech_events` copies its real third-party speech into a sound bank.
- **Agent proxy.** `interaction_gym.agent_proxy` (`python -m interaction_gym.agent_proxy --backends ...`): one
  WebSocket address in front of several duplex agent servers. A vLLM-Omni server runs a single Thinker (no
  data-parallel Thinker yet), so several GPUs mean several servers; duplex sessions are stateful, so the proxy
  balances per connection — each session goes to the server with the fewest open sessions (ties rotate) and stays
  there; sessions queue when every server is at `--max-sessions` (1013 after `--queue-timeout-s`); a server that
  refuses is skipped; close codes pass through; plain HTTP returns the open sessions per server. The reference
  deployment starts it on `AGENT_PORT` (8009) and prints it as the one `IG_AGENT_URL` (the runners used to reach only
  the first of the four servers). `servers_behind(url)` lets the FD-Bench v2 / v3 and Audio MultiChallenge runners
  warm up one throw-away session per server. The extra hop measured well under a millisecond per round trip.
  docs/agent_server.md records the future alternative (Thinker data parallelism inside vLLM-Omni).
- **OpenAI API backends (non-duplex).** `OpenAIChat`, `OpenAISpeech`, `OpenAITranscribe` and the cascaded agent's
  `ToolChat` reach `api.openai.com` (by the URL's host, `api="openai"`, or `.openai(model)`): key from
  `$OPENAI_API_KEY`; retries with backoff on 429 (not an exhausted quota), 5xx and dropped connections, honouring
  `Retry-After`; local-server request fields not sent; `max_tokens` → `max_completion_tokens` with `reasoning_tokens`
  headroom on reasoning models; a field the API refuses for a model (e.g. `temperature` while it reasons) dropped and
  the request retried, then never sent again by that client; 4xx errors carry the API's message. `describe()` records
  `provider` and `not_sent`; every client (local too) counts `usage` (requests, tokens incl. reasoning, TTS characters,
  ASR seconds). Hosted TTS gets raw 24 kHz PCM, no `seed` / `language`, and raises on cloning;
  `user.OPENAI_TTS_VOICES` is a preset-voice catalogue for `pick_voice`. `ToolChat` is always native on the hosted API
  and takes `api_key_env` / `retries` / `timeout`.
- **OpenAI Realtime agent.** `agents.openai_realtime.OpenAIRealtimeAgent`: a `gpt-realtime*` model as the agent (server
  or semantic VAD turns, interruptible, session tools as function tools). Probed first (gpt-realtime-mini): turn
  detection counts received audio, not wall time (4x sending moved no boundary; 6 s send pauses, at a turn's end or
  mid-word, detected nothing; 12 s idle kept the session). So the microphone is real-time paced and the env's stalls
  are simply not sent; a reply starts at the simulated time the server began it plus its first-audio wall latency
  (unaffected by stalls); playback is contiguous (`underrun="wait"` holds the env while a slow stream catches up —
  0.7x real time was measured — or `"silence"`); user speech stops playback and the server's item is truncated to what
  was heard (deleted if nothing was). `usage` from `response.done`, per-response timing in `trace()`;
  `meta.agent.reproducible: false`. **Barriers:** over a WebSocket the server's turn-detection events queue behind
  reply audio already sent (the server produces a reply far faster than real time; an interruption detected at 10.9 s
  of audio reached the client at 29 s), so while the agent is active and the microphone carries sound each step ends
  with an in-order barrier (`conversation.item.retrieve` of a non-existent item) and the env waits for its answer;
  barge-ins are then cut where the server detected them (live: user at 11.0 s, cut at 11.3 s). `trace()` records
  `input_lag` per turn-detection event, `barriers` and `client_events`; a warning fires once when an event lands over
  1 s late (the network is part of the measurement — documented in the module and Getting started).
  `audio_format="pcmu"` (8 kHz mu-law, 8x less to download). Keepalive pongs may be late (they queue too) without
  closing the session. `examples/openai_realtime_demo.py` runs it against an all-OpenAI user;
  `examples/openai_realtime_checks.py` runs live checks (barge-in, backchannel, tools, semantic VAD, silence
  underrun, pcmu — all passed on gpt-realtime-mini, 2026-10-22). Tests against a local stand-in with sample-based VAD,
  a slow stream and a slow link.
- **SOPs: one source format, pluggable presentation.** `interaction_gym.sop`: an SOP as data (actions whose conditions
  combine with `all` / `any` / `chain`, each `ask` or `look_up`, with `on_fail` alternatives), the logic to check an
  agent against it (`holds`, `witnessed`, `refuted`), and formats as plugins: `structured`, `list` (keeps the logic,
  unlike SOPBench's flat list), `code`, `mermaid`, plus `@register_format` and the `interaction_gym.sop_formats` entry
  point. `interaction_gym.sop_task`: spoken SOP tasks for agents without tools (what they would look up is "their
  screen"), judged by an LLM report of what happened plus rules on the SOP tree (`decision_correct`,
  `procedure_followed`, `details_correct`). `examples/sop_calls.py` turns the old service calls (restaurant, clinic,
  shop) into SOP tasks, two to be done and two to be declined, with MiniCPM-o 4.5. The viewer's SOP card also shows the
  agent's screen and the ground truth, and Evaluation shows the judge's report. `OpenAIChat(timeout=)`. docs/SOP.md.
- **Chinese SOPs.** An SOP's `lang` (`en`, `zh`) sets the language of the built-in formats' own words and of the agent's
  frame in `sop_task`. `examples/sop_calls.py` adds four calls in Mandarin: a courier changing a parcel's address and a
  hotel's late checkout, each with a caller to serve and one to decline.
- **Task categories, SOP policy and verifier in the trajectory and viewer.** `meta.categories` tags a task as `chitchat`,
  `sop` and/or `tool_call` (several allowed; from `task.scenario["categories"]`, else inferred by `traj.categories`);
  `meta.policy` holds the procedure the agent had to follow (`sopbench.policy(task)`); `eval.verifier` describes the
  task's own checker (`VERIFIER` in `sopbench`, `colbench`, `browsergym`; `traj.episode(..., verifier=)`). The viewer
  shows category tags in the header, sidebar (with a filter) and run index, an "SOP policy" card (the requested
  action's conditions, then the whole SOP), and splits Evaluation into Content (reward, verifier, each check and what
  it checks; left empty for open conversation) and Duplex timing.
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
