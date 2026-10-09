# InteractionGym: closed-loop simulation for full-duplex spoken agents

InteractionGym (formerly DuplexInteractionGym) provides closed-loop environments for **training and evaluating full-duplex spoken agents**. The agent listens and
speaks at the same time; a simulated user reacts to what it actually hears — answers, interrupts,
backchannels, pauses, talks to someone else in the room — so the agent can be evaluated and trained with
online RL in a closed loop instead of against fixed recordings.

Status: research code, version 0.1.0. APIs may still change.

**Demo:** an interactive page with example episodes — https://yinhongliu.com/interaction-gym-demo/

## Key features

- **Event-driven simulation in virtual time.** Users, tools and the agent share one integer-millisecond
  timeline. Overlaps, barge-ins and cuts are exact, episodes end naturally, and nothing depends on wall-clock
  time, so runs are reproducible and can go faster than real time.
- **Simulated users at three levels.** Replay of pre-timed recordings (open loop); scripted turns with reactive
  timing; or online users whose words come from a text LLM and whose voice comes from a TTS. What the user decides
  is decided by a model in context — backchannels and barge-ins at the agent's phrase boundaries, mid-thought pauses
  inside its turns — informed by a structured persona (gender, age, language, accent, speaking style, listening
  style, hesitancy); what just happens to it (noises, being called away) is random at rates set by its surroundings,
  over an always-on background (a quiet microphone floor, coloured noise or recorded ambience; a script builds a sound
  bank from DEMAND and MUSAN). Every behaviour is a labelled user
  turn the env scores; one consistent voice per user through voice cloning.
- **Real full-duplex models as agents.** An adapter for duplex models served by
  [vLLM-Omni](https://github.com/vllm-project/vllm-omni) (MiniCPM-o 4.5, Nemotron VoiceChat, AURA, ...), in
  real time or in **input-clocked lockstep**: model time advances only with the audio sent, so episodes are
  reproducible and limited only by compute. A cascaded agent (VAD + ASR + LLM + TTS) is included as a baseline.
- **Tools.** Stateful tool backends with simulated latency, all-up-front or discoverable tool schemas, and
  adapters for τ²-bench and AutomationBench tasks with their official evaluators.
- **Trajectories and evaluation.** One JSON record per episode ([docs/FORMAT.md](docs/FORMAT.md), with a JSON
  Schema): turns with exact times, media references, tool calls, the user simulation's configuration. Rule-based
  duplex scores (`eval.scores`: one score per user turn — respond without talking over the user, yield to a
  barge-in and answer, ignore a backchannel / aside / noise, wait through a pause — graded by latency when the agent
  takes over: full score up to ~500 ms, still high at 700 ms, falling off quickly from ~800 ms) usable as metrics and
  as a reward.
- **Benchmarks as test sets.** Loaders that turn existing duplex benchmarks (Full-Duplex-Bench v1 / v1.5 / v2 /
  v3, Easy Turn, HumDial-FDBench, Audio MultiChallenge) into episodes, with ports of their official metrics,
  in open- and closed-loop variants.
- **Viewer.** Self-contained HTML pages: a timeline with playback of every voice, the user simulation, the
  environment context, the scores, and a page with the agent server's per-unit token trace.
- **GPU tuner.** Finds how many GPUs each service (agent, user LLM, TTS) should get for the highest episode
  throughput on one host.

## Install

Python 3.12. With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
git clone https://github.com/williamLyh/InteractionGym
cd InteractionGym
uv sync                      # the core package has no dependencies; adds pytest + jsonschema for development
uv run pytest -q             # deterministic tests, no models or network needed
```

Optional extras:

| extra | for | install |
|---|---|---|
| `vllm-omni` | the vLLM-Omni duplex agent adapter (WebSocket client) | `uv sync --extra vllm-omni` |
| `tau` | τ²-bench tasks (Sierra's tau2-bench, pinned git commit) | see [τ²-bench](#tau-bench-and-automationbench) |
| `automationbench` | AutomationBench tasks | see [AutomationBench](#tau-bench-and-automationbench) |
| `fdbench` | Full-Duplex-Bench v1 / v1.5 / v2 / v3 loaders (`benchmarks.full_duplex_bench`, `fdb2`, `fdb3`): the separate component `interaction-gym-fdbench` in `extras/fdbench`, **CC BY-NC 4.0 (non-commercial only)** | `uv sync --extra fdbench` (the dev group, used by the tests, includes it) |

With pip: `pip install -e ".[vllm-omni]"` (the `tau` extra installs tau2-bench from its git repository);
`pip install ./extras/fdbench` for the Full-Duplex-Bench component.

## Quick start (no GPU)

Every example below runs on a laptop: scripted or fake (deterministic) LLM / TTS clients stand in for real models.

```bash
uv run python examples/minimal.py                                   # a scripted user and agent, one barge-in; prints the log
uv run python examples/minimal.py --html runs/minimal.html          # the same as viewer pages (5 variants)
uv run python examples/user_modes.py --html runs/user_modes.html    # replay / scripted / LLM users (fake LLM + TTS)
uv run python examples/tools_demo.py --html runs/tools_demo.html    # tools: latency, errors, discovery
```

Open the HTML files in a browser. Saved episodes can be turned into a viewer page later:
`uv run python -m interaction_gym.viewer runs/*.json -o viewer.html`.

The loop in `examples/minimal.py`, in short (`RuleUser`, `REPLIES` and `TASK` are defined there):

```python
from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.traj import episode, save

spec = AgentSpec(chunk_ms=200)                       # the agent is driven every 200 ms of simulated time
env = Env({"user": RuleUser()}, spec, max_ms=60_000)  # nodes: users, tools, ...
agent = CannedAgent(REPLIES, spec, yield_after=160)
obs = await env.reset(TASK, seed=0)
done = False
while not done:
    obs, reward, done = await env.step(agent.act(env.t, obs))
save([episode(env, "my-episode")], "runs/episodes.json")
```

[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) walks through install, a first offline episode, the viewer
and a first closed-loop episode against real model servers.

## With real models

A closed loop with real models needs OpenAI-compatible servers for the user simulator — a chat LLM
(`IG_LLM_URL`) and a TTS (`IG_TTS_URL`) plus a voice-clone TTS (`IG_CLONE_URL`: by default every user turn
after the first is cloned from it, see `user.Voice`) — and, for a
full-duplex agent, a vLLM-Omni duplex server (`IG_AGENT_URL`). [examples/serving/](examples/serving/) has an
example deployment for one 8-GPU host (vLLM for the LLM, vLLM-Omni for TTS and MiniCPM-o 4.5).

```bash
# a simulated user on a real LLM + TTS, talking to a scripted agent (no duplex server needed)
IG_LLM_URL=http://localhost:8000/v1 IG_TTS_URL=http://localhost:8001/v1 IG_CLONE_URL=http://localhost:8005/v1 \
  uv run python examples/live_user.py --html runs/live_user.html

# MiniCPM-o 4.5 as the agent: several personas, lockstep, token trace (text output; --audio-out for speech)
uv sync --extra vllm-omni
IG_LLM_URL=http://localhost:8000/v1 IG_TTS_URL=http://localhost:8001/v1 IG_CLONE_URL=http://localhost:8005/v1 \
IG_AGENT_URL='ws://localhost:8010/v1/realtime?duplex=1' IG_MODEL_DIR=/path/to/models \
  uv run python examples/minicpmo_suite.py --out runs/suite
```

**MiniCPM-o output is text only by default.** The MiniCPM-o runners and examples (benchmarks, the suite, the GPU
tuner) run it Thinker-only with text output, as RL rollouts do: `VllmOmniDuplexAgent(audio_out=False)` against
Thinker-only servers (one GPU, 16 sessions each; the reference deployment's default `AGENT_LAYOUT=thinker`, which
needs its `patches/minicpmo_thinker_only.patch`), with the agent's text timed at `speech_cps` (11.3 characters per
second). `--audio-out` runs the full Thinker + Talker + Code2Wav deployment and the agent's real speech. Caveats:
the Thinker-only deployment is not bit-identical to the two-GPU audio one (bf16-level logprob differences from the
first speak unit), timing is estimated rather than taken from real audio, and text-only results are not directly
comparable with full-audio runs ([docs/BENCHMARKS.md](docs/BENCHMARKS.md), "Agent output").

Served model names default to the reference deployment's; override them with `IG_LLM_MODEL`, `IG_TTS_MODEL`,
`IG_CLONE_MODEL`, `IG_AGENT_MODEL` (see each example's docstring). `IG_MODEL_DIR` locates MiniCPM-o's reference
voice (`MiniCPM-o-4_5/assets/system_ref_audio.wav`; or set `IG_AGENT_REF_AUDIO`).

**Which vLLM-Omni?** Realtime mode (`clock="realtime"`: the adapter paces audio like a microphone) works on
stock vLLM-Omni. Three features need our vLLM-Omni patches: input-clocked **lockstep** (`clock="input"`), turning
off server-side `silence_continuation`, and the per-unit **token trace**. They are proposed upstream in
[vllm-project/vllm-omni#8485](https://github.com/vllm-project/vllm-omni/pull/8485); until merged, use the fork
[williamLyh/vllm-omni](https://github.com/williamLyh/vllm-omni), branch `duplex-input-clock`. Details and the
protocol: [docs/agent_server.md](docs/agent_server.md).

## Core concepts

| concept | where | what it is |
|---|---|---|
| `Sim`, `Node`, `Frame` | `core.py` | discrete-event simulation of a node graph; nodes exchange timestamped frames on named streams |
| `Segment`, `Chunk` | `core.py` | an utterance (or any timed output) that plays out over time and can be cut; a step's slice of it |
| `Env`, `VecEnv`, `AgentSpec` | `core.py` | the RL-facing wrapper: `reset` / `step` the agent every `AgentSpec.chunk_ms`; `VecEnv` runs episodes concurrently |
| `Task` | `core.py` | scenario (persona, goal, turns, ...), initial world state, evaluation criteria |
| `ReplayUser` | `user.py` | pre-timed turns, regardless of the agent (open loop) |
| `UserSim` + `ScriptSource` / `LLMSource` | `user.py` | a reactive user: when to speak (turn taking, barge-in) is separate from what to say (script or LLM) |
| `Voice` | `user.py` | renders a turn: given audio, TTS (voice chosen from the persona, voice cloning), or text with an estimated duration |
| `LLMListener` | `user.py` | the user's listening decisions (LISTEN / BACKCHANNEL / INTERRUPT) at the agent's phrase boundaries, one greedy LLM call each (τ-Voice's prompt, extended) |
| `Behaviors`, `ResponseDelay`, `TurnTaking`, `AsideWriter` | `user.py` | random events (noises, asides, being called away) by surroundings and the safety caps; LLM-written aside lines; content-aware reply gaps; turn-taking timing |
| `Soundscape`, `background_spec`, `EventProcess` | `soundscape.py` | the episode's background track (silence / white / pink / brown / ambience), the noise event process (compound Poisson by surroundings) and noise clips, synthetic or from a sound bank (`scripts/fetch_noise_banks.py`) |
| `ToolWorld`, `ToolBackend`, `FunctionBackend` | `tools.py` | stateful tools with simulated latency; all-up-front or discoverable schemas |
| agents | `agents/` | `CannedAgent` (scripted), `vllm_omni.VllmOmniDuplexAgent` (vLLM-Omni duplex server), `cascaded.CascadedAgent` (VAD + ASR + LLM + TTS) |
| `traj.episode` / `save` / `load` | `traj.py` | the trajectory record ([docs/FORMAT.md](docs/FORMAT.md), [schema](docs/trajectory.schema.json)) |
| `eval.scores` | `eval.py` | one rule-based duplex score per user turn and their mean (metric and reward) |
| viewer | `viewer/` | `export_html`, `export_run`, `export_agent_trace_html`; `python -m interaction_gym.viewer` |

Design notes: [docs/DESIGN.md](docs/DESIGN.md). Agent-server requirements and the vLLM-Omni protocol:
[docs/agent_server.md](docs/agent_server.md). Token trace: [docs/AGENT_TRACE.md](docs/AGENT_TRACE.md).

## Benchmarks

`interaction_gym.benchmarks` loads existing benchmarks as episodes: Full-Duplex-Bench v1.0 / v1.5 (replay,
official metrics), v2 (live examiner, open vs. closed loop), v3 (tool use under disfluency), the Easy Turn testset,
HumDial-FDBench and Audio MultiChallenge, plus a clip bank of backchannels / asides / background speech cut from
them to calibrate the user simulator. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Release results (MiniCPM-o 4.5, Thinker-only,
open vs. closed loop on FD-Bench v3 / v2 and Audio MultiChallenge, with 95% CIs and pass by user turn):
[results/BENCHMARKS.md](results/BENCHMARKS.md); the earlier full-audio run with cascaded agents and the open-loop-only
benchmarks is kept in [results/BENCHMARKS_full_audio.md](results/BENCHMARKS_full_audio.md).

**The repository ships loaders only, never benchmark data**: download each dataset yourself and follow its license.
Several are **non-commercial** (CC BY-NC 4.0: Full-Duplex-Bench v2 / v3 and the v1.0 Candor / ICC subsets).
Code and prompts ported from Full-Duplex-Bench are CC BY-NC 4.0 too and are **not part of the Apache-2.0
package**: they form the separate optional component `interaction-gym-fdbench` (`extras/fdbench`, its own
LICENSE / NOTICE / pyproject; extra `fdbench`). `benchmarks.full_duplex_bench`, `fdb2` and `fdb3` import without it
and load it on first use (an `ImportError` says how to install it); using them is therefore limited to
non-commercial use. Details: [THIRD_PARTY.md](THIRD_PARTY.md).

### τ²-bench and AutomationBench

τ²-bench (Sierra's tau2-bench, MIT; not the unrelated `tau2` package on PyPI). The package is installed from git at a
pinned commit (`5bfa7e3`); its domain data is read from a checkout (a full clone at that commit matches exactly):

```bash
git clone --depth 1 https://github.com/sierra-research/tau2-bench third_party/tau2-bench   # data (or set TAU2_DATA_DIR)
uv sync --extra tau
uv run python examples/tau_mock.py --html runs/tau_mock.html
```

AutomationBench ([Zapier, MIT](https://github.com/zapier/AutomationBench)): business workflows across 47 simulated
SaaS apps, scored on the end state of the world; the simulator is local Python (no Zapier account or network). It is
loaded from a checkout (it declares Python ≥ 3.13; the parts used run on 3.12), tested at v1.0.6:

```bash
git clone --depth 1 https://github.com/zapier/AutomationBench third_party/automationbench   # or set AUTOMATIONBENCH_PATH
uv sync --extra automationbench
uv run python examples/automationbench_demo.py --html runs/automationbench_demo.html
```

## GPU tuner

`python -m interaction_gym.gpu_tuner` decides how many GPUs each service gets (and how many sessions each
server takes) for the highest episode throughput: a solver over measured per-replica throughput curves, a dry run on
a mock cluster, and an empirical measure → rebalance loop on a real host (`examples/gpu_tuner.py`). See
[docs/GPU_TUNER.md](docs/GPU_TUNER.md).

```bash
uv run python -m interaction_gym.gpu_tuner dry-run --rounds 6
```

## Project layout

```
src/interaction_gym/
  core.py            Sim, Node, Frame, Segment / Chunk, Env, VecEnv, AgentSpec, Session, Task, Background
  user.py            ReplayUser, UserSim, ScriptSource / LLMSource, Voice, LLMListener, Behaviors, ResponseDelay, TurnTaking
  soundscape.py      background tracks and noise events (synthetic or a sound bank)
  tools.py           ToolWorld, ToolBackend, FunctionBackend
  agents/            CannedAgent, VllmOmniDuplexAgent (vLLM-Omni duplex adapter), CascadedAgent
  clients.py         OpenAI-compatible chat / TTS / ASR clients, fakes and a cache
  traj.py            trajectory records: episode / save / load / conversions
  eval.py            duplex scores and timing metrics
  turn_validity.py   checks that replayed user turns still make sense against the agent's replies
  viewer/            HTML viewer and agent-trace page
  benchmarks/        benchmark loaders and metrics (third_party/: ported MIT material)
  integrations/      τ²-bench and AutomationBench adapters
  gpu_tuner/         serving-layout tuner (hosts/: host presets)
examples/            runnable examples (no-GPU and real-model), serving/ (reference deployment)
extras/fdbench/      interaction-gym-fdbench: the Full-Duplex-Bench component (CC BY-NC 4.0, separate)
docs/                format, design, agent server, benchmarks, GPU tuner
tests/               deterministic tests (no network, no GPU)
```

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)), except the third-party material listed in
[THIRD_PARTY.md](THIRD_PARTY.md). The Full-Duplex-Bench component in `extras/fdbench/` is a separate distribution
under CC BY-NC 4.0 (non-commercial only) and is not part of the `interaction-gym` package.

## Citation

A paper is in preparation. Until then, please cite the repository:

```bibtex
@misc{interactiongym2026,
  title  = {InteractionGym: Closed-Loop Environments for Full-Duplex Spoken Agents},
  author = {Liu, Yinhong},
  year   = {2026},
  url    = {https://github.com/williamLyh/InteractionGym}
}
```
