# InteractionGym design

*InteractionGym: closed-loop simulation for full-duplex spoken agents.*

Design notes; some sections describe planned work. The environment is event-driven: there is no fixed tick, and synchronization happens only at policy steps (dialogue) or at batched physics steps (robotics). The design reuses what τ-Voice (tau2-bench) has already validated wherever possible.

## 0. Scope

A clock-driven interaction simulator for **online closed-loop RL** of full-duplex omni-modal models. The first phase covers dialogue (speech / video + tools); the same abstractions are meant to connect to a physics simulator later for embodied robotics.

- **τ-Voice** (MIT license): a full-duplex voice environment for evaluation. We **use it as a library** (domains, tasks, tools, databases, the user simulator's turn-taking logic, audio effects, metrics) and **rewrite its orchestrator** so that it is trainable, forkable, and supports more than two participants.
- We do **not** write an RL trainer or an inference engine; we connect to RLinf / VeRL / vLLM-Omni.

## 1. Core abstractions (only five)

> Everything is a stream of timestamped **Frames**; participants are **Nodes** with explicit state; **Sim** advances simulated time; outputs are truncatable **Segments**; a **Task** defines the scenario and the evaluation criteria, and the reward is computed from the log after the episode ends.

### 1.1 Frame: timestamped data
```python
@dataclass
class Frame:
    stream: str      # "user.audio" / "policy.text" / "policy.tool_call" / "tool.result" / "world.video" / "world.state"
    t: int           # simulated time in ms; if greater than the current time = delayed delivery (reaction time, tool latency and inference latency are all expressed this way)
    data: Any
    dur: int = 0
```
- This generalizes τ-Voice's `Tick`: `Tick` hard-codes agent and user as two fields; here there is any number of streams, which is what makes room for video, world state, robot actions and multiple users.
- As in τ-Voice, **speech and tools use separate channels**, and a tool result is delivered on the next tick.

### 1.2 Node: a participant with explicit state
```python
class Node:
    reads: tuple[str, ...] = ()     # subscribed streams (wildcard match); the wiring itself is the information boundary
    batched: bool = False           # True: called once per sync point for all episodes (GPU physics simulation)
    def init_state(self, task: Task, rng) -> State: ...
    async def step(self, state: State, t: int, inbox: list[Frame]) -> tuple[list[Frame], int | None]: ...
        # returns (output frames, next wake-up time); state is modified in place
    def fork(self, state: State, n: int) -> list[State]: ...   # deepcopy by default; GPU physics nodes override it with the simulator's get/set_state
```
- **Reused** from τ-Voice's `get_next_chunk(state, incoming, tool_results)`: the node object holds only configuration, **everything that changes lives in an explicit state**, and forking is copying the state.
- τ-Voice treats user and agent as **symmetric participants**; we generalize this to N nodes: user simulators, the world (replay / tool database / physics simulation), and multiple users all implement the same interface.
- There is no access-control system: privileged information (database facts, object poses) is written to a stream that only the user simulator and the judge subscribe to.
- A node is called only when it **receives new frames** or **reaches a wake-up time it scheduled**. For example, the user simulator wakes every 2 seconds while the agent is speaking to decide whether to interrupt.

### 1.3 Sim: event-driven advance of simulated time
```python
class Sim:
    def __init__(self, nodes: dict[str, Node], task: Task, seed: int, log: tuple[str, ...] = ("*",)): ...
    def put(self, frames: list[Frame]) -> None       # inject from outside (the policy's actions)
    async def advance_to(self, t: int) -> None       # process all due events in time order, then set the clock to t
    log: list[Frame]                                 # full log filtered by the log argument = the trajectory, directly replayable
    reward: float; done: bool                        # aggregated from the conventional "reward" / "done" streams
    def fork(self, n: int) -> list["Sim"]            # copies the event queue and log; each node's state is copied via node.fork
```
- **Discrete-event simulation**: time is integer milliseconds, and Sim jumps straight to the next event instead of idling through fixed ticks. The number of node calls depends on the number of events (tens to hundreds in a conversation), not on duration.
- Within one instant: first deliver all frames due at that instant, then call, in a fixed order, the nodes that have input or have reached their wake-up time. Frames they emit with the same timestamp are delivered in the next round at the same instant (causal order is deterministic).
- A Segment that is playing does not need to be sent chunk by chunk: it is emitted once as a frame, and readers slice it by the current time.
- Differences from the τ-Voice orchestrator: (1) simulated time is not bound to wall-clock time; (2) the policy is external; (3) any number of nodes; (4) fork support; (5) event-driven, no idle ticks.
- **Conventional streams**: any node may emit `done` (ends the episode) and `reward` (optional per-step reward).

### 1.4 Segment: one utterance or one action chunk
```python
@dataclass(frozen=True)
class Segment:
    id: str; t0: int; dur: int; data: str | Sequence    # text, audio samples or action sequences
    def play(self, a: int, b: int)                       # content played during [a, b), sliced proportionally
    def heard(self, t: int)                              # prefix played up to time t
    def cut(self, t: int) -> "Segment"                   # interruption: return the executed prefix; the rest is discarded
```
- A Segment is emitted as a frame **once** (for example on `user.speech`); on interruption a truncated version with the same id is emitted again. Readers keep the latest version per id and slice it by the current time.
- This merges three τ-Voice mechanisms: `output_streaming_queue` (generate the whole utterance, then queue it in chunks), proportional text delivery (the other side only sees what has actually been said), and flushing buffers when the user barges in.
- A robot action chunk is also a Segment, so interrupting speech and correcting an action are the same operation. This is the key to sharing one architecture between dialogue and embodiment.
- For text-only policies, a Segment's duration is estimated from a speaking rate (FDGym uses 3.4 words per second), or taken from the real duration after TTS synthesis.

### 1.5 Task and reward: reusing τ-Voice's task structure
```python
@dataclass
class Task:
    scenario: ...        # user persona + structured instructions (reason for call / known info / unknown info / task requirements)
    initial_state: ...   # initial world state (a database; later a simulation scene)
    criteria: ...        # expected actions / world-state assertions / natural-language assertions / information to communicate

def evaluate(log: list[Frame], task: Task, world_final) -> dict[str, float]
```
- **The reward is computed from the log after the episode ends**, following τ-Voice's `evaluate_simulation(sim_run, task)`: outcome reward (is the world state correct), interaction-timing reward (response latency, yield latency, selectivity, etc. computed from events in the log), and LLM-judge reward. No separate judge node and no live reward stream are needed.
- If per-step (dense) reward is needed later, the same function can be called on a prefix of the log.
- Robot tasks fit this structure directly: a world-state assertion becomes "the red cup was lifted by at least 10 cm".

### 1.6 Env: the interface for RL frameworks (the agent-session boundary)
```python
@dataclass(frozen=True)
class AgentSpec:              # properties of the agent session
    chunk_ms: int = 200       # update granularity = the model's frame / chunk length (80 / 160 / 200 / 240 ms, 1 s, etc., set per model)
    obs: tuple[str, ...] = ("user.speech",)   # observe only what the user has said aloud (not the user's own tool calls)
    out: str = "policy.speech"

class Env:
    def __init__(self, nodes, agent: AgentSpec = AgentSpec(), *, log=("*",), max_ms=None, peek=False)
    async def reset(self, task, seed) -> list[Frame]
    async def step(self, action: list[Frame]) -> tuple[list[Frame], float, bool]   # each call advances agent.chunk_ms
    def fork(self, n) -> list["Env"]
```
- **Observations are streamed (implemented)**: Segments on the obs streams are cut into `Chunk`s at step boundaries, and each step contains only what was **actually played** in that time window (`data` slice + aligned `text` + `first` / `last` flags; a truncated Segment gets its `last` early). **The agent never sees content that has not been played yet**; a dedicated test guarantees this. `peek=True` passes the raw frames instead and is only for hard-coded debugging agents.
- **Actions come in two forms (implemented)**: submit a whole `Segment` at once (event-triggered realtime models, which send a `cut` when interrupted), or submit a `Chunk` every step (frame-synchronous full-duplex models). The env **concatenates chunks with the same id into one growing Segment**; stopping is equivalent to finishing or yielding.
- The policy lives outside the env and is owned by the RL framework, so it can batch inference across episodes and record logprobs itself. An action frame's timestamp can be set to t + inference latency.
- The outcome reward is computed after the episode with `evaluate(log, task, world)`.
- `agents.CannedAgent`: a hard-coded stand-in agent that reads only streamed chunks and supports both whole-segment and per-chunk output.

**The two VecEnv execution modes**:

| Mode | Use case | How it advances | Sync points | What happens when the LLM is slow |
|---|---|---|---|---|
| **async** (default) | dialogue / video | each episode advances independently, concurrently via `asyncio.gather` | no global sync; the inference server batches whichever requests are ready (continuous batching) | it waits; only that episode is affected |
| **lockstep** (planned) | batched physics nodes present | all episodes sync on the policy period; a batched node is called once per sync point for all episodes | physics step + policy step | no waiting: the user's reaction is shifted to the time the LLM finishes, and this is logged |

- **Requirements on the agent server** (driven by input rather than wall-clock time, incremental input, truncation at the actual playback position, plus logprobs / hot weight updates / fork for training) are in [agent_server.md](agent_server.md).
- For evaluation, a `PolicyNode` (planned) wraps τ-Voice's cloud API adapters (OpenAI / Gemini / Qwen realtime, etc.) as a node to produce baselines.

## 2. User simulator (`user.py`, implemented and validated with fake LLM / TTS)

The user side is split into three independently replaceable parts: **where content comes from x who decides timing x how the voice is produced**.

| Mode | Node and content source | Timing | Use |
|---|---|---|---|
| **Offline** | `ReplayUser`: predefined or calibrated user turns (text or speech, with absolute timestamps) | fixed timestamps regardless of what the agent does (open loop) | Kyutai-style setups, regression tests, building data from real corpora |
| **Semi-online** | `UserSim` + `ScriptSource`: fixed content | `TurnTaking` decides based on agent behaviour | controlled content, closed-loop timing |
| **Online** | `UserSim` + `LLMSource`: the first turn is given (text, or speech + transcript, from `task.scenario["first_turn"]`), later turns are generated by a text LLM; outputting `###STOP###` ends the episode | `TurnTaking` | fully closed loop, endogenous interaction |

- **Voice**: `Voice` works in priority order: if the turn carries its own audio, use it; else if a TTS is configured, synthesize; otherwise emit text only, with duration estimated at 3.4 words per second. The `data` of an audio Segment is `Audio` (16-bit PCM), and `text` is the time-aligned transcript.
- **Turn-taking** (`TurnTaking`, thresholds following τ-Voice): respond after the agent has been silent for `respond_after_ms=1000`; speak again if there is no reply within `nudge_after_ms=5000` after finishing; yield if the agent talks over the user for more than `yield_after_ms=1000`; while the agent is speaking, decide at each of its phrase boundaries (at least every `max_decision_gap_ms=3000`), and start speaking `reaction_ms=300` after deciding to interrupt.
- **Listening decision** (pluggable): `LLMListener` (τ-Voice's barge-in prompt extended to LISTEN / BACKCHANNEL / INTERRUPT with the persona, goal and listening style; one greedy call per decision point; `LLMInterrupt` is the old name), `KeywordInterrupt` (a rule), or `None` (this user never interrupts; a separate `listener` may still decide backchannels).
- **Subjective vs. random**: backchannels, barge-ins and mid-thought pauses (the user LLM's "(pause)" markers) are decided by models; noises and being called away are random at rates from the persona's surroundings; the background track is context. Every behaviour is a user turn labelled with `kind` / `expects` / `intent` (FORMAT.md §4.4).
- **Information boundary**: the user side can only read **what the agent has already said** via `conversation(...)` / `Segment.heard_text(t)`; the utterance in progress is marked `[CURRENTLY SPEAKING, INCOMPLETE]` (a dedicated test guarantees unspoken content never leaks to the LLM).
- LLM and TTS calls run inside `step`, while simulated time stands still; the real reaction time is represented by scheduling the utterance's timestamp in the future.

**Model services** (`clients.py`): the env does not deploy models itself. The LLM and TTS are **two independent services** behind interfaces: `TextGen.chat` (OpenAI-compatible chat, e.g. vLLM) and `Speech.synth` (a TTS service compatible with OpenAI `/audio/speech`). Each interface has three kinds of implementation: real clients (`OpenAIChat` / `OpenAISpeech`), fakes (`FakeChat` / `FakeSpeech`, for tests and development), and a `Cached` wrapper (cached by arguments, for reproducible replay and to avoid re-synthesizing).

## 2.5 Tool calls: one interface + adapters (`tools.py`, implemented)

The env side knows only three data types and one node; any external tool environment plugs in through an adapter:

```python
ToolSpec(name, description, parameters)     # schema in OpenAI function format
ToolCall(id, name, arguments)
ToolResult(id, name, content, error)

class ToolBackend:                           # adapter base class
    def tools(self, caller="agent") -> list[ToolSpec]
    def instructions(self, caller="agent") -> str        # text that goes into the prompt (e.g. domain rules)
    def reset(self, task, rng) -> State                  # one state per episode
    async def call(self, state, caller, call) -> ToolResult
    def fork(self, state, n) -> list[State]              # deepcopy by default
    def evaluate(self, task, log) -> dict | None         # optional: the backend provides the outcome reward

    state_group: str | None                              # backends (services) in the same group share one episode state

ToolWorld(backend, latency_ms=0 | (call -> ms))          # generic node: routes by caller, virtual latency, fork
```

- Stream conventions: the agent calls via `policy.tool_call` and results appear on `tool.result`; the user calls via `user.tool_call` and results appear on `tool.user_result`.
- **Virtual latency** can be set per tool (e.g. search 900 ms, lookup 0 ms): the tool executes at the moment it is called, and the result is delivered after the latency. τ-Voice's tools return instantly, which it lists as future work.
- Existing adapters:
  - `FunctionBackend`: tools defined as Python functions. The schema is generated from the function signature, and a function can read and write its per-episode state through a `state` parameter.
  - `integrations/tau.TauBackend`: τ-bench's domains, tools, databases and **official evaluation** (DB / ENV_ASSERTION / ACTION / COMMUNICATE; NL assertions need an LLM judge and are currently marked as skipped).
  - `integrations/automationbench`: Zapier's AutomationBench (MIT). Its 47 simulated SaaS apps (Gmail, Salesforce, Sheets, ...) are local Python simulators (a pydantic `WorldState` + per-app tool functions) with no network access.
    - Tasks: the request becomes what the user says (`scenario["turns"]`; an LLM user uses `scenario["instructions"]`); the world becomes `initial_state`; assertions become `criteria`. **Rules stay in the world** (e.g. an email in the inbox, a row in a sheet) and are not put into instructions or context, as in the original benchmark; the agent gets only the benchmark's system prompt and tools.
    - Tool sets: `api` (the official leaderboard setting: `api_search` uses BM25 over the schemas of about 500 REST endpoints, and `api_fetch` calls one by method/URL/body; the discovery mechanism is the benchmark's own), `zapier` (its `search_tools` / `execute_tool`), `limited_zapier` (only the actions the task allows, one service per app), and `apps` (all actions, one service per app, using ToolWorld's `discover` mode). The app services share one `WorldState` through `state_group` and are copied together on fork, so cross-app flows and evaluation see the same world.
    - Evaluation: the official rubric runs on the world at the end of the episode. `reward` is `task_completed_correctly` (1 only if all assertions pass; the leaderboard metric, no partial credit), and `partial_credit` is its dense signal for training (assertions already true in the initial state do not count; breaking one counts as a failure). IDs in the simulator are random, so a log cannot be replayed into the same world; `evaluate(env, task)` therefore reads the world inside the env rather than the log.
    - The benchmark provides no reference solutions; `HANDWRITTEN_SOLUTIONS` are hand-written correct call sequences (replayed by `ReplayAgent`) used to verify that reward=1 is reachable.
- MCP servers, search APIs and robot skill interfaces can be added later as further `ToolBackend` subclasses.

**Session initialization and the two tool-exposure modes** (implemented):
- At t=0, `Env.reset()` tells the agent about its environment on the `session` stream (`Session`: mode, instructions, tools, services, context) **without any world state**. Each node (e.g. `ToolWorld`) contributes its own part, and `task.scenario["agent_context"]` supplies context the agent is allowed to know. When something changes later (e.g. tools get loaded), the merged session is sent again.
- `ToolWorld(backends, mode="all" | "discover")`: several services can be mounted (with several services, tool names are `<service>__<tool>`).
  - `all`: all tools are given up front (the benchmark setting).
  - `discover`: only the service directory and three meta-tools, `list_services` / `search_tools` / `load_tools`, are given; tools can be called only after loading (the realistic setting with many tools). Search and loading have virtual latency too.
- **The tool environment decides what is public**: `ToolBackend.context(caller, state)` returns the information the tool environment is willing to reveal to the agent at load time (e.g. an account overview), which goes into the session context together with the tool schemas. Confidential information that should require a tool call is excluded by the tool environment itself; the env does not judge. In `discover` mode a service's context is revealed only when it is loaded. The τ-bench backend reveals no state at the start (as in the original benchmark).
- A harness (pre-retrieving tools, or leaving lookup to the model) belongs on the agent side and can be built on top of the interface `discover` mode provides.
- The trajectory records what the agent received at the start in `meta.agent.session`.

## Module map

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

## 3. Repository layout

Entries marked "planned" do not exist yet.

```
interaction_gym/
├── pyproject.toml              # optional dependency groups (e.g. [tau], [vllm-omni], [automationbench]) and a dev group
├── docs/DESIGN.md
├── src/interaction_gym/
│   ├── core.py                 # Frame, Node, Sim, Segment, Task, Env, VecEnv (about 300 lines)
│   ├── user.py                 # ReplayUser / UserSim + ScriptSource / LLMSource + Voice + TurnTaking + interruption deciders
│   ├── clients.py              # TextGen / Speech interfaces: OpenAI-compatible clients, fakes, caching
│   ├── audio.py                # Audio: 16-bit PCM that slices like a sequence
│   ├── tools.py                # unified tool interface: ToolSpec / ToolCall / ToolResult, ToolBackend, ToolWorld, FunctionBackend
│   ├── agents/                 # CannedAgent (hard-coded stand-in agent) and server adapters (e.g. vllm_omni)
│   ├── world/                  # planned: replay.py (exogenous stream replay, v1); physics/ in v2
│   ├── eval.py                 # evaluate(log, task, world): outcome / timing / LLM judge
│   ├── linearize.py            # planned: overlapping speech -> sequential text (for the user simulator and judge)
│   ├── traj.py                 # standard output format schema v1 (FORMAT.md): episode / save / load / to_frames / agent_view
│   ├── media.py                # unified content-addressed media store
│   ├── viewer/                 # self-contained HTML trajectory viewer: per-case timeline, state at the cursor, transcript, frame table
│   └── integrations/
│       ├── tau.py              # τ-bench adapter: task loading, TauBackend, official evaluation, OracleAgent (layer 1)
│       ├── automationbench.py  # AutomationBench: tasks, four tool sets, shared world, official rubric
│       ├── rlinf.py            # planned: env worker adapter
│       └── serving/            # planned: local streaming policy inference (HF reference implementation), PolicyNode (wraps cloud APIs)
├── examples/
│   ├── minimal.py              # rule-based user + simple policy, including one interruption
│   └── 01_tau_voice.py         # planned: run a τ-Voice task with this project's Sim
└── tests/
    ├── test_core.py            # deterministic tests: scheduling, latency, Segment truncation, fork
    └── test_tau_parity.py      # planned: same task set, this project's metrics ≈ official tau2 metrics
```

## 4. Deliberately left out (to be added when needed, without affecting the five abstractions above)

| Not done yet | When to add | How |
|---|---|---|
| Implementation of lockstep mode and batched nodes | v2 (when connecting ManiSkill) | the interfaces are defined (`Node.batched`, `Node.fork`); only the VecEnv lockstep loop needs implementing |
| Wall-clock mode (real humans / real robots) | v3 | add a parameter to Sim |
| Strict information-boundary checks | when an information leak is found | add a check when building the node wiring |

## 5. Roadmap

| Stage | Content | Exit criterion |
|---|---|---|
| **v0.0** | `core.py` + `eval.py` + `examples/minimal.py` + `test_core` / `test_minimal` (done) | scheduling, latency, truncation and fork all pass deterministic tests |
| **v0.1** | `user/` + `world/tooldb` + `integrations/tau` + `eval.py` | running the same cloud API model on the same task set gives results roughly matching official tau2 |
| **v0.2** | `integrations/serving` (local streaming policy) | baselines for open-source full-duplex models on τ-Voice / FD-Bench v2 |
| **v0.3** | `integrations/rlinf` + fork for GRPO | **core hypothesis test**: on endogenous tasks, online RL clearly beats offline baselines (the continue-or-stop decision point) |
| v1 | exogenous video streams (`world/replay`) | proactive video-reminder tasks are trainable |
| v2 | `world/physics` (ManiSkill) + action Segments | robot tasks with "corrected mid-execution" are trainable |
| v3 | real humans and real robots, wall-clock mode | tested with real humans or on real robots |

## 6. Open questions

1. **Which policy model**: τ-Voice tasks need tool calls, but Moshi / PersonaPlex cannot call tools. Candidates: MiniCPM-o 4.5, an open-source release of Qwen3-Omni (to be confirmed), or doing tool-free endogenous tasks first in v0.3 (FD-Bench v2 Correction / Entity Tracking).
2. **How deeply to reuse τ-Voice**: can its user simulator be used without its own orchestrator? If not, port the turn-taking functions and prompts and reuse the rest (tasks, tools, databases, audio effects, metrics) via import. To be decided in v0.1.
3. **Timeline for text-only output**: estimate from speaking rate by default, plus a consistency check with TTS.
4. **Compute platform**: on aarch64 clusters, TTS and vLLM need containers; physics-simulation rendering on aarch64 must be validated ahead of time (before v2).
