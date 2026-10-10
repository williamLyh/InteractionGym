# InteractionGym: closed-loop simulation for full-duplex spoken agents

InteractionGym is a set of closed-loop environments for **training and evaluating full-duplex spoken agents**. The agent listens and speaks at the same time. A simulated user reacts to what it actually hears: it answers, interrupts, backchannels, pauses, and talks to someone else in the room. Agents are evaluated, and can be trained with online RL, against that user instead of fixed recordings.

- **Virtual-time simulation.** Episodes are exact and reproducible, and can run faster than real time, including input-clocked lockstep with full-duplex models served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni).
- **LLM + TTS simulated users.** Model-decided backchannels, barge-ins and pauses come from a structured persona. Every user behaviour is a labelled, scored turn.
- **Built-in pieces.** A trajectory format with a JSON Schema, one rule-based score per user turn (usable as metric and reward), an HTML viewer, benchmark loaders run open and closed loop, [pluggable tool environments](#adding-your-own-tool-environment), and a GPU tuner for serving layouts.

Status: research code, version 0.1.0; APIs may still change. **Demo:** https://yinhongliu.com/InteractionGym/ (source in [demo/](demo/))

## Install

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/williamLyh/InteractionGym && cd InteractionGym
uv sync && uv run pytest -q          # deterministic tests, no models or network
```

Optional extras are `vllm-omni`, `tau`, `automationbench` and `fdbench` (Full-Duplex-Bench, CC BY-NC 4.0). See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

### Agent server: vLLM-Omni + our patch (required for full-duplex agents)

> [!IMPORTANT]
> Stock vLLM-Omni does **not** have what InteractionGym needs: lockstep (input-clocked) sessions, the token trace, Thinker-only MiniCPM-o, and the per-session feature-extractor fix. Install `vllm-omni==0.31.0rc1` in its own environment and apply our patch before serving any agent.

```bash
python3.12 -m venv ~/envs/omni
~/envs/omni/bin/pip install vllm==0.31.0 vllm-omni==0.31.0rc1
~/envs/omni/bin/pip install nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88   # CUDA JIT pins, see the patch README
python3 scripts/apply_vllm_omni_patch.py --python ~/envs/omni/bin/python      # --status to check, --revert to undo
```

The patch carries open upstream PRs ([vllm-omni#8485](https://github.com/vllm-project/vllm-omni/pull/8485), [#8638](https://github.com/vllm-project/vllm-omni/pull/8638)) and follow-ups; each part is dropped once it is merged and released. Supported bases (0.31.0rc1, or a source checkout of main `61cae20d`), CUDA pins and other install pitfalls: [patches/vllm-omni/README.md](patches/vllm-omni/README.md). Reinstalling vLLM-Omni removes the patch; apply it again.

## Quick start (no GPU)

```bash
uv run python examples/minimal.py --html runs/minimal.html    # scripted user and agent, open the HTML in a browser
```

```python
from interaction_gym import AgentSpec, Env
from interaction_gym.agents import CannedAgent
from interaction_gym.traj import episode, save

spec = AgentSpec(chunk_ms=200)                          # the agent acts every 200 ms of simulated time
env = Env({"user": user}, spec, max_ms=60_000)          # nodes: a simulated user, tools, ...
agent = CannedAgent(replies, spec)
obs, done = await env.reset(task, seed=0), False
while not done:
    obs, reward, done = await env.step(agent.act(env.t, obs))
save([episode(env, "ep-0")], "runs/episodes.json")      # view: python -m interaction_gym.viewer runs/*.json -o viewer.html
```

## Output data

Each episode is one JSON record (trajectory schema v1). A run is a JSONL file with one episode per line, and audio is kept in a shared, content-addressed media store that the episodes reference.

| | |
|---|---|
| [docs/FORMAT.md](docs/FORMAT.md) | field reference: `meta` (task, env context, user simulation, agent), `background`, `turns`, `tool_calls`, media references, `eval` (scores, reward) |
| [docs/trajectory.schema.json](docs/trajectory.schema.json) | JSON Schema for one episode; [docs/format_example.json](docs/format_example.json) is an example |
| [docs/AGENT_TRACE.md](docs/AGENT_TRACE.md), [docs/agent_trace.schema.json](docs/agent_trace.schema.json) | optional companion `agent_traces.jsonl`: the agent server's per-unit tokens |

```python
from interaction_gym.traj import load
episodes = load("runs/episodes.json")          # list of dicts, as in the schema
```

## With real models

The user simulator needs OpenAI-compatible servers: a chat LLM, a TTS, and a voice-clone TTS (later user turns are cloned from the first). A full-duplex agent also needs a vLLM-Omni duplex server.

```bash
export IG_LLM_URL=http://localhost:8000/v1 IG_TTS_URL=http://localhost:8001/v1 IG_CLONE_URL=http://localhost:8005/v1
uv run python examples/live_user.py --html runs/live_user.html              # LLM + TTS user, scripted agent

uv sync --extra vllm-omni
IG_AGENT_URL='ws://localhost:8010/v1/realtime?duplex=1' IG_MODEL_DIR=/path/to/models \
  uv run python examples/minicpmo_suite.py --out runs/suite                 # MiniCPM-o 4.5 as the agent
```

Notes:
- **MiniCPM-o output.** By default MiniCPM-o runs Thinker-only, with text output timed at speaking rate; `--audio-out` gives real speech.
- **vLLM-Omni patch.** The agent server must run the patched vLLM-Omni from [Install](#agent-server-vllm-omni--our-patch-required-for-full-duplex-agents).
- **Serving.** A reference deployment for one 8-GPU host is in [examples/serving/](examples/serving/), and the details are in [docs/agent_server.md](docs/agent_server.md).

## Adding your own tool environment

Tools are one more node in the env. A `ToolWorld` node runs the calls, delivers each result after a simulated latency, and forks its state with the episode. A tool environment plugs into it through a `ToolBackend` adapter. The agent calls on the `policy.tool_call` stream and hears results on `tool.result`, at the simulated time they arrive, so it can keep talking (or be interrupted) while a call is pending.

**1. Plain Python functions.** The schema comes from the signature and docstring. A `state` argument is the episode's own copy of the world: changes stay inside the episode, and every fork gets its own copy. An exception becomes an error result for the agent, not a crash.

```python
from interaction_gym import AgentSpec, Env, Task
from interaction_gym.tools import RESULT, FunctionBackend, ToolWorld, tool_log

def find_order(state: dict, order_id: str) -> dict:
    """Look up an order by its id."""
    return state["orders"][order_id]

def cancel_order(state: dict, order_id: str, reason: str = "") -> dict:
    """Cancel an order that has not shipped yet."""
    order = state["orders"][order_id]
    if order["status"] == "shipped":
        raise ValueError("already shipped")
    order["status"] = "cancelled"
    return order

shop = FunctionBackend({"find_order": find_order, "cancel_order": cancel_order},
                       instructions="Confirm with the user before cancelling.", name="shop")
tools = ToolWorld(shop, latency_ms=lambda call: 1500 if call.name == "find_order" else 300)
task = Task(id="cancel-1", initial_state={"orders": {"A7": {"status": "processing"}}})
spec = AgentSpec(chunk_ms=200, obs=("user.speech", RESULT["agent"]))   # the agent must observe tool results
env = Env({"user": user, "tools": tools}, spec, max_ms=120_000)
```

At `reset` the agent receives a `Session` on the `session` stream, with the instructions and tool schemas. `CascadedAgent` passes them to its LLM as OpenAI tools.

**2. Score the episode.** `tool_log(env.log)` returns `(time, caller, call, result)` for every call. Turn it into a reward and pass it to `episode(env, ..., reward={"total": r, "parts": {...}})`:

```python
calls = [(c.name, c.arguments, r) for _, caller, c, r in tool_log(env.log) if caller == "agent"]
r = float(any(name == "cancel_order" and args.get("order_id") == "A7" and res and not res.error
              for name, args, res in calls))
```

**3. Any other system** (an MCP server, a REST API, an existing benchmark): subclass `ToolBackend`.

```python
import json
from interaction_gym.tools import ToolBackend, ToolResult, ToolSpec

class CRMBackend(ToolBackend):
    name, description = "crm", "Customer records"

    def tools(self, caller="agent"):          # the schemas this caller may use ("agent" or "user")
        return [ToolSpec("get_customer", "Look up a customer.",
                         {"type": "object", "properties": {"email": {"type": "string"}}, "required": ["email"]})]

    def reset(self, task, rng):               # per-episode state; keep everything mutable in it
        return {"customers": dict(task.initial_state["customers"])}

    async def call(self, state, caller, call):
        c = state["customers"].get(call.arguments.get("email"))
        return ToolResult(call.id, call.name, json.dumps(c) if c else "not found", error=c is None)
```

Optional hooks:
- `instructions(caller)`: a policy text for the prompt.
- `context(caller, state)`: what the agent may know up front, such as an account overview. Anything it should have to look up stays out.
- `fork(state, n)`: how to copy the state. The default is a deep copy.
- `evaluate(task, log)`: the backend's own outcome reward.

**Options.**
- **Several services.** `ToolWorld({"shop": shop, "crm": crm})` exposes their tools as `shop__find_order`, `crm__get_customer`, and so on. Backends with the same `state_group` share one world.
- **`mode="discover"`.** The agent starts with only a list of the services and three meta-tools (`list_services`, `search_tools`, `load_tools`), and loads a service before calling its tools. This is the realistic setting when there are many tools.
- **User-side tools.** `tools(caller="user")` lets the simulated user call tools too, on `user.tool_call`.

**Agent support.** `CascadedAgent` (ASR → tool-calling LLM → TTS) and your own agents call tools. The vLLM-Omni duplex agent does not, because the duplex server takes no tool schemas.

**Worked examples.**
- [examples/tools_demo.py](examples/tools_demo.py): slow tools with filler speech, a user talking while a call is pending, error and retry, discover mode. It needs no GPU.
- Full adapters: [integrations/tau.py](src/interaction_gym/integrations/tau.py) (τ²-bench, with its official evaluator) and [integrations/automationbench.py](src/interaction_gym/integrations/automationbench.py) (47 simulated SaaS apps sharing one world).
- Design notes: [docs/DESIGN.md §2.5](docs/DESIGN.md#25-tool-calls-one-interface--adapters-toolspy-implemented).

## GPU tuner: run it before any large run

A run serves several models at once (the duplex agent, the user LLM, TTS and clone TTS), and **how the GPUs and session caps are split between them decides throughput far more than any single server setting**. A hand-written split typically leaves some services idle while the agent is the bottleneck. The tuner runs real episodes on the current layout, measures GPU util, queues and per-service wait time, and moves GPUs and session caps toward the measured load.

| on one 8× RTX 5090 host (measured) | episodes/hour | |
|---|---|---|
| hand-written layout | 447 | 1.00× |
| after the tuner (GPU split + session caps) | **788** | **1.76×** |
| session cap pushed too high (tuner rejects it) | 460, 24 failed | 1.03× |

Full table and setup: [docs/GPU_TUNER.md, "Why it matters"](docs/GPU_TUNER.md#why-it-matters-measured-effect). Re-tune when the models, the GPUs or the agent mode (Thinker-only vs `--audio-out`) change.

```bash
uv run python -m interaction_gym.gpu_tuner dry-run --rounds 6      # mock services: see what the loop does, no GPU
PYTHONPATH=src:. python examples/gpu_tuner.py loop --layout "agent=2/3/4/5,llm=0+1,tts=6,clone=7" --out runs/gpu_tuner
                                                                   # on the GPU host: measure the running layout, propose a better one
```

## Benchmarks

```bash
uv sync --extra vllm-omni --extra fdbench
uv run python examples/fdb3_ab.py --help                      # e.g. Full-Duplex-Bench v3, open vs closed loop
```

The repository ships loaders only, no benchmark data, and several datasets are non-commercial. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [THIRD_PARTY.md](THIRD_PARTY.md). The release results are in [results/BENCHMARKS.md](results/BENCHMARKS.md).

## Documentation

| | |
|---|---|
| [GETTING_STARTED](docs/GETTING_STARTED.md) | install, extras, first episodes, real models, τ²-bench / AutomationBench |
| [DESIGN](docs/DESIGN.md) | architecture and concepts |
| [FORMAT](docs/FORMAT.md) | output data: trajectory format, scoring ([schema](docs/trajectory.schema.json)) |
| [BENCHMARKS](docs/BENCHMARKS.md) | benchmark loaders, open vs closed loop, metrics |
| [agent_server](docs/agent_server.md), [AGENT_TRACE](docs/AGENT_TRACE.md) | agent-server protocol, token trace |
| [GPU_TUNER](docs/GPU_TUNER.md) | serving-layout tuner, measured speed-up |

## Noise data

No audio is shipped. By default the simulated user's background ambience (home, office, car, cafe, street) is real
recordings from **DEMAND**: about 80 MB, fetched from Zenodo on first use into `~/.cache/interaction_gym/soundbank`.
Set `IG_SOUNDBANK` to use another directory, or `IG_SOUNDBANK=synthetic` to use the synthetic stand-ins. If the fetch
fails, the env falls back to synthetic noise. DEMAND is licensed CC BY-SA 3.0, so a bank derived from it is ShareAlike:

> J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)",
> ICA 2013, doi:[10.5281/zenodo.1227121](https://doi.org/10.5281/zenodo.1227121).

Noise events (coughs, doors, phones, horns ...) are synthetic by default. Recorded events come from **MUSAN** (Snyder,
Chen & Povey 2015, CC BY 4.0), in the optional full bank that `scripts/fetch_noise_banks.py` builds (an 11 GB
download). See [THIRD_PARTY.md](THIRD_PARTY.md) and [docs/FORMAT.md §4.2](docs/FORMAT.md).

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)), except the third-party material in [THIRD_PARTY.md](THIRD_PARTY.md). `extras/fdbench/` is a separate CC BY-NC 4.0 distribution and is not part of the `interaction-gym` package. The demo page in `demo/` carries third-party audio under its own licences ([demo/README.md](demo/README.md)).

## Citation

```bibtex
@misc{interactiongym2026,
  title  = {InteractionGym: Closed-Loop Environments for Full-Duplex Spoken Agents},
  author = {Liu, Yinhong},
  year   = {2026},
  url    = {https://github.com/williamLyh/InteractionGym}
}
```
