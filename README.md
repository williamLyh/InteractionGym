# InteractionGym: closed-loop simulation for full-duplex spoken agents

InteractionGym is a set of closed-loop environments for **training and evaluating full-duplex spoken agents**. The agent listens and speaks at the same time. A simulated user reacts to what it actually hears: it answers, interrupts, backchannels, pauses, and talks to someone else in the room. Agents are evaluated, and can be trained with online RL, against that user instead of fixed recordings.

- **Virtual-time simulation.** Episodes are exact and reproducible, and can run faster than real time, including input-clocked lockstep with full-duplex models served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni).
- **LLM + TTS simulated users.** Model-decided backchannels, barge-ins and pauses come from a structured persona. Every user behaviour is a labelled, scored turn.
- **Built-in pieces.** A trajectory format with a JSON Schema, one rule-based score per user turn (usable as metric and reward), an HTML viewer, benchmark loaders run open and closed loop, tool environments, and a GPU tuner for serving layouts.

Status: research code, version 0.1.0; APIs may still change. **Demo:** https://yinhongliu.com/interaction-gym-demo/

## Install

Python 3.12 with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/williamLyh/InteractionGym && cd InteractionGym
uv sync && uv run pytest -q          # deterministic tests, no models or network
```

Optional extras are `vllm-omni`, `tau`, `automationbench` and `fdbench` (Full-Duplex-Bench, CC BY-NC 4.0). See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

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
- **vLLM-Omni patches.** Lockstep and the token trace need our vLLM-Omni patches ([vllm-omni#8485](https://github.com/vllm-project/vllm-omni/pull/8485)).
- **Serving.** A reference deployment for one 8-GPU host is in [examples/serving/](examples/serving/), and the details are in [docs/agent_server.md](docs/agent_server.md).

## Benchmarks and GPU tuner

```bash
uv sync --extra vllm-omni --extra fdbench
uv run python examples/fdb3_ab.py --help                      # e.g. Full-Duplex-Bench v3, open vs closed loop
uv run python -m interaction_gym.gpu_tuner dry-run --rounds 6 # GPU split across agent / LLM / TTS services
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
| [GPU_TUNER](docs/GPU_TUNER.md) | serving-layout tuner |

## License

Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE)), except the third-party material in [THIRD_PARTY.md](THIRD_PARTY.md). `extras/fdbench/` is a separate CC BY-NC 4.0 distribution and is not part of the `interaction-gym` package.

## Citation

```bibtex
@misc{interactiongym2026,
  title  = {InteractionGym: Closed-Loop Environments for Full-Duplex Spoken Agents},
  author = {Liu, Yinhong},
  year   = {2026},
  url    = {https://github.com/williamLyh/InteractionGym}
}
```
