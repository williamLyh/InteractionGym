# Getting started

InteractionGym is a closed-loop simulator for full-duplex spoken agents. This guide goes from a fresh clone to a closed-loop episode against real models. Steps 1-3 need no GPU and no network access
beyond installing packages.

## 1. Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/williamLyh/InteractionGym
cd InteractionGym
uv sync
uv run pytest -q          # all tests are deterministic; a few skip without optional extras
```

Optional extras:

| extra | for | install |
|---|---|---|
| `vllm-omni` | the vLLM-Omni duplex agent adapter (WebSocket client) | `uv sync --extra vllm-omni` |
| `tau` | τ²-bench tasks (Sierra's tau2-bench, pinned git commit) | see [τ²-bench and AutomationBench](#τ²-bench-and-automationbench) |
| `automationbench` | AutomationBench tasks | see [τ²-bench and AutomationBench](#τ²-bench-and-automationbench) |
| `fdbench` | Full-Duplex-Bench v1 / v1.5 / v2 / v3 loaders: the separate component `interaction-gym-fdbench` in `extras/fdbench`, **CC BY-NC 4.0 (non-commercial only)** | `uv sync --extra fdbench` (the dev group includes it) |

With pip: `pip install -e ".[vllm-omni]"`; `pip install ./extras/fdbench` for the Full-Duplex-Bench component.

## 2. A first offline episode

```bash
uv run python examples/minimal.py
```

A scripted user asks about tomorrow's weather; while the scripted agent answers, the user notices the wrong day and
interrupts (300 ms reaction time); the agent yields and answers again. The output is the event log (time, stream,
segment span, transcript), the reward, and response / yield latencies.

Things to try in `examples/minimal.py`: `AgentSpec(chunk_ms=80)` (a finer agent step), `CannedAgent(...,
yield_after=None)` (an agent that never yields), `streaming=True` (the agent emits its reply in chunks).

## 3. The viewer

```bash
uv run python examples/minimal.py --html runs/minimal.html
uv run python examples/user_modes.py --html runs/user_modes.html
```

Open the pages in a browser: one row per episode variant, a timeline with every turn (click to play its audio when
there is any), the user simulation's settings, the environment context and the duplex scores. `user_modes.py`
compares the three kinds of user (replay, scripted, LLM-driven) on one scenario with fake LLM / TTS clients, so it
produces (synthetic) audio without any model.

Episodes are plain JSON ([FORMAT.md](FORMAT.md)): save them with `interaction_gym.traj.save`, rebuild a page
with `uv run python -m interaction_gym.viewer runs/*.json -o viewer.html`, and score them with
`interaction_gym.eval.scores(episode["turns"], end_ms=episode["meta"]["duration_ms"])`: one score per user
turn (did the agent do what the user expected, graded by latency when it took over: `eval.latency_score`, a
logistic with full score up to ~500 ms, 0.92 at 700 ms, 0.38 at 1 s; docs/FORMAT.md §6.3).

## 4. A closed-loop episode with real models

You need an OpenAI-compatible chat LLM, a TTS server and a voice-cloning TTS server (e.g. vLLM and vLLM-Omni with
Qwen3-TTS CustomVoice and Base; see
[examples/serving/](../examples/serving/) for an example deployment). If they run on another machine, tunnel them:

```bash
ssh -N -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8005:localhost:8005 <gpu-host>
```

Then run an LLM-driven user (its words from the LLM, its voice from the TTS) against a scripted agent:

```bash
IG_LLM_URL=http://localhost:8000/v1 IG_LLM_MODEL=<served LLM name> \
IG_TTS_URL=http://localhost:8001/v1 IG_CLONE_URL=http://localhost:8005/v1 \
  uv run python examples/live_user.py --html runs/live_user.html
```

`IG_TTS_MODEL` / `IG_TTS_VOICE` select another TTS model or voice (defaults: Qwen3-TTS 1.7B CustomVoice, voice
`vivian`). The user's first turn is synthesized with a neutral everyday-speech instruction and every later turn is
cloned from it by the clone TTS (`IG_CLONE_URL`, `IG_CLONE_MODEL`, default Qwen3-TTS 1.7B Base), so the user keeps one
voice, pace and manner for the whole call; each turn is also trimmed and loudness-normalised (`user.Voice`,
docs/FORMAT.md §4.1.2). A `Voice` with a TTS but no clone TTS raises; `Voice(tts, clone=False)` opts out. The page shows what the user said, when, and why it barged in. The same LLM decides, at each phrase boundary of the agent, whether the user listens, backchannels or cuts in (`LLMInterrupt` = `LLMListener`); a persona with `surroundings` (e.g. `{"profile": {"surroundings": "cafe"}}`) also gets random noises, asides and a matching background track — every one a labelled user turn on the page (docs/FORMAT.md §4.4).

Recorded noise instead of the synthetic stand-ins: build a sound bank once (DEMAND ambience + MUSAN noise events,
MUSAN comes as one 11 GB archive of which only the noise part is kept, DEMAND as ~80 MB of range requests; the
bank itself is ~115 MB, 16 kHz mono) and
pass it to the user:

```bash
uv run python scripts/fetch_noise_banks.py --out ~/dig_soundbank      # downloads, then builds ~/dig_soundbank/bank
```

```python
UserSim(..., soundscape=Soundscape(bank="~/dig_soundbank/bank"))
```

Check the datasets' licenses first (THIRD_PARTY.md: DEMAND is CC BY-SA 3.0, MUSAN CC BY 4.0).

## 5. A full-duplex model as the agent

With a vLLM-Omni duplex server for MiniCPM-o 4.5 (on `:8010`) as well:

```bash
uv sync --extra vllm-omni
IG_LLM_URL=http://localhost:8000/v1 IG_TTS_URL=http://localhost:8001/v1 \
IG_AGENT_URL='ws://localhost:8010/v1/realtime?duplex=1' IG_MODEL_DIR=/path/to/models \
  uv run python examples/minicpmo_agent.py --html runs/minicpmo_agent.html
```

This runs in realtime mode (works on stock vLLM-Omni). `--clock input` (lockstep) and `--trace-tokens` need a
vLLM-Omni build with our duplex patches ([agent_server.md](agent_server.md)). For several personas, voice cloning
and the token-trace pages, see `examples/minicpmo_suite.py`.

The agent's output is **text only by default**, timed at `speech_cps` (11.3 characters per second for MiniCPM-o
4.5): `VllmOmniDuplexAgent(audio_out=False)`, served by a Thinker-only MiniCPM-o server (one GPU; the reference
deployment's default, with its `patches/minicpmo_thinker_only.patch`). This is what every MiniCPM-o runner and
example in the repository does. `--audio-out` asks for the agent's real speech and needs the full two-GPU deployment
(`AGENT_LAYOUT=audio`); there, the first session after the server starts may return no audio (lazy initialisation):
run it twice or warm the server up first. Text-only runs are not bit-identical to audio runs (the deployments
differ at bf16 level from the first speak unit), their agent timing is an estimate, and their results are not
directly comparable with full-audio runs ([BENCHMARKS.md](BENCHMARKS.md), "Agent output").

Served model names default to the reference deployment's; override them with `IG_LLM_MODEL`, `IG_TTS_MODEL`,
`IG_CLONE_MODEL`, `IG_AGENT_MODEL` (see each example's docstring). `IG_MODEL_DIR` locates MiniCPM-o's reference
voice (`MiniCPM-o-4_5/assets/system_ref_audio.wav`; or set `IG_AGENT_REF_AUDIO`). An example deployment for one
8-GPU host is in [examples/serving/](../examples/serving/).

## τ²-bench and AutomationBench

τ²-bench (Sierra's tau2-bench, MIT; not the unrelated `tau2` package on PyPI) is installed from git at a pinned
commit (`5bfa7e3`); its domain data is read from a checkout:

```bash
git clone --depth 1 https://github.com/sierra-research/tau2-bench third_party/tau2-bench   # data (or set TAU2_DATA_DIR)
uv sync --extra tau
uv run python examples/tau_mock.py --html runs/tau_mock.html
```

AutomationBench ([Zapier, MIT](https://github.com/zapier/AutomationBench)): business workflows across 47 simulated
SaaS apps, scored on the end state of the world; the simulator is local Python. Loaded from a checkout (tested at
v1.0.6; it declares Python ≥ 3.13, the parts used run on 3.12):

```bash
git clone --depth 1 https://github.com/zapier/AutomationBench third_party/automationbench   # or set AUTOMATIONBENCH_PATH
uv sync --extra automationbench
uv run python examples/automationbench_demo.py --html runs/automationbench_demo.html
```

## Next

- [DESIGN.md](DESIGN.md): the simulation model and its design choices
- [FORMAT.md](FORMAT.md): the trajectory format
- [BENCHMARKS.md](BENCHMARKS.md): benchmark loaders, open- vs. closed-loop evaluation
- [GPU_TUNER.md](GPU_TUNER.md): splitting GPUs between agent, user LLM and TTS
