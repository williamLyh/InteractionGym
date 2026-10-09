# Plan: hosted (closed) model backends

Status: **planned, not implemented**. This is for a release after 0.1.0. Written 2026-10-08. Open decisions are listed at the end.

The goal is that the simulated user and the agent can each run on hosted model APIs (OpenAI, Anthropic, Gemini, ElevenLabs, …), in addition to the local OpenAI-compatible servers used today.

## What exists today

The environment talks to models through small duck-typed protocols:
- `TextGen.chat(messages) -> str`
- `Speech.synth(text, voice, instructions, ref_audio, ref_text, language, seed) -> Audio`
- `Transcriber.transcribe(audio) -> str`
- the agent's `act(t, obs) -> frames`

A hosted backend is therefore a new class and does not touch the core.

The current clients assume a local server:
- `clients._post` sends Bearer auth only, has no retries or backoff, and drops `usage`.
- `OpenAISpeech` relies on vLLM-Omni extras for cloning.
- `ToolChat` takes no API key.
- `describe()` records the model and params but not the provider.
- `LLMSource.messages` can start with an `assistant` message when the user speaks first, which Anthropic and Gemini reject.
- The `LLMListener` / `AsideWriter` token budgets (16 / 96) are too small for reasoning models.

## Reference: τ-Voice (third_party/tau2-bench)

Worth reusing:
- a normalised per-tick result (`TickResult`), including "audio actually played" when the agent is interrupted;
- a usage ledger (`UsageRecord`, delta vs cumulative meters) kept apart from a versioned pricing table, where an unknown price is `None`, never 0;
- the per-provider event parsing for OpenAI Realtime, Gemini Live, xAI, Nova Sonic and Qwen-Omni.

To avoid:
- tying simulated time to wall time, so that user-simulator LLM/TTS calls stall the tick loop while the provider keeps running;
- a hard-coded provider `if/elif`;
- voice, VAD and audio format hard-coded as constants and not logged;
- a hard-coded user turn-taking model with no seed;
- inconsistent interruption handling, since only OpenAI is told how much audio was played;
- for text LLMs, litellm silently dropping unsupported params and costing errors as 0.

## Design

### Three layers
| Layer | Shared between user and agent? | Contents |
|---|---|---|
| Env interface | already one | agent `act(t, obs)`; the user node's audio stream |
| Role logic | **separate** | agent: instructions, tools, turn detection. User: persona, goal, model-decided behaviours, and the per-turn labels (`kind` / `expects` / `intent`) that evaluation needs |
| Model backends | **shared** | `ChatModel`, `ToolModel`, `Speech`, `Transcriber`, `RealtimeProvider`, plus one registry, retries, a disk cache and a usage ledger |

Each role has two engines:
- **Agent:** `CascadedAgent` (ASR + LLM + TTS, any of them hosted) or `RealtimeAgent(RealtimeProvider)`.
- **User:** `UserSim(LLMSource, Voice)`, the default, or an experimental `RealtimeUser(RealtimeProvider, labeler)`.

### Model protocols
```python
class ChatModel(Protocol):
    async def chat(self, messages: list[dict], **params) -> str
    def describe(self) -> dict          # provider, model, host, params, api_version

class ToolModel(Protocol):
    async def complete(self, messages, tools: list[ToolSpec], **params) -> Completion  # content, tool_calls, usage

class Speech(Protocol):                 # synth() as today
    consistency: Literal["clone", "voice_id", "none"]

class RealtimeProvider(Protocol):
    async def connect(self, cfg: RealtimeConfig) -> None   # instructions, tools, voice, vad, formats, reasoning
    async def send_audio(self, pcm: bytes) -> None
    async def send_tool_result(self, call_id, content, is_error=False) -> None
    async def truncate(self, item_id, played_ms) -> None
    async def cancel(self) -> None
    def events(self) -> AsyncIterator[RtEvent]   # AudioDelta, TranscriptDelta, ToolCall, SpeechStarted,
                                                 # ResponseDone, Usage, Error
    capabilities: set[str]                       # truncate, server_vad, manual_turns, tools, ...
```

### Shared backend features
- **Messages:** OpenAI-style messages everywhere. Adapters convert them per provider: the system prompt moves out, roles must alternate, and a conversation must not start with `assistant`.
- **Token budgets:** `max_output_tokens` is kept separate from a reasoning budget. Short decision calls (the listener) either turn reasoning off or get a larger budget.
- **Reliability:** retries with exponential backoff on 429 and 5xx, timeouts, and a concurrency limit per provider.
- **Caching:** a persistent disk cache keyed by provider, model, params and the full request. Reruns with a hosted user simulator are then cheap and reproducible.
- **Usage:** a ledger per call, summed per component into `meta.usage` (`user.llm`, `user.tts`, `agent`), and priced from a versioned table.
- **Seeds:** passed when the provider supports them; otherwise `describe()` records `seed: unsupported`.
- **Selection:** `provider:model` strings resolved by a registry, for example `make_chat("anthropic:<model>")`, `make_tts("elevenlabs:<model>")`, `make_realtime("openai-realtime:<model>")`, or `local:<model>@<url>` for today's servers. Runner variables are `IG_USER_LLM`, `IG_USER_TTS` and `IG_AGENT`.
- **Keys:** read only from each provider's standard environment variable, and never written to trajectories.
- **Native adapters:** OpenAI, Anthropic and Gemini. litellm is an optional catch-all backend, not a dependency.

### Voice consistency for hosted TTS
The user's voice must stay the same across turns, and cloning is the default. Hosted TTS rarely clones from a reference clip, so `Speech.consistency` says how a backend keeps the voice:
- `clone`: our Qwen3-TTS Base;
- `voice_id`: a fixed preset voice id;
- `none`.

`Voice` accepts `clone` and `voice_id`, and raises on `none` unless `clone=False`. Each provider ships a voice catalogue (gender, age, accent), so `pick_voice` works unchanged.

### Clocks (the hard part)
A hosted realtime API cannot be paused. Its VAD timers and its generation run on wall time.

| User \ agent | local full-duplex (lockstep) | cascaded (text + TTS, may be hosted) | hosted realtime |
|---|---|---|---|
| LLM + TTS (default) | virtual clock, reproducible, RL-capable | virtual clock, reproducible | needs one of the modes below; eval only |
| realtime (experimental) | wall | wall | wall |

When one side is realtime, the other side's compute time leaks into the conversation:
- A user LLM+TTS reply takes about 0.5–1.5 s, against human gaps of 0.2–0.7 s.
- A listener decision lands late, so a backchannel ends up in the wrong place.
- The agent then reacts to silence that the simulator caused.

Two candidate modes:

1. **Audio-clocked (preferred; needs a probe first).** This works if a provider's VAD counts received audio samples rather than wall time. Then:
   - user audio is sent in sim time, in bursts;
   - nothing is sent while the user simulator computes, so the provider never sees the gap;
   - provider output, usually generated faster than real time, is buffered and played on the sim timeline;
   - the provider's own response latency is measured in wall time, since that is the agent's real latency;
   - when the user barges in, `truncate(played_ms)` tells the provider how much audio was actually played.

   **Probe:** OpenAI Realtime and Gemini Live, a few dozen calls each, after cost confirmation. Check three things:
   - whether turn detection follows samples when audio is sent at 4× speed;
   - whether a 5–10 s send pause causes a disconnect or a server-side timeout;
   - the output generation speed relative to real time.

2. **Wall-clocked (fallback).** Sim time = wall time.
   - The user simulator's LLM/TTS run asynchronously, and the microphone keeps streaming silence or ambience meanwhile.
   - The user's replies are pre-generated where possible, and a fast user LLM plus streaming TTS are used.
   - Each user turn records `user_late_ms`, and evaluation excludes or marks late turns.

Hosted realtime agents have no logprobs and are not reproducible, so they are for evaluation, not RL. The trajectory records `meta.agent.clock` and every resolved session setting: voice, VAD, formats and transcription model.

What the user hears: by default the agent's own transcript, as in τ-Voice and our current setup. An option runs ASR on the agent's audio, so the agent's pronunciation and audio quality reach the user.

### Cost and safety
- `--max-cost-usd`, plus an estimate printed before a run starts.
- Raw usage is kept per episode, so a run can be re-priced later.
- Runs that call hosted APIs are never scheduled unattended.

## Phases (after the 0.1.0 release)
1. Text LLM backends (OpenAI, Anthropic, Gemini, optional litellm), with the registry, retries, disk cache, usage ledger and `describe()` provenance. This lets the user simulator and the judges run on hosted LLMs.
2. Hosted TTS and ASR, with `Speech.consistency` and voice catalogues.
3. A cascaded agent on hosted parts, on the virtual clock. Benchmarks can use it directly.
4. The realtime probe, then `RealtimeAgent` (OpenAI Realtime first, then Gemini Live) in audio-clocked mode, or wall-clocked if the probe fails. Event parsing follows τ-Voice. `RealtimeProvider` is shared so that a user can use it too.
5. More providers (xAI, Nova Sonic, Qwen-Omni), and the experimental `RealtimeUser`, which also needs a labeler for `kind` / `expects` / `intent`, goal and stop monitoring, and preset voices only.

## Open decisions
- litellm: optional fallback (proposed) or the main path.
- Which providers come first. Proposed: LLM Anthropic, OpenAI and Gemini; TTS OpenAI and ElevenLabs; realtime OpenAI.
- A default budget cap and its value.
- Keys for the probe (an OpenAI key exists locally; there is no Gemini key yet).
