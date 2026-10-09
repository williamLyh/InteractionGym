# Requirements for agent servers

The agent (the full-duplex model being trained or evaluated) is **driven chunk by chunk in simulated time** by the env. Whether it is an in-process model or a WebSocket service (e.g. vLLM-Omni speaking a dialect of the OpenAI Realtime protocol), it must meet the requirements below.

## A. Time semantics (required for both evaluation and training)

1. **Driven by input, not by the wall clock**: the model advances only when it receives an input chunk. There must be no wall-clock-triggered logic, such as timed generation, proactive-speech timers, idle timeouts, or session state that "assumes the client plays back in real time". The wall-clock gap between two inputs must not affect the output.
2. **Incremental input**: each step sends only the **new** chunk of input (audio or video for `[t, t+Δ)`), never the history. Δ is the agent-session property `AgentSpec.chunk_ms` (default 200 ms; set per model to 80 / 160 / 240 ms, 1 s, etc.). The server keeps state per session (KV cache stays resident).
3. **Output may be faster than real time**: output deltas may be returned as fast as they are generated, but must carry their **audio duration** (or allow it to be computed). The env places output on the simulated timeline according to duration and buffers the excess.
4. **Truncation is at the actual playback position**: when the agent is interrupted, the env tells the server to truncate at the **position already played in simulated time** (the Realtime protocol's `conversation.item.truncate` + `audio_end_ms`), so that the model's context keeps only what was actually said.
5. **Reproducible**: with a fixed random seed and greedy decoding, the same input sequence must produce the same output.

## B. Additional requirements for training (not provided by the Realtime protocol; need an extension or an in-process implementation)

6. The **logprob** of every generated token, including silence / PAD / control tokens, since speaking timing is learned through them.
7. **Hot weight updates**: without restarting the server and without losing the configuration of ongoing sessions.
8. **Session fork / snapshot**: copy n sessions from the same intermediate state (GRPO groups), ideally sharing the prefix KV.
9. **Many concurrent sessions + continuous batching**: batch whichever sessions are ready for inference, without requiring all sessions to be in sync.

## C. Both kinds of model must be supported

| Model type | Examples | Output per step | How the env handles it |
|---|---|---|---|
| Frame-synchronous full-duplex models | Moshi, PersonaPlex, CharDuplex (dual-stream) | the output for this frame (silence, or a short piece of speech / text) | **concatenates consecutive output chunks into one growing Segment** (re-emitting the same id); stopping output is equivalent to finishing or yielding |
| Event-triggered "realtime" models | triggered by VAD, generate a whole reply at once | a whole reply (generated faster than real time) | plays it as one Segment by duration; truncates it on interruption |

> Both kinds are supported: the env accepts either a whole `Segment` or per-step `Chunk`s as actions; observations are always `Chunk`s cut at `AgentSpec.chunk_ms` (containing only what has been played).

## D. Validation experiment (run before connecting any server)

With a fixed random seed and greedy decoding, send the same user audio chunk by chunk twice: once continuously, and once with artificial 2-second wall-clock pauses between some chunks. **The two outputs (content and durations) must be identical**; otherwise the server has wall-clock logic and violates A.1.

Something to check in particular: vLLM-Omni's MiniCPM-o full-duplex runtime has "playback-aware session state", which may carry a wall-clock assumption and needs to be confirmed in the code.

## E. Protocols and adapters

The agent side has one adapter per server protocol (`interaction_gym.agents`). An adapter sends the env's microphone audio for each step to the server and turns the speech it receives into `Chunk`s for the env. The first one targets vLLM-Omni's full-duplex protocol: `agents.vllm_omni.VllmOmniDuplexAgent`.

### vLLM-Omni full-duplex protocol (a full-duplex dialect of OpenAI Realtime)

- Endpoint `ws://HOST:PORT/v1/realtime?duplex=1`. The client sends one `session.update` first, then streams `input_audio_buffer.append` continuously (16 kHz PCM16; the microphone never stops, so silence and background sound are sent too).
- No VAD and no `response.create` (`turn_detection: null`): in each unit (1 s of input for MiniCPM-o) the model decides by itself whether to listen or speak.
- Output is `response.output_audio.delta` (24 kHz PCM16) with paired `response.output_audio_transcript.delta`, followed by `response.done`.
- Model-specific session settings are in [MiniCPM-o 4.5 protocol notes](#minicpm-o-45-protocol-notes) and the compatibility table below.

### Two clocks

| `clock` | How time is measured | How it advances | Use |
|---|---|---|---|
| `realtime` | wall clock | the adapter waits at least `chunk_ms` of wall time per step, like a real microphone | works with any vLLM-Omni duplex server; latency includes the model's real compute time; not fully reproducible |
| `input` (lockstep) | amount of input audio: feeding 1 s of audio means 1 s has passed | after sending each step's audio, wait for the server's acknowledgement of that append, then advance the env | training and reproducible evaluation; speed depends only on how fast the model computes: no input means model time stands still, and continuous input can run faster than real time |

The `input` clock is an extension added at vLLM-Omni's model-agnostic duplex engine layer, so it applies to all duplex models; it requires the patches described in [section F](#f-vllm-omni-server-features-and-patches). The adapter disables silence continuation (`extra_body.silence_continuation = false`) under both clocks: the env's microphone always carries background sound, and the server must never insert pure silence on its own.

### Output: talker audio vs Thinker text only

- `audio_out=False` — **the default of the repository's MiniCPM-o runners** (benchmarks, examples, GPU tuner; since
  2026-10-09, as in our RL rollouts): requests only text (`modalities: ["text"]`) and estimates each
  utterance's duration from the agent's speaking rate (`speech_cps`, characters per second).
  - Served by a Thinker-only MiniCPM-o server (one GPU, 16 sessions; the reference deployment's default
    `AGENT_LAYOUT=thinker`, `configs/minicpmo_4_5_thinker_1gpu.yaml`) built with
    `examples/serving/reference/patches/minicpmo_thinker_only.patch`: a text-only session ends at Stage 0, so no
    Talker or Code2Wav runs, and the transcript of each unit is cut from the Thinker's tokens exactly as the Talker
    hand-off would cut it. With the patch, the two-GPU audio deployment also serves text-only sessions at Stage 0.
  - Calibrate the rate with `interaction_gym.agents.speech_rate(episodes)` on episodes of the same model in audio mode; MiniCPM-o 4.5 under lockstep measured about 11.3 characters per second.
  - The estimation error is a few tenths of a second per utterance. When the agent starts speaking is still decided by the model at unit boundaries, and the user simulator responds and interrupts as usual, so the interaction is unaffected; only duplex metrics that depend on exact timing (e.g. overlap duration, yield latency) are biased.
  - Caveats: the Thinker-only deployment is not bit-identical to the two-GPU audio deployment (same inputs, but
    bf16-level logprob differences from the first speak unit, mean |diff| 0.013 / max 0.14 over the first 24
    tokens, so sampled trajectories diverge); timing is estimated, not taken from real audio; and results are not
    directly comparable with full-audio runs. docs/BENCHMARKS.md ("Agent output") lists which benchmark metrics
    would read agent audio and what they use instead.
- `audio_out=True` — the class default, and the runners' `--audio-out`: uses the real speech generated by the talker; the timeline is exact and can be recorded and replayed. Needs the full deployment (`AGENT_LAYOUT=audio`: Thinker on one GPU, Talker + Code2Wav on a second). Use it for recordings, listening and demos, and for any number that must come from the agent's real speech.
- Why the class default stays `True`: `VllmOmniDuplexAgent` is a generic client of any vLLM-Omni duplex model, and
  the others (Nemotron VoiceChat, AURA, ...) have no text-only path; `speech_cps` defaults to MiniCPM-o's rate. The
  MiniCPM-o runners pass `audio_out=False` explicitly. `describe()` records the mode as `meta.agent.output`
  (`"audio"` or `"text @ 11.3 chars/s"`), and `check_output_mode` lets a resumable runner refuse to mix the two in
  one output file.
- Without `minicpmo_thinker_only.patch` (stock or PR-branch vLLM-Omni), a session with `modalities: ["text"]` still passes the Thinker's output to the Talker and still generates audio (the adapter discards it): text-only output then saves no Talker / Code2Wav compute, and a Thinker-only pipeline does not exist.

### Compatibility of duplex models on vLLM-Omni

| Model | Unit | Usage notes | Status |
|---|---|---|---|
| MiniCPM-o 4.5 | 1 s, the model decides to listen or speak | needs `ref_audio`; default configuration otherwise | fully usable: exact lockstep, token trace available |
| Nemotron VoiceChat 11B | 80 ms frames | `AgentSpec.chunk_ms` must be a multiple of 80; output is 22.05 kHz (the adapter resamples automatically); it keeps emitting very quiet audio within a reply, so `split_silence_ms` is needed to split utterances; does not stop when interrupted | usable; with the patch for vLLM-Omni 0.30 the lockstep acknowledgement arrives one frame early (a fix is on the patch branch, not yet verified on GPU) |
| AURA | answers a whole turn after the user commits | requires client-side commit (`commit_after_silence_ms`; the server VAD depends on Silero, which was not installed in our build); every append must carry a video frame (`video_frame`); text arrives before audio | runs end to end, but has server-side problems: results are not reproducible, the TTS language is misconfigured, and later replies repeat text from earlier ones |
| PersonaPlex 7B | 80 ms frames | — | cannot yet run in duplex mode in vLLM-Omni 0.30 |
| Qwen3-Omni 30B | whole turn (commit) | — | needs about 70 GB in bf16 (at least four 32 GB GPUs); not tested |

## F. vLLM-Omni server features and patches

Three server features used by `VllmOmniDuplexAgent` require our vLLM-Omni patches:

- input-clocked lockstep: `session.extra_body.clock = "input"`;
- disabling silence continuation: `extra_body.silence_continuation = false`;
- the per-unit token trace: `session.extra_body.trace_tokens = true`, which also needs `duplex_session.enable_debug_events: true` in the deploy YAML.

For MiniCPM-o 4.5 the reference deployment adds two patches on top (`examples/serving/reference/patches/`, not
upstream): `minicpmo_fe_per_session.patch` (one audio feature extractor per session) and
`minicpmo_thinker_only.patch` (text-only sessions end at the Thinker; the `minicpmo_4_5_thinker` pipeline for the
one-GPU Thinker-only deployment, the default for evaluation).

The patches are tracked upstream in PR [vllm-project/vllm-omni#8485](https://github.com/vllm-project/vllm-omni/pull/8485) and are available on the fork [williamLyh/vllm-omni](https://github.com/williamLyh/vllm-omni), branch `duplex-input-clock` (the adapter's input-clocked mode, including the resend of refused inputs, depends on that PR). Realtime mode (`clock = "realtime"`) works on stock vLLM-Omni. The rest of this section describes a vLLM-Omni build with the patches; a reference deployment is in [examples/serving/reference/](../examples/serving/reference/).

The patch adds generic engine code (an input clock plus changes to the duplex session runner, config, manager and events) and one optional plugin hook. Sessions that do not opt in behave exactly as on stock vLLM-Omni.

### Lockstep (input-clocked) duplex sessions

Opt in per session in the first `session.update`: `session.extra_body.clock = "input"`. The contract below is that of PR #8485 as revised after review (the authoritative text is `docs/serving/realtime_duplex_api.md` → *Input-clocked sessions* on that branch). Then:

- The server never invents input: there is no wall-clock silence continuation (this implies `silence_continuation: false`). If the client sends nothing, model time does not advance and nothing is emitted. Idle handling is that of any duplex session (`idle_timeout_s` and the lease's `idle_ttl_s`, 300 s by default): a simulated user that pauses longer is released like any idle client.
- Every `input_audio_buffer.append`, `input_audio_buffer.commit` and `response.create` gets exactly one acknowledgement, in input order:

  ```
  {"type": "input_audio_buffer.processed", "event_id": "...",
   "audio_end_ms": 3200,      # total input audio received so far
   "unit_end_ms": 3000,       # input covered by model units settled so far
   "input_index": 16,         # 1-based index of the input this acknowledges
   "trigger": "input_audio_buffer.append",
   "units": [{"end_ms": 3000, "decision": "speak"}]}   # units settled since the previous ack
  ```

  It is sent only after everything that input caused has been sent: for each unit it completed, the listen/speak decision was taken and, if the model spoke, all of that unit's text and audio deltas (and `response.done` if the response ended there) went out first. An append that only buffers a partial unit is acknowledged once the acknowledgements before it are out, with `units: []`. `units[].decision` is the model's decision (`listen`, `speak`), or `dropped`, `cancelled`, `aborted` or `timed_out` (with a `reason`) for a unit that will produce no further output.
- **Refused inputs.** An input the session refuses before any of it reaches the model is still acknowledged in input order, with `"decision": "rejected"` and `"reason"` = the error code (`input_backpressure`, `invalid_input_modality`, or for an append whose audio the engine cannot convert `bad_audio` / `bad_event`) at the top level of the acknowledgement; its `error` (with the input's `event_id`) comes first. An input rejected before it reaches the session (transport layer: `engine_backpressure`, `bad_audio`, `bad_event`, `unsupported_audio_format`, `event_too_large`, ...) gets only an `error` carrying the input's `event_id`, and no `input_index`. Since `bad_audio` / `bad_event` can come from either layer, the adapter waits `ack_grace_s` (1 s) after such an error for a possible acknowledgement.
- Client loop (what the adapter does): send one input with a fresh `event_id`, wait for its acknowledgement, repeat; nothing later is sent before the input in flight is acknowledged. A refusal a resend can fix (`input_backpressure`, `engine_backpressure`) is resent after a bounded exponential back-off (`retry_backoff_s` = 0.05 s doubling up to `retry_backoff_max_s` = 1 s), so the model never misses audio. After `max_input_retries` (default 5) resends, or at once for a refusal a resend cannot fix, the input is given up: it is recorded in `agent.input_stats["dropped_inputs"]` (`t`, `type`, `reason`, `attempts`; also `input_retries` and `input_errors`, and in the agent trace) and the step raises `InputDroppedError`, or, with `on_input_dropped="mark"`, the episode goes on and `agent.failed` says why it is invalid. Put `agent.input_stats` into the episode `meta` when you keep it. On a server build without the revised PR (e.g. the earlier prototype patch) refused inputs are never acknowledged, so such a refusal ends in the adapter's `ack_timeout_s` error instead.
- Keep each append at most one model unit long (MiniCPM-o: 1 s; 200 ms works well). The MiniCPM-o PCM buffer submits at most one unit per append, so larger appends build a backlog (`unit_end_ms` lags behind `audio_end_ms`).
- Timeouts: if units are open and the model pipeline produces nothing for `extra_body.input_clock_unit_timeout_s` (default 15 s), the oldest open unit is settled as `timed_out` (`no_progress`); a unit older than `input_clock_unit_max_s` (default 60 s) is settled too (`max_age`). They only release acknowledgements; the session's resources are handled as for any session.
- Silence continuation can also be disabled on its own, without the input clock: `extra_body.silence_continuation = false`.

Measured with MiniCPM-o 4.5 (two questions in 32.6 s of input): pauses of 12 s and 330 s in the middle of an answer produced no events, `unit_end_ms` stayed frozen, and the answer resumed with input; the 32.6 s of input was processed in about 7-8 s of wall time (about 4x real time) with both answers correct; the event sequence (unit decisions, transcript, audio delta durations, input position) was identical across two fast runs and a run with random 0-2 s sleeps between appends.

**How unit completion is tracked** (relevant when adding other models): one Stage-0 submission is one unit; each Stage-0 segment end decides the oldest undecided unit (consumed by a plugin decision = listen, done; forwarded = speak). A speaking unit is done when the plugin hook `DuplexModelPlugin.unit_output_complete(...)` says so. By default the final stage marks one segment end per unit. MiniCPM-o overrides the hook: Code2Wav does not mark unit segments, but its output echoes the connector's per-unit flush flag `tts_is_last_chunk`.

### Per-unit token trace (debug)

Opt in per session with `session.extra_body.trace_tokens = true` (with or without `clock: "input"`; nothing is tracked when it is off). The deploy YAML must set `duplex_session.enable_debug_events: true`. For every model unit, when the unit completes (in lockstep: before the `input_audio_buffer.processed` that covers it), the server emits:

```
{"type": "debug.unit_tokens", "event_id": "...", "session_id": "...", "unit_index": 3, "end_ms": 4000,
 "decision": "listen" | "speak",
 "stages": [{"stage": "thinker", "input": [[id, piece], ...], "output": [[id, piece], ...]},
            {"stage": "talker",  "input": [...], "output": [[codec_id, "s3:<id>"], ...]}],   # talker only if it spoke
 "special_ids": [...]}       # tokenizer special/added ids among the above
```

plus one event with `unit_index: -1, decision: "prompt"` carrying the initial context. `piece` is `tokenizer.convert_ids_to_tokens` output (raw pieces, e.g. `Ġthe`). The adapter collects these events into the agent trace format described in [AGENT_TRACE.md](AGENT_TRACE.md).

MiniCPM-o 4.5 layout as observed:

```
prompt  thinker.in  <|im_start|> system Ċ Streaming ĠOmni ĠConversation .Ċ <|audio_start|> <unit>x168 <|audio_end|> <|im_end|>
unit 0  thinker.in  <unit> <|audio|>x10                             out: <|listen|>
unit 1  thinker.in  <|listen|> </unit> <unit> <|audio|>x10          out: <|listen|>
unit 3  thinker.in  <|chunk_eos|> </unit> <unit> <|audio|>x10       out: <|speak|> ĠFrance Ġis <|chunk_eos|>
        talker.in   ĠFrance Ġis <audio_bos>                         out: s3:x25 s3:6561   (26 codes/unit)
```

Notes: the previous unit's terminator (`<|listen|>` / `<|chunk_eos|>`) is re-fed at the start of each unit; the reference-audio embeddings sit on `<unit>` placeholder ids (168 = 16.8 s at 10 per second); the `<|audio|>` ids are placeholders for the 10 audio embeddings of the 1 s unit. Talker input ids are summed with projected Thinker hidden states (the ids shown are the text ids); `<audio_bos>` (151687) is labelled explicitly because the Thinker tokenizer calls that id `<focus>`.

### MiniCPM-o 4.5 protocol notes

- First message `session.update`, which needs:
  - `ref_audio` as a data URL (`data:audio/wav;base64,...`), e.g. the `assets/system_ref_audio.wav` shipped with the model;
  - `turn_detection: null`;
  - `extra_body.auto_response: true`.
- Then stream `input_audio_buffer.append` (PCM16, 16 kHz, any chunk size; the model decides listen/speak once per 1 s unit).
- Output: `response.output_audio.delta` (PCM16, 24 kHz, about 1 s each) and `response.output_audio_transcript.delta`.
- Audio deployment: the first duplex session after the server starts can come back with listen decisions only and no audio while the Talker and Code2Wav finish lazy initialization. Send one throw-away warm-up session before real episodes.
- Thinker-only deployment: ask for text only (`modalities: ["text"]`); an audio session gets no output there. `ref_audio` is still sent: its embeddings are part of the Thinker's prompt, so leaving it out would change what the model sees.
- Use a vLLM-Omni build that includes upstream PR [vllm-project/vllm-omni#8227](https://github.com/vllm-project/vllm-omni/pull/8227): with it, a unit `<|speak|> ... <|turn_eos|> <|listen|>` goes to the Talker instead of being taken as a listen decision. Without it (e.g. vLLM-Omni 0.30.0), that unit's text and turn end are lost and the response stays open.
