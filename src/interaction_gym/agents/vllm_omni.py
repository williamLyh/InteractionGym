"""Adapter for full-duplex models served by vLLM-Omni (e.g. MiniCPM-o 4.5).

The server (``vllm serve <model> --omni`` with a duplex deploy config) speaks a full-duplex
dialect of the OpenAI Realtime API on ``ws://HOST:PORT/v1/realtime?duplex=1``:

- client → server: ``session.update`` once, then ``input_audio_buffer.append`` (16 kHz PCM16)
  continuously — the microphone never stops, silence included;
- the model decides by itself, once per unit of input (1 s for MiniCPM-o), whether to listen or
  speak (``turn_detection`` is off: no VAD, no ``response.create``);
- server → client: ``response.output_audio.delta`` (24 kHz PCM16, ~1 s each) paired with
  ``response.output_audio_transcript.delta``, then ``response.done``.

Each env step the adapter sends the microphone audio of the window that just played, queues
the speech that arrived (a client's playback buffer) and plays one step of it as a ``Chunk`` —
so the env sees the agent frame by frame and nothing is scheduled ahead. Like a client's jitter
buffer, a reply starts playing once ``prebuffer_ms`` of it has arrived (or it is complete), and
if playback runs dry while the response is still going, the gap is filled with silence — one
response is one utterance. The model handles barge-in itself by stopping; a response that ends
early (``response.done`` not ``completed``) drops whatever has not played yet.

Two clocks:

- ``clock="input"`` (lockstep): time is the input audio. Each step sends its audio and waits for
  the server's ``input_audio_buffer.processed`` acknowledgement of that append — sent only after
  every output the input caused — so the agent's behaviour depends on what it heard, never on
  wall time: steps run as fast as the model computes (faster or slower than real time) and runs
  are reproducible. Needs a server with the lockstep extension (session ``extra_body.clock:
  "input"``; see docs/agent_server.md).
  Inputs go one at a time: each append (and commit) is sent only after the previous one was
  acknowledged. An input the server refuses is resent, so the model never misses audio: an
  acknowledgement with ``decision: "rejected"`` (e.g. ``reason: "input_backpressure"``), or an
  ``error`` the transport answers without acknowledging the input (e.g. ``engine_backpressure``).
  A resend waits a short, bounded, exponentially growing delay and nothing later is sent before it
  is acknowledged. After ``max_input_retries`` resends (or at once, for a refusal a resend cannot
  fix, e.g. undecodable audio) the input is given up: it is recorded in ``input_stats`` and, by
  default, the step raises ``InputDroppedError``; ``on_input_dropped="mark"`` continues the episode
  and marks it failed (``failed``) instead.
- ``clock="realtime"``: a step takes at least ``chunk_ms`` of wall time, so the server sees a live
  microphone. Works with any vLLM-Omni duplex server; latency includes the model's real compute
  time and runs are not exactly reproducible.

The adapter opts out of the server's wall-clock "silence continuation" (it inserts 1 s of
digital silence when input pauses mid-response): the env's microphone never stops and carries
background noise, so the model must only hear what the env sent.

Output: ``audio_out=True`` (the class default) plays the talker's speech (exact timing, recordable audio).
``audio_out=False`` asks only for text and times it with a speaking rate, ``speech_cps`` characters per
second — calibrate it per model from audio episodes with ``interaction_gym.agents.speech_rate``.
The estimate is off by a fraction of a second per utterance, which shifts when the user hears each part
but not what happens. With a vLLM-Omni that ends text-only sessions at the Thinker (the reference
deployment's patched vLLM-Omni, ``patches/vllm-omni/``), a text-only MiniCPM-o session runs no Talker or
Code2Wav, and a Thinker-only server (one GPU) serves it; this is how the repository's MiniCPM-o runners
evaluate by default (``--audio-out`` for the talker's speech; docs/agent_server.md).
The class default stays ``True``: it is a generic client of any vLLM-Omni duplex model, most of which have no
text-only path, and ``speech_cps`` defaults to MiniCPM-o's rate.

Tools are not passed: the duplex server does not take them. The session's instructions are.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import warnings
from pathlib import Path

import websockets

from ..audio import Audio
from ..core import SESSION, AgentSpec, Chunk, Frame, Session

IN_SR = 16000
#: Rejections a resend can fix (lockstep): an acknowledgement ``{"decision": "rejected", "reason": ...}``
#: or a transport ``error`` code (the server acknowledges no input it answered that way).
RETRYABLE = frozenset({"input_backpressure", "engine_backpressure"})
#: ``error`` codes with which the server refuses an input before the session sees it: no acknowledgement follows.
TRANSPORT_ERRORS = frozenset({"engine_backpressure", "bad_audio", "bad_event", "invalid_json", "event_too_large",
                              "unsupported_audio_format"})
#: ... except that the session itself may also refuse an append with these, and then acknowledges it (``rejected``):
#: after such an error the adapter waits ``ack_grace_s`` for that acknowledgement before deciding none follows.
EITHER_LAYER = frozenset({"bad_audio", "bad_event"})


class InputDroppedError(RuntimeError):
    """Lockstep: the server kept refusing an input, so the model did not hear it."""
MINICPMO_CPS = 11.3  # MiniCPM-o 4.5 speaking rate, characters/s (speech_rate over 7 lockstep audio episodes, 2026-10-02)


def output_label(audio_out: bool, speech_cps: float = MINICPMO_CPS) -> str:
    """``meta.agent.output``: ``"audio"`` (the talker's speech) or ``"text @ <cps> chars/s"`` (text timed by estimate)."""
    return "audio" if audio_out else f"text @ {speech_cps:g} chars/s"


def check_output_mode(path: str | Path, audio_out: bool) -> None:
    """For resumable runners that append to an ``episodes.jsonl``: refuse to add episodes of one agent output mode
    to a file that holds the other (text-only timing is estimated, so the two are not comparable). Reads the first
    episode's ``meta.agent`` (``output`` or ``audio_out``); a file without either is accepted."""
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        line = next((x for x in f if x.strip()), None)
    if line is None:
        return
    agent = json.loads(line).get("meta", {}).get("agent") or {}
    if isinstance(agent.get("output"), str):
        was = agent["output"] == "audio"
    elif "audio_out" in agent:
        was = bool(agent["audio_out"])
    else:
        return
    if was != audio_out:
        raise ValueError(f"{p} holds {'audio-output' if was else 'text-only'} agent episodes; this run is "
                         f"{'audio-output' if audio_out else 'text-only'}: use another --out / --tag "
                         f"({'drop' if not audio_out else 'add'} --audio-out to resume that run)")


class _Timed:
    """Text mode's stand-in for audio: only a length (in samples at the output rate), sliceable and
    joinable like ``Audio``, so playback buffering works the same for both outputs."""

    def __init__(self, n: int, sr: int):
        self.n, self.sr = n, sr

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, s: slice) -> _Timed:
        return _Timed(len(range(self.n)[s]), self.sr)

    def __add__(self, other: _Timed) -> _Timed:
        return _Timed(self.n + other.n, self.sr)

    @property
    def dur_ms(self) -> int:
        return round(self.n * 1000 / self.sr)


class VllmOmniDuplexAgent:
    """One duplex session per episode. ``spec.audio`` must be set (the agent hears the mixed
    microphone); ``spec.sr`` is the env's audio rate (input is resampled to ``in_sr``, output is
    resampled from whatever rate each packet says to ``out_sr``, which must equal ``spec.sr``).

    Model-specific options (all off by default; the defaults suit MiniCPM-o):
    ``ref_audio`` / ``voice`` (the model's own voice), ``in_sr`` (model input rate),
    ``turn_detection`` (e.g. ``{"type": "server_vad"}`` for turn-commit models),
    ``commit_after_silence_ms`` + ``vad_rms`` (client-side turn detection: commit the user's turn
    when the microphone goes quiet after speech — for turn-commit models without a server VAD),
    ``video_frame`` (a base64 JPEG sent with every append, for models that require video),
    ``split_silence_ms`` + ``silence_rms`` (frame-synchronous models that keep emitting near-silent
    audio within a response: end the utterance after that much quiet output, start a new one when
    speech resumes)."""

    def __init__(
        self,
        spec: AgentSpec,
        url: str = "ws://127.0.0.1:8010/v1/realtime?duplex=1",
        *,
        model: str = "openbmb/MiniCPM-o-4_5",
        ref_audio: str | Path | None = None,  # the model's own voice (MiniCPM-o needs it for audio output)
        out_sr: int = 24000,
        session: dict | None = None,  # extra session.update fields (e.g. temperature)
        clock: str = "realtime",  # "realtime" | "input" (lockstep)
        # default: 1000 for realtime (network jitter); 0 for lockstep, which follows the model's own timeline —
        # if its first unit yields only a sliver of audio (e.g. 80 ms), that blip and the gap after it are kept
        prebuffer_ms: int | None = None,
        ack_timeout_s: float = 60.0,  # lockstep: no acknowledgement this long means the server is stuck
        trace_tokens: bool = False,  # record the token sequences the model consumes / produces per unit (lockstep only)
        audio_out: bool = True,  # False: text only, timed at speech_cps
        speech_cps: float = MINICPMO_CPS,
        in_sr: int = IN_SR,
        voice: str | None = None,
        turn_detection: dict | None = None,
        commit_after_silence_ms: int | None = None,
        vad_rms: float = 500.0,  # int16 RMS above which the microphone counts as speech
        video_frame: str | None = None,
        split_silence_ms: int | None = None,
        silence_rms: float = 300.0,  # int16 RMS below which output audio counts as silence
        max_input_retries: int = 5,  # lockstep: resends of a refused input before it is given up
        retry_backoff_s: float = 0.05,  # first resend delay; doubles per resend ...
        retry_backoff_max_s: float = 1.0,  # ... up to this
        on_input_dropped: str = "raise",  # "raise" (InputDroppedError) | "mark" (record, set ``failed``, go on)
        ack_grace_s: float = 1.0,  # lockstep: after a bad_audio / bad_event error, how long an acknowledgement may follow
        name: str | None = None,  # how meta.agent names it (default: from the model id)
    ):
        assert clock in ("realtime", "input"), clock
        assert on_input_dropped in ("raise", "mark"), on_input_dropped
        assert max_input_retries >= 0 and 0 <= retry_backoff_s <= retry_backoff_max_s
        assert spec.audio, "the duplex agent listens to the microphone: set AgentSpec(audio=...)"
        assert not audio_out or spec.sr == out_sr, "the env's audio rate must match the server's output rate"
        self.spec, self.url, self.model, self.out_sr = spec, url, model, out_sr
        self.name = name
        self.ref_audio = Path(ref_audio) if ref_audio else None
        self.audio_out, self.speech_cps = audio_out, speech_cps
        self.extra = session or {}
        self.in_sr, self.voice, self.turn_detection, self.video_frame = in_sr, voice, turn_detection, video_frame
        self.commit_after_silence_ms, self.vad_rms = commit_after_silence_ms, vad_rms
        self._heard_speech, self._quiet_in_ms, self.commits = False, 0, 0
        self.split_silence_ms, self.silence_rms = split_silence_ms, silence_rms
        self._quiet_out_ms, self._split = 0, False  # output silence so far; between two utterances of one response
        self._early_text: dict[str, str] = {}  # text that arrived ahead of its response's audio
        self.clock = clock
        self.prebuffer_ms = prebuffer_ms if prebuffer_ms is not None else (0 if clock == "input" else 1000)
        self.ack_timeout_s = ack_timeout_s
        self.max_input_retries, self.on_input_dropped = max_input_retries, on_input_dropped
        self.retry_backoff_s, self.retry_backoff_max_s = retry_backoff_s, retry_backoff_max_s
        self.ack_grace_s = ack_grace_s
        # lockstep bookkeeping: resends, inputs given up ({t, type, reason, attempts}), errors the server answered
        # inputs with while still acknowledging them (e.g. a commit with nothing to commit)
        self.input_stats: dict = {"input_retries": 0, "dropped_inputs": [], "input_errors": []}
        self.failed: str | None = None  # why the episode failed (on_input_dropped="mark")
        self._seq = 0  # inputs sent, for their event ids
        self._inflight: str | None = None  # event_id of the input waiting for its acknowledgement
        self._ack: dict | None = None  # its acknowledgement
        self._err: dict | None = None  # an error the server answered it with
        self.appends = self.acks = 0  # lockstep: one processed acknowledgement per append
        self.unit_ms: int | None = None  # the model's own chunk length, as the server reports it
        # The server's token trace relies on the input clock's unit tracking: it refuses a traced session without
        # clock="input" (token_trace_requires_input_clock), so a realtime session is not traced.
        self.trace_tokens = trace_tokens and clock == "input"
        if trace_tokens and clock != "input":
            warnings.warn("VllmOmniDuplexAgent: trace_tokens needs clock='input'; this realtime session is not traced",
                          stacklevel=2)
        self.units: list[dict] = []  # debug.unit_tokens events, if traced
        self._created = asyncio.Event()
        self._acked = asyncio.Event()
        self.finished: set[str] = set()  # responses the server has ended
        self.ws = None
        self.inbox: list[dict] = []  # server events received since the last step
        self.events: dict[str, int] = {}  # counts, for debugging
        # received, not yet played: [audio left, its whole text, response id, total samples, samples played, chars released];
        # the text plays along with its audio, in proportion (as the env times any segment)
        self.queue: list[list] = []
        self._text_source: str | None = None  # text mode: the one event type the text is taken from
        self.utt: str | None = None  # id of the utterance being played
        self.utt_resp: str | None = None
        self.utt_end = -1  # where it has got to; audio continues it only from exactly there
        self.next_wall = 0.0

    async def _connect(self, instructions: str) -> None:
        self.ws = await websockets.connect(self.url, max_size=None)
        self.reader = asyncio.create_task(self._read())
        session = {
            "model": self.model,
            "modalities": ["audio", "text"] if self.audio_out else ["text"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "audio": {"input": {"sample_rate_hz": self.in_sr}, "output": {"sample_rate_hz": self.out_sr}},
            "turn_detection": self.turn_detection,
            **({"voice": self.voice} if self.voice else {}),
            **({"ref_audio": "data:audio/wav;base64," + base64.b64encode(self.ref_audio.read_bytes()).decode()} if self.ref_audio else {}),
            # silence_continuation: off — the env streams the microphone continuously (background
            # noise included), so the server must not fill gaps with silence of its own on a wall
            # clock (needs the server patch; an unpatched server ignores the field)
            "extra_body": {**({"auto_response": True, "force_listen_count": 0} if self.turn_detection is None else {}),
                           "silence_continuation": False,
                           **({"clock": "input"} if self.clock == "input" else {}),
                           **({"trace_tokens": True} if self.trace_tokens else {})},
            **({"instructions": instructions} if instructions else {}),
            **self.extra,
        }
        await self.ws.send(json.dumps({"type": "session.update", "session": session}))
        # The server answers with its capabilities, or refuses the session (e.g. an ``error`` with
        # ``duplex_session_capacity_exhausted``): fail at once on a refusal instead of waiting out the timeout.
        created = asyncio.ensure_future(self._created.wait())
        await asyncio.wait({created, self.reader}, timeout=30, return_when=asyncio.FIRST_COMPLETED)
        if not created.done():
            created.cancel()
            if self.reader.done():
                self.reader.result()  # raises the server's error
                raise RuntimeError("vLLM-Omni closed the connection before creating the session")
            raise TimeoutError("vLLM-Omni did not create the session within 30 s")
        if self.reader.done():
            self.reader.result()
        self._check_alignment()
        self.next_wall = time.monotonic()

    async def _read(self) -> None:
        async for raw in self.ws:
            ev = json.loads(raw)
            t = ev.get("type", "")
            self.events[t] = self.events.get(t, 0) + 1
            if t == "error":
                err = ev.get("error") if isinstance(ev.get("error"), dict) else {}
                if self.clock == "input" and self._inflight is not None and err.get("event_id") == self._inflight:
                    self._err = err  # about the input in flight: settled by its acknowledgement or a resend
                    self._acked.set()
                    continue
                raise RuntimeError(f"vLLM-Omni error: {ev}")
            if t == "session.created":
                unit = ev.get("session", {}).get("capabilities", {}).get("chunk_period_ms")
                self.unit_ms = int(unit) if unit else None
                self._created.set()
            if t in ("response.output_audio.delta", "response.output_audio_transcript.delta", "response.output_text.delta", "response.done"):
                self.inbox.append(ev)
            elif t == "debug.unit_tokens":
                self.units.append({k: v for k, v in ev.items() if k not in ("type", "session_id", "event_id")})
            elif t == "input_audio_buffer.processed":
                if (ev.get("trigger", "input_audio_buffer.append") == "input_audio_buffer.append"  # commits are acked too
                        and ev.get("decision") != "rejected"):  # a refused append is resent
                    self.acks += 1
                self._ack = ev
                self._acked.set()

    @property
    def busy(self) -> bool:
        """Speech received and not yet played, or a response the server has not ended (an episode loop waits for it
        before ending a run on the env's ``done``)."""
        return bool(self.queue) or (self.utt_resp is not None and self.utt_resp not in self.finished)

    @property
    def output(self) -> str:
        """``"audio"`` or ``"text @ <speech_cps> chars/s"`` (``output_label``)."""
        return output_label(self.audio_out, self.speech_cps)

    def describe(self) -> dict:
        """``meta.agent``: which model, served how."""
        name = self.name or ("MiniCPM-o 4.5" if "minicpm-o-4_5" in self.model.lower() else self.model.rsplit("/", 1)[-1])
        out = {"name": name, "model": self.model, "kind": "full-duplex", "server": f"vllm-omni duplex {self.url}",
               "clock": self.clock, "audio_out": self.audio_out, "output": self.output}
        if self.extra.get("temperature") is not None:
            out["temperature"] = self.extra["temperature"]
        return out

    def trace(self, episode_id: str) -> dict | None:
        """What the server's model saw and produced, unit by unit — one record of the optional
        ``agent_traces.jsonl`` that goes next to the trajectories (docs/AGENT_TRACE.md)."""
        if not self.trace_tokens:
            return None
        return {"episode_id": episode_id, "server": "vllm-omni duplex", "model": self.model, "unit_ms": self.unit_ms,
                "clock": self.clock, "units": self.units, **({"input_stats": self.input_stats} if self.clock == "input" else {})}

    def _check_alignment(self) -> None:
        """The model decides once per unit of input; the env steps by ``chunk_ms``. A unit must be a
        whole number of steps, so unit boundaries fall on step boundaries (and no append carries more
        than one unit, which the server would only work through one unit per append)."""
        step, unit = self.spec.chunk_ms, self.unit_ms
        if unit is None:  # this server does not report its unit: nothing to check
            return
        if unit % step:
            ok = [d for d in range(1, unit + 1) if unit % d == 0 and d % 10 == 0]
            raise ValueError(f"AgentSpec.chunk_ms={step} does not divide the model's unit of {unit} ms "
                             f"(session.capabilities.chunk_period_ms); use a divisor such as {ok[-4:]}")

    async def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        if self.ws is None:
            s = next((f.data for f in obs if f.stream == SESSION and isinstance(f.data, Session)), None)
            await self._connect(s.instructions if s else "")
        for f in obs:
            if f.stream == self.spec.audio:
                pcm = f.data.resample(self.in_sr).samples.tobytes()
                await self._send_input(t, {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode(),
                                           "format": "pcm16", "sample_rate_hz": self.in_sr,
                                           **({"video_frames": [self.video_frame]} if self.video_frame else {})})
                self.appends += 1
                if self.commit_after_silence_ms is not None and await self._end_of_user_turn(f.data, f.dur):
                    await self._send_input(t, {"type": "input_audio_buffer.commit"})
                    self.commits += 1
        if self.clock != "input":  # a live microphone: one step of audio per step of wall time
            self.next_wall = max(self.next_wall + self.spec.chunk_ms / 1000, time.monotonic())
            await asyncio.sleep(self.next_wall - time.monotonic())
        if self.reader.done():
            self.reader.result()
        events, self.inbox = self.inbox, []
        for ev in events:
            self._receive(ev)
        return self._play(t)

    async def _send_input(self, t: int, event: dict) -> None:
        """Send one client input. Lockstep: wait for its acknowledgement, resending it while the server refuses it."""
        if self.clock != "input":
            await self.ws.send(json.dumps(event))
            return
        self._seq += 1
        for attempt in range(self.max_input_retries + 1):
            if attempt:
                self.input_stats["input_retries"] += 1
                await asyncio.sleep(self.backoff_s(attempt))
            refusal = await self._send_and_wait(dict(event, event_id=f"in{self._seq}.{attempt}"))
            if refusal is None:
                return
            if refusal not in RETRYABLE:
                break
        self._give_up(t, event["type"], refusal, attempt + 1)

    def backoff_s(self, attempt: int) -> float:
        """Delay before resend ``attempt`` (1-based): exponential from ``retry_backoff_s``, capped."""
        return min(self.retry_backoff_max_s, self.retry_backoff_s * 2 ** (attempt - 1))

    async def _send_and_wait(self, event: dict) -> str | None:
        """One attempt: ``None`` once acknowledged, else the refusal (the rejection reason or the error code)."""
        self._inflight, self._ack, self._err = event["event_id"], None, None
        self._acked.clear()
        await self.ws.send(json.dumps(event))
        grace_over = False
        try:
            while True:
                if self._ack is not None:
                    if self._ack.get("decision") == "rejected":
                        return str(self._ack.get("reason") or "rejected")
                    if self._err is not None:  # refused while handled (e.g. nothing to commit): it caused nothing
                        self.input_stats["input_errors"].append({"type": event["type"], "code": self._err.get("code")})
                    return None
                code = self._err.get("code") if self._err is not None else None
                if code in TRANSPORT_ERRORS and (code not in EITHER_LAYER or grace_over):
                    return str(code)  # the session never got it: no acknowledgement follows
                if self.reader.done():
                    self.reader.result()
                    raise RuntimeError("vLLM-Omni closed the session")
                self._acked.clear()
                try:
                    await asyncio.wait_for(self._acked.wait(), self.ack_grace_s if code in EITHER_LAYER else self.ack_timeout_s)
                except TimeoutError:
                    if code in EITHER_LAYER:
                        grace_over = True
                        continue
                    raise RuntimeError(f"vLLM-Omni sent no input_audio_buffer.processed for {self.ack_timeout_s:.0f} s "
                                       f"({event['type']} {event['event_id']}); the server is stuck — check its log") from None
        finally:
            self._inflight = None

    def _give_up(self, t: int, kind: str, reason: str, attempts: int) -> None:
        self.input_stats["dropped_inputs"].append({"t": t, "type": kind, "reason": reason, "attempts": attempts})
        msg = (f"vLLM-Omni refused {kind} at t={t} ms {attempts} time(s) (last: {reason}); "
               f"the model did not hear it")
        if self.on_input_dropped == "raise":
            raise InputDroppedError(msg)
        self.failed = self.failed or msg

    async def _end_of_user_turn(self, mic: Audio, dur: int) -> bool:
        """Client-side energy VAD: the user spoke and the microphone has now been quiet long enough."""
        x = mic.samples
        if (sum(v * v for v in x) / max(1, len(x))) ** 0.5 >= self.vad_rms:
            self._heard_speech, self._quiet_in_ms = True, 0
            return False
        self._quiet_in_ms += dur
        if self._heard_speech and self._quiet_in_ms >= self.commit_after_silence_ms:
            self._heard_speech, self._quiet_in_ms = False, 0
            return True
        return False

    def _receive(self, ev: dict) -> None:
        kind = ev["type"]
        rid = ev.get("response_id") or ev.get("response", {}).get("id")
        if kind == "response.output_audio.delta" and self.audio_out:
            sr = int(ev.get("sample_rate_hz") or self.out_sr)
            self._enqueue(Audio.from_pcm16(base64.b64decode(ev["delta"]), sr).resample(self.out_sr), "", rid)
        elif not self.audio_out and kind in ("response.output_text.delta", "response.output_audio_transcript.delta"):
            self._text_source = self._text_source or kind  # a server may send both: take one, never both
            if kind == self._text_source:
                text = ev.get("delta", "")  # as long as it takes to say
                self._enqueue(_Timed(round(len(text) / self.speech_cps * self.out_sr), self.out_sr), text, rid)
        elif kind == "response.output_audio_transcript.delta":
            mine = [q for q in self.queue if q[2] == rid]
            if mine:  # the text of the audio delta just before it
                mine[-1][1] += ev.get("delta", "")
            elif rid == self.utt_resp:  # that audio has already been played out: attach to what follows
                self._enqueue(self._silence(0), ev.get("delta", ""), rid)
            else:  # text ahead of its audio (e.g. sentence-level TTS): held for that audio
                self._early_text[rid] = self._early_text.get(rid, "") + ev.get("delta", "")
        elif kind == "response.done":
            self.finished.add(rid)
            if ev["response"].get("status") != "completed":
                self.queue = [q for q in self.queue if q[2] != rid]  # stopped early: the rest never plays

    def _enqueue(self, audio, text: str, rid: str) -> None:
        text = self._early_text.pop(rid, "") + text
        self.queue.append([audio, text, rid, len(audio), 0, 0])

    def _silence(self, ms: int):
        return Audio.silence(ms, self.out_sr) if self.audio_out else _Timed(round(ms * self.out_sr / 1000), self.out_sr)

    def _play(self, t: int) -> list[Frame]:
        """Release one step of queued speech, starting now."""
        want = round(self.spec.chunk_ms * self.out_sr / 1000)
        left = any(q[2] == self.utt_resp for q in self.queue)
        speaking = self.utt_end == t and (left or self.utt_resp not in self.finished)
        if speaking and self.queue and not left:  # a new response took over
            speaking = False
        if speaking and not self._split:
            rid, first = self.utt_resp, False
        else:
            if not self.queue:
                return []
            rid = self.queue[0][2]
            ready = sum(len(q[0]) for q in self.queue if q[2] == rid) * 1000 // self.out_sr
            if ready < self.prebuffer_ms and rid not in self.finished:
                return []  # let some of the reply arrive before starting to play it
            first = True
            self.utt, self.utt_resp = f"a:{rid}:{t}", rid
        audio, text = self._silence(0), ""
        while self.queue and self.queue[0][2] == rid and len(audio) < want:
            head = self.queue[0]
            take = want - len(audio)
            audio = audio + head[0][:take]
            head[0], head[4] = head[0][take:], head[4] + min(take, len(head[0]))
            due = len(head[1]) if not len(head[0]) else round(len(head[1]) * head[4] / head[3])  # chars said by now
            text, head[5] = text + head[1][head[5]:due], due
            if not len(head[0]):
                self.queue.pop(0)
        if not len(audio) and not text and rid in self.finished:
            return []  # the response ended exactly here
        if len(audio) < want and rid not in self.finished:
            audio = audio + self._silence(round((want - len(audio)) * 1000 / self.out_sr))  # still going: a pause
        if self.split_silence_ms is not None and self.audio_out and len(audio):
            quiet = not text.strip() and (sum(v * v for v in audio.samples) / len(audio)) ** 0.5 < self.silence_rms
            self._quiet_out_ms = self._quiet_out_ms + audio.dur_ms if quiet else 0
            if self._split and quiet:  # between utterances: the model is only emitting near-silence
                self.utt_end = t + audio.dur_ms
                return []
            if self._split:  # voice again: a new utterance of the same response
                self._split, first = False, True
                self.utt = f"a:{rid}:{t}"
            elif self._quiet_out_ms >= self.split_silence_ms:  # quiet long enough: this utterance has ended
                self._split = True
                self.utt_end = t + audio.dur_ms
                return []
        if first:
            text = text.lstrip()
        self.utt_end = t + audio.dur_ms
        data = audio if self.audio_out else text  # text mode: the utterance is its text, timed by the estimate
        return [Frame(self.spec.out, t, Chunk(self.utt, t, audio.dur_ms, data, text, first=first, last=False))]

    async def close(self) -> None:
        if self.ws is not None:
            self.reader.cancel()
            try:
                await self.ws.send(json.dumps({"type": "session.close"}))
            except websockets.ConnectionClosed:
                pass
            await self.ws.close()
