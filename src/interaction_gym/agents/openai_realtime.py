"""Adapter for the OpenAI Realtime API (``gpt-realtime*``) as a full-duplex agent.

The model turns by voice activity: the server's turn detection (``server_vad`` or ``semantic_vad``) commits the
user's turn and starts a response, and user speech during a response interrupts it. It does not backchannel or speak
over the user; the env measures it like any other duplex agent.

**Clocks.** The env's timeline is simulated: it stands still while the simulated user's LLM and TTS compute. The
Realtime API cannot be paused, but its turn detection counts the audio it *receives*, not wall time (probed
2026-10-22 on gpt-realtime-mini, server and semantic VAD: sending at 4x moved no detected turn boundary in audio
time; a 6 s pause in sending, at the end of a turn or in the middle of a word, detected nothing; a session idle for
12 s stayed open). So the adapter:

- sends the microphone at real-time pace (one step of audio per step of wall time, ``chunk_ms``), so that what the
  server hears and when it answers keep their real-time relation; when the env stalls (the simulated user computing)
  it sends nothing and picks up again from there without catching up — the server never sees the stall;
- places each response on the simulated timeline at the simulated time the server started it (``response.created``)
  plus the wall time it then took to produce its first audio — the model's real latency, whatever the env did
  meanwhile;
- plays a response's audio contiguously. The API streams audio at about real time or slower (0.7x measured from
  one client), so playback can run dry mid-utterance: ``underrun="wait"`` (default) then holds the env until more
  audio arrives — the server's turn detection does not notice, so only network throughput, not the model's
  behaviour, is taken out; ``underrun="silence"`` plays the gap as silence, as a live client would, after a
  ``prebuffer_ms`` start-up buffer;
- on user speech during playback (``input_audio_buffer.speech_started``, with ``interrupt_response``) stops playing at
  once and tells the server how much was heard (``conversation.item.truncate``; an item nothing of which was heard
  is deleted), so the model's history holds what the user heard.

Latency therefore includes the network from this client and the provider's load: runs are not reproducible, and
there are no token log-probabilities. Use it for evaluation, not RL.

**Tools.** The session's tools are passed as Realtime function tools (``AgentSpec.obs`` must include ``tool.result``). A call goes out on ``policy.tool_call``; its
``tool.result`` is returned as a ``function_call_output`` item, and once every call of a response has its result the
adapter asks for the next response.

``usage`` sums the token counts of every ``response.done``; ``trace()`` records each response's timing.
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import json
import os
import time
import warnings

import websockets

from ..audio import Audio
from ..core import SESSION, AgentSpec, Chunk, Frame, Session
from ..tools import CALL, RESULT, ToolCall, ToolResult

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"
RT_SR = 24000  # the API's PCM rate, in and out
SERVER_VAD = {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300, "silence_duration_ms": 500,
              "create_response": True, "interrupt_response": True}
# errors about a client request that came too late to matter (e.g. truncating an item already gone)
BENIGN_ERRORS = ("response_cancel_not_active", "item_truncate_invalid_item_id", "conversation_already_has_active_response")


class _Response:
    def __init__(self, rid: str, created_w: float, created_t: int):
        self.rid, self.created_w, self.created_t = rid, created_w, created_t
        self.item: str | None = None  # the assistant message item its audio belongs to
        self.pcm = bytearray()  # all audio received (RT_SR PCM16)
        self.text = ""  # all transcript received
        self.first_w: float | None = None
        self.start_t: int | None = None  # simulated time its speech starts
        self.played = 0  # samples played
        self.chars = 0  # transcript characters released
        self.status: str | None = None  # from response.done
        self.cut_t: int | None = None  # where playback was stopped by the user
        self.fixed = False  # the server's copy was made what the user heard (truncated / deleted)
        self.calls: list[str] = []  # function calls it made

    @property
    def received(self) -> int:
        return len(self.pcm) // 2

    def log(self) -> dict:
        lat = None if self.first_w is None else round((self.first_w - self.created_w) * 1000)
        return {"response": self.rid, "created_t": self.created_t, "first_audio_wall_ms": lat, "start_t": self.start_t,
                "status": self.status, "received_ms": self.received * 1000 // RT_SR, "played_ms": self.played * 1000 // RT_SR,
                **({"cut_t": self.cut_t} if self.cut_t is not None else {}), **({"tool_calls": self.calls} if self.calls else {}),
                "text": self.text}


class OpenAIRealtimeAgent:
    """One Realtime session per episode. ``spec.audio`` must be set (the agent hears the mixed microphone) and
    ``spec.sr`` is the env's audio rate (resampled to and from the API's 24 kHz). ``turn_detection`` is the session's
    (default ``SERVER_VAD``: 500 ms of silence ends a turn, user speech interrupts). ``instructions`` overrides the
    session's. ``session`` adds raw ``session.update`` fields. ``max_wait_s``: how long ``underrun="wait"`` waits for
    audio before it gives up and plays silence."""

    def __init__(self, spec: AgentSpec, model: str = "gpt-realtime-mini", *, voice: str = "marin",
                 turn_detection: dict | None = None, instructions: str | None = None, url: str = OPENAI_REALTIME_URL,
                 api_key_env: str = "OPENAI_API_KEY", underrun: str = "wait", prebuffer_ms: int = 300,
                 max_wait_s: float = 30.0, tools: bool = True, session: dict | None = None, name: str | None = None):
        assert spec.audio, "the realtime agent listens to the microphone: set AgentSpec(audio=...)"
        assert underrun in ("wait", "silence"), underrun
        self.spec, self.model, self.voice, self.url = spec, model, voice, url
        self.turn_detection = dict(SERVER_VAD if turn_detection is None else turn_detection)
        self.instructions, self.extra, self.use_tools, self.name = instructions, session or {}, tools, name
        self.underrun, self.prebuffer_ms, self.max_wait_s = underrun, prebuffer_ms, max_wait_s
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"OpenAIRealtimeAgent: the OpenAI API needs a key in ${api_key_env}")
        self.ws = None
        self.reader: asyncio.Task | None = None
        self.inbox: list[tuple[float, dict]] = []  # (wall time received, event)
        self._arrived = asyncio.Event()
        self._updated = asyncio.Event()
        self.anchors_w: list[float] = []  # wall time each step's audio was sent ...
        self.anchors_t: list[int] = []  # ... and the simulated time it reached
        self.next_wall = 0.0
        self.responses: dict[str, _Response] = {}
        self.order: list[str] = []  # responses in the order they were created
        self.cur: _Response | None = None  # the one playing
        self.calls: dict[str, str | None] = {}  # call_id -> response, while its result is awaited
        self.requested = False  # a response.create sent, its response not yet created
        self.extra_frames: list[Frame] = []  # frames that came up while playback waited for audio
        self.usage: dict[str, int] = {}
        self.errors: list[dict] = []
        self.events: dict[str, int] = {}

    # ---------------------------------------------------------------- session

    def _tools(self, session: Session | None) -> list[dict]:
        if not self.use_tools or session is None:
            return []
        if session.tools and RESULT["agent"] not in self.spec.obs:
            warnings.warn(f"OpenAIRealtimeAgent: the session has tools but AgentSpec.obs lacks {RESULT['agent']!r}: "
                          "tool results will never reach the model", stacklevel=3)
        return [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in session.tools]

    def _session(self, session: Session | None) -> dict:
        instructions = self.instructions if self.instructions is not None else (session.instructions if session else "")
        tools = self._tools(session)
        return {"type": "realtime", "output_modalities": ["audio"],
                **({"instructions": instructions} if instructions else {}),
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": RT_SR}, "turn_detection": self.turn_detection},
                          "output": {"format": {"type": "audio/pcm", "rate": RT_SR}, "voice": self.voice}},
                **({"tools": tools, "tool_choice": "auto"} if tools else {}), **self.extra}

    async def _connect(self, session: Session | None) -> None:
        sep = "&" if "?" in self.url else "?"
        self.ws = await websockets.connect(f"{self.url}{sep}model={self.model}", max_size=None,
                                           additional_headers={"Authorization": f"Bearer {self.api_key}"})
        self.reader = asyncio.create_task(self._read())
        await self._send({"type": "session.update", "session": self._session(session)})
        updated = asyncio.ensure_future(self._updated.wait())
        await asyncio.wait({updated, self.reader}, timeout=30, return_when=asyncio.FIRST_COMPLETED)
        if not updated.done():
            updated.cancel()
            if self.reader.done():
                self.reader.result()
                raise RuntimeError("the Realtime API closed the connection before the session was set up")
            raise TimeoutError("the Realtime API did not confirm the session within 30 s")
        self.next_wall = time.monotonic()

    async def _send(self, event: dict) -> None:
        await self.ws.send(json.dumps(event))

    async def _read(self) -> None:
        async for raw in self.ws:
            ev = json.loads(raw)
            kind = ev.get("type", "")
            self.events[kind] = self.events.get(kind, 0) + 1
            if kind == "session.updated":
                self._updated.set()
            elif kind == "error":
                err = ev.get("error") or {}
                if err.get("code") in BENIGN_ERRORS:
                    self.errors.append(err)
                    continue
                if not self._updated.is_set():
                    raise RuntimeError(f"Realtime API error: {err}")
                self.errors.append(err)
                warnings.warn(f"OpenAIRealtimeAgent: {err.get('code')}: {err.get('message')}", stacklevel=1)
                continue
            self.inbox.append((time.monotonic(), ev))
            self._arrived.set()

    # ---------------------------------------------------------------- time

    def _sim(self, w: float) -> int:
        """The simulated time at wall time ``w``: the step whose audio had just been sent, plus the wall time since,
        at most one step (the env stalled after it: simulated time stood still)."""
        k = bisect.bisect_right(self.anchors_w, w) - 1
        if k < 0:
            return self.anchors_t[0] if self.anchors_t else 0
        return self.anchors_t[k] + min(round((w - self.anchors_w[k]) * 1000), self.spec.chunk_ms)

    # ---------------------------------------------------------------- the step

    @property
    def busy(self) -> bool:
        """A response not yet ended, or speech not yet played (an episode loop waits for it before ending)."""
        return bool(self.calls) or self.requested or any(
            r.status is None or (r.cut_t is None and r.played < r.received) for r in self.responses.values())

    async def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        if self.ws is None:
            s = next((f.data for f in obs if f.stream == SESSION and isinstance(f.data, Session)), None)
            await self._connect(s)
        elif any(f.stream == SESSION for f in obs):
            s = next(f.data for f in obs if f.stream == SESSION)
            await self._send({"type": "session.update", "session": self._session(s)})
        if self.reader.done():
            self.reader.result()
        for f in obs:
            if f.stream == self.spec.audio and isinstance(f.data, Audio):
                pcm = f.data.resample(RT_SR).samples.tobytes()
                await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})
        self.anchors_w.append(time.monotonic())
        self.anchors_t.append(t)
        out = []
        for f in obs:
            if isinstance(f.data, ToolResult):
                await self._tool_result(f.data)
        # a live microphone: one step of audio per step of wall time; after a stall, carry on from now
        self.next_wall = max(self.next_wall + self.spec.chunk_ms / 1000, time.monotonic())
        await asyncio.sleep(self.next_wall - time.monotonic())
        out += await self._process(t)
        out += await self._play(t)
        out, self.extra_frames = self.extra_frames + out, []
        return out

    async def _process(self, t: int) -> list[Frame]:
        events, self.inbox = self.inbox, []
        self._arrived.clear()
        out = []
        for w, ev in events:
            out += await self._handle(w, ev, t)
        return out

    async def _handle(self, w: float, ev: dict, t: int) -> list[Frame]:
        kind = ev["type"]
        if kind == "response.created":
            rid = ev["response"]["id"]
            self.responses[rid] = _Response(rid, w, self._sim(w))
            self.order.append(rid)
            self.requested = False
        elif kind == "response.output_audio.delta":
            r = self.responses.get(ev.get("response_id"))
            if r is not None:
                r.item = r.item or ev.get("item_id")
                if r.first_w is None:
                    r.first_w = w
                    r.start_t = r.created_t + round((w - r.created_w) * 1000)
                if r.cut_t is None:
                    r.pcm += base64.b64decode(ev["delta"])
        elif kind == "response.output_audio_transcript.delta":
            r = self.responses.get(ev.get("response_id"))
            if r is not None:
                r.item = r.item or ev.get("item_id")
                r.text += ev.get("delta", "")
        elif kind == "response.function_call_arguments.done":
            r = self.responses.get(ev.get("response_id"))
            try:
                args = json.loads(ev.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": ev.get("arguments")}
            call = ToolCall(ev["call_id"], ev.get("name", ""), args if isinstance(args, dict) else {"_value": args})
            self.calls[call.id] = r.rid if r else None
            if r is not None:
                r.calls.append(call.name)
            return [Frame(CALL["agent"], max(t, self._sim(w)), call)]
        elif kind == "response.done":
            resp = ev.get("response", {})
            r = self.responses.get(resp.get("id"))
            self._count(resp.get("usage") or {})
            if r is not None:
                r.status = resp.get("status")
                if r.status != "completed" and r.cut_t is None:  # cancelled (or failed) before it was heard in full
                    self._cut(r, t)
                if r.cut_t is not None:
                    await self._fix(r)
        elif kind == "input_audio_buffer.speech_started" and self.turn_detection.get("interrupt_response", True):
            # the user talks: a client stops playing and drops whatever it has not played yet
            for r in self.responses.values():
                if r.cut_t is None and (r.status is None or r.played < r.received):
                    self._cut(r, t)
                    if r.status is not None:  # already ended on the server: fix its copy now (else at response.done)
                        await self._fix(r)
        return []

    def _count(self, usage: dict) -> None:
        for k, v in usage.items():
            if isinstance(v, int):
                self.usage[k] = self.usage.get(k, 0) + v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, int):
                        key = f"{k.replace('_details', '')}.{k2}"
                        self.usage[key] = self.usage.get(key, 0) + v2

    def _cut(self, r: _Response, t: int) -> None:
        """Stop playing ``r`` at ``t``: nothing more of it is heard."""
        r.cut_t = t
        if self.cur is r:
            self.cur = None

    async def _fix(self, r: _Response) -> None:
        """Make the server's copy of an ended, cut response what the user heard of it."""
        if r.fixed or r.item is None:
            return
        r.fixed = True
        if r.played:
            await self._send({"type": "conversation.item.truncate", "item_id": r.item, "content_index": 0,
                              "audio_end_ms": r.played * 1000 // RT_SR})
        else:  # nothing of it was heard
            await self._send({"type": "conversation.item.delete", "item_id": r.item})

    async def _tool_result(self, res: ToolResult) -> None:
        if res.id not in self.calls:
            return
        rid = self.calls.pop(res.id)
        await self._send({"type": "conversation.item.create",
                          "item": {"type": "function_call_output", "call_id": res.id, "output": res.content}})
        if not any(v == rid for v in self.calls.values()):  # every call of that response is answered
            await self._send({"type": "response.create"})
            self.requested = True

    # ---------------------------------------------------------------- playback

    def _next(self, t: int, horizon: int) -> _Response | None:
        for rid in self.order:
            r = self.responses[rid]
            if r.cut_t is not None or r.start_t is None or r.played >= r.received and r.status is not None:
                continue
            if r.start_t >= horizon:
                return None  # its first audio is due later
            if self.underrun == "silence" and r.status is None and r.received * 1000 // RT_SR < self.prebuffer_ms:
                return None
            return r
        return None

    async def _play(self, t: int) -> list[Frame]:
        """One step of speech: from the current response, or a new one that is due in this step."""
        horizon = t + self.spec.chunk_ms
        r, first = self.cur, False
        if r is None:
            r = self._next(t, horizon)
            if r is None:
                return []
            self.cur, first = r, True
        t0 = max(t, r.start_t) if first else t
        want = (horizon - t0) * RT_SR // 1000
        if self.underrun == "wait":
            deadline = time.monotonic() + self.max_wait_s
            while r.received - r.played < want and r.status is None and r.cut_t is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    warnings.warn(f"OpenAIRealtimeAgent: no audio for {self.max_wait_s:.0f} s; playing silence", stacklevel=2)
                    break
                try:
                    await asyncio.wait_for(self._arrived.wait(), left)
                except asyncio.TimeoutError:
                    pass
                self.extra_frames += await self._process(t)  # a reply may also end, or be cut, or call a tool
            if r.cut_t is not None:
                return []
        take = min(want, r.received - r.played)
        a = bytes(r.pcm[r.played * 2:(r.played + take) * 2])
        r.played += take
        done = r.status is not None and r.played >= r.received
        if take < want and not done:  # still coming: the gap is silence (underrun="silence", or waited too long)
            a += bytes((want - take) * 2)
        total = r.received if not done else max(r.received, 1)
        due = len(r.text) if done else len(r.text) * r.played // max(total, 1)
        text, r.chars = r.text[r.chars:due], max(r.chars, due)
        if done:
            self.cur = None
        if not a and not text:
            return []
        audio = Audio.from_pcm16(a, RT_SR).resample(self.spec.sr)
        return [Frame(self.spec.out, t0, Chunk(f"a:{r.rid}", t0, audio.dur_ms, audio, text.lstrip() if first else text,
                                               first=first, last=False))]

    # ---------------------------------------------------------------- records

    def describe(self) -> dict:
        """``meta.agent``: which model, served how, on which clock."""
        return {"name": self.name or self.model, "model": self.model, "provider": "openai", "kind": "full-duplex (VAD turns)",
                "server": f"openai realtime {self.url}", "clock": "wall, audio-paced", "underrun": self.underrun,
                "voice": self.voice, "turn_detection": self.turn_detection, "output": "audio", "reproducible": False,
                **({"usage": dict(self.usage)} if self.usage else {})}

    def trace(self, episode_id: str) -> dict:
        """Each response's timing (``agent_traces.jsonl``)."""
        return {"episode_id": episode_id, "server": "openai realtime", "model": self.model,
                "responses": [self.responses[rid].log() for rid in self.order], "usage": dict(self.usage),
                **({"errors": self.errors} if self.errors else {})}

    async def close(self) -> None:
        if self.ws is not None:
            if self.reader is not None:
                self.reader.cancel()
            await self.ws.close()
