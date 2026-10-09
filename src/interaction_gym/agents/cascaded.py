"""A cascaded voice agent (VAD endpointing → ASR → tool-calling LLM → TTS) that runs in simulated time.

The pipeline of the usual open-source voice-agent stacks (e.g. LiveKit's ``AgentSession`` with
Silero VAD + STT + LLM + TTS, which Full-Duplex-Bench v3 uses as its cascaded baseline), stepped by the env:

- **Endpointing**: an energy VAD on the microphone (``Endpointing``: 20 ms frames, RMS threshold). A user turn
  opens with ``min_speech_ms`` of voiced audio and ends after ``silence_ms`` of silence — so a mid-sentence pause
  longer than that ends the turn too, which is what makes a cascaded agent cut into hesitant speakers.
- **ASR** on the audio of the turn (from its onset minus ``preroll_ms`` to its last voiced frame).
- **LLM** with the session's tools. Tool calls go out on ``policy.tool_call``; results come back on
  ``tool.result``; the LLM is called again until it answers in text (at most ``max_rounds`` rounds). Text that
  comes with tool calls is spoken too (a filler, as LiveKit does).
- **TTS** of each text reply, played as one segment.
- **Interruptions**: ``barge_in_ms`` of user speech while the agent talks cuts its speech (the heard part stays in
  the history); the same amount of speech before a scheduled reply starts cancels whatever has not gone out yet
  (calls already made stay made, and their results are kept in the history).

**Time.** Model calls run while simulated time stands still; their outputs are scheduled at simulated times
from a fixed ``LatencyModel`` (ASR, LLM time-to-first-token + per-token decode, TTS time to first audio) — not
from wall time, so that latency metrics are meaningful and episodes are deterministic whatever the load on the
servers. Measured wall times are recorded in ``events`` for comparison. ``cache`` (a dict) memoizes every model
call by its full request (LLM requests include the seed): two episodes that hear the same thing produce the same
outputs, e.g. an open-loop and a closed-loop run of the same recording up to the point where they diverge.

**Tool calling.** ``ToolChat`` calls an OpenAI-compatible chat endpoint. With ``native=True`` it relies on the
server's tool-call parser (``tool_choice="auto"``); otherwise (``native=False``, for servers started without
``--enable-auto-tool-choice``) it passes the tools with ``tool_choice="none"`` — the chat template still renders
them — and parses the model's own tool-call markup from the text (Qwen3 / Qwen3-Coder ``<tool_call>`` XML or
Hermes JSON).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from array import array
from dataclasses import dataclass, replace

from ..audio import Audio
from ..core import SESSION, AgentSpec, Frame, Segment, Session
from ..tools import CALL, ToolCall, ToolResult


@dataclass(frozen=True)
class Endpointing:
    rms_threshold: float = 300.0  # int16 RMS of a 20 ms frame counted as voiced
    frame_ms: int = 20
    min_speech_ms: int = 100  # voiced run that opens a user turn
    silence_ms: int = 800  # silence that ends it (end-of-turn delay)
    barge_in_ms: int = 300  # voiced run that interrupts the agent's speech / cancels a pending reply
    preroll_ms: int = 300  # audio kept before the onset for ASR
    min_words: int = 1  # shorter transcripts are ignored (a cough, a click)


@dataclass(frozen=True)
class LatencyModel:
    """Simulated processing time (ms). Defaults resemble a local streaming cascade on one GPU: Qwen3-ASR on a
    whole turn, a ~27B LLM at ~65 tokens/s, a streaming TTS's first audio."""

    asr_base_ms: float = 150.0
    asr_per_s_ms: float = 10.0  # per second of audio transcribed
    llm_ttft_ms: float = 350.0
    llm_per_token_ms: float = 15.0
    tts_first_audio_ms: float = 250.0

    def asr(self, audio_ms: int) -> int:
        return round(self.asr_base_ms + self.asr_per_s_ms * audio_ms / 1000)

    def llm(self, tokens: int) -> int:
        return round(self.llm_ttft_ms + self.llm_per_token_ms * tokens)

    def tts(self) -> int:
        return round(self.tts_first_audio_ms)


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


_THINK = re.compile(r"<think>.*?</think>", re.S)
_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FUNC = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)", re.S)


def _param_value(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """(spoken text, [{"name", "arguments"}]) from a reply carrying ``<tool_call>`` blocks, in either the
    Qwen3-Coder XML form (``<function=f><parameter=p>v</parameter></function>``) or Hermes JSON
    (``{"name": f, "arguments": {...}}``). Values that parse as JSON (numbers, lists) are taken as such."""
    text = _THINK.sub("", text).split("<think>")[0]
    calls = []
    for body in _CALL.findall(text):
        body = body.strip()
        m = _FUNC.search(body)
        if m:
            calls.append({"name": m.group(1).strip(), "arguments": {k: _param_value(v) for k, v in _PARAM.findall(m.group(2))}})
            continue
        try:
            obj = json.loads(body)
            args = obj.get("arguments", obj.get("parameters", {}))
            calls.append({"name": obj["name"], "arguments": json.loads(args) if isinstance(args, str) else args})
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            pass
    spoken = _CALL.sub("", text).strip()
    return spoken, calls


class ToolChat:
    """Chat completions with tools against an OpenAI-compatible endpoint (vLLM). ``complete`` returns
    ``{"content", "tool_calls": [{"name", "arguments"}], "completion_tokens", "wall_ms"}``."""

    def __init__(self, base_url: str, model: str, native: bool = False, **defaults):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.native, self.defaults = model, native, defaults

    def _post(self, payload: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(self.url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    async def complete(self, messages: list[dict], tools: list[dict], **kw) -> dict:
        payload = {"model": self.model, "messages": messages, **self.defaults, **kw}
        if tools:
            payload.update(tools=tools, tool_choice="auto" if self.native else "none")
        t0 = time.monotonic()
        out = await asyncio.to_thread(self._post, payload)
        wall = round((time.monotonic() - t0) * 1000)
        msg = out["choices"][0]["message"]
        if self.native:
            content = _THINK.sub("", msg.get("content") or "").strip()
            calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"].get("arguments") or "{}")}
                     for c in msg.get("tool_calls") or []]
        else:
            content, calls = parse_tool_calls(msg.get("content") or "")
        return {"content": content, "tool_calls": calls, "completion_tokens": out.get("usage", {}).get("completion_tokens", 0),
                "wall_ms": wall}

    def describe(self) -> dict:
        return {"model": self.model, "tool_calls": "native" if self.native else "prompted (template-rendered, parsed from text)",
                **({"params": self.defaults} if self.defaults else {})}


@dataclass
class _Pending:
    t: int  # when it goes out (simulated)
    kind: str  # "call" | "say"
    data: object  # ToolCall | Segment (t0 to be set on release)
    msg: dict | None = None  # the history message it belongs to


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")


def speakable(text: str) -> str:
    """What a TTS front end reads out: markdown emphasis, headings, code marks and emoji removed (LiveKit's
    default ``filter_markdown`` / ``filter_emoji`` transforms do the same)."""
    text = _EMOJI.sub("", text)
    text = re.sub(r"(\*\*|__|`+|~~)", "", text)
    text = re.sub(r"(?m)^\s*#+\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", " ", text).strip()


def _first_sentence_frac(text: str) -> float:
    m = re.search(r"[.!?](\s|$)", text)
    return 1.0 if m is None or not text else max(0.1, (m.end()) / len(text))


class CascadedAgent:
    """See the module docstring. ``asr``: a ``Transcriber``; ``llm``: a ``ToolChat``; ``tts``: a ``Speech``.
    ``instructions`` replaces the session's instructions as the system prompt (the session's tools are used
    either way). ``seed`` goes into every LLM request."""

    def __init__(self, spec: AgentSpec, asr, llm: ToolChat, tts, *, instructions: str | None = None, voice: str = "aiden",
                 language: str | None = "English", endpointing: Endpointing = Endpointing(), latency: LatencyModel = LatencyModel(),
                 seed: int = 0, cache: dict | None = None, max_rounds: int = 6, asr_language: str | None = "en",
                 out_sr: int = 16000):
        assert spec.audio, "the cascaded agent listens to the microphone: AgentSpec(audio=...)"
        self.spec, self.asr, self.llm, self.tts = spec, asr, llm, tts
        self.instructions, self.voice, self.language, self.asr_language = instructions, voice, language, asr_language
        self.ep, self.lat, self.seed, self.max_rounds, self.out_sr = endpointing, latency, seed, max_rounds, out_sr
        self.cache = cache if cache is not None else {}
        self.session = Session()
        self.history: list[dict] = []
        self.events: list[dict] = []
        # microphone + VAD
        self.mic = array("h")
        self._frame_n = round(spec.sr * endpointing.frame_ms / 1000)
        self._vad_pos = 0  # samples analysed
        self._run_voiced = 0  # ms of consecutive voiced frames
        self._last_voiced_end: int | None = None
        self._turn_start: int | None = None  # onset of the open user turn (ms), None when no turn is open
        self._barged = False  # this voiced run already triggered a barge-in
        # output
        self.pending: list[_Pending] = []
        self.awaiting: dict[str, int | None] = {}  # tool call id -> result time
        self.chain: dict | None = None  # the response in progress: {"round", "cancelled"}
        self.speaking: Segment | None = None
        self._speech_msg: dict | None = None
        self._n_calls = self._n_says = 0

    # ---------------------------------------------------------------- bookkeeping

    @property
    def busy(self) -> bool:
        """Something is still going on inside the agent (a user turn being listened to, a reply being prepared,
        tool results awaited, or speech scheduled) — the episode should not end yet."""
        return self._turn_start is not None or bool(self.pending) or bool(self.awaiting) or self.chain is not None

    def describe(self) -> dict:
        return {"name": "cascaded ASR-LLM-TTS", "kind": "cascaded", "model": getattr(self.llm, "model", type(self.llm).__name__),
                "type": "cascaded", "endpointing": self.ep.__dict__, "latency_model": self.lat.__dict__,
                "llm": self.llm.describe(), "asr": getattr(self.asr, "model", type(self.asr).__name__),
                "tts": {"model": getattr(self.tts, "model", type(self.tts).__name__), "voice": self.voice},
                "seed": self.seed, "max_rounds": self.max_rounds, "system_prompt": self._system()}

    def _event(self, t: int, kind: str, **kw) -> None:
        self.events.append({"t": t, "kind": kind, **kw})

    def _system(self) -> str:
        return self.instructions if self.instructions is not None else self.session.instructions

    # ---------------------------------------------------------------- model calls (memoized)

    async def _transcribe(self, audio: Audio) -> tuple[str, int]:
        k = _key("asr", getattr(self.asr, "model", ""), self.asr_language, hashlib.sha256(audio.samples.tobytes()).hexdigest())
        if k not in self.cache:
            t0 = time.monotonic()
            text = await self.asr.transcribe(audio, self.asr_language)
            self.cache[k] = (text, round((time.monotonic() - t0) * 1000))
        return self.cache[k]

    async def _complete(self, messages: list[dict]) -> dict:
        tools = [t.openai() for t in self.session.tools]
        k = _key("llm", self.llm.model, messages, tools, self.seed, self.llm.defaults)
        if k not in self.cache:
            self.cache[k] = await self.llm.complete(messages, tools, seed=self.seed)
        return self.cache[k]

    async def _synth(self, text: str) -> tuple[Audio, int]:
        k = _key("tts", getattr(self.tts, "model", ""), text, self.voice, self.language)
        if k not in self.cache:
            t0 = time.monotonic()
            audio = await self.tts.synth(text, self.voice, None, language=self.language)
            if audio.sr != self.out_sr:
                audio = audio.resample(self.out_sr)
            self.cache[k] = (audio, round((time.monotonic() - t0) * 1000))
        return self.cache[k]

    # ---------------------------------------------------------------- listening

    def _hear(self, audio: Audio) -> list[tuple]:
        """Append microphone audio and run the VAD over the new complete frames. Returns events in time order:
        ``("onset", t)``, ``("barge", t)``, ``("endpoint", t_end, start_ms, last_voiced_end)``."""
        if audio.sr != self.spec.sr:
            audio = audio.resample(self.spec.sr)
        self.mic.extend(audio.samples)
        ep, n, out = self.ep, self._frame_n, []
        while self._vad_pos + n <= len(self.mic):
            seg = self.mic[self._vad_pos:self._vad_pos + n]
            self._vad_pos += n
            t_end = round(self._vad_pos * 1000 / self.spec.sr)
            rms = (sum(v * v for v in seg) / n) ** 0.5
            if rms >= ep.rms_threshold:
                self._run_voiced += ep.frame_ms
                if self._run_voiced >= ep.min_speech_ms:
                    self._last_voiced_end = t_end
                    if self._turn_start is None:
                        self._turn_start = t_end - self._run_voiced
                        out.append(("onset", self._turn_start))
                if self._run_voiced >= ep.barge_in_ms and not self._barged:
                    self._barged = True
                    out.append(("barge", t_end))
            else:
                self._run_voiced = 0
                self._barged = False
                if self._turn_start is not None and t_end - self._last_voiced_end >= ep.silence_ms:
                    out.append(("endpoint", self._last_voiced_end + ep.silence_ms, self._turn_start, self._last_voiced_end))
                    self._turn_start = None
        return out

    def _audio(self, a_ms: int, b_ms: int) -> Audio:
        sr = self.spec.sr
        return Audio(self.mic[max(0, round(a_ms * sr / 1000)):round(b_ms * sr / 1000)], sr)

    # ---------------------------------------------------------------- acting

    async def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        out: list[Frame] = []
        events: list[tuple] = []
        results: list[Frame] = []
        for f in obs:
            if f.stream == SESSION:
                self.session = f.data
            elif f.stream == self.spec.audio and isinstance(f.data, Audio):
                events += self._hear(f.data)
            elif isinstance(f.data, ToolResult):
                results.append(f)
        for f in results:
            await self._on_result(f)
        for ev in events:
            if ev[0] == "barge":
                out += self._on_barge(ev[1], t)
            elif ev[0] == "endpoint":
                await self._on_endpoint(*ev[1:])
        out += self._release(t)
        return out

    def _on_barge(self, t_b: int, t: int) -> list[Frame]:
        out = []
        if self.speaking is not None and t_b < self.speaking.end and t < self.speaking.end:
            cut = self.speaking.cut(max(t, self.speaking.t0))
            out.append(Frame(self.spec.out, cut.end, cut))
            if self._speech_msg is not None:  # only what was said stays in the history
                self._speech_msg["content"] = cut.text or ""
                self._drop_if_empty(self._speech_msg)
            self._event(t, "yield", heard_ms=t_b, said=cut.text)
            self.speaking = replace(self.speaking, dur=cut.dur)
        if self.pending or self.chain is not None:
            self._cancel(t_b)
        return out

    def _cancel(self, t: int) -> None:
        """The user started speaking again before the reply went out: drop everything not yet sent."""
        dropped = []
        for p in self.pending:
            dropped.append(p.kind)
            if p.kind == "call":
                self.awaiting.pop(p.data.id, None)  # never sent: no result will come
            if p.msg is None:
                continue
            if p.kind == "call":
                p.msg["tool_calls"] = [c for c in p.msg.get("tool_calls", []) if c["id"] != p.data.id]
                if not p.msg["tool_calls"]:
                    p.msg.pop("tool_calls")
            else:
                p.msg["content"] = ""
            self._drop_if_empty(p.msg)
        self.pending = []
        if self.chain is not None and not self.chain["cancelled"]:
            self.chain["cancelled"] = True
            dropped.append("chain")
        if self.chain is not None and not self.awaiting:
            self.chain = None
        if dropped:
            self._event(t, "cancel", dropped=dropped)

    def _drop_if_empty(self, msg: dict) -> None:
        if msg["role"] == "assistant" and not msg.get("content") and not msg.get("tool_calls"):
            self.history = [m for m in self.history if m is not msg]

    async def _on_endpoint(self, t_end: int, start: int, last_voiced: int) -> None:
        audio = self._audio(start - self.ep.preroll_ms, last_voiced + 100)
        text, wall = await self._transcribe(audio)
        t_asr = t_end + self.lat.asr(audio.dur_ms)
        self._event(t_end, "endpoint", turn_start=start, speech_end=last_voiced, audio_ms=audio.dur_ms, transcript=text,
                    asr_wall_ms=wall, asr_done=t_asr)
        if len(text.split()) < self.ep.min_words:
            return
        if self.chain is not None:  # a reply still in progress (tool results pending): this turn supersedes it
            self._cancel(t_end)
        self.history.append({"role": "user", "content": text})
        self.chain = {"round": 0, "cancelled": False}
        await self._round(t_asr)

    async def _round(self, t0: int) -> None:
        chain = self.chain
        chain["round"] += 1
        messages = [{"role": "system", "content": self._system()}] + self.history
        res = await self._complete(messages)
        content, calls = res["content"], res["tool_calls"][:8]
        if chain["round"] > self.max_rounds:  # stop calling tools
            calls = []
        tokens = res["completion_tokens"]
        if calls:
            t_out = t0 + self.lat.llm(tokens)  # a tool call is usable once it has been fully generated
        else:
            t_out = t0 + self.lat.llm(round(tokens * _first_sentence_frac(content)))  # speech starts after the first sentence
        msg: dict = {"role": "assistant", "content": content}
        self._event(t0, "llm", round=chain["round"], tokens=tokens, wall_ms=res["wall_ms"], done=t_out, content=content,
                    tool_calls=calls)
        if calls:
            msg["tool_calls"] = []
            for c in calls:
                self._n_calls += 1
                call = ToolCall(f"call_{self._n_calls}", str(c["name"]), c["arguments"] if isinstance(c["arguments"], dict) else {})
                msg["tool_calls"].append({"id": call.id, "type": "function",
                                          "function": {"name": call.name, "arguments": json.dumps(call.arguments)}})
                self.pending.append(_Pending(t_out, "call", call, msg))
                self.awaiting[call.id] = None
        self.history.append(msg)
        if content:
            await self._say(content, t_out + self.lat.tts(), msg)
        if not calls:
            self.chain = None

    async def _say(self, text: str, t: int, msg: dict) -> None:
        text = speakable(text)
        if not text:
            return
        audio, wall = await self._synth(text)
        self._n_says += 1
        seg = Segment(f"a{self._n_says}", t, audio.dur_ms, audio, text=text)
        self.pending.append(_Pending(t, "say", seg, msg))
        self._event(t, "tts", text=text, audio_ms=audio.dur_ms, wall_ms=wall)

    async def _on_result(self, f: Frame) -> None:
        res: ToolResult = f.data
        if res.id not in self.awaiting:
            return
        self.awaiting[res.id] = f.t
        self.history.append({"role": "tool", "tool_call_id": res.id, "content": res.content})
        if any(v is None for v in self.awaiting.values()):
            return
        t_all = max(self.awaiting.values())
        self.awaiting = {}
        if self.chain is None or self.chain["cancelled"]:
            self.chain = None
            return
        await self._round(t_all)

    def _release(self, t: int) -> list[Frame]:
        """Send what is due in this step's window [t, t + chunk): tool calls at their time, speech once the
        agent's previous speech has finished."""
        out, keep = [], []
        horizon = t + self.spec.chunk_ms
        for p in sorted(self.pending, key=lambda p: p.t):
            at = max(p.t, t)
            if p.kind == "say":
                at = max(at, self.speaking.end if self.speaking is not None else 0)
            if at >= horizon:
                keep.append(p)
                continue
            if p.kind == "call":
                out.append(Frame(CALL["agent"], at, p.data))
                self._event(at, "tool_call", name=p.data.name, arguments=p.data.arguments)
            else:
                seg = replace(p.data, t0=at)
                self.speaking, self._speech_msg = seg, p.msg
                out.append(Frame(self.spec.out, at, seg))
        self.pending = keep
        return out
