"""Instrumentation for demand profiling: wrappers that count what an episode asks of each service.

Nothing in the core changes: wrap the clients an episode is built with, the same way
``examples/live_user.py`` wraps them in ``Counting``::

    meter = Meter()
    llm = MeteredChat(OpenAIChat(url, model), meter, "llm")
    tts = Cached(MeteredSpeech(OpenAISpeech(...), meter, "tts"))      # inside Cached: cache hits are free
    clone = MeteredSpeech(OpenAISpeech(...), meter, "clone")
    agent = MeteredAgent(VllmOmniDuplexAgent(...), meter, "agent")
    ... run the episode, then meter.end_episode(env.t) ...
    demand = meter.demand()          # a DemandProfile (tune.profile)

Every call is one ``Event``: service, episode, wall start/end, units of work (``call`` for a chat
completion, ``audio_s`` synthesized for TTS, simulated seconds stepped for a duplex agent) and,
for chat, prompt/completion tokens. Concurrency over time (how many agent sessions / requests were
in flight) falls out of the start/end times.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import re
import time
from dataclasses import asdict, dataclass, field

from ..clients import OpenAIChat, _post

# which episode a call belongs to: set by Meter.episode(); concurrent episodes each run in their own task
_EPISODE: contextvars.ContextVar[str | None] = contextvars.ContextVar("dig_tune_episode", default=None)


@dataclass
class Event:
    service: str
    episode: str | None
    start: float  # wall clock (perf_counter)
    end: float
    units: float  # work, in the service's unit
    calls: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_tokens: bool = False


@dataclass
class EpisodeRecord:
    id: str
    sim_s: float = 0.0
    wall_s: float = 0.0
    start: float = 0.0


@dataclass
class Meter:
    events: list[Event] = field(default_factory=list)
    episodes: dict[str, EpisodeRecord] = field(default_factory=dict)

    # -- episodes
    def start_episode(self, eid: str) -> contextvars.Token:
        self.episodes[eid] = EpisodeRecord(eid, start=time.perf_counter())
        return _EPISODE.set(eid)

    def end_episode(self, sim_ms: int, token: contextvars.Token | None = None, eid: str | None = None) -> None:
        eid = eid or _EPISODE.get()
        rec = self.episodes[eid]
        rec.sim_s, rec.wall_s = sim_ms / 1000, time.perf_counter() - rec.start
        if token is not None:
            _EPISODE.reset(token)

    def record(self, service: str, start: float, units: float, **kw) -> Event:
        ev = Event(service, _EPISODE.get(), start, time.perf_counter(), units, **kw)
        self.events.append(ev)
        return ev

    # -- summaries
    def services(self) -> list[str]:
        return sorted({e.service for e in self.events})

    def concurrency(self, service: str) -> dict:
        """Mean and peak number of in-flight calls of a service while any was in flight
        (for an agent: sessions stepping at once)."""
        evs = [e for e in self.events if e.service == service]
        if not evs:
            return {"mean": 0.0, "peak": 0}
        edges = sorted([(e.start, 1) for e in evs] + [(e.end, -1) for e in evs], key=lambda x: (x[0], x[1]))
        cur = peak = 0
        area = busy = 0.0
        last = edges[0][0]
        for t, d in edges:
            if cur:
                area += cur * (t - last)
                busy += t - last
            cur += d
            peak = max(peak, cur)
            last = t
        return {"mean": area / busy if busy else 0.0, "peak": peak}

    def demand(self, units: dict[str, str] | None = None):
        from .profile import DemandProfile

        return DemandProfile.from_meter(self, units)

    def to_json(self) -> dict:
        return {"events": [asdict(e) for e in self.events], "episodes": {k: asdict(v) for k, v in self.episodes.items()}}

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f)


def _tokens(text: str) -> int:
    """Rough token count (~4 characters a token in English) when the server reports no usage."""
    return max(1, round(len(text) / 4))


class MeteredChat:
    """A ``TextGen`` that records each call (units = 1 call, plus tokens). For an ``OpenAIChat``
    it makes the request itself to read the server's ``usage``; any other client is timed and its
    tokens estimated from text length."""

    def __init__(self, inner, meter: Meter, service: str = "llm"):
        self.inner, self.meter, self.service = inner, meter, service

    def __getattr__(self, k):  # describe() and friends see the wrapped client
        return getattr(self.inner, k)

    async def chat(self, messages, **kw) -> str:
        t = time.perf_counter()
        if isinstance(self.inner, OpenAIChat):
            c = self.inner
            payload = {"model": c.model, "messages": messages, **c.defaults, **kw}
            data = json.loads(await asyncio.to_thread(_post, c.url, payload, c.api_key))
            usage = data.get("usage") or {}
            text = data["choices"][0]["message"].get("content") or ""
            pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
            est = pt is None
            self.meter.record(self.service, t, 1, prompt_tokens=pt if pt is not None else _tokens(json.dumps(messages)),
                              completion_tokens=ct if ct is not None else _tokens(text), estimated_tokens=est)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # as OpenAIChat.chat does
            return text.split("<think>")[0].strip()
        text = await self.inner.chat(messages, **kw)
        self.meter.record(self.service, t, 1, prompt_tokens=_tokens(json.dumps(messages)), completion_tokens=_tokens(text),
                          estimated_tokens=True)
        return text


class MeteredSpeech:
    """A ``Speech`` client that records each synthesis (units = seconds of audio produced)."""

    def __init__(self, inner, meter: Meter, service: str = "tts"):
        self.inner, self.meter, self.service = inner, meter, service

    def __getattr__(self, k):
        return getattr(self.inner, k)

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None, **kw):
        t = time.perf_counter()
        audio = await self.inner.synth(text, voice, instructions, ref_audio, ref_text, language, **kw)
        self.meter.record(self.service, t, audio.dur_ms / 1000)
        return audio


class MeteredAgent:
    """Wraps a duplex agent (``act`` / ``close``): each step is one event whose units are the
    simulated seconds it covered and whose wall time is what the agent server took (in lockstep,
    its compute; in realtime, the pacing)."""

    def __init__(self, inner, meter: Meter, service: str = "agent"):
        self.inner, self.meter, self.service = inner, meter, service
        self._last_t: int | None = None

    def __getattr__(self, k):
        return getattr(self.inner, k)

    async def act(self, t: int, obs):
        start = time.perf_counter()
        out = self.inner.act(t, obs)
        if inspect.isawaitable(out):  # VllmOmniDuplexAgent.act is async, CannedAgent.act is not
            out = await out
        step_ms = self.inner.spec.chunk_ms if hasattr(self.inner, "spec") else (t - (self._last_t or 0))
        self._last_t = t
        self.meter.record(self.service, start, step_ms / 1000, calls=0)  # steps are not calls: units = sim seconds
        return out

    async def close(self):
        if hasattr(self.inner, "close"):
            await self.inner.close()
