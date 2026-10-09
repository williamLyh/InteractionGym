"""Core abstractions: Frame, Segment, Task, Node, Sim, Env.

Everything is a stream of timestamped frames; every participant is a node with
explicit state; the Sim advances simulated time event by event; outputs that
take time (an utterance, an action chunk) are Segments that can be cut.
"""

from __future__ import annotations

import asyncio
import copy
import heapq
import itertools
import random
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from typing import Any

REWARD = "reward"
SESSION = "session"


@dataclass(frozen=True)
class Frame:
    """A timestamped piece of data on a named stream. Times are integer ms.

    A frame whose ``t`` lies in the future is delivered when the clock reaches
    it; this is how reaction time, tool latency and inference latency are
    expressed.
    """

    stream: str
    t: int
    data: Any = None
    dur: int = 0
    src: str = ""


@dataclass(frozen=True)
class Segment:
    """Something that plays out over time: an utterance, audio, an action chunk.

    ``data`` is any sliceable sequence (text, audio samples, actions); playback
    maps elapsed time linearly onto it. ``text`` is an optional transcript
    aligned the same way (e.g. for audio data), so listeners can read what has
    been said so far. A segment is emitted once; an interruption is expressed by
    emitting ``cut(t)`` under the same id.
    """

    id: str
    t0: int
    dur: int
    data: Any
    text: str | None = None
    kind: str | None = None  # set by the producer: None (a normal turn) | "backchannel" | "aside" | "noise" | "away"
    expects: str | None = None  # set by the producer when it differs from the kind's default (docs/FORMAT.md §4.4)
    intent: str | None = None  # a barge-in's purpose, set by the producer: "correction" | "question" | "stop" | "other"
    label: str | None = None  # what a sound is, set by the producer (e.g. "door_slam" for a noise, "called_away")

    @property
    def end(self) -> int:
        return self.t0 + self.dur

    def _index(self, n: int, t: int) -> int:
        if self.dur <= 0:
            return n if t >= self.t0 else 0
        frac = min(max((t - self.t0) / self.dur, 0.0), 1.0)
        return round(n * frac)

    def play(self, a: int, b: int):
        """Portion of ``data`` played during [a, b)."""
        n = len(self.data)
        return self.data[self._index(n, a) : self._index(n, b)]

    def heard(self, t: int):
        """Prefix of ``data`` played by time t."""
        return self.data[: self._index(len(self.data), t)]

    @property
    def transcript(self) -> str:
        return self.text if self.text is not None else self.data if isinstance(self.data, str) else ""

    def heard_text(self, t: int) -> str:
        """Prefix of the transcript spoken by time t."""
        s = self.transcript
        return s[: self._index(len(s), t)]

    def active(self, t: int) -> bool:
        return self.t0 <= t < self.end

    def cut(self, t: int) -> Segment:
        """Truncate at t: keep what already played, drop the rest."""
        t = min(max(t, self.t0), self.end)
        text = None if self.text is None else self.heard_text(t)
        return replace(self, dur=t - self.t0, data=self.heard(t), text=text)


@dataclass(frozen=True)
class Chunk:
    """One step's worth of a segment, as exchanged with the agent.

    Observations: the portion of a segment played during the step window (the
    agent never sees what has not been played yet); ``last`` marks that the
    segment ended in this window (finished or cut). Actions: a frame-synchronous
    agent sends one chunk per step; consecutive chunks with the same ``id`` are
    appended into one growing segment.
    """

    id: str
    t0: int
    dur: int
    data: Any
    text: str = ""
    first: bool = False
    last: bool = False


@dataclass(frozen=True)
class Session:
    """What the agent is told about its environment: sent on the ``session`` stream at t=0 and
    again whenever it changes (e.g. after tools are loaded). Never contains world state —
    only instructions, callable tools (or a catalog to discover them) and the context the
    task explicitly grants (``task.scenario["agent_context"]``)."""

    mode: str = "all"  # "all": every tool up front | "discover": a catalog + meta-tools
    instructions: str = ""
    tools: tuple = ()  # ToolSpec
    services: tuple = ()  # discover mode: catalog entries {"service", "description", "tools"}
    context: dict = field(default_factory=dict)


@dataclass
class Task:
    """Scenario, initial world state and evaluation criteria (τ-bench shaped)."""

    id: str = ""
    scenario: dict = field(default_factory=dict)
    initial_state: dict = field(default_factory=dict)
    criteria: dict = field(default_factory=dict)


class Node:
    """A participant. The node object holds config; everything mutable lives in ``state``.

    ``step`` is called only when the node has new input or its wake time has
    come. It returns the frames it emits and optionally the next time it wants
    to be woken (``None`` keeps any pending wake). State is mutated in place.
    """

    reads: tuple[str, ...] = ()
    batched: bool = False  # reserved for lockstep mode (batched physics), not used yet

    def init_state(self, task: Task, rng: random.Random) -> Any:
        return None

    async def step(self, state: Any, t: int, inbox: list[Frame]) -> tuple[list[Frame], int | None]:
        return [], None

    def fork(self, state: Any, n: int) -> list[Any]:
        return [copy.deepcopy(state) for _ in range(n)]


def matches(stream: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(stream, p) for p in patterns)


class Sim:
    """Discrete-event simulation of a node graph for one episode.

    The clock jumps from event to event. At each instant, due frames are routed
    to the nodes that read their stream (never back to their source), then every
    node with input or a due wake is stepped; frames they emit for the same
    instant are delivered in the next round of that instant.
    """

    def __init__(self, nodes: dict[str, Node], task: Task, seed: int = 0, log: tuple[str, ...] = ("*",)):
        self.nodes = nodes
        self.task = task
        self.log_patterns = log
        self.t = 0
        self.log: list[Frame] = []
        self.reward = 0.0
        self.busy_until = 0  # when the last delivered activity ends (a segment's end, or a frame's time)
        self._ends: dict[tuple[str, str], int] = {}  # each segment's latest version's end (a cut moves it back)
        self._last_t = 0
        rng = random.Random(seed)
        self.states = {name: node.init_state(task, random.Random(rng.getrandbits(64))) for name, node in nodes.items()}
        self._queue: list[tuple[int, int, Frame]] = []
        self._seq = itertools.count()
        self._wakes: dict[str, int | None] = dict.fromkeys(nodes, 0)  # every node runs once at t=0

    def put(self, frames: list[Frame], src: str = "env") -> None:
        for f in frames:
            if not f.src:
                f = replace(f, src=src)
            if f.t < self.t:
                f = replace(f, t=self.t)
            heapq.heappush(self._queue, (f.t, next(self._seq), f))

    def _next_time(self) -> int | None:
        times = [w for w in self._wakes.values() if w is not None]
        if self._queue:
            times.append(self._queue[0][0])
        return min(times) if times else None

    def idle(self) -> bool:
        """Nothing scheduled: no queued frames and no node waiting to wake."""
        return not self._queue and all(w is None for w in self._wakes.values())

    async def advance_to(self, t_end: int) -> None:
        while True:
            t = self._next_time()
            if t is None or t > t_end:
                break
            self.t = t
            await self._run_instant(t)
        self.t = max(self.t, t_end)

    async def _run_instant(self, t: int) -> None:
        while True:
            due = []
            while self._queue and self._queue[0][0] <= t:
                due.append(heapq.heappop(self._queue)[2])
            woken = {name for name, w in self._wakes.items() if w is not None and w <= t}
            if not due and not woken:
                return
            for name in woken:
                self._wakes[name] = None

            inboxes: dict[str, list[Frame]] = {name: [] for name in self.nodes}
            for f in due:
                self._record(f)
                for name, node in self.nodes.items():
                    if name != f.src and matches(f.stream, node.reads):
                        inboxes[name].append(f)

            to_step = [name for name in self.nodes if inboxes[name] or name in woken]
            results = await asyncio.gather(
                *(self.nodes[name].step(self.states[name], t, inboxes[name]) for name in to_step)
            )
            for name, (frames, wake) in zip(to_step, results):
                self.put(frames, src=name)
                if wake is not None:
                    self._wakes[name] = max(wake, t)

    def _record(self, f: Frame) -> None:
        if f.stream == REWARD:
            self.reward += f.data
        self._last_t = max(self._last_t, f.t)
        if isinstance(f.data, Segment):
            self._ends[(f.stream, f.data.id)] = f.data.end
        self.busy_until = max(self._last_t, max(self._ends.values(), default=0))
        if matches(f.stream, self.log_patterns):
            self.log.append(f)

    def fork(self, n: int) -> list[Sim]:
        states = {name: node.fork(self.states[name], n) for name, node in self.nodes.items()}
        sims = []
        for i in range(n):
            s = copy.copy(self)
            s.states = {name: states[name][i] for name in self.nodes}
            s.log = list(self.log)
            s._queue = list(self._queue)
            s._wakes = dict(self._wakes)
            s._ends = dict(self._ends)
            s._seq = itertools.count(next(self._seq))
            sims.append(s)
        return sims


@dataclass(frozen=True)
class AgentSpec:
    """How the agent session talks to the env. ``chunk_ms`` is the session's update
    granularity (the model's frame / chunk length: 80, 160, 200, 240 ms, 1 s, ...).

    With ``audio`` set (e.g. ``"user.audio"``), every step also delivers one continuous
    audio frame on that stream: the audio segments on ``obs`` streams playing in the
    window, mixed with the env's background tracks — like a real microphone, which
    never goes silent.
    """

    chunk_ms: int = 200
    obs: tuple[str, ...] = ("user.speech",)  # what the user says out loud — not, e.g., its own tool calls
    out: str = "policy.speech"
    audio: str | None = None
    sr: int = 16000


@dataclass(frozen=True)
class Background:
    """A track laid under the whole episode (or ``start_time``–``end_time``), starting
    ``offset_ms`` into ``audio``. ``spec`` describes what it is (type, level, source; see
    ``soundscape.background_spec``) for the trajectory and the viewer."""

    audio: Any  # interaction_gym.audio.Audio
    gain_db: float = 0.0
    loop: bool = False
    offset_ms: int = 0
    start_time: int = 0
    end_time: int | None = None
    spec: dict | None = None

    def slice(self, a: int, b: int):
        """Samples (list of int) covering timeline window [a, b), zero outside the track."""
        sr, n = self.audio.sr, round((b - a) * self.audio.sr / 1000)
        out = [0] * n
        src = self.audio.samples
        g = 10 ** (self.gain_db / 20)
        for i in range(n):
            t = a + i * 1000 / sr
            if t < self.start_time or (self.end_time is not None and t >= self.end_time):
                continue
            j = round((t - self.start_time + self.offset_ms) * sr / 1000)
            if self.loop:
                j %= len(src)
            if 0 <= j < len(src):
                out[i] = src[j] * g
        return out


class _PolicyPort(Node):
    """The env side of the agent session: collects what the agent may observe and
    assembles the agent's streamed output chunks into segments. Lives inside the
    Sim so that its state forks with the episode."""

    def __init__(self, reads: tuple[str, ...]):
        self.reads = reads

    def init_state(self, task, rng):
        return {"raw": [], "segs": {}, "said": {}, "t_prev": 0, "out": {}, "session": Session()}

    async def step(self, state, t, inbox):
        for f in inbox:
            if f.stream == SESSION:  # merge partial updates into the full session the agent sees
                upd = f.data if isinstance(f.data, dict) else {k: getattr(f.data, k) for k in Session.__dataclass_fields__}
                upd = {k: tuple(v) if isinstance(v, list) else v for k, v in upd.items()}
                if "context" in upd:  # revealed context accumulates
                    upd["context"] = {**state["session"].context, **upd["context"]}
                state["session"] = replace(state["session"], **upd)
                f = replace(f, data=state["session"])
            state["raw"].append(f)
            if isinstance(f.data, Segment):
                state["segs"][(f.stream, f.data.id)] = f.data
        return [], None


def _mix(env: "Env", segs, a: int, b: int):
    """Mixed microphone audio for [a, b): playing audio segments + background tracks."""
    from .audio import Audio  # core stays free of media dependencies unless audio is used
    from array import array

    sr = env.agent.sr
    n = round((b - a) * sr / 1000)
    acc = [0.0] * n
    for seg in segs:
        if not isinstance(seg.data, Audio) or seg.end <= a or seg.t0 >= b:
            continue
        part = seg.play(max(a, seg.t0), min(b, seg.end))
        part = (part if part.sr == sr else part.resample(sr)).samples  # e.g. a 48 kHz voice into a 24 kHz mic
        off = round((max(a, seg.t0) - a) * sr / 1000)
        for i, v in enumerate(part[: n - off]):
            acc[off + i] += v
    for bg in env.background:
        for i, v in enumerate(bg.slice(a, b)):
            acc[i] += v
    return Audio(array("h", (max(-32768, min(32767, int(v))) for v in acc)), sr)


def _chunk(seg: Segment, a: int, b: int, delivered: int = 0) -> Chunk:
    """The part of ``seg`` played during [a, b); its text is whatever of the transcript has been heard
    by ``b`` beyond the ``delivered`` characters already sent (so a cut, which re-times the segment,
    never drops or repeats characters)."""
    a, b = max(a, seg.t0), min(b, seg.end)
    a = min(a, b)
    text = seg.heard_text(b)[delivered:]
    return Chunk(seg.id, a, b - a, seg.play(a, b), text, first=a == seg.t0, last=b == seg.end)


class Env:
    """RL-facing wrapper. The agent lives outside and is driven every ``agent.chunk_ms``.

    Observations are streamed: segments on the agent's ``obs`` streams arrive as
    ``Chunk``s covering exactly what was played during the step window, so the
    agent never sees future content (``peek=True`` passes raw frames instead, for
    scripted debugging agents). Other frames pass through as they are.

    Actions are frames on ``agent.out`` whose data is either a whole ``Segment``
    (event-driven models) or a ``Chunk`` (frame-synchronous models; chunks with the
    same id grow one segment). A frame's ``t`` may lie in the future to model
    inference latency.
    """

    POLICY = "policy"

    def __init__(
        self,
        nodes: dict[str, Node],
        agent: AgentSpec = AgentSpec(),
        *,
        log: tuple[str, ...] = ("*",),
        max_ms: int | None = None,
        end_idle_ms: int = 3000,
        peek: bool = False,
        background: list[Background] = (),
    ):
        self.nodes = {**nodes, self.POLICY: _PolicyPort(tuple(agent.obs) + (SESSION,))}
        self.agent = agent
        # mixed into the microphone at the agent's rate; nodes may add their own per episode (``background``, at reset)
        self.fixed_background = [self._at_rate(bg) for bg in background]
        self.background = list(self.fixed_background)
        self.background_specs: list[dict] = []  # what the episode's background is (also for silence: nothing to mix)
        self.log_patterns = log
        self.max_ms = max_ms
        self.end_idle_ms = end_idle_ms
        self.peek = peek
        self.sim: Sim | None = None
        self.seed: int | None = None

    def _at_rate(self, bg: Background) -> Background:
        return bg if bg.audio.sr == self.agent.sr else replace(bg, audio=bg.audio.resample(self.agent.sr))

    @property
    def chunk_ms(self) -> int:
        return self.agent.chunk_ms

    @property
    def t(self) -> int:
        return self.sim.t

    @property
    def log(self) -> list[Frame]:
        return self.sim.log

    def _drain(self) -> list[Frame]:
        st, t = self.sim.states[self.POLICY], self.sim.t
        raw, st["raw"] = st["raw"], []
        if self.peek:
            return raw
        out = [f for f in raw if not isinstance(f.data, Segment)]
        if self.agent.audio and t > st["t_prev"]:
            mic = _mix(self, st["segs"].values(), st["t_prev"], t)
            out.append(Frame(self.agent.audio, st["t_prev"], mic, dur=t - st["t_prev"], src=self.POLICY))
        for key, seg in list(st["segs"].items()):
            a, b = max(st["t_prev"], seg.t0), min(t, seg.end)
            ended = seg.end <= t  # finished or cut by now: its last chunk goes out (empty if nothing new played)
            if (b > a or ended) and seg.t0 <= t:
                c = _chunk(seg, a, b, st["said"].get(key, 0))
                st["said"][key] = st["said"].get(key, 0) + len(c.text)
                out.append(Frame(key[0], c.t0, c, dur=c.dur, src=self.POLICY))
            if ended:
                del st["segs"][key]
                st["said"].pop(key, None)
        st["t_prev"] = t
        return sorted(out, key=lambda f: f.t)

    def _assemble(self, action: list[Frame]) -> list[Frame]:
        """Turn streamed output chunks into growing segments."""
        grown = self.sim.states[self.POLICY]["out"]
        frames = []
        for f in action:
            c = f.data
            if isinstance(c, Chunk):
                prev = grown.get(c.id)
                if prev is None:
                    seg = Segment(c.id, c.t0, c.dur, c.data, c.text)
                else:
                    seg = replace(prev, dur=c.t0 + c.dur - prev.t0, data=prev.data + c.data, text=(prev.text or "") + c.text)
                grown[c.id] = seg
                f = replace(f, t=c.t0, data=seg)
            frames.append(f)
        return frames

    @property
    def truncated(self) -> bool:
        """The episode hit ``max_ms`` (anything still playing was cut there)."""
        return self.max_ms is not None and self.sim.t >= self.max_ms

    @property
    def done(self) -> bool:
        """The conversation ended by itself: nothing is scheduled, everything that was being said
        has been said, and ``end_idle_ms`` passed without new activity. Nobody is cut off; only
        ``max_ms`` truncates."""
        sim = self.sim
        return self.truncated or (sim.idle() and sim.t - sim.busy_until >= self.end_idle_ms)

    def initial_session(self, task: Task) -> Session:
        """Merged from every node that contributes one (e.g. ToolWorld) plus the task's agent context."""
        parts = [node.session(task, self.sim.states[name]) for name, node in self.nodes.items() if callable(getattr(node, "session", None))]
        modes = [p["mode"] for p in parts if p.get("mode")]
        return Session(
            mode="discover" if "discover" in modes else "all",
            instructions="\n\n".join(p["instructions"] for p in parts if p.get("instructions")),
            tools=tuple(t for p in parts for t in p.get("tools", ())),
            services=tuple(x for p in parts for x in p.get("services", ())),
            context={**task.scenario.get("agent_context", {}), **{k: v for p in parts for k, v in p.get("context", {}).items()}},
        )

    def _episode_background(self, task: Task, seed: int) -> None:
        """The fixed tracks plus those the nodes contribute for this task (``node.background(task, seed, sr)`` →
        ``(specs, tracks)``, e.g. the simulated user's surroundings)."""
        self.background, self.background_specs = list(self.fixed_background), [bg.spec for bg in self.fixed_background if bg.spec]
        for node in self.nodes.values():
            if callable(getattr(node, "background", None)):
                specs, tracks = node.background(task, seed, self.agent.sr)
                self.background_specs += specs
                self.background += [self._at_rate(bg) for bg in tracks]

    async def reset(self, task: Task, seed: int = 0) -> list[Frame]:
        self.seed = seed
        self._episode_background(task, seed)
        self.sim = Sim(self.nodes, task, seed, self.log_patterns)
        self.sim.put([Frame(SESSION, 0, self.initial_session(task))], src="env")
        await self.sim.advance_to(0)
        return self._drain()

    async def step(self, action: list[Frame]) -> tuple[list[Frame], float, bool]:
        self.sim.put(self._assemble(action), src=self.POLICY)
        r0 = self.sim.reward
        t_end = self.sim.t + self.agent.chunk_ms
        await self.sim.advance_to(t_end if self.max_ms is None else min(t_end, self.max_ms))  # never past max_ms
        return self._drain(), self.sim.reward - r0, self.done

    def fork(self, n: int) -> list[Env]:
        envs = []
        for sim in self.sim.fork(n):
            e = copy.copy(self)
            e.sim = sim
            envs.append(e)
        return envs


class VecEnv:
    """Async mode: episodes advance independently and concurrently."""

    def __init__(self, envs: list[Env]):
        self.envs = envs

    async def reset(self, tasks: list[Task], seeds: list[int]) -> list[list[Frame]]:
        return await asyncio.gather(*(e.reset(task, s) for e, task, s in zip(self.envs, tasks, seeds)))

    async def step(self, actions: list[list[Frame]]) -> list[tuple[list[Frame], float, bool]]:
        return await asyncio.gather(*(e.step(a) for e, a in zip(self.envs, actions)))
