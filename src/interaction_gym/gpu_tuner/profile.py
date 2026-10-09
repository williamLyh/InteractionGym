"""Demand and supply profiles: what one episode asks of each service, and what one replica of each
service delivers at a given concurrency.

Units of work, per service:

- ``sim_s``   duplex agent: simulated seconds of the session stepped (one session per episode);
- ``call``    chat LLM: one completion (with the episode's typical prompt / completion sizes);
- ``audio_s`` TTS / voice clone: seconds of speech synthesized.

A ``Curve`` is a replica's speed as a function of concurrency ``c`` (in-flight streams on that
replica): ``unit_time(c)`` = wall seconds one stream needs per unit of work (per-call latency for
``call``, inverse real-time factor for ``audio_s``, wall seconds per simulated second for
``sim_s``). The replica's throughput is ``c / unit_time(c)`` units per wall second.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from statistics import mean

UNITS = {"agent": "sim_s", "llm": "call", "tts": "audio_s", "clone": "audio_s"}


# ---------------------------------------------------------------- demand


@dataclass
class ServiceDemand:
    unit: str
    units: float  # per episode
    calls: float = 0.0  # per episode
    prompt_tokens: float = 0.0  # per call
    completion_tokens: float = 0.0  # per call
    source: str = "measured"  # measured | estimated | synthetic

    @property
    def units_per_call(self) -> float:
        return self.units / self.calls if self.calls else 0.0


@dataclass
class DemandProfile:
    sim_s: float  # simulated seconds per episode
    services: dict[str, ServiceDemand]
    episodes: int = 1
    wall_s: float | None = None  # measured wall seconds per episode (at the concurrency it ran)
    concurrency: dict[str, dict] = field(default_factory=dict)  # observed in-flight (Meter.concurrency)
    notes: list[str] = field(default_factory=list)

    def per_minute(self) -> dict[str, dict]:
        """Demand per simulated minute of conversation."""
        k = 60 / self.sim_s if self.sim_s else 0.0
        return {s: {"unit": d.unit, "units": d.units * k, "calls": d.calls * k} for s, d in self.services.items()}

    # -- sources
    @classmethod
    def from_meter(cls, meter, units: dict[str, str] | None = None) -> DemandProfile:
        units = {**UNITS, **(units or {})}
        eps = list(meter.episodes) or [None]
        tot: dict[str, dict[str | None, list]] = defaultdict(lambda: defaultdict(list))
        for e in meter.events:
            tot[e.service][e.episode].append(e)
        services = {}
        for s, by_ep in tot.items():
            per = [by_ep.get(ep, []) for ep in eps]
            calls = [sum(e.calls for e in evs) for evs in per]
            all_evs = [e for evs in per for e in evs if e.calls]
            unit = units.get(s, "call")
            services[s] = ServiceDemand(
                unit=unit, units=mean(sum(e.units for e in evs) for evs in per),
                calls=1.0 if unit == "sim_s" else mean(calls),  # an agent: one session per episode
                prompt_tokens=mean(e.prompt_tokens for e in all_evs) if all_evs else 0.0,
                completion_tokens=mean(e.completion_tokens for e in all_evs) if all_evs else 0.0,
                source="estimated" if any(e.estimated_tokens for e in all_evs) and unit == "call" else "measured")
        recs = [meter.episodes[e] for e in eps if e is not None]
        sim = mean(r.sim_s for r in recs) if recs else services.get("agent", ServiceDemand("sim_s", 0)).units
        return cls(sim_s=sim, services=services, episodes=len(recs) or 1,
                   wall_s=mean(r.wall_s for r in recs) if recs else None,
                   concurrency={s: meter.concurrency(s) for s in services})

    @classmethod
    def from_episodes(cls, episodes: list[dict], traces: list[dict] | None = None, *, words_per_sec: float = 2.6,
                      prompt_tokens: float = 700, completion_tokens: float = 30) -> DemandProfile:
        """Estimate demand from saved trajectories (``episodes.jsonl`` + optional ``agent_traces.jsonl``).

        Exact: simulated seconds, agent units (from the trace), TTS calls and seconds (the user's
        audio turns; with a clone voice the first turn is TTS and later ones are cloned). Estimated:
        user-LLM calls — one per LLM-written user turn plus the listener's decisions (``meta.user.decisions.llm_calls``
        when recorded, else one per phrase boundary of the agent's speech once ``min_words`` have been heard, or
        every ``max_decision_gap_ms`` without one; episodes from before decision points: every ``check_ms``) — and
        their token sizes; measure them live (``Meter``) when it matters."""
        traces_by = {t["episode_id"]: t for t in traces or []}
        rows = []
        for ep in episodes:
            m, user = ep["meta"], ep["meta"].get("user", {})
            sim = m.get("duration_ms", 0) / 1000
            voice = user.get("voice", {})
            has_clone = "clone" in voice
            online = user.get("mode") == "online"
            tt = user.get("turn_taking", {})
            gap_ms = tt.get("max_decision_gap_ms")
            listening = user.get("listening") or {}
            min_words = ((listening.get("model") or user.get("barge_in") or {}).get("min_words")) or 5
            bi = user.get("barge_in") or {}
            has_listener = bool(listening.get("model")) if listening else bool(bi) and bi.get("type", "llm") == "llm"
            tts = clone = 0.0
            tts_calls = clone_calls = 0
            user_turns = [t for t in ep["turns"] if t["role"] == "user" and t.get("text")]
            for i, t in enumerate(user_turns):
                med = t.get("media") or {}
                dur = (med["end_ms"] - med["start_ms"]) / 1000 if "end_ms" in med else (t["end_time"] - t["start_time"]) / 1000
                if has_clone and i > 0:
                    clone, clone_calls = clone + dur, clone_calls + 1
                else:
                    tts, tts_calls = tts + dur, tts_calls + 1
            llm_calls = 0.0
            if online:
                scripted_first = 1 if (m.get("task", {}).get("scenario", {}).get("first_turn")) else 0
                llm_calls += max(0, len(user_turns) - scripted_first) + 1  # +1: the reply that ends the call
                if has_listener:
                    d = user.get("decisions") or {}
                    if "llm_calls" in d:  # recorded by the user simulator
                        llm_calls += d["llm_calls"]
                    else:
                        for t in ep["turns"]:
                            if t["role"] == "agent":
                                llm_calls += _decision_points(t, min_words, gap_ms, words_per_sec)
            tr = traces_by.get(m.get("episode_id"))
            units = len([u for u in tr["units"] if u.get("unit_index", 0) >= 0]) * tr["unit_ms"] / 1000 if tr and tr.get("unit_ms") else sim
            rows.append(dict(sim=sim, agent=units, llm=llm_calls, tts=tts, tts_calls=tts_calls, clone=clone,
                             clone_calls=clone_calls, wall=m.get("wall_s")))
        avg = lambda k: mean(r[k] for r in rows)  # noqa: E731
        services = {"agent": ServiceDemand("sim_s", avg("agent"), 1.0)}
        if avg("llm"):
            services["llm"] = ServiceDemand("call", avg("llm"), avg("llm"), prompt_tokens, completion_tokens, source="estimated")
        if avg("tts"):
            services["tts"] = ServiceDemand("audio_s", avg("tts"), avg("tts_calls"))
        if avg("clone"):
            services["clone"] = ServiceDemand("audio_s", avg("clone"), avg("clone_calls"))
        walls = [r["wall"] for r in rows if r["wall"]]
        return cls(sim_s=avg("sim"), services=services, episodes=len(rows), wall_s=mean(walls) if walls else None,
                   notes=["from saved episodes: user-LLM calls and tokens are estimates"])

    @classmethod
    def synthetic(cls) -> DemandProfile:
        """Illustrative numbers in the range of the MiniCPM-o suite (runs/suite): ~55 s calls, 3-4
        user turns, one TTS turn then cloned turns, ~10 barge-in checks."""
        return cls(sim_s=55.0, episodes=0, wall_s=None, notes=["synthetic demand (dry run)"], services={
            "agent": ServiceDemand("sim_s", 55.0, 1.0, source="synthetic"),
            "llm": ServiceDemand("call", 14.0, 14.0, 700, 25, source="synthetic"),
            "tts": ServiceDemand("audio_s", 2.5, 1.0, source="synthetic"),
            "clone": ServiceDemand("audio_s", 8.0, 3.0, source="synthetic"),
        })

    # -- io
    def to_json(self) -> dict:
        return {"sim_s": self.sim_s, "episodes": self.episodes, "wall_s": self.wall_s, "concurrency": self.concurrency,
                "notes": self.notes, "services": {k: asdict(v) for k, v in self.services.items()}}

    @classmethod
    def from_json(cls, d: dict) -> DemandProfile:
        return cls(sim_s=d["sim_s"], episodes=d.get("episodes", 1), wall_s=d.get("wall_s"), concurrency=d.get("concurrency", {}),
                   notes=d.get("notes", []), services={k: ServiceDemand(**v) for k, v in d["services"].items()})

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path) -> DemandProfile:
        with open(path) as f:
            return cls.from_json(json.load(f))


# ---------------------------------------------------------------- supply


@dataclass
class Curve:
    """One replica's speed vs concurrency: ``points`` = [(c, wall seconds per unit per stream)]."""

    unit: str
    points: list[tuple[float, float]]
    max_concurrency: int | None = None  # hard cap (duplex sessions per server)
    source: str = "measured"
    meta: dict = field(default_factory=dict)  # probe settings and raw stats

    def __post_init__(self):
        self.points = sorted((float(c), float(u)) for c, u in self.points)
        assert self.points and all(c > 0 and u > 0 for c, u in self.points), self.points

    def unit_time(self, c: float) -> float:
        """Linear between measured points; below the first, its value; past the last the replica is
        taken to be saturated (throughput flat, so per-stream time grows linearly with c)."""
        pts = self.points
        c = max(c, 0.0)
        if c <= pts[0][0]:
            return pts[0][1]
        for (c0, u0), (c1, u1) in zip(pts, pts[1:]):
            if c <= c1:
                return u0 + (u1 - u0) * (c - c0) / (c1 - c0)
        cl, ul = pts[-1]
        return ul * c / cl

    def throughput(self, c: float) -> float:
        return c / self.unit_time(c) if c > 0 else 0.0

    def peak(self) -> tuple[float, float]:
        """(concurrency, units per second) at the best measured point within the cap."""
        cands = [c for c, _ in self.points if self.max_concurrency is None or c <= self.max_concurrency]
        if self.max_concurrency is not None and self.max_concurrency not in cands:
            cands.append(float(self.max_concurrency))
        best = max(cands, key=self.throughput)
        return best, self.throughput(best)


@dataclass
class SupplyProfile:
    curves: dict[str, Curve]
    notes: list[str] = field(default_factory=list)

    @classmethod
    def synthetic(cls) -> SupplyProfile:
        """Illustrative per-replica curves for 8x RTX 5090: MiniCPM-o duplex (2 GPUs, lockstep ~4x real
        time alone, 4 sessions max), Qwen3.8-27B FP8 TP2 (~25-token replies), Qwen3-TTS 1.7B replicas."""
        return cls(notes=["synthetic supply (dry run)"], curves={
            "agent": Curve("sim_s", [(1, 0.24), (2, 0.28), (3, 0.34), (4, 0.42)], max_concurrency=4, source="synthetic"),
            "llm": Curve("call", [(1, 0.45), (2, 0.47), (4, 0.50), (8, 0.58), (16, 0.75), (32, 1.10), (64, 1.90)], source="synthetic"),
            "tts": Curve("audio_s", [(1, 0.35), (2, 0.40), (4, 0.55), (8, 0.95), (16, 1.80)], source="synthetic"),
            "clone": Curve("audio_s", [(1, 0.45), (2, 0.52), (4, 0.70), (8, 1.20), (16, 2.30)], source="synthetic"),
        })

    def merge(self, other: SupplyProfile) -> SupplyProfile:
        return SupplyProfile({**self.curves, **other.curves}, self.notes + other.notes)

    def to_json(self) -> dict:
        return {"notes": self.notes, "curves": {k: asdict(v) for k, v in self.curves.items()}}

    @classmethod
    def from_json(cls, d: dict) -> SupplyProfile:
        return cls({k: Curve(**v) for k, v in d["curves"].items()}, d.get("notes", []))

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)

    @classmethod
    def load(cls, path) -> SupplyProfile:
        with open(path) as f:
            return cls.from_json(json.load(f))


def _decision_points(turn: dict, min_words: int, gap_ms: int | None, words_per_sec: float) -> float:
    """How many listener decisions an agent turn gets: its phrase boundaries after ``min_words``, plus a fallback
    point every ``gap_ms`` without one (``gap_ms`` None: an episode from before decision points, polled every second)."""
    from ..core import Segment
    from ..user import n_words, phrase_boundaries

    dur = turn["end_time"] - turn["start_time"]
    if gap_ms is None:
        return max(0.0, (dur / 1000 - min_words / words_per_sec))
    seg = Segment(turn["id"], turn["start_time"], dur, turn.get("text", ""))
    pts, last = 0, seg.t0
    for b in phrase_boundaries(seg) + [seg.end]:
        while b - last > gap_ms:
            last += gap_ms
            pts += n_words(seg.heard_text(last)) >= min_words
        if b < seg.end:
            pts += n_words(seg.heard_text(b)) >= min_words
            last = b
    return float(pts)
