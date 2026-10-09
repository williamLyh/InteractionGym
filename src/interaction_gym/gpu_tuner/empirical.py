"""Running real end-to-end episodes at a given env concurrency, and a mock cluster to do it on a laptop.

An episode runner is any ``async (index) -> simulated ms``. ``run_batch`` keeps ``concurrency``
episodes in flight for a while and measures episodes/hour. ``MockCluster`` runs episodes against
fake services with configurable capacity (per-stream speed vs load, batch slots that queue beyond
capacity, a session cap for the duplex agent) and reports fake GPU utilisation / vLLM-style queue
metrics, so the measure → rebalance loop (tune.rebalance) is testable without GPUs.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .monitor import Sample
from .profile import Curve, DemandProfile, SupplyProfile

Runner = Callable[[int], Awaitable[int]]


@dataclass
class Level:
    concurrency: int
    episodes: int
    wall_s: float
    eph: float  # measured episodes/hour
    sim_s: float  # mean simulated seconds per episode
    errors: int = 0
    t0: float = 0.0  # perf_counter window of the batch (to cut monitor samples)
    t1: float = 0.0
    error_kinds: list[str] = field(default_factory=list)  # first line of each failed episode's exception
    aborted: bool = False  # stopped early: more than max(3, concurrency) episodes failed


async def run_batch(runner: Runner, concurrency: int, episodes: int | None = None, seconds: float | None = None,
                    start_index: int = 0, clock=time.perf_counter, scale: float = 1.0) -> Level:
    """``concurrency`` workers run episodes back to back until ``episodes`` have started (or ``seconds``
    have passed). Each worker is busy without a break, so the system's rate is the sum of the
    workers' rates (episodes / busy time each) — the drain at the end, when fewer than
    ``concurrency`` episodes are left running, does not dilute it. ``scale`` converts mock
    (time-compressed) wall seconds into real ones. Failed episodes are counted (with their error
    message) and lower the rate; after more than ``max(3, concurrency)`` failures the batch stops
    (``aborted``) instead of hammering a broken server."""
    assert episodes or seconds
    counter = itertools.count(start_index)
    started = 0
    sims: list[int] = []
    rates: list[float] = []
    errors = 0
    kinds: list[str] = []
    t0 = clock()

    async def worker():
        nonlocal started, errors
        n, last = 0, t0
        while True:
            if episodes is not None and started >= episodes:
                break
            if seconds is not None and clock() - t0 >= seconds:
                break
            if errors > max(3, concurrency):
                break
            started += 1
            try:
                sims.append(await runner(next(counter)))
                n, last = n + 1, clock()
            except Exception as e:  # noqa: BLE001 — a failed episode lowers the measured rate
                errors += 1
                kinds.append(f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"[:200])
        if n:
            rates.append(n / ((last - t0) * scale))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    t1 = clock()
    return Level(concurrency, len(sims), (t1 - t0) * scale, sum(rates) * 3600, sum(sims) / len(sims) / 1000 if sims else 0.0,
                 errors, t0, t1, kinds, errors > max(3, concurrency))


# ---------------------------------------------------------------- mock cluster


@dataclass
class _Pool:
    """One service: ``replicas`` (effective, after co-location slowdown) each serving up to ``slots``
    streams at the speed ``curve`` gives for its load; streams beyond that queue."""

    curve: Curve
    replicas: float
    slots: int
    inflight: int = 0
    waiting: int = 0
    _sem: asyncio.Semaphore | None = None

    def __post_init__(self):
        self._sem = asyncio.Semaphore(max(1, round(self.replicas * self.slots)))

    async def do(self, units: float, time_scale: float) -> None:
        self.waiting += 1
        async with self._sem:
            self.waiting -= 1
            self.inflight += 1
            try:
                await asyncio.sleep(units * self.curve.unit_time(self.inflight / self.replicas) * time_scale)
            finally:
                self.inflight -= 1

    def util(self) -> float:
        """Fraction of the replicas' peak throughput in use (stands in for GPU utilisation)."""
        if not self.inflight:
            return 0.0
        _, peak = self.curve.peak()
        return min(1.0, self.curve.throughput(self.inflight / self.replicas) / peak)


@dataclass
class MockCluster:
    """Fake services for a layout (``start`` / ``apply_layout``): each episode alternates agent compute
    (in ``steps`` pieces) with the user-LLM / TTS / clone work its demand profile asks for; ``supply``
    gives each fake service's per-replica speed, ``slots`` the streams per replica beyond which
    requests queue (agent: its session cap; a layout's ``caps`` override it). Records the env-side
    waits into a ``Meter`` and reports ``Sample``s (GPU util and memory per GPU, running / waiting
    per service) like the real monitor. ``mem_gb`` = a service's memory per GPU at start (default:
    its spec's, or 90 % of the GPU), ``session_mem_gb`` = what each open session adds on every GPU
    of its replica (the Talker/Code2Wav state that grows with sessions). ``time_scale`` compresses
    time (0.002: a 30 s episode takes 60 ms)."""

    demand: DemandProfile
    supply: SupplyProfile
    specs: dict
    hw: object
    time_scale: float = 0.002
    steps: int = 6
    overhead_s: float = 3.0
    slots: dict[str, int] = field(default_factory=lambda: {"agent": 4, "llm": 64, "tts": 16, "clone": 16})
    mem_gb: dict[str, float] = field(default_factory=dict)
    session_mem_gb: float = 0.0
    applied: list[dict] = field(default_factory=list)
    applied_caps: list[dict] = field(default_factory=list)
    pools: dict[str, _Pool] = field(default_factory=dict)
    placement: dict[str, list[list[int]]] = field(default_factory=dict)
    sessions: asyncio.Semaphore | None = None
    open_sessions: int = 0
    caps: dict[str, int] = field(default_factory=dict)

    def start(self, placement: dict[str, list[list[int]]], caps: dict[str, int] | None = None) -> None:
        from .solve import effective_replicas

        self.placement = {s: r for s, r in placement.items() if r}
        self.caps = dict(caps or {})
        slots = {**self.slots, **self.caps}
        eff = effective_replicas(self.placement, self.specs, self.hw)
        self.pools = {s: _Pool(self.supply.curves[s], eff[s], slots.get(s, 16))
                      for s in self.demand.services if s in self.supply.curves and s in eff}
        # a duplex session is held for the whole episode (also while it waits on the user simulator)
        self.sessions = asyncio.Semaphore(len(self.placement.get("agent", [])) * slots.get("agent", 4) or 10**6)
        self.open_sessions = 0
        self.applied.append({s: len(r) for s, r in self.placement.items()})
        self.applied_caps.append(dict(self.caps))

    async def apply_layout(self, layout) -> None:
        await asyncio.sleep(0)
        self.start(layout.placement, getattr(layout, "caps", None))

    def runner(self, meter=None) -> Runner:
        async def episode(i: int) -> int:
            d = self.demand.services
            async with self.sessions:
                self.open_sessions += 1
                try:
                    return await body(i, d)
                finally:
                    self.open_sessions -= 1

        async def body(i: int, d) -> int:
            tok = meter.start_episode(f"mock-{i}") if meter is not None else None
            await asyncio.sleep(self.overhead_s * self.time_scale)
            for _ in range(self.steps):
                for s, x in d.items():
                    if s in self.pools:
                        t = time.perf_counter()
                        await self.pools[s].do(x.units / self.steps, self.time_scale)
                        if meter is not None:
                            meter.record(s, t, x.units / self.steps, calls=0 if s == "agent" else max(1, round(x.calls / self.steps)))
            if meter is not None:
                meter.end_episode(round(self.demand.sim_s * 1000), tok)
            return round(self.demand.sim_s * 1000)

        return episode

    def sample(self) -> Sample:
        s = Sample(time.perf_counter())
        gpu: dict[int, float] = {g: 0.0 for g in self.hw.gpus}
        mem: dict[int, float] = {g: 0.0 for g in self.hw.gpus}
        for svc, reps in self.placement.items():
            pool = self.pools.get(svc)
            sp = self.specs[svc]
            per_rep_sessions = self.open_sessions / len(reps) if svc == "agent" and reps else 0.0
            for r in reps:
                for g in r:
                    gpu[g] = min(1.0, gpu[g] + (pool.util() if pool else 0.0))
                    base = self.mem_gb.get(svc, sp.mem_gb or self.hw.mem_gb * 0.9)
                    mem[g] += (base + self.session_mem_gb * per_rep_sessions) * 1024
            if pool:
                s.metrics[svc] = {"running": float(pool.inflight), "waiting": float(pool.waiting)}
        s.gpu = {g: (gpu[g], mem[g]) for g in gpu}
        return s


def mock_truth(supply: SupplyProfile, factor: dict[str, float]) -> SupplyProfile:
    """A copy of ``supply`` with some services slower by ``factor`` (the 'real world' in a dry run)."""
    return SupplyProfile({s: Curve(c.unit, [(x, u * factor.get(s, 1.0)) for x, u in c.points], c.max_concurrency, c.source, c.meta)
                          for s, c in supply.curves.items()}, supply.notes + [f"mock truth: {factor}"])


# A duplex agent bound by its per-unit pipeline rather than compute (as MiniCPM-o on an 8x RTX 5090 host: all sessions taken,
# GPUs ~35 % busy): per-session speed degrades slowly with more sessions on a server, so raising the session cap
# raises throughput, sublinearly.
PIPELINE_AGENT = Curve("sim_s", [(1, 0.24), (2, 0.26), (4, 0.30), (8, 0.40), (12, 0.55), (16, 0.75)], source="synthetic")


def session_bound_mock(specs, hw, *, time_scale: float = 0.002, agent_mem_gb: float = 24.0, session_mem_gb: float = 0.6) -> MockCluster:
    """A mock cluster whose agent is session-bound with low GPU util (``PIPELINE_AGENT``) and whose agent GPUs fill
    up by ``session_mem_gb`` per open session (24 GB + 0.6 GB x sessions: 92 % of 32 GB is passed above ~9 sessions
    per server)."""
    from dataclasses import replace

    sup = SupplyProfile.synthetic()
    truth = SupplyProfile({**sup.curves, "agent": PIPELINE_AGENT}, sup.notes + ["pipeline-bound agent"])
    return MockCluster(DemandProfile.synthetic(), mock_truth(truth, {"llm": 1.1}), specs, replace(hw, colocation_penalty=1.0),
                       time_scale=time_scale, mem_gb={"agent": agent_mem_gb}, session_mem_gb=session_mem_gb)
