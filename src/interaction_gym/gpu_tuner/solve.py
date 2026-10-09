"""Allocate GPUs to service replicas so that episodes/hour is as high as it can be.

Model (one episode = one agent session, stepped in lockstep with the user simulator):

- An episode's wall time is ``T = overhead + sum_s W_s``: the agent's compute for every step plus
  every user-LLM / TTS / clone call the env awaits in between (lockstep coupling: an episode waits
  on whichever call is in flight, so per-call *latency* matters, not just throughput). In the
  realtime clock the agent's share is at least the simulated duration.
- ``W_s = demand_s * unit_time_s(c_s)``: the service's work per episode times its per-stream speed
  at ``c_s`` in-flight streams per (effective) replica.
- With ``N`` episodes running at once, Little's law gives ``c_s = N * (W_s / T) / R_s``; solved as a
  fixed point together with ``T``. Throughput ``X = N / T``.
- A duplex agent holds a session for the whole episode (also while the episode waits on TTS), so
  ``N <= agent replicas * max_sessions`` — the usual reason extra LLM GPUs sit idle.

The solver enumerates replica counts per service (whole-GPU services take aligned GPU blocks,
fractional ones are packed by memory, spread over the free GPUs), picks the best ``N`` for each, and
returns the allocation with the highest predicted episodes/hour (ties: fewer GPUs, then smaller N).
Also reported: each service's capacity bound ``R_s * peak_s / demand_s`` — the plain
"min over services of supply/demand" figure, reachable only with enough concurrency.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

from .profile import Curve, DemandProfile, SupplyProfile


@dataclass
class ServiceSpec:
    name: str
    gpus: int = 1  # whole GPUs per replica; 0 = a fractional replica packed by mem_gb
    mem_gb: float = 0.0  # fractional replicas: memory each needs on a shared GPU
    max_sessions: int | None = None  # session-bound services (the duplex agent): default sessions per replica
    cap: int | None = None  # per-replica concurrency limit passed to the launch command as {cap} (agent: max_sessions)
    cap_steps: list[int] | None = None  # values the tuner may raise the cap through, one step per round (e.g. 4,6,8,12)
    min_replicas: int = 0
    max_replicas: int = 8
    max_per_gpu: int | None = None  # fractional: at most this many replicas of this service per GPU
    spare_gb: float = 0.0  # whole-GPU replicas: memory left on each of their GPUs for fractional ones
    reserve: bool = False  # not a served dependency (e.g. a trainer): its replicas are fixed, no demand
    gpu_ids: list[int] | None = None  # pin to these GPUs
    pool: str | None = None  # "dp": one server, replicas data-parallel; "proxy": replicas behind a balancer; None: one endpoint each
    port: int = 0
    backend_port: int = 0  # proxy pools: first replica port
    url: str = ""  # client URL template, "{port}"
    model: str = ""
    launch: str = ""  # shell command template: {gpus} {gpu} {port} {dp}
    proxy_launch: str = ""  # proxy pools: {port} {backends}

    @property
    def fractional(self) -> bool:
        return self.gpus == 0

    @property
    def default_cap(self) -> int | None:
        return self.cap if self.cap is not None else self.max_sessions


@dataclass
class Hardware:
    gpus: list[int]
    mem_gb: float = 32.0
    colocation_penalty: float = 0.5  # each extra replica sharing a GPU slows every replica on it by this factor
    name: str = "host"
    workdir: str = "."
    session_prefix: str = "dig_"
    env: str = ""  # shell lines every service needs (sourced before its launch command)


@dataclass
class Row:
    service: str
    replicas: int
    gpus: list[list[int]]
    effective: float  # replicas after co-location slowdown
    inflight: float  # per effective replica
    wait_s: float  # wall seconds per episode spent in this service
    util: float  # fraction of the replicas' peak throughput used
    capacity_eph: float  # episodes/hour this service alone could sustain at peak


@dataclass
class Plan:
    alloc: dict[str, int]
    placement: dict[str, list[list[int]]]
    concurrency: int  # episodes in flight
    eph: float  # predicted episodes/hour
    episode_wall_s: float
    rows: list[Row] = field(default_factory=list)
    idle_gpus: list[int] = field(default_factory=list)
    clock: str = "input"
    caps: dict[str, int] = field(default_factory=dict)  # per-replica concurrency limits that differ from the specs' defaults

    @property
    def bottleneck(self) -> str:
        return max(self.rows, key=lambda r: r.util).service if self.rows else ""

    @property
    def gpus_used(self) -> int:
        return len({g for reps in self.placement.values() for r in reps for g in r})


# ---------------------------------------------------------------- placement


def _blocks(free: list[int], order: list[int], n: int) -> list[int] | None:
    """n GPUs for one replica: an aligned block of consecutive ids if possible, else any consecutive, else any."""
    pos = {g: i for i, g in enumerate(order)}
    fs = sorted(free, key=pos.get)
    for aligned in (True, False):
        for i, g in enumerate(fs):
            blk = fs[i:i + n]
            if len(blk) < n:
                break
            if all(pos[b] == pos[g] + k for k, b in enumerate(blk)) and (not aligned or pos[g] % n == 0):
                return blk
    return fs[:n] if len(fs) >= n else None


def place(alloc: dict[str, int], specs: dict[str, ServiceSpec], hw: Hardware) -> dict[str, list[list[int]]] | None:
    """GPU ids for every replica, or None if the allocation does not fit."""
    free = list(hw.gpus)
    out: dict[str, list[list[int]]] = {s: [] for s in alloc}
    shared: dict[int, float] = {}  # GPU -> memory left for fractional replicas
    count: dict[tuple[int, str], int] = {}
    whole = sorted((s for s in alloc if not specs[s].fractional), key=lambda s: (specs[s].gpu_ids is None, -specs[s].gpus))
    for s in whole:
        sp = specs[s]
        for _ in range(alloc[s]):
            if sp.gpu_ids is not None:
                blk = [g for g in sp.gpu_ids if g in free][: sp.gpus]
                blk = blk if len(blk) == sp.gpus else None
            else:
                blk = _blocks(free, hw.gpus, sp.gpus)
            if blk is None:
                return None
            for g in blk:
                free.remove(g)
                if sp.spare_gb > 0:
                    shared[g] = sp.spare_gb
            out[s].append(blk)
    frac = sorted(((s, i) for s in alloc if specs[s].fractional for i in range(alloc[s])), key=lambda x: -specs[x[0]].mem_gb)
    # round-robin across services so replicas of one service land on different GPUs
    frac = sorted(frac, key=lambda x: x[1])
    for s, _ in frac:
        sp = specs[s]
        ok = lambda g: shared[g] >= sp.mem_gb and (sp.max_per_gpu is None or count.get((g, s), 0) < sp.max_per_gpu)  # noqa: E731
        allowed_free = [g for g in free if sp.gpu_ids is None or g in sp.gpu_ids]
        if allowed_free and sp.mem_gb <= hw.mem_gb:  # spread: a fresh GPU while any is free
            g = allowed_free[0]
            free.remove(g)
            shared[g] = hw.mem_gb
        else:
            cands = [g for g in shared if ok(g) and (sp.gpu_ids is None or g in sp.gpu_ids)]
            if not cands:
                return None
            # replicas of one service on different GPUs first, then the least shared GPU
            g = min(cands, key=lambda g: (count.get((g, s), 0), sum(n for (h, _), n in count.items() if h == g), -shared[g]))
        shared[g] -= sp.mem_gb
        count[(g, s)] = count.get((g, s), 0) + 1
        out[s].append([g])
    return out


def effective_replicas(placement: dict[str, list[list[int]]], specs: dict[str, ServiceSpec], hw: Hardware) -> dict[str, float]:
    """Replicas discounted for sharing a GPU's compute with other fractional replicas."""
    on: dict[int, int] = {}
    for s, reps in placement.items():
        if specs[s].fractional:
            for r in reps:
                on[r[0]] = on.get(r[0], 0) + 1
    eff = {}
    for s, reps in placement.items():
        if specs[s].fractional:
            eff[s] = sum(1 / (1 + hw.colocation_penalty * (on[r[0]] - 1)) for r in reps)
        else:
            eff[s] = float(len(reps))
    return eff


# ---------------------------------------------------------------- the coupled model


def _session_service(specs, supply: SupplyProfile, services) -> str | None:
    for s in services:
        cap = specs[s].max_sessions if s in specs else None
        if cap is not None or supply.curves[s].max_concurrency is not None and supply.curves[s].unit == "sim_s":
            return s
    return None


def simulate(n: int, eff: dict[str, float], demand: DemandProfile, curves: dict[str, Curve], *, clock: str = "input",
             overhead_s: float = 3.0, session: str | None = None, iters: int = 200) -> tuple[float, dict[str, float], dict[str, float]]:
    """Episode wall time T, wait per service W_s and in-flight per replica c_s for n concurrent episodes."""
    svc = list(curves)
    c = {s: n / eff[s] for s in svc}
    T = overhead_s
    W: dict[str, float] = {}
    for _ in range(iters):
        W = {s: demand.services[s].units * curves[s].unit_time(c[s]) for s in svc}
        if clock == "realtime" and session in W:
            W[session] = max(W[session], demand.sim_s)
        T = overhead_s + sum(W.values())
        new = {s: n * W[s] / T / eff[s] for s in svc}
        delta = max(abs(new[s] - c[s]) for s in svc) if svc else 0.0
        c = {s: 0.5 * c[s] + 0.5 * new[s] for s in svc}
        if delta < 1e-5:
            break
    return T, W, c


def evaluate(alloc: dict[str, int], specs: dict[str, ServiceSpec], demand: DemandProfile, supply: SupplyProfile, hw: Hardware, *,
             clock: str = "input", overhead_s: float = 3.0, max_env: int = 64, concurrency: int | None = None,
             tolerance: float = 0.01) -> Plan | None:
    """The best plan for a fixed allocation (or None if it does not fit / misses a demanded service)."""
    placement = place(alloc, specs, hw)
    if placement is None:
        return None
    needed = [s for s, d in demand.services.items() if d.units > 0]
    for s in needed:
        if s not in supply.curves:
            raise ValueError(f"no supply curve for demanded service {s!r}: probe it or drop it from the demand")
        if alloc.get(s, 0) == 0:
            return None
    eff = effective_replicas(placement, specs, hw)
    curves = {s: supply.curves[s] for s in needed}
    session = _session_service(specs, supply, needed)
    cap = max_env
    if session:
        per = specs[session].max_sessions if session in specs and specs[session].max_sessions else curves[session].max_concurrency
        cap = min(cap, alloc[session] * per)
    run = lambda n: simulate(n, eff, demand, curves, clock=clock, overhead_s=overhead_s, session=session)  # noqa: E731
    if concurrency is not None:
        n = min(concurrency, cap)
    else:  # throughput rises with n and saturates: the smallest n within tolerance of the cap's
        T, _, _ = run(cap)
        best = cap / T
        n = cap
        for k in range(1, cap):
            Tk, _, _ = run(k)
            if k / Tk >= (1 - tolerance) * best:
                n = k
                break
    T, W, c = run(n)
    X = n / T * 3600
    rows = []
    for s in needed:
        peak_c, peak = curves[s].peak()
        rate = X / 3600 * demand.services[s].units  # units per second the plan asks of s
        rows.append(Row(s, alloc[s], placement[s], eff[s], c[s], W[s], rate / (eff[s] * peak),
                        eff[s] * peak / demand.services[s].units * 3600))
    for s in alloc:  # reserved / undemanded services still show where they sit
        if s not in needed and alloc[s]:
            rows.append(Row(s, alloc[s], placement[s], eff[s], 0.0, 0.0, 0.0, math.inf))
    used = {g for reps in placement.values() for r in reps for g in r}
    return Plan(dict(alloc), placement, n, X, T, rows, [g for g in hw.gpus if g not in used], clock)


def solve(specs: dict[str, ServiceSpec], demand: DemandProfile, supply: SupplyProfile, hw: Hardware, *, clock: str = "input",
          overhead_s: float = 3.0, max_env: int = 64, tolerance: float = 0.01, top: int = 5) -> list[Plan]:
    """The best plans, best first."""
    ranges = {}
    for s, sp in specs.items():
        if sp.reserve:
            ranges[s] = [max(sp.min_replicas, 1)] if sp.min_replicas or sp.gpu_ids else [0]
        elif demand.services.get(s) and demand.services[s].units > 0:
            hi = sp.max_replicas
            if not sp.fractional:
                hi = min(hi, len(hw.gpus) // sp.gpus)
            ranges[s] = list(range(max(1, sp.min_replicas), hi + 1))
        else:
            ranges[s] = [sp.min_replicas]
    missing = [s for s, d in demand.services.items() if d.units > 0 and s not in specs]
    if missing:
        raise ValueError(f"demanded services without a spec: {missing}")
    names = list(ranges)
    plans = []
    for combo in itertools.product(*(ranges[s] for s in names)):
        alloc = dict(zip(names, combo))
        whole = sum(alloc[s] * specs[s].gpus for s in names)
        if whole > len(hw.gpus):
            continue
        p = evaluate({s: r for s, r in alloc.items() if r}, specs, demand, supply, hw, clock=clock, overhead_s=overhead_s,
                     max_env=max_env, tolerance=tolerance)
        if p is not None:
            plans.append(p)
    if not plans:
        raise ValueError("no allocation fits the hardware")
    best = max(p.eph for p in plans)
    # within tolerance of the best, prefer fewer GPUs, fewer replicas, then fewer concurrent episodes
    near = lambda p: p.eph >= (1 - tolerance) * best  # noqa: E731
    plans.sort(key=lambda p: (not near(p), (p.gpus_used, sum(p.alloc.values()), p.concurrency) if near(p) else (0, 0, 0), -p.eph))
    # drop plans that only add replicas to a better-or-equal plan without raising episodes/h (more TTS, same result)
    kept: list[Plan] = []
    for p in plans:
        if not any(q.eph >= (1 - tolerance) * p.eph and all(p.alloc.get(s, 0) >= n for s, n in q.alloc.items()) for q in kept):
            kept.append(p)
    return kept[:top]


def session_cap(plan: Plan, specs: dict[str, ServiceSpec], supply: SupplyProfile | None = None) -> int | None:
    """Most episodes a plan can run at once (agent replicas x sessions each), or None if unbounded."""
    for s, n in plan.alloc.items():
        per = specs[s].max_sessions if s in specs else None
        if per is None and supply is not None and s in supply.curves and supply.curves[s].unit == "sim_s":
            per = supply.curves[s].max_concurrency
        if per:
            return n * per
    return None
