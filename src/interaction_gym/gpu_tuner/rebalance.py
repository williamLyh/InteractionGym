"""The empirical GPU tuner: measure the running layout, rebalance GPUs toward the measured load, repeat.

One round (``measure``):
  run real end-to-end episodes on the current layout, ramping env concurrency (2, 4, 8, ... up to
  the agent's session capacity) until episodes/hour stops rising or a service queues; while they
  run, sample GPU util / memory and vLLM ``/metrics`` (``Monitor``) and record, via wrapper clients
  (``Meter``), how long each episode waits on each service. The level with the best measured
  episodes/hour is the round's result and its window is analysed.

Analysis (``analyze``): per service, busy GPU-seconds per second (GPU util attributed to the
services on each GPU), queue depth, KV usage, wait per episode and wait share; the bottleneck is the
service that queues, else the one at ~100% util, else the agent if its sessions are all taken while
throughput still rose, else the one with the dominant wait share. Idle: low util, no queue.

Proposal (``propose`` → ``allocate``): a GPU split proportional to measured busy GPU-seconds (the
bottleneck's measured load is capped by its saturation, so it gets a ``boost``), rounded to whole
replicas respecting GPUs per replica, memory and co-location (``solve.place``); plus a recommended
env concurrency. The projected episodes/hour is a linear extrapolation and is labelled as such.

Concurrency caps: a service's per-replica limit (the agent's ``max_sessions``, the LLM's
``--max-num-seqs``, ...) is a knob too (``Layout.caps``, ``ServiceSpec.cap_steps``). When the
bottleneck is concurrency-bound (its cap is reached and requests queue / episodes/hour still rose)
while its GPUs are mostly idle (util < ``cap_util``), raising the cap one step costs a restart and
no GPUs, so it is proposed before more replicas; with high util, more replicas as before.

``loop``: measure → propose → (flag-gated) apply and re-measure, until episodes/hour stops
improving or the bottleneck flips back. A cap raise is kept only if it gains ≥ ``cap_gain`` with
no failed episodes, no stage eviction in the server logs and GPU memory ≤ ``mem_limit``; otherwise
the cap is rolled back and frozen for that service. Every episodes/hour figure says whether it was
measured.
"""

from __future__ import annotations

import math
from collections import defaultdict
import inspect
from dataclasses import dataclass, field
from statistics import mean
from typing import Awaitable, Callable

from .empirical import Level, run_batch
from .meter import Meter
from .monitor import Monitor
from .solve import Hardware, ServiceSpec, place


# ---------------------------------------------------------------- layouts


@dataclass
class Layout:
    placement: dict[str, list[list[int]]]  # service -> replicas -> GPU ids
    caps: dict[str, int] = field(default_factory=dict)  # per-replica concurrency limits set explicitly (else the spec's)

    @property
    def alloc(self) -> dict[str, int]:
        return {s: len(r) for s, r in self.placement.items() if r}

    @classmethod
    def parse(cls, text: str) -> Layout:
        """``agent=4+5@6,llm=0+1/2+3,tts=6/7,clone=6``: replicas split by ``/``, a replica's GPUs by ``+``,
        an optional ``@n`` = the service's per-replica concurrency cap (agent: sessions per server)."""
        out, caps = {}, {}
        for part in filter(None, (p.strip() for p in text.split(","))):
            s, reps = part.split("=")
            if "@" in reps:
                reps, cap = reps.split("@")
                caps[s.strip()] = int(cap)
            out[s.strip()] = [[int(g) for g in r.split("+")] for r in reps.split("/") if r]
        return cls(out, caps)

    def __str__(self) -> str:
        return ",".join(f"{s}=" + "/".join("+".join(map(str, r)) for r in reps) + (f"@{self.caps[s]}" if s in self.caps else "")
                        for s, reps in self.placement.items() if reps)

    def cap(self, s: str, specs: dict[str, ServiceSpec]) -> int | None:
        return self.caps.get(s, specs[s].default_cap if s in specs else None)

    def with_cap(self, s: str, cap: int) -> Layout:
        return Layout({k: [list(r) for r in v] for k, v in self.placement.items()}, {**self.caps, s: cap})

    def same(self, other: Layout, specs: dict[str, ServiceSpec]) -> bool:
        """Same replicas per service and the same effective caps (GPU ids may differ)."""
        return self.alloc == other.alloc and all(self.cap(s, specs) == other.cap(s, specs) for s in self.alloc)

    def gpu_share(self) -> dict[str, float]:
        """GPU-equivalents each service holds (a GPU shared by k services counts 1/k to each)."""
        on: dict[int, set] = defaultdict(set)
        for s, reps in self.placement.items():
            for r in reps:
                for g in r:
                    on[g].add(s)
        return {s: sum(1 / len(on[g]) for g in {g for r in reps for g in r}) for s, reps in self.placement.items() if reps}

    def gpus_of(self, s: str) -> list[int]:
        return sorted({g for r in self.placement.get(s, []) for g in r})


def session_service(layout: Layout, specs: dict[str, ServiceSpec]) -> str | None:
    return next((s for s in layout.alloc if s in specs and specs[s].max_sessions), None)


def session_capacity(layout: Layout, specs: dict[str, ServiceSpec]) -> int | None:
    """Most episodes the layout can run at once: agent servers x sessions each."""
    s = session_service(layout, specs)
    return layout.alloc[s] * layout.cap(s, specs) if s else None


def next_cap(spec: ServiceSpec, cur: int | None) -> int | None:
    """The next value on the spec's cap ladder above ``cur`` (None at the top / without a ladder)."""
    return next((c for c in sorted(spec.cap_steps or []) if cur is not None and c > cur), None)


# ---------------------------------------------------------------- measurement


@dataclass
class ServiceStats:
    service: str
    replicas: int
    gpus: list[int]
    held: float  # GPU-equivalents held
    busy: float | None  # busy GPUs (mean util x GPUs, attributed); None without GPU samples
    util: float | None  # busy / held
    mem_gb: float | None
    running: float | None  # vLLM num_requests_running (mean over the window)
    waiting: float | None  # vLLM num_requests_waiting
    kv: float | None
    wait_s: float  # env-side wall seconds per episode spent on this service
    wait_share: float  # of the episode's wall time
    calls: float  # per episode
    mem_peak: float | None = None  # highest memory used / total on any of its GPUs during the window (0..1)
    unit_s: float | None = None  # env-side wall seconds per unit of work (agent: per simulated second, i.e. per 1 s unit)
    cap: int | None = None  # per-replica concurrency cap it ran with


@dataclass
class Round:
    layout: Layout
    levels: list[Level]
    stats: dict[str, ServiceStats]
    bottleneck: str = ""
    why: str = ""
    idle: list[str] = field(default_factory=list)
    session_cap: int | None = None
    stop_reason: str = ""
    session_bound: bool = False  # every agent session was taken and episodes/hour was still rising
    session_service: str | None = None
    caps: dict[str, int] = field(default_factory=dict)  # effective per-replica caps of the layout
    cap_bound: list[str] = field(default_factory=list)  # services at their concurrency cap with work waiting
    problems: list[str] = field(default_factory=list)  # server-log lines: stage eviction, OOM
    guard: str = ""  # why the loop rejected this round's setting (memory, failures, no gain), if it did

    @property
    def best(self) -> Level:
        return max(self.levels, key=lambda lv: lv.eph)

    @property
    def failures(self) -> int:
        return sum(lv.errors for lv in self.levels)

    @property
    def mem_peak(self) -> float | None:
        xs = [x.mem_peak for x in self.stats.values() if x.mem_peak is not None]
        return max(xs) if xs else None


def service_stats(layout: Layout, meter: Meter, samples, episode_wall_s: float, baseline: dict[str, dict] | None = None, *,
                  gpu_mem_mb: float | None = None, specs: dict[str, ServiceSpec] | None = None) -> dict[str, ServiceStats]:
    """Per-service numbers for one measured window. ``baseline`` = metrics sampled before the episodes
    started: queue depth is reported as the excess over it (some servers report a constant non-zero
    ``num_requests_waiting`` at idle, e.g. vLLM-Omni pipeline stages, and other users' load is not ours)."""
    held = layout.gpu_share()
    eps = max(1, len(meter.episodes))
    busy_time: dict[str, float] = defaultdict(float)  # env-side busy seconds, to split shared GPUs
    waits: dict[str, float] = defaultdict(float)
    calls: dict[str, float] = defaultdict(float)
    units: dict[str, float] = defaultdict(float)
    for e in meter.events:
        busy_time[e.service] += e.end - e.start
        waits[e.service] += e.end - e.start
        calls[e.service] += e.calls
        units[e.service] += e.units
    on: dict[int, list[str]] = defaultdict(list)
    for s, reps in layout.placement.items():
        for g in {g for r in reps for g in r}:
            on[g].append(s)
    gpu_samples = [x for x in samples if x.gpu]
    busy: dict[str, float] = defaultdict(float)
    mem: dict[str, float] = defaultdict(float)
    peak: dict[str, float] = {}
    for g, svcs in on.items():
        vals = [x.gpu[g] for x in gpu_samples if g in x.gpu]
        if not vals:
            continue
        u, m = mean(v[0] for v in vals), mean(v[1] for v in vals) / 1024
        fr = [v[1] / (v[2] if len(v) > 2 and v[2] else gpu_mem_mb) for v in vals if (len(v) > 2 and v[2]) or gpu_mem_mb]
        for s in svcs:  # the fullest GPU a service sits on (a GPU's memory is not split: any process can OOM it)
            if fr:
                peak[s] = max(peak.get(s, 0.0), max(fr))
        tot = sum(busy_time[s] for s in svcs)
        for s in svcs:  # a shared GPU's work goes to its services in proportion to their env-side busy time
            w = busy_time[s] / tot if tot else 1 / len(svcs)
            busy[s] += u * w
            mem[s] += m / len(svcs)

    baseline = baseline or {}

    def metric(s, k):
        xs = [x.metrics[s][k] for x in samples if s in x.metrics and k in x.metrics[s]]
        if not xs:
            return None
        base = baseline.get(s, {}).get(k, 0.0) if k in ("running", "waiting") else 0.0
        return max(0.0, mean(xs) - base)

    out = {}
    for s, reps in layout.placement.items():
        if not reps:
            continue
        b = busy[s] if gpu_samples else None
        w = waits[s] / eps
        out[s] = ServiceStats(s, len(reps), layout.gpus_of(s), held[s], b, b / held[s] if b is not None and held[s] else None,
                              mem[s] if gpu_samples else None, metric(s, "running"), metric(s, "waiting"), metric(s, "kv"),
                              w, w / episode_wall_s if episode_wall_s else 0.0, calls[s] / eps, peak.get(s),
                              waits[s] / units[s] if units[s] else None, layout.cap(s, specs) if specs else layout.caps.get(s))
    return out


def analyze(r: Round, *, sat_util: float = 0.9, idle_util: float = 0.3, queue_tol: float = 0.5) -> Round:
    st = r.stats.values()
    queued = [x for x in st if (x.waiting or 0) >= queue_tol]
    saturated = [x for x in st if x.util is not None and x.util >= sat_util]
    rising = len(r.levels) >= 2 and r.levels[-1].eph >= 1.05 * r.levels[-2].eph
    agent = r.stats.get("agent")
    session_bound = agent is not None and r.session_cap is not None and r.best.concurrency >= r.session_cap and (rising or len(r.levels) == 1)
    r.session_bound = session_bound
    if queued:
        x = max(queued, key=lambda x: (x.waiting or 0) / max(1.0, x.running or 1.0))
        r.bottleneck, r.why = x.service, f"queues ({x.waiting:.1f} waiting, {x.running or 0:.1f} running)"
    elif saturated:
        x = max(saturated, key=lambda x: x.util)
        r.bottleneck, r.why = x.service, f"GPU util {x.util:.0%}"
    elif session_bound:
        r.bottleneck, r.why = "agent", (f"all {r.session_cap} agent sessions in use and episodes/hour still rising "
                                        f"(sessions are held while episodes wait on other services)")
    elif st:
        x = max(st, key=lambda x: x.wait_share)
        r.bottleneck, r.why = x.service, f"largest share of episode time ({x.wait_share:.0%})"
    r.idle = [x.service for x in st if x.util is not None and x.util < idle_util and (x.waiting or 0) < queue_tol and x.service != r.bottleneck]
    r.cap_bound = [x.service for x in st if _cap_bound(r, x, queue_tol)]
    return r


def _cap_bound(r: Round, x: ServiceStats, queue_tol: float) -> bool:
    """At its concurrency cap with work waiting: for the session service, every session taken and
    either requests queue or episodes/hour was still rising; for others, requests queue while about
    cap x replicas are running."""
    queued = (x.waiting or 0) >= queue_tol
    if x.service == r.session_service:
        return r.session_cap is not None and r.best.concurrency >= r.session_cap and (queued or r.session_bound)
    cap = r.caps.get(x.service)
    return bool(cap) and queued and (x.running or 0) >= 0.9 * cap * x.replicas


async def measure(layout: Layout, make_runner: Callable[[Meter], Callable], specs: dict[str, ServiceSpec], *,
                  monitor: Monitor | None = None, levels=(2, 4, 8, 16), seconds: float = 60.0, min_gain: float = 0.05,
                  queue_tol: float = 0.5, scale: float = 1.0, max_concurrency: int | None = None, start_at: int | None = None,
                  watch=None, gpu_mem_mb: float | None = None, log=print) -> Round:
    """Ramp env concurrency on a running layout; returns the round (analysed) at the best level.
    The ramp goes up to the agent's session capacity (agent servers x sessions each), which it always
    tries: when a service starts to queue below it, the ramp jumps straight to it. ``start_at`` skips
    the levels below it (a cap round re-measures from the previous best level up). ``watch``
    (``LogWatch``) collects stage-eviction lines from the server logs written during the round."""
    cap = session_capacity(layout, specs)
    lv = sorted({c for c in levels if cap is None or c < cap} | ({cap} if cap else set()))
    if start_at:
        lv = sorted({c for c in lv if c >= start_at} | ({min(start_at, cap)} if cap else {start_at}))
    if max_concurrency:
        lv = [c for c in lv if c <= max_concurrency] or [max_concurrency]
    base_sample = monitor.sample() if monitor else None  # idle / other users' load before our episodes
    baseline = base_sample.metrics if base_sample else {}
    if watch is not None:
        watch.mark()
    meters, out = [], []
    best = 0.0
    stop = "ramp finished"
    idx = 0
    todo = list(lv)
    jumped = None
    while todo:
        c = todo.pop(0)
        meter = Meter()
        level = await run_batch(make_runner(meter), c, seconds=seconds, start_index=idx, scale=scale)
        idx += level.episodes + level.errors
        meters.append(meter)
        out.append(level)
        win = monitor.window(level.t0, level.t1) if monitor else []
        waiting = max((mean(x.metrics[s]["waiting"] for x in win if s in x.metrics and "waiting" in x.metrics[s])
                       - baseline.get(s, {}).get("waiting", 0.0)
                       for s in {s for x in win for s in x.metrics} if any("waiting" in x.metrics.get(s, {}) for x in win)), default=0.0)
        log(f"  N={c:3d}: {level.episodes:4d} episodes, {level.eph:7.0f} ep/h (measured)" + (f", max queue {waiting:.1f}" if win else "")
            + (f", {level.errors} failed ({'; '.join(sorted(set(level.error_kinds)))[:300]})" if level.errors else ""))
        if level.aborted:
            stop = f"episodes kept failing at N={c}"
            break
        if level.eph < (1 + min_gain) * best:
            stop = f"episodes/hour stopped rising at N={c}"
            break
        best = max(best, level.eph)
        if waiting >= queue_tol:
            if cap and c < cap:  # the session capacity is always tried
                todo, jumped = [cap], c
                continue
            stop = f"a service queues at N={c}"
            break
    if jumped:
        stop += f" (queued from N={jumped}: jumped to the session capacity)"
    i = max(range(len(out)), key=lambda k: out[k].eph)
    b = out[i]
    meter = meters[i]
    wall = mean(rec.wall_s for rec in meter.episodes.values()) if meter.episodes else 0.0
    samples = monitor.window(b.t0, b.t1) if monitor else []
    stats = service_stats(layout, meter, samples, wall, baseline, gpu_mem_mb=gpu_mem_mb, specs=specs)
    if monitor:  # memory: the peak over the whole round (it grows with sessions and time, not just at the best level)
        allw = monitor.window(out[0].t0, out[-1].t1)
        for s, x in service_stats(layout, meter, allw, wall, baseline, gpu_mem_mb=gpu_mem_mb, specs=specs).items():
            if x.mem_peak is not None:
                stats[s].mem_peak = x.mem_peak
    for x in stats.values():  # mock runs compress time: report real seconds
        x.wait_s *= scale
        if x.unit_s is not None:
            x.unit_s *= scale
    problems = watch.check() if watch is not None else []
    caps = {s: layout.cap(s, specs) for s in layout.alloc if layout.cap(s, specs) is not None}
    return analyze(Round(layout, out, stats, session_cap=cap, stop_reason=stop, session_service=session_service(layout, specs),
                         caps=caps, problems=problems), queue_tol=queue_tol)


# ---------------------------------------------------------------- allocation


def allocate(targets: dict[str, float], specs: dict[str, ServiceSpec], hw: Hardware, *, keep: dict[str, int] | None = None) -> Layout:
    """Integer replicas whose GPU-equivalents follow ``targets`` (GPU-equivalents per service):
    one replica each to start, then repeatedly one more replica to the service furthest below its
    target (smallest held/target) whose replica still fits (GPUs per replica, memory, co-location,
    max replicas), until nothing more fits. ``keep`` pins replica counts (e.g. a trainer)."""
    keep = keep or {}
    alloc = {s: max(1, specs[s].min_replicas) for s in targets}
    alloc.update(keep)
    if place(alloc, specs, hw) is None:
        raise ValueError(f"even one replica per service does not fit: {alloc}")
    while True:
        cur = Layout(place(alloc, specs, hw)).gpu_share()
        order = sorted((s for s in targets if s not in keep and alloc[s] < specs[s].max_replicas),
                       key=lambda s: (cur.get(s, 0) / targets[s] if targets[s] > 0 else math.inf, -targets[s]))
        for s in order:
            trial = {**alloc, s: alloc[s] + 1}
            if place(trial, specs, hw) is not None and _improves(s, trial, alloc, targets, specs, hw):
                alloc = trial
                break
        else:
            return Layout(place(alloc, specs, hw))


def _improves(s, trial, alloc, targets, specs, hw) -> bool:
    """A fractional replica added onto GPUs that are all in use only splits them further; allow it
    only if the service is still below target. Whole-GPU replicas use free GPUs, always allowed."""
    if not specs[s].fractional:
        return True
    before = Layout(place(alloc, specs, hw))
    after = Layout(place(trial, specs, hw))
    used = lambda lay: len({g for reps in lay.placement.values() for r in reps for g in r})  # noqa: E731
    return used(after) > used(before) or before.gpu_share().get(s, 0) < targets[s]


@dataclass
class Proposal:
    layout: Layout
    targets: dict[str, float]  # GPU-equivalents, proportional to measured load
    concurrency: int  # recommended env concurrency (projected)
    projected_eph: float  # linear extrapolation from the round — not measured
    limits: dict[str, float]  # per-service projected cap on episodes/hour
    basis: str  # what the split is proportional to
    action: str = "replicas"  # "cap": raise one service's concurrency cap (no GPUs move); "replicas": move GPUs
    why: str = ""  # why this action rather than the other
    service: str = ""  # the service whose cap changes (action "cap")


def propose(r: Round, specs: dict[str, ServiceSpec], hw: Hardware, *, boost: float = 1.25, target_util: float = 0.9,
            keep: dict[str, int] | None = None, cap_util: float = 0.6, frozen: dict[str, str] | None = None) -> Proposal:
    """The cheaper of two actions for the bottleneck: if it is concurrency-bound (``Round.cap_bound``)
    with GPU util below ``cap_util`` and its cap can go one step higher (``ServiceSpec.cap_steps``,
    not ``frozen`` by an earlier failed raise), raise the cap — a restart, no GPUs. Otherwise split the
    GPUs in proportion to the measured load (more replicas for the bottleneck)."""
    frozen = frozen or {}
    b = r.bottleneck
    note = ""
    if b in r.cap_bound and b in specs and specs[b].cap_steps:
        cur = r.layout.cap(b, specs)
        nxt = next_cap(specs[b], cur)
        util = r.stats[b].util if b in r.stats else None
        if b in frozen:
            note = f"{b} cap {cur} stays (raising it failed: {frozen[b]}); "
        elif nxt is None:
            note = f"{b} cap {cur} is the last step; "
        elif util is not None and util >= cap_util:
            note = f"{b} is at its cap but its GPUs are busy ({util:.0%} util >= {cap_util:.0%}): more replicas, not a higher cap; "
        else:
            return _cap_proposal(r, specs, b, cur, nxt, util, target_util)
    p = _replica_proposal(r, specs, hw, boost=boost, target_util=target_util, keep=keep)
    p.why = note + (f"GPU split follows the measured load (bottleneck {b})" if b else "GPU split follows the measured load")
    return p


def _cap_proposal(r: Round, specs, s: str, cur: int, nxt: int, util: float | None, target_util: float) -> Proposal:
    lay = r.layout.with_cap(s, nxt)
    b = r.best
    limits = {}
    if s == r.session_service and r.session_cap and b.concurrency:
        limits[f"{s} sessions"] = b.eph * session_capacity(lay, specs) / b.concurrency
    else:
        limits[f"{s} cap"] = b.eph * nxt / cur
    x = r.stats.get(s)
    if x is not None and x.util:
        limits[s] = b.eph * target_util / x.util  # its GPUs: busy GPU-seconds per episode stay as measured
    proj = min(limits.values())
    n = session_capacity(lay, specs) if s == r.session_service else b.concurrency
    why = (f"{s} is concurrency-bound (cap {cur} reached, " + ("requests queue" if (x and (x.waiting or 0) >= 0.5) else "episodes/hour still rising")
           + f") with its GPUs at {util:.0%} util: raise its cap {cur} -> {nxt} (a restart, no extra GPUs) before adding replicas"
           if util is not None else f"{s} is concurrency-bound (cap {cur} reached), GPU util unknown: raise its cap {cur} -> {nxt}")
    return Proposal(lay, {k: v.held for k, v in r.stats.items()}, max(1, n or b.concurrency), proj, limits,
                    "concurrency cap (no GPUs move)", "cap", why, s)


def _replica_proposal(r: Round, specs: dict[str, ServiceSpec], hw: Hardware, *, boost: float, target_util: float,
                      keep: dict[str, int] | None) -> Proposal:
    keep = keep or {}
    svcs = [s for s in r.stats if s not in keep and not specs[s].reserve]
    have_gpu = all(r.stats[s].busy is not None for s in svcs)
    load = {s: (r.stats[s].busy if have_gpu else r.stats[s].wait_share * r.stats[s].held) for s in svcs}
    basis = "busy GPU-seconds (nvidia-smi)" if have_gpu else "env-side wait share x GPUs held (no GPU samples)"
    if r.bottleneck in load:  # its measured load is capped by its own saturation
        load[r.bottleneck] *= boost
    keep = {s: r.layout.alloc.get(s) or n for s, n in keep.items()}  # replicas held out of the split (e.g. a trainer)
    total = len(hw.gpus) - sum(n * max(1, specs[s].gpus) for s, n in keep.items())
    tot = sum(load.values()) or 1.0
    targets = {s: max(1e-6, load[s] / tot * total) for s in svcs}
    agent = next((s for s in svcs if specs[s].max_sessions), None)
    more = {s: max(1, specs[s].min_replicas) for s in svcs} | keep | {agent: r.layout.alloc.get(agent, 0) + 1} if agent else None
    if r.session_bound and agent and place(more, specs, hw) is not None:
        # more sessions are needed whatever the GPU load says: at least one more agent replica (if one fits at all)
        need = r.stats[agent].held + max(1, specs[agent].gpus)
        if targets[agent] < need:
            rest = sum(v for s, v in targets.items() if s != agent) or 1.0
            left = max(0.0, total - need)
            targets = {s: (need if s == agent else max(1e-6, v / rest * left)) for s, v in targets.items()}
    layout = allocate(targets, specs, hw, keep=keep)
    layout.caps = dict(r.layout.caps)  # tuned caps stay
    # projection: each service's busy GPU-seconds per episode stays as measured; sessions and wall per episode too
    b = r.best
    held_new = layout.gpu_share()
    limits = {}
    for s in svcs:
        x = r.stats[s]
        if have_gpu and x.busy and x.busy > 0:
            limits[s] = b.eph * held_new.get(s, 0) * target_util / x.busy
    cap_new = session_capacity(layout, specs)
    if cap_new and b.concurrency:
        limits["agent sessions"] = b.eph * cap_new / b.concurrency
    proj = min(limits.values()) if limits else b.eph
    n = math.ceil(b.concurrency * proj / b.eph) if b.eph else b.concurrency
    if cap_new:
        n = min(n, cap_new)
    return Proposal(layout, targets, max(1, n), proj, limits, basis)


# ---------------------------------------------------------------- the loop


@dataclass
class LoopResult:
    rounds: list[Round]
    proposals: list[Proposal]
    best: Round
    recommendation: Layout
    concurrency: int
    measured: bool  # is the recommended layout's episodes/hour measured (True) or projected (False)
    reason: str
    actions: list[str] = field(default_factory=list)  # per round: how its layout came about ("start", "cap", "replicas")
    frozen: dict[str, str] = field(default_factory=dict)  # caps the loop stopped raising, and why


def guard(nxt: Round, prev: Round, p: Proposal, *, mem_limit: float = 0.92, max_failures: int = 0, cap_gain: float = 0.05,
          min_gain: float = 0.03) -> str:
    """Why a new setting must be rolled back ('' = keep it): GPU memory above ``mem_limit`` on any GPU (and higher
    than in the previous round: static preallocation is not growth), failed
    episodes (lockstep ack timeouts, closed sessions), stage eviction in the server logs, or (for a cap raise)
    a gain under ``cap_gain``."""
    def grew(s, x):  # vLLM preallocates (e.g. 0.9 of a GPU): a static level just under the limit is fine, growth is not
        before = prev.stats[s].mem_peak if s in prev.stats else None
        return before is None or x.mem_peak > before + 0.005

    over = {s: x.mem_peak for s, x in nxt.stats.items() if x.mem_peak is not None and x.mem_peak > mem_limit and grew(s, x)}
    if over:
        s = max(over, key=over.get)
        return f"GPU memory {over[s]:.0%} > {mem_limit:.0%} on {s}'s GPUs {nxt.stats[s].gpus}"
    if nxt.problems:
        return f"stage eviction / OOM in the server logs: {nxt.problems[0]}"
    if nxt.failures > max_failures:
        kinds = sorted({k for lv in nxt.levels for k in lv.error_kinds})
        return f"{nxt.failures} failed episodes ({'; '.join(kinds)[:200]})"
    if p.action == "cap" and nxt.best.eph < (1 + cap_gain) * prev.best.eph:
        return f"gain under {cap_gain:.0%} ({prev.best.eph:.0f} -> {nxt.best.eph:.0f} ep/h measured)"
    return ""


async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x


async def loop(layout: Layout, make_runner: Callable[[Layout, Meter], Callable], specs: dict[str, ServiceSpec], hw: Hardware, *,
               apply: Callable[[Layout], Awaitable[None]] | None = None, monitor_for: Callable[[Layout], Monitor | None] = lambda lay: None,
               rounds: int = 3, min_gain: float = 0.03, levels=(2, 4, 8, 16), seconds: float = 60.0, scale: float = 1.0,
               keep: dict[str, int] | None = None, max_concurrency: int | None = None, cap_util: float = 0.6, cap_gain: float = 0.05,
               mem_limit: float = 0.92, max_failures: int = 0, caps: bool = True, alive=None, watch_for=None, retries: int = 2,
               log=print) -> LoopResult:
    """measure → propose → (if ``apply``) restart + re-measure, at most ``rounds`` measurements.

    ``alive(layout) -> bool`` (optional): checked after each round; if the servers died meanwhile (e.g. an
    external kill), the layout is restarted and the round measured again (``retries`` times). A failed start
    is retried too. ``watch_for(layout)`` gives a ``LogWatch`` for stage-eviction lines. ``caps=False``
    turns the concurrency-cap knob off (GPU counts only)."""
    hist: list[Round] = []
    props: list[Proposal] = []
    actions: list[str] = []
    frozen: dict[str, str] = {} if caps else {s: "cap tuning off" for s in specs}
    gpu_mem_mb = hw.mem_gb * 1024

    async def start(lay: Layout) -> None:
        for k in range(retries + 1):
            try:
                await apply(lay)
                return
            except Exception as e:  # noqa: BLE001 — e.g. killed while loading; a second failure is the layout's
                if k == retries:
                    raise
                log(f"  start failed ({str(e).splitlines()[0][:200]}); retrying")

    async def one(lay: Layout, start_at: int | None = None) -> Round:
        for k in range(retries + 1):
            mon = monitor_for(lay)
            watch = watch_for(lay) if watch_for else None
            log(f"round {len(hist) + 1}: measuring {lay}")
            kw = dict(levels=levels, seconds=seconds, scale=scale, max_concurrency=max_concurrency, start_at=start_at, watch=watch,
                      gpu_mem_mb=gpu_mem_mb, log=log)
            if mon is None:
                rr = await measure(lay, lambda m: make_runner(lay, m), specs, **kw)
            else:
                async with mon.running():
                    rr = await measure(lay, lambda m: make_runner(lay, m), specs, monitor=mon, **kw)
            if alive is not None and apply is not None and k < retries and not await _maybe_await(alive(lay)):
                log("  the servers died during the round (external kill?): restarting the layout and measuring again")
                await start(lay)
                continue
            break
        log(f"  -> {rr.best.eph:.0f} ep/h measured at N={rr.best.concurrency}; bottleneck {rr.bottleneck} ({rr.why})"
            + (f"; idle: {', '.join(rr.idle)}" if rr.idle else "")
            + (f"; mem peak {rr.mem_peak:.0%}" if rr.mem_peak is not None else "") + (f"; {rr.failures} failed" if rr.failures else ""))
        return rr

    cur = await one(layout)
    hist.append(cur)
    actions.append("start")
    best = cur
    running = layout
    reason = ""
    while True:
        p = propose(cur, specs, hw, keep=keep, cap_util=cap_util, frozen=frozen)
        props.append(p)
        if p.layout.same(cur.layout, specs):
            reason = "the measured load already matches the layout (proposal = current)"
            break
        if p.projected_eph < (1 + min_gain) * cur.best.eph:
            # e.g. the bottleneck cannot get another replica, only an idle service would grow: a restart buys nothing
            reason = (f"the proposal {p.layout} projects no gain ({p.projected_eph:.0f} vs {cur.best.eph:.0f} ep/h measured; "
                      f"limits: {', '.join(f'{k} {v:.0f}' for k, v in sorted(p.limits.items(), key=lambda kv: kv[1])[:3])})")
            break
        if apply is None:
            reason = f"restarts not allowed: proposal ({p.action}) not applied (its episodes/hour is projected, not measured)"
            return LoopResult(hist, props, best, p.layout, p.concurrency, False, reason, actions, frozen)
        if len(hist) >= rounds:
            reason = f"round limit ({rounds}) reached"
            break
        log(f"applying {p.layout} ({p.action}: {p.why}; projected {p.projected_eph:.0f} ep/h at N={p.concurrency})")
        try:
            await start(p.layout)
            running = p.layout
            nxt = await one(p.layout, start_at=cur.best.concurrency if p.action == "cap" else None)
        except Exception as e:  # noqa: BLE001 — the layout does not come up (e.g. out of memory at load with a larger cap)
            msg = f"failed to start: {str(e).splitlines()[0][:200]}"
            log(f"  {p.layout} {msg}")
            if p.action != "cap":
                reason = f"{p.layout} {msg}"
                break
            frozen[p.service] = msg
            cur = best
            continue
        hist.append(nxt)
        actions.append(p.action)
        g = guard(nxt, cur, p, mem_limit=mem_limit, max_failures=max_failures, cap_gain=cap_gain, min_gain=min_gain)
        nxt.guard = g
        if g:
            log(f"  rejected: {g}; rolling back to {best.layout}")
        elif nxt.best.eph > best.best.eph:
            best = nxt
        if p.action == "cap":
            if g:  # freeze this cap and go on from the best setting (another action may still help)
                frozen[p.service] = g
                cur = best
            else:
                cur = nxt
            continue
        if g:
            reason = f"{p.layout} rejected: {g}"
            break
        if nxt.best.eph < (1 + min_gain) * cur.best.eph:
            reason = f"episodes/hour stopped improving ({cur.best.eph:.0f} -> {nxt.best.eph:.0f} measured)"
            break
        if len(hist) >= 3 and nxt.bottleneck == hist[-3].bottleneck and nxt.bottleneck != cur.bottleneck:
            reason = f"bottleneck flipped back to {nxt.bottleneck}"
            break
        cur = nxt
    if apply is not None and not running.same(best.layout, specs):
        log(f"restoring the best measured layout {best.layout}")
        await start(best.layout)
    return LoopResult(hist, props, best, best.layout, best.best.concurrency, True, reason, actions, frozen)
