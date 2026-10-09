"""Plain-text tables for profiles and plans."""

from __future__ import annotations

import math

from .profile import DemandProfile, SupplyProfile
from .solve import Plan


def demand_table(d: DemandProfile) -> str:
    out = [f"Demand per episode ({d.sim_s:.1f} s simulated, {d.episodes} episode(s)"
           + (f", {d.wall_s:.1f} s wall measured" if d.wall_s else "") + ")",
           f"  {'service':8s} {'unit':8s} {'per episode':>12s} {'per sim-min':>12s} {'calls/ep':>9s} {'tokens/call (p+c)':>18s}  source"]
    pm = d.per_minute()
    for s, x in d.services.items():
        tok = f"{x.prompt_tokens:.0f}+{x.completion_tokens:.0f}" if x.unit == "call" else ""
        out.append(f"  {s:8s} {x.unit:8s} {x.units:12.2f} {pm[s]['units']:12.2f} {x.calls:9.2f} {tok:>18s}  {x.source}")
    return "\n".join(out + [f"  note: {n}" for n in d.notes])


def supply_table(sp: SupplyProfile) -> str:
    out = ["Supply per replica (unit_time = wall s per unit per stream; throughput = units/s)"]
    for s, c in sp.curves.items():
        pc, pk = c.peak()
        pts = "  ".join(f"c={x:g}:{u:.3f}" for x, u in c.points)
        out.append(f"  {s:8s} {c.unit:8s} peak {pk:7.2f}/s at c={pc:g}"
                   + (f" (cap {c.max_concurrency})" if c.max_concurrency else "") + f"  [{c.source}]  {pts}")
    return "\n".join(out + [f"  note: {n}" for n in sp.notes])


def plan_table(p: Plan, title: str = "Recommended layout", demand: DemandProfile | None = None) -> str:
    out = [f"{title}: {p.eph:.0f} episodes/h with {p.concurrency} concurrent episodes "
           f"({p.episode_wall_s:.1f} s wall per episode" + (f" of {demand.sim_s:.1f} s simulated" if demand else "")
           + f", clock={p.clock}); bottleneck: {p.bottleneck}",
           f"  {'service':8s} {'repl':>4s} {'eff':>5s}  {'GPUs':22s} {'in-flight/repl':>14s} {'wait s/ep':>9s} {'util':>6s} {'cap ep/h':>9s}"]
    for r in p.rows:
        gpus = " ".join("[" + ",".join(map(str, g)) + "]" for g in r.gpus)
        cap = "-" if math.isinf(r.capacity_eph) else f"{r.capacity_eph:.0f}"
        out.append(f"  {r.service:8s} {r.replicas:4d} {r.effective:5.2f}  {gpus:22s} {r.inflight:14.2f} {r.wait_s:9.2f} {r.util:6.0%} {cap:>9s}")
    if p.idle_gpus:
        out.append(f"  idle GPUs: {p.idle_gpus} (extra replicas would not raise episodes/h)")
    sessions = next((r for r in p.rows if r.service == "agent"), None)
    if sessions and p.episode_wall_s:
        out.append(f"  agent sessions spend {sessions.wait_s / p.episode_wall_s:.0%} of an episode computing; "
                   f"the rest they are held while the env waits on the user simulator / TTS")
    return "\n".join(out)


def alternatives(plans: list[Plan]) -> str:
    out = ["Alternatives:"]
    for p in plans:
        alloc = " ".join(f"{s}={n}" for s, n in p.alloc.items())
        out.append(f"  {p.eph:7.0f} ep/h  N={p.concurrency:3d}  GPUs={p.gpus_used}  {alloc}  (bottleneck {p.bottleneck})")
    return "\n".join(out)


def _f(x, fmt):
    return format(x, fmt) if x is not None else "-"


def round_table(r, title: str = "") -> str:
    """One measurement round: the ramp and the per-service table at the best level (all measured)."""
    b = r.best
    out = [f"{title or 'Round'}: {r.layout}",
           "  ramp (measured): " + "  ".join(f"N={lv.concurrency}:{lv.eph:.0f}" + (f"({lv.errors} failed)" if lv.errors else "")
                                            for lv in r.levels) + f" ep/h  ({r.stop_reason})",
           f"  best: {b.eph:.0f} episodes/h at N={b.concurrency}" + (f" (session capacity {r.session_cap})" if r.session_cap else "")
           + f"; bottleneck: {r.bottleneck} — {r.why}" + (f"; idle: {', '.join(r.idle)}" if r.idle else "")
           + (f"; at their cap: {', '.join(r.cap_bound)}" if r.cap_bound else ""),
           f"  {'service':8s} {'repl':>4s} {'cap':>4s} {'GPUs':14s} {'held':>5s} {'busy':>5s} {'util':>5s} {'mem GB':>6s} {'mem pk':>6s} "
           f"{'run':>5s} {'queue':>5s} {'KV':>5s} {'wait s/ep':>9s} {'share':>6s} {'calls/ep':>8s} {'s/unit':>6s}"]
    for x in r.stats.values():
        out.append(f"  {x.service:8s} {x.replicas:4d} {_f(x.cap, '4d'):>4s} {','.join(map(str, x.gpus)):14s} {x.held:5.2f} {_f(x.busy, '5.2f'):>5s} "
                   f"{_f(x.util, '5.0%'):>5s} {_f(x.mem_gb, '6.1f'):>6s} {_f(x.mem_peak, '6.0%'):>6s} {_f(x.running, '5.1f'):>5s} "
                   f"{_f(x.waiting, '5.1f'):>5s} {_f(x.kv, '5.0%'):>5s} {x.wait_s:9.2f} {x.wait_share:6.0%} {x.calls:8.1f} "
                   f"{_f(x.unit_s, '6.3f'):>6s}")
    if r.failures:
        kinds = sorted({k for lv in r.levels for k in lv.error_kinds})
        out.append(f"  failed episodes: {r.failures} ({'; '.join(kinds)[:300]})")
    for line in r.problems:
        out.append(f"  server log: {line}")
    if r.guard:
        out.append(f"  REJECTED: {r.guard}")
    return "\n".join(out)


def proposal_table(r, p) -> str:
    if p.action == "cap":
        lim = ", ".join(f"{k} {v:.0f}" for k, v in sorted(p.limits.items(), key=lambda kv: kv[1]))
        return "\n".join([f"Proposal (cap): {p.service} cap {r.caps.get(p.service)} -> "
                          f"{p.layout.caps[p.service]} per replica, same GPUs: {p.layout}",
                          f"  why: {p.why}",
                          f"  projected (linear extrapolation, NOT measured): {p.projected_eph:.0f} ep/h at N={p.concurrency}"
                          f" vs {r.best.eph:.0f} measured now  [limits: {lim}]"])
    cur, new = r.layout, p.layout
    cs, ns = cur.gpu_share(), new.gpu_share()
    out = [f"Proposal (replicas; split proportional to {p.basis}):"] + ([f"  why: {p.why}"] if p.why else []) + [
           f"  {'service':8s} {'current':18s} {'proposed':18s} {'target GPUs':>11s} {'GPU-eq now':>10s} {'GPU-eq new':>10s}"]
    for s in dict.fromkeys(list(cur.placement) + list(new.placement)):
        fmt = lambda lay: "/".join("+".join(map(str, rr)) for rr in lay.placement.get(s, [])) or "-"  # noqa: E731
        out.append(f"  {s:8s} {fmt(cur):18s} {fmt(new):18s} {p.targets.get(s, 0):11.2f} {cs.get(s, 0):10.2f} {ns.get(s, 0):10.2f}")
    lim = ", ".join(f"{k} {v:.0f}" for k, v in sorted(p.limits.items(), key=lambda kv: kv[1]))
    out.append(f"  projected (linear extrapolation, NOT measured): {p.projected_eph:.0f} ep/h at N={p.concurrency}"
               f" vs {r.best.eph:.0f} measured now  [limits: {lim}]")
    return "\n".join(out)


def loop_summary(res) -> str:
    out = ["Rounds:"]
    for i, r in enumerate(res.rounds):
        act = res.actions[i] if i < len(res.actions) else ""
        mem = f"mem {r.mem_peak:.0%}" if r.mem_peak is not None else ""
        out.append(f"  {i + 1}. {str(r.layout):44s} {r.best.eph:7.0f} ep/h measured at N={r.best.concurrency:<3d} "
                   f"bottleneck {r.bottleneck:6s} [{act}] {mem}" + (f" {r.failures} failed" if r.failures else "")
                   + (f"  REJECTED: {r.guard}" if r.guard else ""))
    for s, why in res.frozen.items():
        if why != "cap tuning off":
            out.append(f"  {s} cap frozen: {why}")
    kind = "measured" if res.measured else "projected"
    out.append(f"Recommendation: {res.recommendation} at N={res.concurrency} ({kind}) — {res.reason}")
    return "\n".join(out)
