"""Host presets (TOML) in, launch configs out: a YAML summary of the plan and a shell script that
starts every replica in tmux on the GPUs the plan gives it."""

from __future__ import annotations

import datetime as _dt
import json
import re
import shlex
import tomllib
from dataclasses import fields
from pathlib import Path

from .solve import Hardware, Plan, ServiceSpec

HOSTS = Path(__file__).parent / "hosts"


def load_host(path: str | Path | None = None) -> tuple[Hardware, dict[str, ServiceSpec]]:
    """A host preset: ``[hardware]`` + ``[services.<name>]`` tables (see hosts/example-8gpu.toml)."""
    p = Path(path) if path else HOSTS / "example-8gpu.toml"
    if not p.exists() and (HOSTS / f"{path}.toml").exists():
        p = HOSTS / f"{path}.toml"
    with open(p, "rb") as f:
        cfg = tomllib.load(f)
    hw_keys = {f.name for f in fields(Hardware)}
    hw = Hardware(**{k: v for k, v in cfg["hardware"].items() if k in hw_keys})
    sp_keys = {f.name for f in fields(ServiceSpec)}
    specs = {name: ServiceSpec(name=name, **{k: v for k, v in d.items() if k in sp_keys}) for name, d in cfg.get("services", {}).items()}
    return hw, specs


def fmt(template: str, **kw) -> str:
    """Fill {gpus} {gpu} {port} {dp} {backends} {cap}; any other braces (e.g. JSON in a command) stay as they are."""
    return re.sub(r"\{(\w+)\}", lambda m: str(kw[m.group(1)]) if m.group(1) in kw else m.group(0), template)


def endpoints(plan: Plan, specs: dict[str, ServiceSpec]) -> dict[str, list[dict]]:
    """The servers a plan starts, per service: [{name, gpus, port, cmd, url?, backends?}]."""
    out: dict[str, list[dict]] = {}
    for s, reps in plan.placement.items():
        sp = specs[s]
        if not reps:
            continue
        cap = plan.caps.get(s, sp.default_cap)
        knobs = {"cap": cap} if cap is not None else {}
        fill = lambda tpl, gpus, port: fmt(tpl, gpus=",".join(map(str, gpus)), gpu=gpus[0], port=port, dp=len(reps), **knobs)  # noqa: E731
        eps = []
        if sp.pool == "dp":
            gpus = [g for r in reps for g in r]
            eps.append({"name": s, "gpus": gpus, "port": sp.port, "cmd": fill(sp.launch, gpus, sp.port), "url": fmt(sp.url, port=sp.port)})
        elif sp.pool == "proxy" and len(reps) > 1:
            ports = [sp.backend_port + i for i in range(len(reps))]
            for i, (r, port) in enumerate(zip(reps, ports)):
                eps.append({"name": f"{s}{i}", "gpus": r, "port": port, "cmd": fill(sp.launch, r, port)})
            backends = ",".join(f"http://127.0.0.1:{p}" for p in ports)
            eps.append({"name": f"{s}proxy", "gpus": [], "port": sp.port, "backends": backends,
                        "cmd": fmt(sp.proxy_launch, port=sp.port, backends=backends), "url": fmt(sp.url, port=sp.port)})
        else:
            for i, r in enumerate(reps):
                port = sp.port + i if sp.port else 0
                eps.append({"name": f"{s}{i}" if len(reps) > 1 else s, "gpus": r, "port": port, "cmd": fill(sp.launch, r, port),
                            **({"url": fmt(sp.url, port=port)} if sp.url else {})})
        out[s] = eps
    return out


def _yaml(v, indent: int = 0) -> str:
    """A small YAML writer (no dependency): dicts, lists of scalars / dicts, scalars as JSON."""
    pad = "  " * indent
    lines = []
    if isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, (dict, list)) and x and not (isinstance(x, list) and all(not isinstance(i, (dict, list)) for i in x)):
                lines.append(f"{pad}{k}:")
                lines.append(_yaml(x, indent + 1))
            elif isinstance(x, str) and "\n" in x:
                lines.append(f"{pad}{k}: |")
                lines.extend(f"{pad}  {line}" for line in x.strip("\n").splitlines())
            else:
                lines.append(f"{pad}{k}: {json.dumps(x)}")
    elif isinstance(v, list):
        for x in v:
            if isinstance(x, dict):
                sub = _yaml(x, indent + 1).splitlines()
                lines.append(f"{pad}- {sub[0].strip()}")
                lines.extend(sub[1:])
            else:
                lines.append(f"{pad}- {json.dumps(x)}")
    return "\n".join(lines)


def launch_config(plan: Plan, specs: dict[str, ServiceSpec], hw: Hardware, demand=None) -> dict:
    eps = endpoints(plan, specs)
    env = {"concurrent_episodes": plan.concurrency, "clock": plan.clock}
    urls = {s: [e["url"] for e in es if "url" in e] for s, es in eps.items()}
    for s, us in urls.items():
        if us:
            env[f"{s}_urls" if len(us) > 1 else f"{s}_url"] = us if len(us) > 1 else us[0]
    agent_urls = urls.get("agent", [])
    exports = {"IG_LLM_URL": (urls.get("llm") or [None])[0], "IG_TTS_URL": (urls.get("tts") or [None])[0],
               "IG_CLONE_URL": (urls.get("clone") or [None])[0], "IG_AGENT_URL": agent_urls[0] if agent_urls else None,
               "IG_AGENT_URLS": ",".join(agent_urls) or None, "IG_CONCURRENCY": str(plan.concurrency)}
    return {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "host": hw.name,
        "prediction": {"episodes_per_hour": round(plan.eph, 1), "episode_wall_s": round(plan.episode_wall_s, 1),
                       "bottleneck": plan.bottleneck, **({"sim_s_per_episode": round(demand.sim_s, 1)} if demand else {})},
        "env": env,
        "env_exports": {k: v for k, v in exports.items() if v},
        "idle_gpus": plan.idle_gpus,
        "services": {s: {"replicas": plan.alloc[s], "gpus_per_replica": specs[s].gpus or f"shared ({specs[s].mem_gb:g} GB)",
                         "replica_gpus": plan.placement[s],
                         **({"cap": plan.caps.get(s, specs[s].default_cap)} if specs[s].default_cap is not None else {}),
                         **({"utilization": round(next((r.util for r in plan.rows if r.service == s), 0.0), 3)} if plan.rows else {}),
                         "servers": [{k: e[k] for k in ("name", "gpus", "port", "url", "backends") if k in e} for e in eps.get(s, [])]}
                     for s in plan.placement if plan.placement[s]},
    }


def write_yaml(cfg: dict, path: str | Path) -> Path:
    p = Path(path)
    p.write_text("# serving layout from interaction_gym.gpu_tuner (docs/GPU_TUNER.md)\n" + _yaml(cfg) + "\n")
    return p


def launch_script(plan: Plan, specs: dict[str, ServiceSpec], hw: Hardware) -> str:
    """A bash script that starts the plan's servers in tmux (it does not stop anything: stop the old
    layout first, e.g. ./stop.sh, or pick free GPUs/ports)."""
    eps = endpoints(plan, specs)
    out = ["#!/usr/bin/env bash",
           f"# Serving layout for {hw.name}: {plan.eph:.0f} episodes/h predicted at {plan.concurrency} concurrent episodes",
           "# (interaction_gym.gpu_tuner). Starts servers only — stop the current layout first; nothing here kills sessions.",
           "set -euo pipefail", f"cd {shlex.quote(hw.workdir)}", "mkdir -p logs tuned", ""]
    for s, es in eps.items():
        for e in es:
            sess = f"{hw.session_prefix}{e['name']}"
            body = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(hw.workdir)}", hw.env.strip(), e["cmd"].strip()])
            out += [f"# {s}: GPUs {e['gpus'] or '-'} port {e['port']}",
                    f"cat > tuned/{sess}.sh <<'IG_EOF'", body, "IG_EOF",
                    f"tmux has-session -t ={sess} 2>/dev/null && echo 'skip {sess}: session exists' || "
                    f"tmux new-session -d -s {sess} \"bash tuned/{sess}.sh 2>&1 | tee logs/{sess}.log\"", ""]
    exports = launch_config(plan, specs, hw)["env_exports"]
    out.append("echo 'env: " + " ".join(f"{k}={v}" for k, v in exports.items()) + "'")
    return "\n".join(out) + "\n"


def layout_plan(layout, concurrency: int, eph: float = 0.0, clock: str = "input") -> Plan:
    """A ``Plan`` view of a measured / proposed ``rebalance.Layout`` (for ``launch_config`` / ``launch_script``)."""
    return Plan(layout.alloc, {s: r for s, r in layout.placement.items() if r}, concurrency, eph, 0.0, clock=clock,
                caps=dict(getattr(layout, "caps", {}) or {}))


def write_outputs(out, layout, concurrency: int, specs, hw, *, eph: float, measured: bool, report: str, extra: dict | None = None):
    """layout.yaml + launch.sh + report.txt for a recommended layout."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    plan = layout_plan(layout, concurrency, eph)
    cfg = launch_config(plan, specs, hw)
    cfg["prediction"] = {("measured" if measured else "projected") + "_episodes_per_hour": round(eph, 1),
                         "source": "measured end-to-end" if measured else "linear extrapolation from the last measured round"}
    cfg["layout"] = str(layout)
    if extra:
        cfg.update(extra)
    write_yaml(cfg, out / "layout.yaml")
    (out / "launch.sh").write_text(launch_script(plan, specs, hw).replace(f"{eph:.0f} episodes/h predicted",
                                                                          f"{eph:.0f} episodes/h {'measured' if measured else 'projected'}"))
    (out / "report.txt").write_text(report + "\n")
    return out
