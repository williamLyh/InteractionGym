"""GPU tuner CLI (docs/GPU_TUNER.md).

Main path — the empirical measure -> rebalance loop (real episodes: examples/gpu_tuner.py loop):

    python -m interaction_gym.gpu_tuner dry-run --out runs/tune_dry        # the loop on a mock cluster
    python -m interaction_gym.gpu_tuner dry-run --no-apply                 # one round + proposal, no "restart"
    python -m interaction_gym.gpu_tuner dry-run --session-bound --rounds 6 # the session-cap sweep on a mock

Fallback — predictive (per-service probes + solver), when episodes cannot be run:

    python -m interaction_gym.gpu_tuner demand --from runs/suite --out d.json
    python -m interaction_gym.gpu_tuner supply --demand d.json --llm http://localhost:8000/v1 --replicas llm=1 --out s.json
    python -m interaction_gym.gpu_tuner predict --demand d.json --supply s.json --current agent=1,llm=2,tts=2,clone=1
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
from dataclasses import replace
from pathlib import Path

from .launch import launch_config, launch_script, load_host, write_outputs, write_yaml
from .profile import DemandProfile, SupplyProfile
from .report import alternatives, demand_table, plan_table, supply_table
from .solve import Plan, evaluate, solve

# an example starting layout on 8 GPUs (LLM TP2xDP2 on 0-3, MiniCPM-o on 4-5, TTS on 6 and 7, clone sharing 6)
EXAMPLE_LAYOUT = "agent=4+5,llm=0+1/2+3,tts=6/7,clone=6"
# the layout the tuner found best on an 8x RTX 5090 host (two MiniCPM-o servers), 4 sessions per server
EXAMPLE_TUNED_LAYOUT = "agent=0+1/2+3,llm=4+5,tts=6,clone=7"


def parse_kv(s: str | None, cast=int) -> dict:
    if not s:
        return {}
    return {k.strip(): cast(v) for k, v in (x.split("=") for x in s.split(",") if x.strip())}


def host_with(args):
    hw, specs = load_host(args.host)
    if getattr(args, "gpus", None):
        hw = replace(hw, gpus=[int(g) for g in args.gpus.split(",")])
    for s, n in parse_kv(getattr(args, "reserve", None)).items():  # --reserve trainer=2: keep 2 GPUs out of serving
        base = specs.get(s)
        specs[s] = replace(base, gpus=n, min_replicas=1, reserve=True) if base else type(next(iter(specs.values())))(name=s, gpus=n, min_replicas=1, reserve=True)
    if getattr(args, "penalty", None) is not None:
        hw = replace(hw, colocation_penalty=args.penalty)
    return hw, specs


def plan_and_report(demand: DemandProfile, supply: SupplyProfile, hw, specs, args, out: Path | None) -> tuple[list[Plan], Plan | None, str]:
    kw = dict(clock=args.clock, overhead_s=args.overhead)
    plans = solve(specs, demand, supply, hw, top=args.top, **kw)
    cur = None
    current = parse_kv(args.current) if args.current else None
    if current:
        cur = evaluate({k: v for k, v in current.items() if v and (k in demand.services or specs[k].reserve)}, specs, demand, supply, hw, **kw)
    text = [demand_table(demand), "", supply_table(supply), "", plan_table(plans[0], "Recommended layout (predicted)", demand), ""]
    if cur:
        text += [plan_table(cur, "Current layout (predicted)", demand), "",
                 f"Predicted gain over current: {plans[0].eph / cur.eph - 1:+.0%}", ""]
    elif current:
        text += [f"Current layout {current} does not fit the host / demand", ""]
    text.append(alternatives(plans))
    report = "\n".join(text)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        demand.save(out / "demand.json")
        supply.save(out / "supply.json")
        write_yaml(launch_config(plans[0], specs, hw, demand), out / "layout.yaml")
        (out / "launch.sh").write_text(launch_script(plans[0], specs, hw))
    return plans, cur, report


def cmd_demand(args):
    src = Path(args.src)
    eps = [json.loads(line) for line in open(src / "episodes.jsonl")]
    tr = src / "agent_traces.jsonl"
    traces = [json.loads(line) for line in open(tr)] if tr.exists() else None
    d = DemandProfile.from_episodes(eps, traces)
    print(demand_table(d))
    if args.out:
        d.save(args.out)
        print("->", args.out)


async def cmd_supply(args):
    from .probe import probe_agent, probe_llm, probe_tts

    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=256))  # urllib calls run in threads
    d = DemandProfile.load(args.demand) if args.demand else DemandProfile.synthetic()
    reps = parse_kv(args.replicas)
    levels = [int(x) for x in args.levels.split(",")]
    curves = {}
    if args.llm:
        x = d.services.get("llm")
        curves["llm"] = await probe_llm(args.llm, args.llm_model, prompt_tokens=round(x.prompt_tokens) if x else 700,
                                        completion_tokens=max(1, round(x.completion_tokens)) if x else 25, levels=levels,
                                        seconds=args.seconds, replicas=reps.get("llm", 1))
    if args.tts:
        x = d.services.get("tts")
        curves["tts"] = await probe_tts(args.tts, args.tts_model, audio_s=x.units_per_call if x and x.calls else 3.0, levels=levels,
                                        seconds=args.seconds, replicas=reps.get("tts", 1))
    if args.clone:
        from ..audio import Audio
        from ..clients import OpenAISpeech

        x = d.services.get("clone")
        if args.ref_wav:
            ref = Audio.read_wav(args.ref_wav)
        elif args.tts:
            ref = await OpenAISpeech(args.tts, args.tts_model).synth("Hi, I'd like to book a table for tonight, please.", "vivian")
        else:
            from .probe import speechlike

            ref = speechlike(3000, 24000)
        curves["clone"] = await probe_tts(args.clone, args.clone_model, audio_s=x.units_per_call if x and x.calls else 3.0,
                                          levels=levels, seconds=args.seconds, replicas=reps.get("clone", 1), ref_audio=ref,
                                          ref_text="Hi, I'd like to book a table for tonight, please.", name="clone")
    if args.agent:
        curves["agent"] = await probe_agent(args.agent, ref_audio=args.agent_ref_audio, levels=[int(k) for k in args.agent_levels.split(",")],
                                            sim_s=args.agent_sim_s, audio_out=args.agent_audio_out)
    sp = SupplyProfile(curves, [f"probed: {', '.join(curves)}"])
    if args.merge:
        sp = SupplyProfile.load(args.merge).merge(sp)
    print(supply_table(sp))
    if args.out:
        sp.save(args.out)
        print("->", args.out)


def cmd_solve(args):
    hw, specs = host_with(args)
    d = DemandProfile.load(args.demand) if args.demand else DemandProfile.synthetic()
    sp = SupplyProfile.load(args.supply) if args.supply else SupplyProfile.synthetic()
    missing = [s for s in d.services if s not in sp.curves]
    if missing and args.fill_synthetic:
        sp = SupplyProfile.synthetic().merge(sp)
        sp.notes.append(f"synthetic curves for unprobed: {missing}")
    out = Path(args.out) if args.out else None
    _, _, report = plan_and_report(d, sp, hw, specs, args, out)
    print(report)
    if out:
        (out / "report.txt").write_text(report + "\n")
        print(f"-> {out}/layout.yaml, launch.sh, report.txt")


async def cmd_dry_run(args):
    """The measure -> rebalance loop against a mock cluster: fake services with the synthetic capacities,
    except that the 'true' world is harsher than the profiles (co-located replicas halve each other's
    speed, the LLM is 10% slower). Starts from ``EXAMPLE_LAYOUT`` unless --layout is given."""
    from .empirical import MockCluster, mock_truth, session_bound_mock
    from .monitor import Monitor
    from .rebalance import Layout, loop
    from .report import loop_summary, proposal_table, round_table

    hw, specs = host_with(args)
    d = DemandProfile.synthetic()
    if args.session_bound:  # the agent is session-bound at low GPU util and its memory grows per session
        mock = session_bound_mock(specs, hw, time_scale=args.time_scale)
    else:
        mock = MockCluster(d, mock_truth(SupplyProfile.synthetic(), {"llm": 1.1}), specs, replace(hw, colocation_penalty=1.0),
                           time_scale=args.time_scale)
    layout = Layout.parse(args.layout or (EXAMPLE_TUNED_LAYOUT if args.session_bound else EXAMPLE_LAYOUT))
    mock.start(layout.placement)
    print("DRY RUN on a mock cluster: 'measured' numbers come from simulated services, not GPUs.\n")
    res = await loop(layout, lambda lay, meter: mock.runner(meter), specs, hw, apply=None if args.no_apply else mock.apply_layout,
                     monitor_for=lambda lay: Monitor(sample_fn=mock.sample, interval=2.0 * args.time_scale), rounds=args.rounds,
                     levels=[int(x) for x in args.levels.split(",")], seconds=args.seconds * args.time_scale, scale=1 / args.time_scale,
                     keep={k: 1 for k in parse_kv(args.reserve)} or None, cap_util=args.cap_util, cap_gain=args.cap_gain,
                     mem_limit=args.mem_limit, caps=not args.no_caps)
    text = []
    for i, r in enumerate(res.rounds):
        text += ["", round_table(r, f"Round {i + 1}"), proposal_table(r, res.proposals[i]) if i < len(res.proposals) else ""]
    text += ["", loop_summary(res)]
    report = "\n".join(text)
    print(report)
    if args.out:
        eph = res.best.best.eph if res.measured else res.proposals[-1].projected_eph
        out = write_outputs(args.out, res.recommendation, res.concurrency, specs, hw, eph=eph, measured=res.measured,
                            report="DRY RUN (mock cluster)\n" + report, extra={"dry_run": True})
        print(f"-> {out}/layout.yaml, launch.sh, report.txt")


def add_cap_args(ap):
    ap.add_argument("--cap-util", type=float, default=0.6, help="raise a concurrency-bound bottleneck's cap (instead of adding "
                    "replicas) while its GPU util is below this")
    ap.add_argument("--cap-gain", type=float, default=0.05, help="keep a cap raise only if episodes/hour rose at least this much")
    ap.add_argument("--mem-limit", type=float, default=0.92, help="roll back a setting that fills any GPU above this fraction")
    ap.add_argument("--no-caps", action="store_true", help="tune GPU counts only, not concurrency caps")


def add_solver_args(ap):
    ap.add_argument("--host", default=None, help="host preset: a TOML path or a name in gpu_tuner/hosts (default example-8gpu)")
    ap.add_argument("--gpus", help="override the usable GPU ids, e.g. 0,1,2,3,4,5")
    ap.add_argument("--reserve", help="keep GPUs out of serving, e.g. trainer=2")
    ap.add_argument("--penalty", type=float, help="co-location slowdown per extra replica on a GPU (default from the preset)")
    ap.add_argument("--current", help="the running layout, e.g. agent=1,llm=2,tts=2,clone=1 (to compare against)")
    ap.add_argument("--clock", default="input", choices=["input", "realtime"])
    ap.add_argument("--overhead", type=float, default=3.0, help="per-episode fixed wall seconds (connect, reset)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m interaction_gym.gpu_tuner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dry-run", help="the measure -> rebalance loop on a mock cluster")
    p.add_argument("--host", default=None)
    p.add_argument("--gpus")
    p.add_argument("--reserve", help="keep replicas fixed / GPUs out of the split, e.g. trainer=1")
    p.add_argument("--penalty", type=float)
    p.add_argument("--layout", help=f"starting layout (default {EXAMPLE_LAYOUT})")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--no-apply", action="store_true", help="stop after the first proposal (as without --allow-restart)")
    p.add_argument("--levels", default="2,4,8,16")
    p.add_argument("--seconds", type=float, default=120.0, help="(mock) seconds per concurrency level")
    p.add_argument("--time-scale", type=float, default=0.002)
    p.add_argument("--session-bound", action="store_true",
                   help="mock agent session-bound at low GPU util, memory growing per session (exercises the cap knob)")
    add_cap_args(p)
    p.add_argument("--out")
    p = sub.add_parser("demand")
    p.add_argument("--from", dest="src", required=True, help="a run dir with episodes.jsonl (+ agent_traces.jsonl)")
    p.add_argument("--out")
    p = sub.add_parser("supply")
    p.add_argument("--demand", help="demand.json (probe sizes follow it); default synthetic")
    p.add_argument("--llm")
    p.add_argument("--llm-model", default="Qwen3.8-27B")
    p.add_argument("--tts")
    p.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    p.add_argument("--clone")
    p.add_argument("--clone-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--ref-wav")
    p.add_argument("--agent", help="ws://.../v1/realtime?duplex=1 (lockstep sessions; keep levels small on shared servers)")
    p.add_argument("--agent-ref-audio")
    p.add_argument("--agent-levels", default="1,2")
    p.add_argument("--agent-sim-s", type=float, default=20.0)
    p.add_argument("--agent-audio-out", action="store_true", help="the agent speaks (full audio deployment); default: text-only "
                   "sessions for a Thinker-only server")
    p.add_argument("--levels", default="1,2,4,8,16")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--replicas", help="replicas behind each probed endpoint, e.g. llm=2,tts=2")
    p.add_argument("--merge", help="add these curves to an existing supply.json")
    p.add_argument("--out")
    p = sub.add_parser("predict", help="fallback: solve from per-service profiles")
    add_solver_args(p)
    p.add_argument("--demand")
    p.add_argument("--supply")
    p.add_argument("--fill-synthetic", action="store_true", help="use synthetic curves for services not probed")
    args = ap.parse_args(argv)
    if args.cmd == "dry-run":
        asyncio.run(cmd_dry_run(args))
    elif args.cmd == "demand":
        cmd_demand(args)
    elif args.cmd == "supply":
        asyncio.run(cmd_supply(args))
    elif args.cmd == "predict":
        cmd_solve(args)


if __name__ == "__main__":
    main()
