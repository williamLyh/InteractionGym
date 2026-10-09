"""GPU tuner with real episodes (docs/GPU_TUNER.md): the empirical measure -> rebalance loop.
Runs on the GPU host (or from the Mac through ssh tunnels, with --gpu-ssh HOST for nvidia-smi).

    # one measurement round on the running layout + a proposed split (no restart), a few minutes
    PYTHONPATH=src:. python examples/gpu_tuner.py loop --layout "agent=2/3/4/5,llm=0+1,tts=6/7,clone=6" --out runs/tune
    # apply each proposal and re-measure, up to 3 rounds (RESTARTS services; the stop command is yours to give)
    PYTHONPATH=src:. python examples/gpu_tuner.py loop --layout ... --allow-restart --ssh gpu-host \\
        --stop-cmd "./stop.sh; ./stop_extras.sh" --rounds 3 --out runs/tune
    # fallback: per-episode demand with metered clients (input to the predictive solver)
    PYTHONPATH=src:. python examples/gpu_tuner.py demand --episodes 4 --concurrency 2 --out runs/tune

Episodes are the MiniCPM-o suite scenarios (examples/minicpmo_suite.py), spread over the layout's agent
servers. MiniCPM-o output is text only, timed at ``speech_cps``, against Thinker-only servers (one GPU each; host
preset gpu_tuner/hosts/example-8gpu-thinker.toml, layouts like ``agent=2/3/4/5``) unless ``--audio-out`` (the talker
speaks; preset example-8gpu.toml, two GPUs per server, e.g. ``agent=4+5``). ``loop`` takes endpoints from the layout +
host preset (``--host`` overrides it); ``demand`` uses IG_LLM_URL / IG_TTS_URL / IG_CLONE_URL / IG_AGENT_URLS.
``--agent canned`` replaces MiniCPM-o by a scripted agent (user-side services only).
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import sys
from pathlib import Path

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.clients import Cached, OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.gpu_tuner import Meter, MeteredAgent, MeteredChat, MeteredSpeech, SupplyProfile
from interaction_gym.gpu_tuner.__main__ import add_cap_args, host_with, parse_kv
from interaction_gym.gpu_tuner.launch import endpoints, layout_plan, write_outputs
from interaction_gym.gpu_tuner.monitor import GpuSampler, LogWatch, Monitor
from interaction_gym.gpu_tuner.probe import supply_from_meter
from interaction_gym.gpu_tuner.rebalance import Layout, loop
from interaction_gym.gpu_tuner.report import demand_table, loop_summary, proposal_table, round_table, supply_table
from interaction_gym.user import QWEN3_TTS_VOICES, LLMInterrupt, LLMSource, ResponseDelay, TurnTaking, UserSim, Voice

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for `examples.*`
from examples.minicpmo_suite import AGENT_MODEL, CLONE_MODEL, LLM_MODEL, REF_AUDIO, SCENARIOS, SR, TTS_MODEL  # noqa: E402

URLS = {"llm": getenv("IG_LLM_URL", "http://localhost:8000/v1"),
        "tts": getenv("IG_TTS_URL", "http://localhost:8001/v1"),
        "clone": getenv("IG_CLONE_URL", "http://localhost:8005/v1"),
        "agent": getenv("IG_AGENT_URLS", getenv("IG_AGENT_URL", "ws://127.0.0.1:8010/v1/realtime?duplex=1")).split(",")}
CANNED = ["Hello, how can I help you today?", "Sure, could you tell me a bit more?", "Okay, let me check that for you.",
          "That's done. Anything else?", "You're welcome, goodbye!"]


def make_runner(urls: dict, *, meter: Meter | None = None, agent_kind: str = "minicpmo", max_ms: int = 90_000, clone: bool = True,
                audio_out: bool = False):
    """An episode runner ``async (i) -> simulated ms``: scenario i (cycling through the suite), on the agent
    server with the fewest open sessions (round robin by episode index drifts: with N = servers x cap, one
    server would get cap + 1 sessions and refuse one); every client metered if ``meter`` is given."""
    names = list(SCENARIOS)
    open_sessions = {u: 0 for u in urls["agent"]}
    wrap = (lambda c, s: c) if meter is None else (lambda c, s: {"llm": MeteredChat, "agent": MeteredAgent}.get(s, MeteredSpeech)(c, meter, s))

    async def episode(i: int) -> int:
        name = names[i % len(names)]
        sc = SCENARIOS[name]
        llm = wrap(OpenAIChat(urls["llm"], LLM_MODEL, max_tokens=80), "llm")
        voice = Voice(Cached(wrap(OpenAISpeech(urls["tts"], TTS_MODEL, sr=SR), "tts")), voices=QWEN3_TTS_VOICES,
                      clone=wrap(OpenAISpeech(urls["clone"], CLONE_MODEL, sr=SR), "clone") if clone else False)
        user = UserSim(LLMSource(llm), voice, interrupt=LLMInterrupt(llm), timing=TurnTaking(response_delay=ResponseDelay()))
        task = Task(id=f"{name}-{i}", scenario={k: sc[k] for k in ("persona", "profile", "first_turn", "turn_taking", "background") if k in sc}
                    | {"instructions": sc["goal"]})
        spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)
        env = Env({"user": user}, spec, max_ms=max_ms)  # the user lays its own background (scenario / surroundings)
        url = None
        if agent_kind == "canned":
            inner = CannedAgent(CANNED, spec, min_words=2)
        else:
            from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent

            url = min(open_sessions, key=lambda u: (open_sessions[u], (urls["agent"].index(u) - i) % len(urls["agent"])))
            inner = VllmOmniDuplexAgent(spec, url, model=AGENT_MODEL, ref_audio=REF_AUDIO, clock="input", audio_out=audio_out,
                                        session={"instructions": sc["agent"]})
            open_sessions[url] += 1
        agent = wrap(inner, "agent") if meter is not None else inner
        token = meter.start_episode(task.id) if meter is not None else None
        done = False
        try:
            obs = await env.reset(task, seed=i)
            while not done:
                act = agent.act(env.t, obs)
                obs, _, done = await env.step(await act if asyncio.iscoroutine(act) else act)
        finally:
            try:
                if hasattr(agent, "close"):
                    await agent.close()
            finally:
                if url:
                    open_sessions[url] -= 1
                if meter is not None:
                    meter.end_episode(env.t, token)
        return env.t

    return episode


def plan_urls(plan, specs) -> dict:
    urls = dict(URLS)
    for s, es in endpoints(plan, specs).items():
        us = [e["url"] for e in es if "url" in e]
        if us:
            urls[s] = us if s == "agent" else us[0]
    return urls


async def cmd_demand(args):
    meter = Meter()
    run = make_runner(URLS, meter=meter, agent_kind=args.agent, max_ms=args.max_ms, clone=not args.no_clone, audio_out=args.audio_out)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(i):
        async with sem:
            sim = await run(i)
            print(f"  episode {i}: {sim / 1000:.1f} s simulated", flush=True)

    await asyncio.gather(*(one(i) for i in range(args.episodes)))
    d = meter.demand()
    insitu = SupplyProfile(supply_from_meter(meter, parse_kv(args.replicas)), ["in-situ: one point per service at the load of the demand run"])
    print(demand_table(d), "\n", supply_table(insitu), sep="\n")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d.save(out / "demand.json")
    insitu.save(out / "supply_insitu.json")
    meter.save(out / "meter.json")
    print(f"-> {out}/demand.json, supply_insitu.json, meter.json")


async def cmd_loop(args):
    hw, specs = host_with(args)
    layout = Layout.parse(args.layout)

    def metrics_urls(lay):  # every server's own port (a proxy has no /metrics; its backends do)
        eps = endpoints(layout_plan(lay, 1), specs)
        return {s: [f"http://127.0.0.1:{e['port']}" for e in es if e.get("port") and "backends" not in e] for s, es in eps.items()}

    def monitor_for(lay):
        return Monitor(gpu=None if args.no_gpu else GpuSampler(ssh=args.gpu_ssh), metrics=metrics_urls(lay), interval=args.interval)

    def runner_for(lay, meter):
        return make_runner(plan_urls(layout_plan(lay, 1), specs), meter=meter, agent_kind=args.agent, max_ms=args.max_ms,
                           clone=not args.no_clone, audio_out=args.audio_out)

    apply = alive = watch_for = None
    if args.allow_restart:
        from interaction_gym.gpu_tuner.remote import SSHLayout

        assert args.ssh and args.stop_cmd, "--allow-restart needs --ssh HOST and --stop-cmd (what to stop is your call)"

        async def warmup(plan):  # the first MiniCPM-o session after a start returns no audio: one throw-away episode per server
            urls = plan_urls(plan, specs)
            for i, u in enumerate(urls.get("agent", [])):
                await make_runner({**urls, "agent": [u]}, agent_kind=args.agent, max_ms=10_000, clone=not args.no_clone,
                                  audio_out=args.audio_out)(i)

        Path(args.out).mkdir(parents=True, exist_ok=True)
        ssh = SSHLayout(args.ssh, specs, hw, args.stop_cmd, scratch=Path(args.out), warmup=warmup)

        async def apply(lay):
            await ssh.apply(layout_plan(lay, 1))

        async def alive(lay):  # every server still answering (an external process may have killed them all)
            return await ssh.alive(layout_plan(lay, 1))

        def watch_for(lay):  # stage eviction / OOM lines in the servers' logs during a round
            return LogWatch(ssh.run, ssh.log_paths(layout_plan(lay, 1)))

    if args.start:  # bring the host into --layout first (e.g. after the services were stopped)
        assert apply is not None, "--start needs --allow-restart"
        print(f"starting {layout}")
        await apply(layout)
    res = await loop(layout, runner_for, specs, hw, apply=apply, monitor_for=monitor_for, rounds=args.rounds,
                     levels=[int(x) for x in args.levels.split(",")], seconds=args.seconds,
                     keep={k: 1 for k in parse_kv(args.reserve)} or None, max_concurrency=args.max_concurrency,
                     cap_util=args.cap_util, cap_gain=args.cap_gain, mem_limit=args.mem_limit, max_failures=args.max_failures,
                     caps=not args.no_caps, alive=alive, watch_for=watch_for)
    text = []
    for i, r in enumerate(res.rounds):
        text += ["", round_table(r, f"Round {i + 1}"), proposal_table(r, res.proposals[i]) if i < len(res.proposals) else ""]
    text += ["", loop_summary(res)]
    report = "\n".join(text)
    print(report)
    eph = res.best.best.eph if res.measured else res.proposals[-1].projected_eph
    out = write_outputs(args.out, res.recommendation, res.concurrency, specs, hw, eph=eph, measured=res.measured, report=report,
                        extra={"rounds": [{"layout": str(r.layout), "action": res.actions[i] if i < len(res.actions) else "",
                                           "measured_episodes_per_hour": round(r.best.eph, 1), "concurrency": r.best.concurrency,
                                           "bottleneck": r.bottleneck, "failed_episodes": r.failures,
                                           **({"gpu_mem_peak": round(r.mem_peak, 3)} if r.mem_peak is not None else {}),
                                           **({"agent_s_per_unit": round(r.stats["agent"].unit_s, 3)}
                                              if "agent" in r.stats and r.stats["agent"].unit_s else {}),
                                           **({"rejected": r.guard} if r.guard else {})}
                                          for i, r in enumerate(res.rounds)]})
    print(f"-> {out}/layout.yaml, launch.sh, report.txt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--agent", default="minicpmo", choices=["minicpmo", "canned"])
    common.add_argument("--max-ms", type=int, default=90_000, help="cap on an episode's simulated length")
    common.add_argument("--no-clone", action="store_true", help="users without a cloned voice (no clone TTS)")
    common.add_argument("--audio-out", action="store_true", help="MiniCPM-o speaks (Thinker + Talker + Code2Wav servers, "
                        "host preset example-8gpu); default: text-only output, Thinker-only servers (example-8gpu-thinker)")
    p = sub.add_parser("loop", parents=[common], help="measure the running layout, propose a split, optionally apply + re-measure")
    p.add_argument("--host", default=None, help="host preset (default example-8gpu-thinker; example-8gpu with --audio-out)")
    p.add_argument("--gpus")
    p.add_argument("--penalty", type=float)
    p.add_argument("--reserve", help="services held out of the split, e.g. trainer=1")
    p.add_argument("--layout", required=True, help='the running layout, e.g. "agent=4+5,llm=0+1/2+3,tts=6/7,clone=6" '
                   '(@n after a service = its per-replica cap, e.g. agent=0+1/2+3@6)')
    p.add_argument("--levels", default="2,4,8,16", help="env concurrency ramp (capped at the agent session capacity)")
    p.add_argument("--seconds", type=float, default=60.0, help="seconds per concurrency level")
    p.add_argument("--max-concurrency", type=int, help="never run more episodes at once (keep it light on shared services)")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--interval", type=float, default=2.0, help="monitor sampling period (s)")
    p.add_argument("--gpu-ssh", help="run nvidia-smi on this host over ssh (default: locally)")
    p.add_argument("--no-gpu", action="store_true", help="no GPU sampling (the split then follows env-side wait shares)")
    p.add_argument("--allow-restart", action="store_true", help="apply proposals by restarting services (disruptive)")
    p.add_argument("--ssh", help="host to restart, e.g. gpu-host, or local (the tuner runs on the GPU host)")
    p.add_argument("--start", action="store_true", help="(re)start the host into --layout before the first round")
    p.add_argument("--stop-cmd", help="run in the host workdir before each launch, e.g. './stop.sh; ./stop_extras.sh'")
    add_cap_args(p)
    p.add_argument("--max-failures", type=int, default=0, help="failed episodes a new setting may have before it is rolled back")
    p.add_argument("--out", default="runs/tune")
    p = sub.add_parser("demand", parents=[common], help="fallback: metered demand per episode")
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--replicas", help="replicas behind each endpoint, for the in-situ supply points, e.g. llm=2,tts=2")
    p.add_argument("--out", default="runs/tune")
    args = ap.parse_args()
    if getattr(args, "host", "") is None:
        args.host = "example-8gpu" if args.audio_out else "example-8gpu-thinker"
    loop_ = asyncio.new_event_loop()
    loop_.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=256))  # urllib calls run in threads
    loop_.run_until_complete({"demand": cmd_demand, "loop": cmd_loop}[args.cmd](args))


if __name__ == "__main__":
    main()
