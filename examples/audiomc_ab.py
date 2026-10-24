"""Audio MultiChallenge: open loop (recorded user turns replayed) vs closed loop (the script replanned to the agent).

    # once: python -c "from interaction_gym.benchmarks import audiomc; audiomc.extract('test.parquet', 'data/audiomc/wav')"
    PYTHONPATH=src:. python examples/audiomc_ab.py run --data data/audiomc/wav --out runs/audiomc_ab --agent minicpmo --per-axis 25
    PYTHONPATH=src:. python examples/audiomc_ab.py report --data data/audiomc/wav --out runs/audiomc_ab

Benchmark port and conditions: ``interaction_gym.benchmarks.audiomc`` (docs/BENCHMARKS.md §13). Agents as in
``fdb2_ab.py`` (``minicpmo``: MiniCPM-o 4.5, lockstep, text-only output timed at ``speech_cps`` on a Thinker-only
server by default, ``--audio-out`` for the talker's speech; ``cascaded``). Services: IG_LLM_URL (replanning
user, cascaded LLM, judge), IG_CLONE_URL (user voice cloned from the first recording), IG_TTS_URL, IG_ASR_URL,
IG_AGENT_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdb2_ab import AGENTS, CASCADED_SPEC, SR, TTS_MODEL, TTS_URL, Judge, _done, _strip, boot_ci, describe, llm, make_agent, timing, timing_table, validity, validity_table  # noqa: E402

from interaction_gym import Env
from interaction_gym.agent_proxy import servers_behind
from interaction_gym.agents.vllm_omni import check_output_mode, duplex_spec
from interaction_gym.benchmarks import audiomc, benchmark_user
from interaction_gym.clients import OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, save
from interaction_gym.user import ResponseDelay, TurnTaking, Voice
from interaction_gym.benchmarks.reports.audiomc import report as _report


async def report(args) -> None:
    """The report (interaction_gym.benchmarks.reports.audiomc) with the local proxy LLM as the judge."""
    await _report(args, llm(0, 0.0, 800))

CLONE_URL = getenv("IG_CLONE_URL", "http://127.0.0.1:8005/v1")
CLONE_MODEL = getenv("IG_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MAX_MS = 900_000
AGENT_PROMPTS = {"minicpmo": "You are a helpful AI assistant. Always speak in English.",
                 "gander": "You are a helpful AI assistant. Always speak in English.",
                 "personaplex": "You are a helpful AI assistant. Always speak in English.",
                 "cascaded": "You are a helpful AI assistant. Always speak in English. Your replies are spoken aloud on a live call: answer in at most four short sentences, no lists or markdown."}


class CloneWithFallback:
    def __init__(self, clone, plain, voice="ryan"):
        self.inner, self.plain, self.voice = clone, plain, voice
        self.model, self.sr = clone.model, clone.sr

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None, **kw):
        for t in (text, text.rstrip(".!?") + "."):
            try:
                return await self.inner.synth(t, voice, instructions, ref_audio, ref_text, language, **kw)
            except Exception:  # noqa: BLE001
                continue
        return await self.plain.synth(text, self.voice, None, language=language)


def make_env(task, cond: str, seed: int, spec) -> Env:
    timing = TurnTaking(response_delay=ResponseDelay(), yield_after_ms=None, nudge_after_ms=10_000)
    if cond == "A":
        user = benchmark_user(audiomc.ScriptSource(), voice=Voice(), timing=timing)
    else:
        # voice: cloned from the first recording (Voice uses the first turn with words as the reference)
        clone = CloneWithFallback(OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR), OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR))
        user = benchmark_user(audiomc.ReplanSource(llm(seed, 0.7, 600)), voice=Voice(clone, clone=clone), timing=timing)
    return Env({"user": user}, spec, max_ms=MAX_MS, end_idle_ms=8000)


async def run_one(task, cond, seed, name, cache):
    kind = AGENTS[name][0]
    env = make_env(task, cond, seed, duplex_spec(name, SR) if kind == "duplex" else CASCADED_SPEC)
    import fdb2_ab

    fdb2_ab.AGENTS[name] = (kind, AGENT_PROMPTS[name])
    agent = make_agent(name, seed, cache)
    wall = time.monotonic()
    try:
        obs = await env.reset(task, seed)
        while True:
            obs, _, done = await env.step(await agent.act(env.t, obs))
            if env.truncated or (done and not getattr(agent, "busy", False)):
                break
    finally:
        if hasattr(agent, "close"):
            await agent.close()
    return env, agent, time.monotonic() - wall


async def run(args) -> None:
    out, tag = Path(args.out), args.agent + (args.tag or "")
    media = MediaStore(out)
    tasks = audiomc.load(args.data, only=args.ids.split(",") if args.ids else None, per_axis=args.per_axis or None)
    conds, seeds = args.conditions.split(","), [int(s) for s in args.seeds.split(",")]
    eid = lambda t, s, c: f"{tag}/{t}/s{s}/{c}"  # noqa: E731
    done = {c: _done(out / tag / c / "episodes.jsonl") for c in conds}
    jobs = [(t, s) for s in seeds for t in tasks if any(eid(t.id, s, c) not in done[c] for c in conds)]
    print(f"{len(tasks)} conversations, {len(jobs)} pairs to run, agent {args.agent} -> {tag}", flush=True)
    duplex = AGENTS[args.agent][0] == "duplex"
    import fdb2_ab

    fdb2_ab.AUDIO_OUT = args.audio_out  # make_agent reads it
    for c in conds if duplex else ():
        check_output_mode(out / tag / c / "episodes.jsonl", args.audio_out)  # never resume a run of the other output mode
    if duplex and jobs:  # the first session after a server start returns no audio: one throw-away episode per server
        n = servers_behind(fdb2_ab.AGENT_URL)  # behind the agent proxy, at once: one lands on each server
        warm = await asyncio.gather(*[run_one(jobs[0][0], "A", jobs[0][1], args.agent, {}) for _ in range(n)])
        env, agent, wall = warm[0]
        print(f"warm-up ({n} server{'s' * (n > 1)}): {env.t / 1000:.1f}s sim in {wall:.1f}s wall", flush=True)
    sem, t_start, n_ok = asyncio.Semaphore(args.concurrency), time.monotonic(), [0]

    async def pair(task, seed):
        async with sem:
            cache: dict = {}
            for cond in conds:
                e = eid(task.id, seed, cond)
                if e in done[cond]:
                    continue
                for attempt in range(args.retries):
                    try:
                        env, agent, wall = await run_one(task, cond, seed, args.agent, cache)
                        break
                    except Exception as ex:  # noqa: BLE001
                        print(f"{e} attempt {attempt + 1}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                        await asyncio.sleep(45 + random.random() * 30)
                else:
                    print(f"GAVE UP {e}", flush=True)
                    return
                ep = episode(env, e, media=media, run_id=f"audiomc-ab-{tag}",
                             meta={"agent": describe(agent, args.agent), "condition": cond, "agent_name": tag, "agent_base": args.agent,
                                   "task_id": task.id, "wall_s": round(wall, 1)})
                save([ep], out / tag / cond / "episodes.jsonl", append=True)
                if duplex and agent.trace(e):
                    save([agent.trace(e)], out / tag / cond / "agent_traces.jsonl", append=True)
                n_ok[0] += 1
                users = [t for t in ep["turns"] if t["role"] == "user" and t["text"]]
                print(f"{n_ok[0]} {e:45s} {env.t / 1000:6.1f}s sim {wall:6.1f}s wall users={len(users)} "
                      f"{n_ok[0] / (time.monotonic() - t_start) * 3600:.0f} ep/h | {audiomc.final_reply(ep)[:80]}", flush=True)

    await asyncio.gather(*(pair(*j) for j in jobs))


# ---------------------------------------------------------------- report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/audiomc_ab")
    ap.add_argument("--agent", default="minicpmo", choices=sorted(AGENT_PROMPTS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--conditions", default="A,B")
    ap.add_argument("--ids", default="")
    ap.add_argument("--per-axis", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--audio-out", action="store_true", help="MiniCPM-o speaks (Thinker + Talker + Code2Wav server); "
                    "default: text-only output timed at speech_cps, Thinker-only server")
    args = ap.parse_args()
    asyncio.run({"run": run, "report": report}[args.cmd](args))


if __name__ == "__main__":
    main()
