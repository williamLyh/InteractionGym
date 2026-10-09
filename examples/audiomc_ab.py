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

from fdb2_ab import AGENTS, CASCADED_SPEC, DUPLEX_SPEC, SR, TTS_MODEL, TTS_URL, Judge, _done, _strip, boot_ci, describe, llm, make_agent, timing, timing_table, validity, validity_table  # noqa: E402

from interaction_gym import Env
from interaction_gym.agents.vllm_omni import check_output_mode
from interaction_gym.benchmarks import audiomc, benchmark_user
from interaction_gym.clients import OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, save
from interaction_gym.user import ResponseDelay, TurnTaking, Voice

CLONE_URL = getenv("IG_CLONE_URL", "http://127.0.0.1:8005/v1")
CLONE_MODEL = getenv("IG_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
MAX_MS = 900_000
AGENT_PROMPTS = {"minicpmo": "You are a helpful AI assistant. Always speak in English.",
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
    env = make_env(task, cond, seed, DUPLEX_SPEC if kind == "duplex" else CASCADED_SPEC)
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
    if duplex and jobs:
        env, agent, wall = await run_one(jobs[0][0], "A", jobs[0][1], args.agent, {})
        print(f"warm-up: {env.t / 1000:.1f}s sim in {wall:.1f}s wall", flush=True)
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

async def report(args) -> None:
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max(32, args.concurrency)))  # LLM calls run in threads
    out = Path(args.out)
    tasks = {t.id: t for t in audiomc.load(args.data)}
    judge = Judge(llm(0, 0.0, 800), out / "judge_cache.json")
    eps = []
    for p in sorted(out.glob("*/*/episodes.jsonl")):
        eps += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(ep):
        task = tasks[ep["meta"]["task_id"]]
        conv = audiomc.history(ep)
        async with sem:
            verdicts = []
            for item in task.criteria["rubric"]:
                raw = await judge.ask(audiomc.judge_prompt(conv, item), max_tokens=800)
                try:
                    verdicts.append(bool(json.loads(_strip(raw)).get("criteria_met")))
                except Exception:  # noqa: BLE001
                    verdicts.append(None)
            val = await validity(ep, judge)
        users = [t for t in sorted(ep["turns"], key=lambda t: t["start_time"]) if t["role"] == "user" and t["text"]]
        script = task.scenario["script"]
        return {"id": ep["meta"]["episode_id"], "task_id": task.id, "axis": task.criteria["axis"], "agent": ep["meta"]["agent_name"],
                "cond": ep["meta"]["condition"], "seed": ep["meta"].get("seed"), "verdicts": verdicts,
                "rubric_rate": sum(v for v in verdicts if v is not None) / max(1, sum(v is not None for v in verdicts)),
                "all_pass": all(v for v in verdicts if v is not None) and any(v is not None for v in verdicts),
                "final_reply": audiomc.final_reply(ep), "user_turns": len(users), "script_turns": len(script),
                "complete": len(users) >= len(script), "duration_ms": ep["meta"]["duration_ms"], "end_reason": ep["meta"]["end_reason"],
                "timing": timing(ep), "validity": val,
                "u1_ms": users[1]["start_time"] if len(users) > 1 else None,
                "speech_before_u1": [(t["start_time"], t["text"]) for t in sorted(ep["turns"], key=lambda t: t["start_time"])
                                     if t["role"] != "user" and t["text"] and len(users) > 1 and t["start_time"] < users[1]["start_time"]]}

    rows = await asyncio.gather(*(one(ep) for ep in eps))
    judge.flush()
    by = {(r["agent"], r["task_id"], r["seed"], r["cond"]): r for r in rows}
    for r in rows:
        a = by.get((r["agent"], r["task_id"], r["seed"], "A"))
        if r["cond"] == "B" and a:
            cut = min(x for x in (a["u1_ms"], r["u1_ms"], 10**9) if x is not None)
            r["consistent"] = [x for x in a["speech_before_u1"] if x[0] < cut] == [x for x in r["speech_before_u1"] if x[0] < cut]
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    agents = sorted({r["agent"] for r in rows})
    L = ["# Audio MultiChallenge: open loop (recorded turns) vs closed loop (script replanned to the agent)", "",
         "Judge: official prompt, per rubric item, local proxy LLM; history = the conversation as it happened.", "",
         "| agent | axis | n pairs | rubric pass A | rubric pass B | B−A [95% CI] | all items A | all items B | B−A [95% CI] | complete A / B |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        for axis in (None, *audiomc.AXES):
            P = [(by[(ag, r["task_id"], r["seed"], "A")], r) for r in rows if r["agent"] == ag and r["cond"] == "B"
                 and (axis is None or r["axis"] == axis) and (ag, r["task_id"], r["seed"], "A") in by]
            if not P:
                continue
            m = lambda xs: sum(xs) / len(xs)  # noqa: E731
            d1 = [b["rubric_rate"] - a["rubric_rate"] for a, b in P]
            d2 = [float(b["all_pass"]) - float(a["all_pass"]) for a, b in P]
            c1, c2 = boot_ci(d1), boot_ci(d2)
            L.append(f"| {ag} | {axis or 'all'} | {len(P)} | {m([a['rubric_rate'] for a, _ in P]):.3f} | {m([b['rubric_rate'] for _, b in P]):.3f} | "
                     f"{m(d1):+.3f} [{c1[0]:+.3f}, {c1[1]:+.3f}] | {m([a['all_pass'] for a, _ in P]):.3f} | {m([b['all_pass'] for _, b in P]):.3f} | "
                     f"{m(d2):+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}] | {m([a['complete'] for a, _ in P]):.2f} / {m([b['complete'] for _, b in P]):.2f} |"
                     if c1[0] is not None else f"| {ag} | {axis or 'all'} | {len(P)} | (too few) |")
    L += ["", "Consistency (agent speech before the user's 2nd turn, A vs B):"]
    for ag in agents:
        C = [r for r in rows if r["agent"] == ag and "consistent" in r]
        if C:
            L.append(f"- {ag}: {sum(r['consistent'] for r in C)}/{len(C)}")
    L += timing_table(rows, agents) + validity_table(rows, agents)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


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
