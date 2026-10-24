"""Full-Duplex-Bench v2: open-loop (static script) vs closed-loop (live examiner, the official protocol) for one agent.

    # 1. reference scripts: one agent-free examiner/assistant dialogue per (task, seed), examiner lines rendered once
    PYTHONPATH=src:. python examples/fdb2_ab.py prepare --data prompts_staged_200.json --out runs/fdb2_ab --seeds 0
    # 2. episodes: condition A (replay of the script) and B (live examiner with the script's first line)
    PYTHONPATH=src:. python examples/fdb2_ab.py run --data prompts_staged_200.json --out runs/fdb2_ab --agent minicpmo --seeds 0
    # 3. official judge (proxy LLM), tables, A = B check up to the examiner's 2nd line
    PYTHONPATH=src:. python examples/fdb2_ab.py report --data prompts_staged_200.json --out runs/fdb2_ab

Benchmark port and conditions: ``interaction_gym.benchmarks.fdb2`` (docs/BENCHMARKS.md §12). Agents: ``minicpmo`` (MiniCPM-o 4.5 on vLLM-Omni, official
examinee prompt; text-only output timed at ``speech_cps`` against a Thinker-only server by default, ``--audio-out`` for
the talker's speech on the full audio deployment — the two are not comparable, docs/BENCHMARKS.md), ``minicpmo-confirm`` (same model, a voice-assistant prompt that confirms details back),
``cascaded`` (energy-VAD endpointing, Qwen3-ASR, Qwen3.8-27B, Qwen3-TTS; no tools). Services: IG_LLM_URL (examiner,
reference assistant, cascaded LLM, judge), IG_TTS_URL (examiner voice, cascaded voice), IG_CLONE_URL (examiner lines cloned from its first), IG_ASR_URL, IG_AGENT_URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym import turn_validity as tv
from interaction_gym.agents.cascaded import CascadedAgent, Endpointing, LatencyModel, ToolChat
from interaction_gym.agent_proxy import servers_behind
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent, check_output_mode, duplex_model, duplex_spec
from interaction_gym.benchmarks import benchmark_user, fdb2
from interaction_gym.clients import OpenAIChat, OpenAISpeech, OpenAITranscribe
from interaction_gym.envvars import getenv
from interaction_gym.eval import timing_counts, timing_summary
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, save
from interaction_gym.user import ReplayUser, ResponseDelay, TurnTaking, Voice
from interaction_gym.benchmarks.reports import (  # noqa: F401 — audiomc_ab.py imports the shared pieces from here
    Judge, _dialogue_upto, _fmt, _mean, _ms, _pct, _speech, _strip, _users, boot_ci, overlap_stats, timing, timing_table, validity, validity_table)
from interaction_gym.benchmarks.reports.fdb2 import COHERENCE_PROMPT, score_ep, stage_tables  # noqa: F401
from interaction_gym.benchmarks.reports.fdb2 import report as _report


async def report(args) -> None:
    """The report (interaction_gym.benchmarks.reports.fdb2) with the local proxy LLM as the judge."""
    await _report(args, llm(0, 0.0, 1500))

SR = 16000
LLM_URL = getenv("IG_LLM_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = getenv("IG_LLM_MODEL", "Qwen3.8-27B")
ASR_URL = getenv("IG_ASR_URL", "http://127.0.0.1:8006/v1")
ASR_MODEL = getenv("IG_ASR_MODEL", "Qwen3-ASR-1.7B")
TTS_URL = getenv("IG_TTS_URL", "http://127.0.0.1:8001/v1")
# the examiner keeps one voice: every line after its first is cloned from it (Qwen3-TTS Base)
CLONE_URL = getenv("IG_CLONE_URL", "http://127.0.0.1:8005/v1")
CLONE_MODEL = getenv("IG_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
TTS_MODEL = getenv("IG_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
AGENT_URL = getenv("IG_AGENT_URL", "ws://127.0.0.1:8010/v1/realtime?duplex=1")
AUDIO_OUT = False  # MiniCPM-o output: False = text timed at speech_cps (Thinker-only server); --audio-out = the talker's speech
EXAMINER_VOICE = "serena"
EXAMINER_MAX_TURNS = 12
CASCADED_SPEC = AgentSpec(chunk_ms=100, obs=("user.speech",), audio="user.audio", sr=SR)
CONFIRM_PROMPT = ("You are a helpful voice assistant on a live phone call. Always speak in English. Let the caller finish speaking "
                  "before you reply. Help them with what they ask; when they give details (items, names, times, numbers), confirm "
                  "them back with the latest values, and if they change something, use the new value from then on. Ask only for "
                  "what you still need. Keep every reply short and natural: it is spoken aloud.")
CASCADED_PROMPT = fdb2.EXAMINEE_PROMPT + " Your replies are spoken aloud on a live call: keep them short and conversational."
AGENTS = {"minicpmo": ("duplex", fdb2.EXAMINEE_PROMPT), "minicpmo-confirm": ("duplex", CONFIRM_PROMPT),
          "gander": ("duplex", fdb2.EXAMINEE_PROMPT), "personaplex": ("duplex", fdb2.EXAMINEE_PROMPT),
          "cascaded": ("cascaded", CASCADED_PROMPT)}


def llm(seed: int, temperature: float = 0.7, max_tokens: int = 200) -> OpenAIChat:
    return OpenAIChat(LLM_URL, LLM_MODEL, temperature=temperature, seed=seed, max_tokens=max_tokens)


def tts() -> OpenAISpeech:
    return OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)


def clone_tts() -> OpenAISpeech:
    return OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR)


# ---------------------------------------------------------------- prepare (agent-free reference scripts)

SCRIPTS: Path | None = None  # --scripts (default <out>/scripts): shared by every agent's run


def script_dir(out: Path, task_id: str, seed: int) -> Path:
    return (SCRIPTS or out / "scripts") / task_id / f"s{seed}"


async def prepare_one(task, seed: int, out: Path, speech) -> dict:
    d = script_dir(out, task.id, seed)
    if (d / "script.json").exists():
        return json.loads((d / "script.json").read_text())
    lines = await fdb2.make_script(task, llm(seed), llm(seed + 10_000), EXAMINER_MAX_TURNS)
    d.mkdir(parents=True, exist_ok=True)
    voice = Voice(speech, voice=EXAMINER_VOICE, clone=clone_tts())
    durs, k, ref = [], 0, None
    for x in lines:
        if x["role"] != "examiner":
            continue
        from interaction_gym.user import UserTurn

        seg = await voice.render(f"e{k}", 0, UserTurn(x["text"]), task, ref)
        ref = voice.reference(seg, ref)
        seg.data.write_wav(d / f"e{k:02d}.wav")
        x["audio"] = f"e{k:02d}.wav"
        durs.append(seg.data.dur_ms)
        k += 1
    starts = fdb2.schedule(lines, durs)
    ex = [x for x in lines if x["role"] == "examiner"]
    for x, t0, dur in zip(ex, starts, durs):
        x["t"], x["dur"] = t0, dur
    rec = {"task": task.id, "seed": seed, "lines": lines, "reached_end": fdb2._end(ex[-1]["text"]) if ex else False,
           "script_end_ms": (starts[-1] + durs[-1]) if ex else 0}
    (d / "script.json").write_text(json.dumps(rec, indent=1))
    return rec


async def prepare(args) -> None:
    out = Path(args.out)
    tasks = fdb2.load(args.data, only=args.ids.split(",") if args.ids else None)
    seeds = [int(s) for s in args.seeds.split(",")]
    sem, speech, n = asyncio.Semaphore(args.concurrency), tts(), [0]

    async def one(task, seed):
        async with sem:
            for attempt in range(4):
                try:
                    rec = await prepare_one(task, seed, out, speech)
                    break
                except Exception as ex:  # noqa: BLE001
                    print(f"{task.id} s{seed} attempt {attempt + 1}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                    await asyncio.sleep(20 + random.random() * 20)
            else:
                return
            n[0] += 1
            ex = [x for x in rec["lines"] if x["role"] == "examiner"]
            print(f"{n[0]} {task.id} s{seed}: {len(ex)} examiner lines, end={rec['reached_end']}, {rec['script_end_ms'] / 1000:.0f}s | {ex[0]['text'][:80]}", flush=True)

    await asyncio.gather(*(one(t, s) for s in seeds for t in tasks))


# ---------------------------------------------------------------- run

MAX_MS = fdb2.MAX_MS  # --max-ms (both conditions)
A_MAX_MS = fdb2.MAX_MS  # --a-max-ms: cap of condition A only (default: --max-ms)
CLOSING = False  # --closing: stage-progress-aware closing rules for the live examiner (fdb2.StageClosing)


def a_turns(rec: dict, d: Path) -> list[dict]:
    ex = [x for x in rec["lines"] if x["role"] == "examiner" and x["t"] < A_MAX_MS]
    return [{"t": x["t"], "text": x["text"], "audio": str(d / x["audio"]), "final": i == len(ex) - 1} for i, x in enumerate(ex)]


def make_env(task, rec: dict, d: Path, cond: str, seed: int) -> Env:
    spec = duplex_spec(task.scenario["_agent"], SR) if AGENTS[task.scenario["_agent"]][0] == "duplex" else CASCADED_SPEC
    if cond == "A":
        task.scenario["turns"] = a_turns(rec, d)
        end = min(rec["script_end_ms"] + 15_000, A_MAX_MS)
        return Env({"user": ReplayUser(voice=Voice())}, spec, max_ms=end, end_idle_ms=8000)
    first = [x for x in rec["lines"] if x["role"] == "examiner"][0]
    task.scenario["first_turn"] = {"text": first["text"], "audio": str(d / first["audio"])}
    closing = fdb2.StageClosing(llm(seed, 0.0, 60)) if CLOSING else None
    # no background floor / noise events: A (replayed) and B hear the same first line on the same silent line
    user = benchmark_user(fdb2.ExaminerSource(llm(seed), EXAMINER_MAX_TURNS, closing=closing), voice=Voice(tts(), voice=EXAMINER_VOICE, clone=clone_tts()),
                   timing=TurnTaking(response_delay=ResponseDelay(), yield_after_ms=1000, nudge_after_ms=10_000))
    return Env({"user": user}, spec, max_ms=MAX_MS, end_idle_ms=3000)


def make_agent(name: str, seed: int, cache: dict):
    kind, prompt = AGENTS[name]
    if kind == "duplex":
        return VllmOmniDuplexAgent(duplex_spec(name, SR), AGENT_URL, **{"audio_out": AUDIO_OUT, **duplex_model(name)}, out_sr=SR, clock="input",
                                   trace_tokens=True, session={"instructions": prompt})
    return CascadedAgent(CASCADED_SPEC, OpenAITranscribe(ASR_URL, ASR_MODEL), ToolChat(LLM_URL, LLM_MODEL, native=False, max_tokens=300),
                         tts(), instructions=prompt, voice="aiden", seed=seed, cache=cache,
                         endpointing=Endpointing(silence_ms=800), latency=LatencyModel())


async def run_one(task, rec, d, cond, seed, name, cache):
    task.scenario["_agent"] = name
    env = make_env(task, rec, d, cond, seed)
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


def describe(agent, name: str) -> dict:
    if isinstance(agent, VllmOmniDuplexAgent):
        return {"name": name, "kind": "full-duplex", "server": "vLLM-Omni duplex", "model": agent.model, "clock": agent.clock, "output": agent.output,
                "unit_ms": agent.unit_ms, "system_prompt": AGENTS[name][1]}
    return {**agent.describe(), "name": name}


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["meta"]["episode_id"] for line in path.read_text().splitlines() if line.strip()}


def eid(task_id: str, seed: int, cond: str, agent: str) -> str:
    return f"{agent}/{task_id}/s{seed}/{cond}"


async def run(args) -> None:
    out = Path(args.out)
    tag = args.agent + (args.tag or "")
    media = MediaStore(out)
    tasks = fdb2.load(args.data, only=args.ids.split(",") if args.ids else None)
    if args.limit:
        tasks = tasks[:: max(1, len(tasks) // args.limit)][: args.limit]
    conds, seeds = args.conditions.split(","), [int(s) for s in args.seeds.split(",")]
    done = {c: _done(out / tag / c / "episodes.jsonl") for c in conds}
    jobs = [(t, s) for s in seeds for t in tasks if any(eid(t.id, s, c, tag) not in done[c] for c in conds)]
    print(f"{len(tasks)} tasks, {len(jobs)} (task, seed) pairs to run, conditions {conds}, agent {args.agent} -> {tag}", flush=True)
    duplex = AGENTS[args.agent][0] == "duplex"
    for c in conds if duplex else ():
        check_output_mode(out / tag / c / "episodes.jsonl", AUDIO_OUT)  # never resume a run of the other output mode
    if duplex and jobs:  # the first session after a server start returns no audio: one throw-away episode per server
        t0, s0 = jobs[0]
        d0 = script_dir(out, t0.id, s0)
        n = servers_behind(AGENT_URL)  # behind the agent proxy, at once: one lands on each server
        warm = await asyncio.gather(*[run_one(fdb2.load(args.data, only=[t0.id])[0], json.loads((d0 / "script.json").read_text()),
                                              d0, "A", s0, args.agent, {}) for _ in range(n)])
        print(f"warm-up ({n} server{'s' * (n > 1)}): {warm[0][0].t / 1000:.1f}s sim in {max(w for _, _, w in warm):.1f}s wall", flush=True)
    sem, t_start, n_ok = asyncio.Semaphore(args.concurrency), time.monotonic(), [0]

    async def pair(task, seed):
        d = script_dir(out, task.id, seed)
        if not (d / "script.json").exists():
            print(f"no script for {task.id} s{seed}", flush=True)
            return
        rec = json.loads((d / "script.json").read_text())
        async with sem:
            cache: dict = {}
            for cond in conds:
                e = eid(task.id, seed, cond, tag)
                if e in done[cond]:
                    continue
                for attempt in range(args.retries):
                    try:
                        tk = fdb2.load(args.data, only=[task.id])[0]
                        env, agent, wall = await run_one(tk, rec, d, cond, seed, args.agent, cache)
                        break
                    except Exception as ex:  # noqa: BLE001  (a service restarting: wait and retry)
                        print(f"{e} attempt {attempt + 1}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                        await asyncio.sleep(45 + random.random() * 30)
                else:
                    print(f"GAVE UP {e}", flush=True)
                    return
                src = getattr(env.nodes.get("user") if hasattr(env, "nodes") else None, "source", None)
                ep = episode(env, e, media=media, run_id=f"fdb2-ab-{tag}",
                             meta={"agent": describe(agent, args.agent), "agent_events": getattr(agent, "events", []),
                                   "condition": cond, "agent_name": tag, "agent_base": args.agent, "task_id": task.id, "wall_s": round(wall, 1),
                                   "max_ms": A_MAX_MS if cond == "A" else MAX_MS, "closing": CLOSING and cond == "B",
                                   **({"examiner_log": src.log} if getattr(src, "log", None) else {})})
                save([ep], out / tag / cond / "episodes.jsonl", append=True)
                if duplex and agent.trace(e):
                    save([agent.trace(e)], out / tag / cond / "agent_traces.jsonl", append=True)
                n_ok[0] += 1
                users = [t for t in ep["turns"] if t["role"] == "user" and t["text"]]
                rate = n_ok[0] / (time.monotonic() - t_start)
                print(f"{n_ok[0]} {e:55s} {env.t / 1000:5.1f}s sim {wall:5.1f}s wall users={len(users)} end={fdb2.reached_end(ep)} "
                      f"{rate * 3600:.0f} ep/h", flush=True)

    await asyncio.gather(*(pair(*j) for j in jobs))


# ---------------------------------------------------------------- report

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run", "report"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="runs/fdb2_ab")
    ap.add_argument("--agent", default="minicpmo", choices=sorted(AGENTS))
    ap.add_argument("--tag", default="", help="suffix of the run folder (e.g. -c1 for a batch-1 rerun)")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--conditions", default="A,B")
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--scripts", default="", help="folder of the reference scripts (default <out>/scripts)")
    ap.add_argument("--max-ms", type=int, default=fdb2.MAX_MS, help="time cap of both conditions (official: 120 s)")
    ap.add_argument("--a-max-ms", type=int, default=0, help="time cap of condition A only (default: --max-ms); "
                    "lets A (official 120 s) and B (e.g. 180 s) run in one invocation, sharing the cascaded agent's model-call cache")
    ap.add_argument("--window-ms", type=int, default=0, help="report: score every conversation on its first window-ms (default: its run's cap)")
    ap.add_argument("--name", default="", help="report: suffix of summary / rows file names")
    ap.add_argument("--closing", action="store_true", help="stage-progress-aware closing rules for the live examiner (B)")
    ap.add_argument("--audio-out", action="store_true", help="MiniCPM-o speaks (Thinker + Talker + Code2Wav server); "
                    "default: text-only output timed at speech_cps, Thinker-only server")
    args = ap.parse_args()
    global SCRIPTS, MAX_MS, A_MAX_MS, CLOSING, AUDIO_OUT
    SCRIPTS = Path(args.scripts) if args.scripts else None
    MAX_MS, CLOSING, AUDIO_OUT = args.max_ms, args.closing, args.audio_out
    A_MAX_MS = args.a_max_ms or args.max_ms
    asyncio.run({"prepare": prepare, "run": run, "report": report}[args.cmd](args))


if __name__ == "__main__":
    main()
