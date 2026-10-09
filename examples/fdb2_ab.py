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
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent, check_output_mode
from interaction_gym.benchmarks import benchmark_user, fdb2
from interaction_gym.clients import OpenAIChat, OpenAISpeech, OpenAITranscribe
from interaction_gym.envvars import getenv
from interaction_gym.eval import timing_counts, timing_summary
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, save
from interaction_gym.user import ReplayUser, ResponseDelay, TurnTaking, Voice

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
AGENT_MODEL = getenv("IG_AGENT_MODEL", "openbmb/MiniCPM-o-4_5")
REF_AUDIO = getenv("IG_AGENT_REF_AUDIO", str(Path(getenv("IG_MODEL_DIR", "models")) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
AUDIO_OUT = False  # MiniCPM-o output: False = text timed at speech_cps (Thinker-only server); --audio-out = the talker's speech
EXAMINER_VOICE = "serena"
EXAMINER_MAX_TURNS = 12
DUPLEX_SPEC = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)
CASCADED_SPEC = AgentSpec(chunk_ms=100, obs=("user.speech",), audio="user.audio", sr=SR)
CONFIRM_PROMPT = ("You are a helpful voice assistant on a live phone call. Always speak in English. Let the caller finish speaking "
                  "before you reply. Help them with what they ask; when they give details (items, names, times, numbers), confirm "
                  "them back with the latest values, and if they change something, use the new value from then on. Ask only for "
                  "what you still need. Keep every reply short and natural: it is spoken aloud.")
CASCADED_PROMPT = fdb2.EXAMINEE_PROMPT + " Your replies are spoken aloud on a live call: keep them short and conversational."
AGENTS = {"minicpmo": ("duplex", fdb2.EXAMINEE_PROMPT), "minicpmo-confirm": ("duplex", CONFIRM_PROMPT),
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
    spec = DUPLEX_SPEC if AGENTS[task.scenario["_agent"]][0] == "duplex" else CASCADED_SPEC
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
        return VllmOmniDuplexAgent(DUPLEX_SPEC, AGENT_URL, model=AGENT_MODEL, ref_audio=REF_AUDIO, out_sr=SR, clock="input",
                                   audio_out=AUDIO_OUT, trace_tokens=True, session={"instructions": prompt})
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
    if duplex and jobs:  # the first session after a server start returns no audio: one throw-away episode
        t0, s0 = jobs[0]
        d0 = script_dir(out, t0.id, s0)
        env, agent, wall = await run_one(fdb2.load(args.data, only=[t0.id])[0], json.loads((d0 / "script.json").read_text()), d0, "A", s0, args.agent, {})
        print(f"warm-up: {env.t / 1000:.1f}s sim in {wall:.1f}s wall", flush=True)
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

COHERENCE_PROMPT = """Below is a spoken conversation between a CALLER and an ASSISTANT, up to the caller's next line.

{dialogue}

The caller's next line is:
CALLER: "{line}"

Check strictly whether this line fits what the ASSISTANT ACTUALLY SAID right before it (not what a typical assistant would have said):
1. What did the assistant's last turn ask or say? (If it asked a question, which one?)
2. Does the caller's line answer that question / react to that statement? A line that answers a DIFFERENT question (e.g. gives a coffee size when the assistant asked how the eggs should be cooked), confirms or thanks for something the assistant never said or offered, supplies information nobody asked for while ignoring the question that was asked, or acts as if a step happened that did not, does NOT fit.
3. Volunteering extra details is fine only if the caller ALSO handles what the assistant asked.

Respond with ONLY a JSON object: {{"assistant_last": "...", "fits": true/false, "explanation": "brief reason"}}"""


class Judge:
    def __init__(self, llm_, cache_path: Path):
        self.llm, self.path = llm_, cache_path
        self.memo = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.new = 0

    async def ask(self, prompt: str, max_tokens: int = 1500) -> str:
        if prompt not in self.memo:
            for attempt in range(4):
                try:
                    self.memo[prompt] = await self.llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens)
                    self.new += 1
                    if self.new % 500 == 0:  # long reports: do not lose the cache to a crash
                        self.flush()
                    break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(20)
            else:
                return ""
        return self.memo[prompt]

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.memo))
        tmp.replace(self.path)


def _strip(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return raw


def _speech(ep, before=None):
    """Agent speech that is complete before ``before`` (a turn still running there can be cut differently once the
    two conditions diverge at the examiner's 2nd line, so it is left out)."""
    return [(t["start_time"], t["text"]) for t in sorted(ep["turns"], key=lambda t: t["start_time"])
            if t["role"] != "user" and t["text"] and t["end_time"] > t["start_time"] and (before is None or t["end_time"] <= before)]


def _users(ep):
    return [t for t in sorted(ep["turns"], key=lambda t: t["start_time"]) if t["role"] == "user" and t["text"]]


def _dialogue_upto(ep, t_cut):
    lines = []
    for t in sorted(ep["turns"], key=lambda t: t["start_time"]):
        if t["start_time"] >= t_cut or not t["text"] or t["end_time"] <= t["start_time"]:
            continue
        lines.append(("CALLER: " if t["role"] == "user" else "ASSISTANT: ") + t["text"])
    return "\n".join(lines) or "(nothing yet)"


def overlap_stats(ep) -> dict:
    """Examiner lines that start while the agent is talking (talked over / collisions) and agent turns started
    while the examiner talks (cut-ins), from the turn spans."""
    us, ag = _users(ep), [t for t in ep["turns"] if t["role"] != "user" and t["end_time"] > t["start_time"]]
    u_into_a = sum(any(a["start_time"] < u["start_time"] < a["end_time"] for a in ag) for u in us[1:])
    a_into_u = sum(any(u["start_time"] < a["start_time"] < u["end_time"] for u in us) for a in ag)
    return {"examiner_over_agent": u_into_a, "agent_over_examiner": a_into_u, "examiner_lines": len(us), "agent_turns": len(ag)}


async def validity(ep, judge: Judge, max_ms: int | None = None, context: str = "") -> list[dict]:
    """Per checked user turn (after the first): open-loop user-turn validity (``turn_validity``: timing rule + LLM
    content / ignored question / stale), with the turn id and start."""
    rules, out = tv.rule_flags(ep["turns"], end_ms=max_ms), []
    for u in tv.user_turns(ep["turns"], end_ms=max_ms):
        raw = await judge.ask(tv.prompt(ep["turns"], u, context=context), max_tokens=250)
        out.append({"turn": u["id"], "t": u["start_time"], **tv.combine(rules[u["id"]], tv.parse(_strip(raw) if raw else raw))})
    return out


def timing(ep) -> dict:
    return timing_counts(ep["turns"], (ep["eval"].get("scores") or {}).get("events"), end_ms=ep["meta"]["duration_ms"])


async def score_ep(ep, task, judge: Judge, window_ms: int | None = None) -> dict:
    """One conversation's row, scored on its first ``window_ms`` (default: the run's own cap)."""
    max_ms = window_ms or ep["meta"].get("max_ms") or fdb2.MAX_MS
    raw = await judge.ask(fdb2.judge_prompt(task, ep))
    j = fdb2.parse_judgement(_strip(raw))
    row = {"id": ep["meta"]["episode_id"], "task_id": task.id, "split": task.criteria["split"], "seed": ep["meta"].get("seed"),
           "cond": ep["meta"]["condition"], "agent": ep["meta"]["agent_name"], "duration_ms": ep["meta"]["duration_ms"],
           "end_reason": ep["meta"]["end_reason"], "reached_end": fdb2.reached_end(ep, max_ms), **fdb2.episode_scores(j, task.criteria["split"]),
           "judge_parsed": bool(j["events"]) or j["task"] is not None, "scores_total": (ep["eval"].get("scores") or {}).get("total"),
           "by_expectation": (ep["eval"].get("scores") or {}).get("by_expectation"), **overlap_stats(ep), "max_ms": max_ms,
           "run_max_ms": ep["meta"].get("max_ms") or fdb2.MAX_MS}
    if max_ms != fdb2.MAX_MS:  # the official judge reads the official 120 s window; a longer run is judged on all of it
        raw = await judge.ask(fdb2.judge_prompt(task, ep, max_ms=max_ms))
        row.update(fdb2.episode_scores(fdb2.parse_judgement(_strip(raw)), task.criteria["split"]))
    # coherence of each examiner line after the first, given what the agent actually said before it
    us, coh = _users(ep), []
    for u in us[1:]:
        if u["start_time"] >= max_ms:
            break
        r = await judge.ask(COHERENCE_PROMPT.format(dialogue=_dialogue_upto(ep, u["start_time"]), line=u["text"]), max_tokens=300)
        try:
            coh.append(bool(json.loads(_strip(r)).get("fits")))
        except Exception:  # noqa: BLE001
            pass
    row["coherent"] = coh
    row["u1_ms"] = us[1]["start_time"] if len(us) > 1 else None
    # duplex timing (collisions apart), open-loop user-turn validity, stage analysis + pacing, reached-stage judge
    row["timing"] = timing(ep)
    row["validity"] = await validity(ep, judge, max_ms)
    st = fdb2.parse_stages(await judge.ask(fdb2.stage_prompt(task, ep, max_ms), max_tokens=700))
    row["stages"], row["pacing"] = st, fdb2.pacing(ep, max_ms)
    row["timeout"] = not row["reached_end"]
    row["cap_cause"] = fdb2.cap_cause(st, row["pacing"]) if st and row["timeout"] else None
    if st is not None and row["timeout"]:
        jr = fdb2.parse_judgement(_strip(await judge.ask(fdb2.judge_prompt(task, ep, stages=st["reached"], max_ms=max_ms))))
        r = fdb2.episode_scores(jr, task.criteria["split"])
    else:
        r = {k: row[k] for k in ("tt", "if", "task")}
    row.update(tt_r=r["tt"], if_r=r["if"], task_r=r["task"])
    log = ep["meta"].get("examiner_log") or []
    row["forced_end"] = any(x.get("forced_end") for x in log)
    row["close_reason"] = next((x.get("close") for x in log if x.get("close")), None)
    return row


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def boot_ci(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    diffs = [d for d in diffs if d is not None]
    if len(diffs) < 3:
        return None, None
    ms = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


async def report(args) -> None:
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max(32, args.concurrency)))  # LLM calls run in threads
    out = Path(args.out)
    tasks = {t.id: t for t in fdb2.load(args.data)}
    judge = Judge(llm(0, 0.0, 1500), out / "judge_cache.json")
    agents = [p.name for p in sorted(out.iterdir()) if (p / "A").exists() or (p / "B").exists()]
    rows, sem = [], asyncio.Semaphore(args.concurrency)
    eps_by = {}
    for ag in agents:
        for cond in ("A", "B"):
            p = out / ag / cond / "episodes.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if line.strip():
                    ep = json.loads(line)
                    eps_by[ep["meta"]["episode_id"]] = ep

    async def one(ep):
        async with sem:
            return await score_ep(ep, tasks[ep["meta"]["task_id"]], judge, args.window_ms or None)

    rows = await asyncio.gather(*(one(ep) for ep in eps_by.values()))
    judge.flush()
    # consistency: agent speech before the examiner's 2nd line identical in A and B
    by = {(r["agent"], r["task_id"], r["seed"], r["cond"]): r for r in rows}
    for r in rows:
        if r["cond"] != "B" or (r["agent"], r["task_id"], r["seed"], "A") not in by:
            continue
        a = by[(r["agent"], r["task_id"], r["seed"], "A")]
        ea, eb = eps_by[a["id"]], eps_by[r["id"]]
        cut = min(x for x in (a["u1_ms"], r["u1_ms"], 10**9) if x is not None)
        sa, sb = _speech(ea, cut), _speech(eb, cut)
        r["consistent_text"] = sa == sb
        first = lambda ep: min((t["start_time"] for t in ep["turns"] if t["role"] != "user" and t["text"]), default=None)  # noqa: E731
        r["consistent_onset"] = first(ea) == first(eb)
    (out / f"rows{args.name}.json").write_text(json.dumps(rows, indent=1))
    lines = ["# FD-Bench v2: open loop (static script) vs closed loop (live examiner)", "",
             "Judge: official prompt on a local proxy LLM; TT / IF = mean over the agent's turn-taking events (per conversation, "
             "then averaged); event-weighted (official aggregation) in brackets; task = task-specific score (1–5).", ""]
    hdr = "| agent | split | n A/B | TT A | TT B | IF A | IF B | IF B−A [95% CI] | task A | task B | task B−A [95% CI] | examiner reached end (B) | examiner lines A / B | incoherent examiner lines A / B | examiner over agent A / B |"
    lines += [hdr, "|" + "---|" * hdr.count("|")[:-1] if False else "|" + "---|" * (hdr.count("|") - 1)]
    for ag in agents:
        for split in (None, *fdb2.SPLITS):
            R = [r for r in rows if r["agent"] == ag and (split is None or r["split"] == split)]
            A, B = [r for r in R if r["cond"] == "A"], [r for r in R if r["cond"] == "B"]
            pa = {(r["task_id"], r["seed"]): r for r in A}
            pairs = [(pa[(r["task_id"], r["seed"])], r) for r in B if (r["task_id"], r["seed"]) in pa]

            def ew(rs, k):  # event-weighted
                num = sum((r[k] or 0) * r["n_events"] for r in rs if r[k] is not None)
                den = sum(r["n_events"] for r in rs if r[k] is not None)
                return num / den if den else None

            dif = lambda k: [(b[k] - a[k]) if a[k] is not None and b[k] is not None else None for a, b in pairs]  # noqa: E731
            ci_if, ci_task = boot_ci(dif("if")), boot_ci(dif("task"))
            inco = lambda rs: _mean([1 - _mean(r["coherent"]) for r in rs if r["coherent"]])  # noqa: E731
            lines.append(f"| {ag} | {split or 'all'} | {len(A)}/{len(B)} | {_fmt(_mean([r['tt'] for r in A]))} [{_fmt(ew(A, 'tt'))}] | "
                         f"{_fmt(_mean([r['tt'] for r in B]))} [{_fmt(ew(B, 'tt'))}] | {_fmt(_mean([r['if'] for r in A]))} | {_fmt(_mean([r['if'] for r in B]))} | "
                         f"{_fmt(_mean(dif('if')))} [{_fmt(ci_if[0])}, {_fmt(ci_if[1])}] | {_fmt(_mean([r['task'] for r in A]))} | "
                         f"{_fmt(_mean([r['task'] for r in B]))} | {_fmt(_mean(dif('task')))} [{_fmt(ci_task[0])}, {_fmt(ci_task[1])}] | "
                         f"{_fmt(_mean([r['reached_end'] for r in B]))} | {_fmt(_mean([r['examiner_lines'] for r in A]), 1)} / {_fmt(_mean([r['examiner_lines'] for r in B]), 1)} | "
                         f"{_fmt(inco(A))} / {_fmt(inco(B))} | {_fmt(_mean([r['examiner_over_agent'] for r in A]))} / {_fmt(_mean([r['examiner_over_agent'] for r in B]))} |")
    lines += ["", "## Consistency (agent speech before the examiner's 2nd line, A vs B)", ""]
    for ag in agents:
        Bs = [r for r in rows if r["agent"] == ag and "consistent_text" in r]
        if Bs:
            lines.append(f"- {ag}: identical text {sum(r['consistent_text'] for r in Bs)}/{len(Bs)}, same first onset "
                         f"{sum(r['consistent_onset'] for r in Bs)}/{len(Bs)}")
    lines += ["", "## Ranking (all splits)", ""]
    for k in ("tt", "if", "task"):
        for cond in ("A", "B"):
            ms = {ag: _mean([r[k] for r in rows if r["agent"] == ag and r["cond"] == cond]) for ag in agents}
            order = sorted((a for a in ms if ms[a] is not None), key=lambda a: -ms[a])
            lines.append(f"- {k} {cond}: " + " > ".join(f"{a} {ms[a]:.2f}" for a in order))
    lines += timing_table(rows, agents) + validity_table(rows, agents) + stage_tables(rows, agents)
    unparsed = sum(not r["judge_parsed"] for r in rows)
    lines += ["", f"Judge replies not parsed: {unparsed}/{len(rows)}"]
    (out / f"summary{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def _pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def _ms(x):
    return "—" if x is None else f"{x / 1000:.2f}"


def timing_table(rows, agents, title="## Duplex timing (eval.scores events; open-loop collisions kept apart)") -> list[str]:
    """Turn-take rate (FD-Bench-style take-over after a user turn), response latency, yield on intended barge-ins,
    agent cut-ins on non-collision turns, and collisions (unintended overlaps of a replayed line) with the agent's
    behaviour on them, which are not counted in its yield / cut-in scores."""
    L = ["", title, "",
         "Collision = a user line that starts while the agent talks although it was not meant to cut in (the static script "
         "always waited for the reference assistant). Turn-take rate and latency in brackets: collision turns left out.", "",
         "| agent | cond | episodes | user turns | turn-take rate [clean] | latency median / p90 s [clean median] | intended barge-ins | yield rate | "
         "agent cut-in rate | collisions / episode | collision share of user turns | agent yielded on collisions | agent cut in during collisions | non-directed ignored |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        for cond in ("A", "B"):
            R = [r["timing"] for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("timing")]
            if not R:
                continue
            s = timing_summary(R)
            L.append(f"| {ag} | {cond} | {s['episodes']} | {s['user_turns']} | {_pct(s['turn_take_rate'])} [{_pct(s['turn_take_rate_clean'])}] | "
                     f"{_ms(s['latency_median_ms'])} / {_ms(s['latency_p90_ms'])} [{_ms(s['latency_median_clean_ms'])}] | {s['barge_ins']} | "
                     f"{_pct(s['yield_rate'])} | {_pct(s['cut_in_rate'])} | {_fmt(s['collisions_per_episode'])} | {_pct(s['collision_share'])} | "
                     f"{_pct(s['collision_yield_rate'])} | {_pct(s['collision_cut_in_rate'])} | "
                     f"{_pct(s['nondirected_ok_rate']) if s['nondirected'] else 'none'} |")
    return L


def validity_table(rows, agents, title="## Open-loop user-turn failure rate (turn_validity; B = noise floor of the LLM checker)") -> list[str]:
    """Invalid user turns (after the first) by type, A (replayed) vs B (live user: the checker's noise floor), and the
    excess A − B with a paired bootstrap over conversations."""
    L = ["", title, "",
         "Per user turn after the first: timing = starts inside the agent's speech without meaning to cut in (rule); "
         "content = responds to something the agent never said; ignored = leaves the agent's open question unaddressed "
         "(rule: a question was asked, and LLM); stale = re-gives / re-asks something already settled (LLM). "
         "LLM-only = content, ignored or stale.", "",
         "| agent | cond | turns | invalid (any) | timing | content | ignored question | stale | LLM-only | conversations with ≥1 invalid | ≥1 LLM-only |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        S = {}
        for cond in ("A", "B"):
            R = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("validity") is not None]
            if not R:
                continue
            s = S[cond] = tv.summarize([r["validity"] for r in R])
            L.append(f"| {ag} | {cond} | {s['turns']} | {_pct(s['invalid'])} | {_pct(s['timing'])} | {_pct(s['content'])} | "
                     f"{_pct(s['ignored_question'])} | {_pct(s['stale'])} | {_pct(s['invalid_llm'])} | {_pct(s['episodes_with_invalid'])} | "
                     f"{_pct(s['episodes_with_invalid_llm'])} |")
        if len(S) == 2:
            ex = {k: S["A"][k] - S["B"][k] for k in ("invalid", "timing", "content", "ignored_question", "stale", "invalid_llm",
                                                      "episodes_with_invalid", "episodes_with_invalid_llm")}
            # paired bootstrap over (task, seed) of the per-conversation invalid-turn rate difference
            by = {(r["task_id"], r["seed"], r["cond"]): r for r in rows if r["agent"] == ag and r.get("validity")}
            d = [_mean([f["invalid_llm"] for f in by[(t, s_, "A")]["validity"]]) - _mean([f["invalid_llm"] for f in by[(t, s_, "B")]["validity"]])
                 for (t, s_, c) in by if c == "A" and (t, s_, "B") in by and by[(t, s_, "A")]["validity"] and by[(t, s_, "B")]["validity"]]
            lo, hi = boot_ci(d)
            L.append(f"| {ag} | **A − B** | | **{100 * ex['invalid']:+.0f}** | {100 * ex['timing']:+.0f} | {100 * ex['content']:+.0f} | "
                     f"{100 * ex['ignored_question']:+.0f} | {100 * ex['stale']:+.0f} | {100 * ex['invalid_llm']:+.0f} "
                     f"(per conversation {100 * _mean(d):+.0f} [{_fmt(None if lo is None else 100 * lo, 0)}, {_fmt(None if hi is None else 100 * hi, 0)}]) | "
                     f"{100 * ex['episodes_with_invalid']:+.0f} | {100 * ex['episodes_with_invalid_llm']:+.0f} |")
    return L


def stage_tables(rows, agents) -> list[str]:
    """Stage-based rescoring (time cap is not a failure: the agent is scored on the stages reached) and the diagnosis of
    conversations that hit the cap without the end phrase."""
    from collections import Counter

    L = ["", "## Stages reached and stage-based scores (the time cap is not a failure)", "",
         "Stage analysis: one LLM call per conversation (T = 0) marks each goal T1–T4 reached / completed / the agent's "
         "behaviour on it (yes 1, partial 0.5, no 0). Stage score = mean over reached goals. IF_r / TT_r / task_r = the "
         "official judge with only the reached goals listed and told the call was cut by the time limit (= IF / TT / task "
         "when the examiner reached its end phrase).", "",
         "| agent | cond | n | ended (end phrase) | timeout | forced end (closing rule) | stages reached | stages completed | all 4 completed | "
         "stage score | IF_r | TT_r | task_r | IF (official) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        for cond in ("A", "B"):
            R = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("stages")]
            if not R:
                continue
            L.append(f"| {ag} | {cond} | {len(R)} | {_pct(_mean([r['reached_end'] for r in R]))} | {_pct(_mean([r['timeout'] for r in R]))} | "
                     f"{_pct(_mean([r.get('forced_end', False) for r in R]))} | {_fmt(_mean([len(r['stages']['reached']) for r in R]))} | "
                     f"{_fmt(_mean([len(r['stages']['completed']) for r in R]))} | {_pct(_mean([len(r['stages']['completed']) == 4 for r in R]))} | "
                     f"{_fmt(_mean([r['stages']['stage_score'] for r in R]))} | {_fmt(_mean([r['if_r'] for r in R]))} | "
                     f"{_fmt(_mean([r['tt_r'] for r in R]))} | {_fmt(_mean([r['task_r'] for r in R]))} | {_fmt(_mean([r['if'] for r in R]))} |")
    L += ["", "Paired B − A [95% CI] (same task and seed):", ""]
    for ag in agents:
        by = {(r["task_id"], r["seed"], r["cond"]): r for r in rows if r["agent"] == ag and r.get("stages")}
        P = [(by[(t, s_, "A")], by[(t, s_, "B")]) for (t, s_, c) in by if c == "B" and (t, s_, "A") in by]
        if len(P) < 3:
            continue
        parts = []
        for k, f in (("stage score", lambda r: r["stages"]["stage_score"]), ("IF_r", lambda r: r["if_r"]),
                     ("stages completed", lambda r: len(r["stages"]["completed"]))):
            d = [None if f(a) is None or f(b) is None else f(b) - f(a) for a, b in P]
            lo, hi = boot_ci(d)
            parts.append(f"{k} {_mean(d):+.2f} [{_fmt(lo)}, {_fmt(hi)}]")
        L.append(f"- {ag} (n = {len(P)}): " + "; ".join(parts))
    L += ["", "## Cap diagnosis: conversations without the end phrase", "",
          "Cause (first match): agent_failed_stage = a reached goal the examiner retried ≥ 2 times and the agent did not "
          "complete (or failed); pleasantry_loop; examiner_drift; all_done_no_end = every goal completed, no end phrase; "
          "slow_pacing = the agent talked ≥ 60 % of the window; slow_other.", "",
          "| agent | cond | timeouts | max stage reached T0/T1/T2/T3/T4 | agent_failed_stage | pleasantry_loop | examiner_drift | all_done_no_end | "
          "slow_pacing | slow_other | examiner lines | agent talk share | agent turn s | gap before examiner s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    causes = ("agent_failed_stage", "pleasantry_loop", "examiner_drift", "all_done_no_end", "slow_pacing", "slow_other")
    for ag in agents:
        for cond in ("A", "B"):
            R = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("stages") and r["timeout"]]
            E = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("stages") and not r["timeout"]]
            if not R:
                continue
            mx, cc = Counter(r["stages"]["max_reached"] for r in R), Counter(r["cap_cause"] for r in R)
            pm = lambda k, rs: _fmt(_mean([r["pacing"][k] for r in rs]), 2)  # noqa: E731
            L.append(f"| {ag} | {cond} | {len(R)} | " + "/".join(_pct(mx[i] / len(R)) for i in range(5)) + " | "
                     + " | ".join(_pct(cc[c] / len(R)) for c in causes)
                     + f" | {pm('examiner_lines', R)} (ended: {pm('examiner_lines', E)}) | {pm('agent_share', R)} ({pm('agent_share', E)}) | "
                     f"{pm('agent_turn_s', R)} ({pm('agent_turn_s', E)}) | {pm('gap_s', R)} ({pm('gap_s', E)}) |")
    return L


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
