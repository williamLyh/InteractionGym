"""Full-Duplex-Bench v3: open-loop (official) vs closed-loop evaluation of the same agent on the same recordings.

    # run: every recording x seed, condition A then B with a shared model-call cache (resumable)
    PYTHONPATH=src:. python examples/fdb3_ab.py run --data /path/to/fdb_v3_data --out runs/fdb3_ab --seeds 0 --concurrency 16
    # score (LLM judges), tables, consistency check, browsable pages
    PYTHONPATH=src:. python examples/fdb3_ab.py report --out runs/fdb3_ab

Conditions (docs/BENCHMARKS.md §11):

- **A, open loop (official)**: ``ReplayUser`` plays the recording (up to the end of the request); the episode ends
  once the agent has settled (nothing pending, 3 s of quiet) or at ``max_ms``. Pass@1 over all its tool calls.
- **B, closed loop**: the same recording is the user's first turn; then a ``UserSim`` (``benchmarks.benchmark_user``: no background floor; LLM user with a scenario
  card from the annotation: goal, facts — corrected values only —, persona from the acting notes; voice cloned
  from the recording; reply gaps from ``ResponseDelay``) answers questions, corrects mistakes and confirms / ends.
  Reported: Pass@1 on the calls before the user's second turn (must equal A: consistency check) and the final
  outcome (``fdb3.effective_calls``: corrections replace earlier calls of the same function).

Agent: ``CascadedAgent`` (energy-VAD endpointing at 800 ms, Qwen3-ASR-1.7B, Qwen3.8-27B with the official
cascaded agent's instructions and tool schemas, Qwen3-TTS), in simulated time with a fixed latency model.

Services (env vars): IG_LLM_URL (:8000, agent LLM, user LLM, judge), IG_ASR_URL (:8006), IG_TTS_URL (:8001,
agent voice), IG_CLONE_URL (:8005, user voice cloning).

``--agent minicpmo``: MiniCPM-o 4.5 served by vLLM-Omni (IG_AGENT_URL, lockstep ``clock="input"``, text-only output
timed at ``speech_cps`` against a Thinker-only server by default, ``--audio-out`` for the talker's speech; token trace
in ``<out>/<agent>/<cond>/agent_traces.jsonl``) with a per-domain voice-assistant system prompt (``DUPLEX_AGENTS``).
It cannot call tools, so the report scores a spoken-fulfilment outcome instead of Pass@1 (``benchmarks.fdb3_spoken``:
did its speech confirm every parameter of the request with its final value?), plus the official timing and response-quality judges. A ends after ``--a-idle-ms`` of quiet (the model decides by itself
when to speak, so it is given longer than the cascaded agent's 3 s to start).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.cascaded import CascadedAgent, Endpointing, LatencyModel, ToolChat
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent, check_output_mode
from interaction_gym.benchmarks import benchmark_user, fdb3
from interaction_gym.benchmarks import fdb3_spoken as spoken
from interaction_gym.clients import OpenAIChat, OpenAISpeech, OpenAITranscribe
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.tools import RESULT, ToolWorld
from interaction_gym.traj import episode, load, save
from interaction_gym.user import ReplayUser, ResponseDelay, TurnTaking, Voice

SR = 16000
LLM_URL = getenv("IG_LLM_URL", "http://127.0.0.1:8000/v1")
LLM_MODEL = getenv("IG_LLM_MODEL", "Qwen3.8-27B")
ASR_URL = getenv("IG_ASR_URL", "http://127.0.0.1:8006/v1")
ASR_MODEL = getenv("IG_ASR_MODEL", "Qwen3-ASR-1.7B")
TTS_URL = getenv("IG_TTS_URL", "http://127.0.0.1:8001/v1")
TTS_MODEL = getenv("IG_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
CLONE_URL = getenv("IG_CLONE_URL", "http://127.0.0.1:8005/v1")
CLONE_MODEL = getenv("IG_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
SPEC = AgentSpec(chunk_ms=100, obs=("user.speech", RESULT["agent"]), audio="user.audio", sr=SR)
AGENTS = {  # name -> system prompt (None: the official cascaded agent's instructions)
    "cascaded-official": fdb3.AGENT_INSTRUCTIONS,
    "cascaded-natural": ("You are a helpful voice AI assistant on a phone line. Keep your responses concise and conversational "
                         "since they will be spoken aloud. Use the provided tools to carry out the caller's requests and answer "
                         "from their results; never make up data. If something you need is missing or unclear, ask briefly."),
}
USER_MAX_TURNS = 8

# Native duplex agents (no tools): the model hears the microphone and decides by itself when to speak.
AGENT_URL = getenv("IG_AGENT_URL", "ws://127.0.0.1:8010/v1/realtime?duplex=1")
AGENT_MODEL = getenv("IG_AGENT_MODEL", "openbmb/MiniCPM-o-4_5")
REF_AUDIO = getenv("IG_AGENT_REF_AUDIO", str(Path(getenv("IG_MODEL_DIR", "models")) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
DUPLEX_SPEC = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)  # MiniCPM-o decides once per 1 s unit
DOMAINS = {"travel_identity": "a travel agency (flight search and booking, passport and ID document updates)",
           "finance_billing": "a bank (card benefits, currency exchange, autopay and bill payments)",
           "housing_location": "a rental housing agency (apartment search, commute times, search filters)",
           "ecommerce_support": "an online shop (product search, shopping cart, order tracking)"}
DUPLEX_PROMPT = ("You are the voice assistant of {domain}, answering a customer phone line. Callers tell you what they need done. "
                 "Let the caller finish speaking before you reply. Then confirm the request back to them in one or two short sentences, "
                 "repeating every detail they gave (names, dates, places, amounts, numbers and codes) with the final values they settled on: "
                 "if they corrected themselves, use the corrected value. Then say that you are taking care of it. "
                 "Only ask a question if something essential is missing. Your replies are spoken aloud: keep them short and natural.")
DUPLEX_AGENTS = {"minicpmo": DUPLEX_PROMPT}


def duplex_prompt(task) -> str:
    return DUPLEX_AGENTS["minicpmo"].format(domain=DOMAINS.get(task.scenario["benchmark"]["domain"], "a customer service line"))


class CloneWithFallback:
    """The voice-cloning TTS, which occasionally refuses a text (no end-of-speech within its token budget -> HTTP
    500): retry once with the text re-punctuated, then fall back to a stock voice of the plain TTS (logged in
    ``fallbacks``)."""

    def __init__(self, clone: OpenAISpeech, plain: OpenAISpeech, voice: str = "ryan"):
        self.inner, self.plain, self.voice = clone, plain, voice
        self.model, self.sr = clone.model, clone.sr
        self.fallbacks: list[str] = []

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None, **kw):
        for t in (text, text.rstrip(".!?") + "."):
            try:
                return await self.inner.synth(t, voice, instructions, ref_audio, ref_text, language, **kw)
            except Exception:  # noqa: BLE001
                continue
        self.fallbacks.append(text)
        return await self.plain.synth(text, self.voice, None, language=language)


def make_env(task, condition: str, seed: int, duplex: bool = False, a_idle_ms: int = 3000) -> Env:
    tools = ToolWorld(fdb3.MockAPIBackend(), fdb3.latency_ms(task))
    if condition == "A":
        user = ReplayUser(voice=Voice())
        max_ms = task.scenario["benchmark"]["user_end_ms"] + 60_000
        if duplex:  # no tools; the model is given a_idle_ms of quiet to start / continue
            return Env({"user": user}, DUPLEX_SPEC, max_ms=max_ms, end_idle_ms=a_idle_ms)
    else:
        llm = OpenAIChat(LLM_URL, LLM_MODEL, temperature=0.7, seed=seed, max_tokens=200)
        clone = CloneWithFallback(OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR), OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR))
        user = benchmark_user(fdb3.UserSource(llm, USER_MAX_TURNS), voice=Voice(clone, clone=clone),
                       timing=TurnTaking(response_delay=ResponseDelay(), yield_after_ms=None, nudge_after_ms=10_000))
        max_ms = 180_000
    if duplex:
        return Env({"user": user}, DUPLEX_SPEC, max_ms=max_ms, end_idle_ms=3000)
    return Env({"user": user, "tools": tools}, SPEC, max_ms=max_ms, end_idle_ms=3000)


AUDIO_OUT = False  # MiniCPM-o output: False = text timed at speech_cps (Thinker-only server); --audio-out = the talker's speech


def make_duplex_agent(task) -> VllmOmniDuplexAgent:
    return VllmOmniDuplexAgent(DUPLEX_SPEC, AGENT_URL, model=AGENT_MODEL, ref_audio=REF_AUDIO, out_sr=SR, clock="input",
                               audio_out=AUDIO_OUT, trace_tokens=True, session={"instructions": duplex_prompt(task)})


def make_agent(name: str, seed: int, cache: dict, endpoint_ms: int = 800) -> CascadedAgent:
    return CascadedAgent(SPEC, OpenAITranscribe(ASR_URL, ASR_MODEL), ToolChat(LLM_URL, LLM_MODEL, native=False, max_tokens=512),
                         OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR), instructions=AGENTS[name], voice="aiden", seed=seed, cache=cache,
                         endpointing=Endpointing(silence_ms=endpoint_ms), latency=LatencyModel())


async def run_one(task, condition: str, seed: int, agent_name: str, cache: dict, endpoint_ms: int = 800, a_idle_ms: int = 3000):
    duplex = agent_name in DUPLEX_AGENTS
    env = make_env(task, condition, seed, duplex, a_idle_ms)
    agent = make_duplex_agent(task) if duplex else make_agent(agent_name, seed, cache, endpoint_ms)
    wall = time.monotonic()
    try:
        obs = await env.reset(task, seed)
        while True:
            obs, _, done = await env.step(await agent.act(env.t, obs))
            if env.truncated or (done and not agent.busy):
                break
    finally:
        if duplex:
            await agent.close()
    return env, agent, time.monotonic() - wall


def describe(agent, name: str, task) -> dict:
    if isinstance(agent, VllmOmniDuplexAgent):
        return {"name": name, "server": "vLLM-Omni duplex", "model": agent.model, "clock": agent.clock, "output": agent.output,
                "url": agent.url, "unit_ms": agent.unit_ms, "system_prompt": duplex_prompt(task)}
    return agent.describe()


def quick_metrics(ep: dict) -> dict:
    b = ep["meta"]["task"]["scenario"]["benchmark"]
    users = fdb3.user_turns(ep)
    u1 = users[1]["start_time"] if len(users) > 1 else None
    return {"calls": fdb3.agent_calls(ep), "u1_ms": u1, "timing": fdb3.timing(ep, b["user_end_ms"], before_ms=u1),
            "user_turns": len(users)}


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["meta"]["episode_id"] for line in path.read_text().splitlines() if line.strip()}


def run_tag(args) -> str:
    """Output folder / episode-id prefix: the agent's name, plus the end-of-turn silence when not the default."""
    return args.agent + (f"-ep{args.endpoint_ms}" if args.endpoint_ms != 800 else "")


def eid(rid: str, seed: int, cond: str, agent: str) -> str:
    return f"{agent}/{rid}/s{seed}/{cond}"


async def run(args) -> None:
    out = Path(args.out)
    media = MediaStore(out)
    recs = fdb3.load(args.data, only=args.ids.split(",") if args.ids else None)
    if args.limit:
        recs = recs[: args.limit]
    for rec, task in recs:
        task.scenario["profile"] = {"language": "en"}
    conds = args.conditions.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    tag = run_tag(args)
    done = {c: _done(out / tag / c / "episodes.jsonl") for c in conds}
    jobs = [(rec, task, s) for s in seeds for rec, task in recs if any(eid(rec.id, s, c, tag) not in done[c] for c in conds)]
    print(f"{len(recs)} recordings, {len(jobs)} (recording, seed) pairs to run, conditions {conds}, agent {args.agent}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    duplex = args.agent in DUPLEX_AGENTS
    for c in conds if duplex else ():
        check_output_mode(out / tag / c / "episodes.jsonl", AUDIO_OUT)  # never resume a run of the other output mode
    if duplex and jobs:  # the first session after a server start returns no audio: one throw-away episode
        env, agent, wall = await run_one(recs[0][1], "A", 0, args.agent, {}, a_idle_ms=args.a_idle_ms)
        print(f"warm-up: {env.t / 1000:.1f}s sim in {wall:.1f}s wall, agent events {agent.events}", flush=True)
    t_start, n_ok = time.monotonic(), 0

    async def pair(rec, task, seed):
        nonlocal n_ok
        async with sem:
            cache: dict = {}  # shared by A and B of this (recording, seed): same input -> same model outputs
            for cond in conds:
                e = eid(rec.id, seed, cond, tag)
                if e in done[cond]:
                    continue
                for attempt in range(args.retries):
                    try:
                        env, agent, wall = await run_one(task, cond, seed, args.agent, cache, args.endpoint_ms, args.a_idle_ms)
                        break
                    except Exception as ex:  # a service restarting (hourly kill on this host): wait and retry
                        body = ex.read().decode(errors="replace")[:300] if hasattr(ex, "read") else ""
                        print(f"{e} attempt {attempt + 1}: {type(ex).__name__}: {str(ex)[:200]} {getattr(ex, 'url', '')} {body}", flush=True)
                        await asyncio.sleep(45 + random.random() * 30)
                else:
                    print(f"GAVE UP {e}", flush=True)
                    return
                ep = episode(env, e, media=media, run_id=f"fdb3-ab-{tag}",
                             meta={"agent": describe(agent, args.agent, task), "agent_events": agent.events, "condition": cond,
                                   "agent_name": args.agent, "recording": rec.id, "wall_s": round(wall, 1)})
                ep["eval"]["benchmark"] = quick_metrics(ep)
                save([ep], out / tag / cond / "episodes.jsonl", append=True)
                if duplex and agent.trace(e):
                    save([agent.trace(e)], out / tag / cond / "agent_traces.jsonl", append=True)
                n_ok += 1
                q = ep["eval"]["benchmark"]
                rate = n_ok / (time.monotonic() - t_start)
                said = " | ".join(t["text"] for t in fdb3.agent_speech(ep))[:120] if duplex else [c["function"] for c in q["calls"]]
                print(f"{n_ok} {e:70s} {env.t / 1000:5.1f}s sim {wall:5.1f}s wall {said} "
                      f"d={q['timing']['delta_ms']} users={q['user_turns']} {rate * 3600:.0f} ep/h", flush=True)

    await asyncio.gather(*(pair(*j) for j in jobs))


# ---------------------------------------------------------------- report

ASKS_PROMPT = """A customer asked a voice assistant on the phone:
"{request}"

The assistant replied (everything it said):
"{reply}"

Did the assistant ask the customer a question — for confirmation, for missing or unclear details, or which option to take — instead of (or before) completing everything requested?
Respond with ONLY a JSON object: {{"asks": true/false, "explanation": "brief reason"}}"""

LABEL_PROMPT = """Below is a phone conversation between a CUSTOMER and a voice ASSISTANT. The customer's first message was a pre-recorded request.

{dialogue}

For each CUSTOMER turn after the first (numbered), label what it does, choosing one:
- "answer": answers a question the assistant asked (gives a detail, chooses an option)
- "confirm": confirms what the assistant proposed or did
- "correct": corrects a mistake (a wrong detail or action) or points out something the assistant skipped
- "repeat": repeats or re-asks the request because the assistant did not act on it or did not respond
- "close": thanks / goodbye
- "other"
Respond with ONLY a JSON object: {{"labels": ["...", ...]}} (one label per numbered customer turn, in order)."""


class Judge:
    """The official judges' prompts on a local LLM (a proxy for GPT-4o), memoized."""

    def __init__(self, llm):
        self.llm, self.memo = llm, {}

    async def _ask(self, prompt: str, max_tokens: int = 200) -> dict:
        if prompt not in self.memo:
            raw = await self.llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens)
            try:
                self.memo[prompt] = json.loads(fdb3._strip_json_fences(raw))
            except Exception:
                self.memo[prompt] = {"error": raw[:200]}
        return self.memo[prompt]

    async def chat(self, messages, **kw):  # TextGen for fdb3.pass_at_1 / judge_response (memoized)
        key = json.dumps(messages)
        if key not in self.memo:
            self.memo[key] = await self.llm.chat(messages, **kw)
        return self.memo[key]


def _text(turns: list[dict]) -> str:
    return " ".join(t["text"] for t in turns if t["text"]).strip()


def _dialogue(ep: dict) -> str:
    lines, k = [], 0
    for t in sorted((t for t in ep["turns"] if t["end_time"] > t["start_time"] and t["text"]), key=lambda t: t["start_time"]):
        if t["role"] == "user":
            lines.append(f"CUSTOMER{'' if k == 0 else f' ({k})'}: {t['text']}")
            k += 1
        else:
            lines.append(f"ASSISTANT: {t['text']}")
    return "\n".join(lines)


def _sig_calls(calls):
    return [(c["function"], json.dumps(c["args"], sort_keys=True), c["time"]) for c in calls]


def _sig_speech(turns):
    return [(t["start_time"], t["end_time"], t["text"]) for t in turns]


LENIENT_RULES = """6. Only the expected arguments are checked: extra arguments in the actual call are fine (the API schema may require them).
7. Hyphens, spaces or dots inside codes and IDs are formatting: "K-2" == "K2", "D-E-L-I-V" == "DELIV".

Respond with ONLY a JSON object:"""


class Lenient:
    """The official argument prompt plus two rules that state what the official exact-match fallback already
    assumes (only expected keys are compared; IDs compared after normalisation): a sensitivity check on the proxy
    judge's strictness. Memoized through the wrapped ``Judge``."""

    def __init__(self, judge):
        self.judge = judge

    async def chat(self, messages, **kw):
        c = messages[0]["content"]
        if "called a function with correct arguments" in c:
            c = c.replace("\nRespond with ONLY a JSON object:", LENIENT_RULES, 1)
        return await self.judge.chat([{"role": "user", "content": c}], **kw)


def _judges(judge):
    return {"pass": judge, "lenient": Lenient(judge)}


async def score_pair(a: dict, b: dict | None, judge: Judge) -> dict:
    task = a["meta"]["task"]
    bench, expected = task["scenario"]["benchmark"], task["criteria"]["expected_tool_calls"]
    user_end = bench["user_end_ms"]
    row = {"recording": bench["recording"], "scenario": bench["id"], "speaker": bench["speaker"], "difficulty": bench["difficulty"],
           "domain": bench["domain"], "disfluency": bench["disfluency_features"], "rollback": bench["state_rollback_test"],
           "seed": a["meta"].get("seed"), "user_end_ms": user_end, "expected": expected}
    a_calls = fdb3.agent_calls(a)
    tm = fdb3.timing(a, user_end)
    resp_a, _ = await fdb3.judge_response(judge, bench["reference_reply"], _text(fdb3.agent_speech(a)))
    row["A"] = {"calls": [[c["function"], c["args"]] for c in a_calls], "response_quality": resp_a, "timing": tm,
                "speech": _text(fdb3.agent_speech(a)), "duration_ms": a["meta"]["duration_ms"], "end_reason": a["meta"]["end_reason"]}
    for key, j in _judges(judge).items():
        pa = await fdb3.pass_at_1(expected, a_calls, j)
        sel = pa["checks"]["tool_selection"]
        row["A"][key] = pa["passed"]
        row["A"][key + "_why"] = pa["failure_reason"]
        row["A"][key + "_sel"] = {"ok": sel["passed"], "missing": sel["missing"], "unexpected": sel["unexpected"]}
    evs = a["meta"].get("agent_events", [])
    endpoints = [e for e in evs if e["kind"] == "endpoint"]
    row["A"]["endpoints"] = len(endpoints)
    row["A"]["premature_endpoints"] = sum(e["speech_end"] < user_end - 200 for e in endpoints)
    first_out = min([c["time"] for c in a_calls] + [t["start_time"] for t in fdb3.agent_speech(a)], default=None)
    row["A"]["acted_before_end"] = first_out is not None and first_out < user_end  # spoke or called a tool mid-request
    row["A"]["asr"] = " | ".join(e["transcript"] for e in endpoints)
    if not (row["A"]["pass"] and row["A"]["lenient"]):
        asks = await judge._ask(ASKS_PROMPT.format(request=bench["script"], reply=row["A"]["speech"] or "(nothing)"))
        row["A"]["asks_user"] = bool(asks.get("asks"))
    if b is None:
        return row
    users = fdb3.user_turns(b)
    u1 = users[1]["start_time"] if len(users) > 1 else None
    b_first = fdb3.agent_calls(b, before_ms=u1)
    b_all = fdb3.agent_calls(b)
    b_eff = fdb3.effective_calls(b_all, expected)
    resp_b, _ = await fdb3.judge_response(judge, bench["reference_reply"], _text(fdb3.agent_speech(b)))
    labels = []
    if len(users) > 1:
        lab = await judge._ask(LABEL_PROMPT.format(dialogue=_dialogue(b)), max_tokens=300)
        labels = lab.get("labels", []) if isinstance(lab.get("labels"), list) else []
    row["B"] = {"calls": [[c["function"], c["args"]] for c in b_all], "response_quality": resp_b,
                "timing": fdb3.timing(b, user_end, before_ms=u1), "user_turns_after_first": len(users) - 1, "user_labels": labels,
                "u1_ms": u1, "duration_ms": b["meta"]["duration_ms"], "end_reason": b["meta"]["end_reason"], "dialogue": _dialogue(b)}
    for key, j in _judges(judge).items():
        pb1 = await fdb3.pass_at_1(expected, b_first, j)
        pbf = await fdb3.pass_at_1(expected, b_eff, j)
        pbs = await fdb3.pass_at_1(expected, b_all, j)
        row["B"].update({f"{key}_first": pb1["passed"], f"{key}_final": pbf["passed"], f"{key}_strict": pbs["passed"],
                         f"{key}_final_why": pbf["failure_reason"]})
    cut = u1 if u1 is not None else 10**12
    a_before = [c for c in a_calls if c["time"] < cut]
    same_calls = _sig_calls(a_before) == _sig_calls(b_first)
    same_speech = _sig_speech(fdb3.agent_speech(a, cut)) == _sig_speech(fdb3.agent_speech(b, cut))
    same_timing = fdb3.timing(a, user_end, before_ms=u1) == row["B"]["timing"]
    row["consistency"] = {"calls": same_calls, "speech": same_speech, "first_response": same_timing,
                          "ok": same_calls and same_speech and same_timing, "a_calls_after_u1": len(a_calls) - len(a_before),
                          "pass_equal": row["A"]["pass"] == row["B"]["pass_first"]}
    return row


def _rate(xs) -> str:
    xs = [bool(x) for x in xs]
    return f"{sum(xs) / len(xs):.3f} ({sum(xs)}/{len(xs)})" if xs else "—"


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs)) if xs else None


def _seeds(rows, f) -> str:
    """mean ± sd over seeds of a per-seed rate."""
    by = {}
    for r in rows:
        by.setdefault(r["seed"], []).append(bool(f(r)))
    vals = [sum(v) / len(v) for v in by.values()]
    if len(vals) < 2:
        return ""
    return f"{statistics.mean(vals):.3f} ± {statistics.stdev(vals):.3f}"


CATS = ("interrupted_in_pause", "asked_user", "no_tool_call", "wrong_tools", "wrong_args")


def categorize(r: dict, key: str, patient: dict | None) -> str | None:
    """Why an open-loop run failed. ``interrupted_in_pause``: the agent acted (spoke or called a tool) before the
    user had finished AND the same agent with a 3 s end-of-turn silence (``patient``, same recording and seed)
    passes — the cut-in is what lost the information. Otherwise by what is visible: the agent asked the user
    instead of completing the request, made no call, chose the wrong tools, or got arguments wrong."""
    a = r["A"]
    if a[key]:
        return None
    p = patient.get((r["recording"], r["seed"])) if patient else None
    if a["acted_before_end"] and p is not None and p["A"][key]:
        return "interrupted_in_pause"
    sel = a[key + "_sel"]
    if a.get("asks_user") and sel["missing"]:
        return "asked_user"
    if not a["calls"]:
        return "no_tool_call"
    if not sel["ok"]:
        return "wrong_tools"
    return "wrong_args"


def tables(rows: list[dict], key: str = "pass", patient: dict | None = None) -> str:
    pairs = [r for r in rows if "B" in r]
    seeds = sorted({r["seed"] for r in pairs})
    L = [f"Pairs (A, B): {len(pairs)} = {len(pairs) // max(1, len(seeds))} recordings × seeds {seeds}. Judge: "
         + ("official argument prompt" if key == "pass" else "official prompt + 2 lenient rules (extra args fine, IDs normalised)")
         + " on Qwen3.8-27B (proxy for GPT-4o).", ""]
    L += ["### Pass@1", "", "| subset | n | A (open loop, official) | B first response | B final | B strict (all calls) | recovered (A fail → B final pass) | broken (A pass → B final fail) |",
          "|---|---|---|---|---|---|---|---|"]

    def line(name, rs):
        fails = [r for r in rs if not r["A"][key]]
        passes = [r for r in rs if r["A"][key]]
        rec = _rate([r["B"][key + "_final"] for r in fails]) if fails else "—"
        brk = _rate([not r["B"][key + "_final"] for r in passes]) if passes else "—"
        return (f"| {name} | {len(rs)} | {_rate([r['A'][key] for r in rs])} | {_rate([r['B'][key + '_first'] for r in rs])} | "
                f"{_rate([r['B'][key + '_final'] for r in rs])} | {_rate([r['B'][key + '_strict'] for r in rs])} | {rec} | {brk} |")

    L.append(line("all", pairs))
    for d in ("easy", "medium", "hard"):
        L.append(line(d, [r for r in pairs if r["difficulty"] == d]))
    for f in ("FILLER", "PAUSE", "HESITATION", "FALSE_START", "SELF_CORRECTION"):
        L.append(line(f, [r for r in pairs if f in r["disfluency"]]))
    L.append(line("no disfluency tag", [r for r in pairs if not r["disfluency"]]))
    L.append(line("state rollback", [r for r in pairs if r["rollback"]]))
    if len(seeds) > 1:
        L += ["", f"Per-seed mean ± sd: A {_seeds(pairs, lambda r: r['A'][key])}, B first {_seeds(pairs, lambda r: r['B'][key + '_first'])}, "
                  f"B final {_seeds(pairs, lambda r: r['B'][key + '_final'])}, B strict {_seeds(pairs, lambda r: r['B'][key + '_strict'])}."]
    fails = [r for r in pairs if not r["A"][key]]
    cat = {id(r): categorize(r, key, patient) for r in fails}
    L += ["", "### Why open-loop runs failed (A)", "",
          "| category | n | share of A failures | of which the agent acted before the user finished | recovered in B (final) |", "|---|---|---|---|---|"]
    for c in CATS:
        rs = [r for r in fails if cat[id(r)] == c]
        L.append(f"| {c} | {len(rs)} | {len(rs) / max(1, len(fails)):.2f} | {sum(r['A']['acted_before_end'] for r in rs)} | "
                 f"{_rate([r['B'][key + '_final'] for r in rs]) if rs else '—'} |")
    L += ["", "| subset | A failures | " + " | ".join(CATS) + " |", "|---|---|" + "---|" * len(CATS)]
    subsets = [(d, lambda r, d=d: r["difficulty"] == d) for d in ("easy", "medium", "hard")]
    subsets += [(f, lambda r, f=f: f in r["disfluency"]) for f in ("FILLER", "PAUSE", "HESITATION", "FALSE_START", "SELF_CORRECTION")]
    for name, pred in subsets:
        rs = [r for r in fails if pred(r)]
        L.append(f"| {name} | {len(rs)} | " + " | ".join(str(sum(cat[id(r)] == c for r in rs)) for c in CATS) + " |")
    L += ["", f"Agent acted before the user finished (all A runs): {_rate([r['A']['acted_before_end'] for r in pairs])}; "
              f"A pass rate when it did: {_rate([r['A'][key] for r in pairs if r['A']['acted_before_end']])}, when it did not: "
              f"{_rate([r['A'][key] for r in pairs if not r['A']['acted_before_end']])}."]
    L += ["", "### User effort (B)", ""]
    turns = [r["B"]["user_turns_after_first"] for r in pairs]
    labs: dict = {}
    for r in pairs:
        for x in r["B"]["user_labels"]:
            labs[x] = labs.get(x, 0) + 1
    L.append(f"User turns after the recording: mean {statistics.mean(turns):.2f}, median {statistics.median(turns)}, max {max(turns)}; "
             f"episodes with ≥1 correction: {_rate(['correct' in r['B']['user_labels'] for r in pairs])}; "
             f"≥1 answer to a question: {_rate(['answer' in r['B']['user_labels'] for r in pairs])}; "
             f"≥1 repeat: {_rate(['repeat' in r['B']['user_labels'] for r in pairs])}. Turn labels: {dict(sorted(labs.items(), key=lambda kv: -kv[1]))}.")
    for name, rs in (("A pass", [r for r in pairs if r["A"][key]]), ("A fail", fails),
                     ("A fail → B pass", [r for r in fails if r["B"][key + "_final"]])):
        if rs:
            L.append(f"- {name} (n={len(rs)}): user turns {statistics.mean(r['B']['user_turns_after_first'] for r in rs):.2f}, "
                     f"corrections {statistics.mean(r['B']['user_labels'].count('correct') for r in rs):.2f}, "
                     f"answers {statistics.mean(r['B']['user_labels'].count('answer') for r in rs):.2f}, "
                     f"repeats {statistics.mean(r['B']['user_labels'].count('repeat') for r in rs):.2f} per episode")
    L += ["", "### Timing (first user turn; official definitions)", "",
          "| | turn-take rate | interruption rate (Δt<0) | first response ms (median, non-interrupted) | first tool call ms (median) |", "|---|---|---|---|---|"]
    for cond, k in (("A", "A"), ("B (before the user's 2nd turn)", "B")):
        ts = [r[k]["timing"] for r in pairs]
        L.append(f"| {cond} | {_rate([t['turn_taken'] for t in ts])} | {_rate([t['interrupted'] for t in ts])} | "
                 f"{_med([t['first_response_ms'] for t in ts])} | {_med([t['first_tool_call_ms'] for t in ts])} |")
    L += ["", f"Response quality (official prompt, proxy judge): A {_rate([r['A']['response_quality'] == 1.0 for r in pairs])}, "
              f"B (whole conversation) {_rate([r['B']['response_quality'] == 1.0 for r in pairs])}.", ""]
    L += ["### Consistency check (A vs B up to the user's second turn)", ""]
    cs = [r["consistency"] for r in pairs]
    L.append(f"Identical agent speech: {_rate([c['speech'] for c in cs])}; identical tool calls (name, args, time): {_rate([c['calls'] for c in cs])}; "
             f"identical first-response metrics: {_rate([c['first_response'] for c in cs])}; all three: {_rate([c['ok'] for c in cs])}; "
             f"Pass@1 A = B first: {_rate([c['pass_equal'] for c in cs])}; A runs with tool calls after B's 2nd user turn: "
             f"{sum(c['a_calls_after_u1'] > 0 for c in cs)}.")
    return "\n".join(L)


# ---------------------------------------------------------------- report: duplex agents (no tools) — spoken fulfilment

CLAIM_PROMPT = """A customer asked a voice assistant on the phone:
"{request}"

The assistant has NO access to any system: it cannot search, book, look up or change anything. Everything the assistant said:
"{reply}"

Did the assistant claim that an action had been completed, or report a concrete result it could not know (e.g. a price, a converted amount, a booking reference, an order status, a commute time, a product it found)? Saying that it will do something, or is doing it now, is not such a claim.
Respond with ONLY a JSON object: {{"claims": true/false, "explanation": "brief reason"}}"""

SPOKEN_CATS = ("silent", "wrong_after_self_correction", "misheard", "interrupted_in_pause", "asked_question", "missed_detail")


def spoken_category(r: dict) -> str | None:
    """Why the agent's first response (A) did not confirm every parameter with its final value, first match wins:
    ``silent`` (said nothing); ``wrong_after_self_correction`` (a self-corrected parameter ended on another value —
    usually the original); ``misheard`` (another parameter stated with a different value); ``interrupted_in_pause``
    (started speaking before the request was over — Δt < 0, i.e. in a pause / disfluency — and left parameters out);
    ``asked_question`` (asked the caller something instead of confirming everything); ``missed_detail`` (left
    parameters out, no question)."""
    a = r["A"]
    if a["spoken"]["fulfilled"]:
        return None
    sl = a["spoken"]["slots"]
    if not a["timing"]["turn_taken"]:
        return "silent"
    if any(x["superseded"] and x["status"] == "wrong" for x in sl):
        return "wrong_after_self_correction"
    if any(x["status"] == "wrong" for x in sl):
        return "misheard"
    if a["timing"]["interrupted"]:
        return "interrupted_in_pause"
    if a.get("asks_user"):
        return "asked_question"
    return "missed_detail"


async def score_pair_spoken(a: dict, b: dict | None, judge: Judge) -> dict:
    task = a["meta"]["task"]
    bench = task["scenario"]["benchmark"]
    user_end = bench["user_end_ms"]
    slots = spoken.slots_of(task)
    row = {"recording": bench["recording"], "scenario": bench["id"], "speaker": bench["speaker"], "difficulty": bench["difficulty"],
           "domain": bench["domain"], "disfluency": bench["disfluency_features"], "rollback": bench["state_rollback_test"],
           "seed": a["meta"].get("seed"), "user_end_ms": user_end, "slots": [[s["key"], s["value"], s["superseded"]] for s in slots]}
    speech_a = _text(fdb3.agent_speech(a))
    resp_a, _ = await fdb3.judge_response(judge, bench["reference_reply"], speech_a)
    sp_a = await spoken.outcome(a, judge, slots=slots)
    first = fdb3.agent_speech(a)[:1]
    row["A"] = {"speech": speech_a, "timing": fdb3.timing(a, user_end), "response_quality": resp_a, "spoken": sp_a,
                "first_turn": [first[0]["start_time"], first[0]["text"]] if first else None,
                "duration_ms": a["meta"]["duration_ms"], "end_reason": a["meta"]["end_reason"],
                "agent_turns": len(fdb3.agent_speech(a)),
                "spoke_before_end": sum(t["start_time"] < user_end for t in fdb3.agent_speech(a))}
    if speech_a:
        row["A"]["claims_result"] = bool((await judge._ask(CLAIM_PROMPT.format(request=bench["script"], reply=speech_a))).get("claims"))
    if not sp_a["fulfilled"] and speech_a:
        asks = await judge._ask(ASKS_PROMPT.format(request=bench["script"], reply=speech_a))
        row["A"]["asks_user"] = bool(asks.get("asks"))
    if b is None:
        return row
    users = fdb3.user_turns(b)
    u1 = users[1]["start_time"] if len(users) > 1 else None
    speech_b = _text(fdb3.agent_speech(b))
    resp_b, _ = await fdb3.judge_response(judge, bench["reference_reply"], speech_b)
    labels = []
    if len(users) > 1:
        lab = await judge._ask(LABEL_PROMPT.format(dialogue=_dialogue(b)), max_tokens=300)
        labels = lab.get("labels", []) if isinstance(lab.get("labels"), list) else []
    row["B"] = {"timing": fdb3.timing(b, user_end, before_ms=u1), "response_quality": resp_b,
                "spoken_first": await spoken.outcome(b, judge, before_ms=u1, slots=slots),
                "spoken_final": await spoken.outcome(b, judge, slots=slots),
                "user_turns_after_first": len(users) - 1, "user_labels": labels, "u1_ms": u1,
                # parameters the simulated user said again after the recording (rule match): what the agent could repeat back
                "user_restated": [s["key"] for s in slots if any(spoken.mentions(s, s["value"], u["text"]) for u in users[1:])],
                "duration_ms": b["meta"]["duration_ms"], "end_reason": b["meta"]["end_reason"], "dialogue": _dialogue(b)}
    if speech_b:
        row["B"]["claims_result"] = bool((await judge._ask(CLAIM_PROMPT.format(request=bench["script"], reply=speech_b))).get("claims"))
    cut = u1 if u1 is not None else 10**12
    # start + text: what the model decided (the end of each utterance also depends on the talker's audio length)
    same_speech = [(t["start_time"], t["text"]) for t in fdb3.agent_speech(a, cut)] == [(t["start_time"], t["text"]) for t in fdb3.agent_speech(b, cut)]
    same_text = _sig_speech(fdb3.agent_speech(a, cut)) == _sig_speech(fdb3.agent_speech(b, cut))
    same_timing = fdb3.timing(a, user_end, before_ms=u1) == row["B"]["timing"]
    a_first = await spoken.outcome(a, judge, before_ms=u1, slots=slots)
    fb = fdb3.agent_speech(b, cut)[:1]
    row["B"]["first_turn"] = [fb[0]["start_time"], fb[0]["text"]] if fb else None
    row["consistency"] = {"speech": same_speech, "text": same_text, "first_response": same_timing, "ok": same_speech and same_timing,
                          "first_turn": row["A"]["first_turn"] == row["B"]["first_turn"],
                          "outcome_equal": a_first["slots"] == row["B"]["spoken_first"]["slots"],
                          "a_speech_after_u1": len(fdb3.agent_speech(a)) - len(fdb3.agent_speech(a, cut))}
    return row


def _mean(xs) -> str:
    xs = [x for x in xs if x is not None]
    return f"{statistics.mean(xs):.3f}" if xs else "—"


def tables_spoken(rows: list[dict]) -> str:
    pairs = [r for r in rows if "B" in r]
    seeds = sorted({r["seed"] for r in pairs})
    L = [f"Pairs (A, B): {len(pairs)} = {len(pairs) // max(1, len(seeds))} recordings × seeds {seeds}. Spoken fulfilment: every "
         "parameter of the request confirmed in the agent's speech with its final value (rules + Qwen3.8-27B fallback).", ""]
    L += ["### Spoken fulfilment (all parameters confirmed) · mean slot outcome", "",
          "| subset | n | A (open loop) | B first response | B final | recovered (A fail → B final ok) | broken (A ok → B final fail) | slot outcome A / B final |",
          "|---|---|---|---|---|---|---|---|"]

    def line(name, rs):
        fails = [r for r in rs if not r["A"]["spoken"]["fulfilled"]]
        oks = [r for r in rs if r["A"]["spoken"]["fulfilled"]]
        rec = _rate([r["B"]["spoken_final"]["fulfilled"] for r in fails]) if fails else "—"
        brk = _rate([not r["B"]["spoken_final"]["fulfilled"] for r in oks]) if oks else "—"
        return (f"| {name} | {len(rs)} | {_rate([r['A']['spoken']['fulfilled'] for r in rs])} | "
                f"{_rate([r['B']['spoken_first']['fulfilled'] for r in rs])} | {_rate([r['B']['spoken_final']['fulfilled'] for r in rs])} | "
                f"{rec} | {brk} | {_mean([r['A']['spoken']['outcome'] for r in rs])} / {_mean([r['B']['spoken_final']['outcome'] for r in rs])} |")

    L.append(line("all", pairs))
    for d in ("easy", "medium", "hard"):
        L.append(line(d, [r for r in pairs if r["difficulty"] == d]))
    for f in ("FILLER", "PAUSE", "HESITATION", "FALSE_START", "SELF_CORRECTION"):
        L.append(line(f, [r for r in pairs if f in r["disfluency"]]))
    L.append(line("no disfluency tag", [r for r in pairs if not r["disfluency"]]))
    L.append(line("state rollback", [r for r in pairs if r["rollback"]]))
    for dom in sorted({r["domain"] for r in pairs}):
        L.append(line(dom, [r for r in pairs if r["domain"] == dom]))
    if len(seeds) > 1:
        L += ["", f"Per-seed mean ± sd: A {_seeds(pairs, lambda r: r['A']['spoken']['fulfilled'])}, "
                  f"B first {_seeds(pairs, lambda r: r['B']['spoken_first']['fulfilled'])}, B final {_seeds(pairs, lambda r: r['B']['spoken_final']['fulfilled'])}."]
    # self-corrected parameters
    sa = [x for r in pairs for x in r["A"]["spoken"]["slots"] if x["superseded"]]
    sb = [x for r in pairs for x in r["B"]["spoken_final"]["slots"] if x["superseded"]]
    allslots = [x for r in pairs for x in r["A"]["spoken"]["slots"]]
    L += ["", f"Self-corrected parameters (n={len(sa)}): A correct {_rate([x['status'] == 'correct' for x in sa])}, wrong (another value, "
              f"usually the original) {_rate([x['status'] == 'wrong' for x in sa])}, missing {_rate([x['status'] == 'missing' for x in sa])}; "
              f"B final correct {_rate([x['status'] == 'correct' for x in sb])}, wrong {_rate([x['status'] == 'wrong' for x in sb])}.",
          f"Slots settled by the rules (A): {_rate([x['source'] == 'rule' for x in allslots])}; all A slots: correct "
          f"{_rate([x['status'] == 'correct' for x in allslots])}, wrong {_rate([x['status'] == 'wrong' for x in allslots])}, "
          f"missing {_rate([x['status'] == 'missing' for x in allslots])}."]
    fails = [r for r in pairs if not r["A"]["spoken"]["fulfilled"]]
    cat = {id(r): spoken_category(r) for r in fails}
    L += ["", "### Why the first response (A) fell short", "",
          "| category | n | share of A failures | recovered in B (final) |", "|---|---|---|---|"]
    for c in SPOKEN_CATS:
        rs = [r for r in fails if cat[id(r)] == c]
        L.append(f"| {c} | {len(rs)} | {len(rs) / max(1, len(fails)):.2f} | {_rate([r['B']['spoken_final']['fulfilled'] for r in rs]) if rs else '—'} |")
    L += ["", "| subset | A failures | " + " | ".join(SPOKEN_CATS) + " |", "|---|---|" + "---|" * len(SPOKEN_CATS)]
    subsets = [(d, lambda r, d=d: r["difficulty"] == d) for d in ("easy", "medium", "hard")]
    subsets += [(f, lambda r, f=f: f in r["disfluency"]) for f in ("FILLER", "PAUSE", "HESITATION", "FALSE_START", "SELF_CORRECTION")]
    for name, pred in subsets:
        rs = [r for r in fails if pred(r)]
        L.append(f"| {name} | {len(rs)} | " + " | ".join(str(sum(cat[id(r)] == c for r in rs)) for c in SPOKEN_CATS) + " |")
    L += ["", f"A runs where the agent spoke before the request was over: {_rate([r['A']['spoke_before_end'] > 0 for r in pairs])}; "
              f"A fulfilled when it did: {_rate([r['A']['spoken']['fulfilled'] for r in pairs if r['A']['spoke_before_end'] > 0])}, "
              f"when it did not: {_rate([r['A']['spoken']['fulfilled'] for r in pairs if r['A']['spoke_before_end'] == 0])}. "
              f"Agent turns in A: mean {statistics.mean(r['A']['agent_turns'] for r in pairs):.2f}. "
              f"Δt of the interruptions (ms): median {_med([r['A']['timing']['delta_ms'] for r in pairs if r['A']['timing']['interrupted']])}, "
              f"min {min([r['A']['timing']['delta_ms'] for r in pairs if r['A']['timing']['interrupted']], default=None)}; "
              f"≤ −1000 ms: {sum(r['A']['timing']['interrupted'] and r['A']['timing']['delta_ms'] <= -1000 for r in pairs)}."]
    L += ["", "### User effort (B)", ""]
    turns = [r["B"]["user_turns_after_first"] for r in pairs]
    labs: dict = {}
    for r in pairs:
        for x in r["B"]["user_labels"]:
            labs[x] = labs.get(x, 0) + 1
    L.append(f"User turns after the recording: mean {statistics.mean(turns):.2f}, median {statistics.median(turns)}, max {max(turns)}; "
             f"episodes with ≥1 correction: {_rate(['correct' in r['B']['user_labels'] for r in pairs])}; "
             f"≥1 answer: {_rate(['answer' in r['B']['user_labels'] for r in pairs])}; ≥1 repeat: {_rate(['repeat' in r['B']['user_labels'] for r in pairs])}. "
             f"Turn labels: {dict(sorted(labs.items(), key=lambda kv: -kv[1]))}. B end reasons: "
             f"{dict(sorted({e: sum(r['B']['end_reason'] == e for r in pairs) for e in {r['B']['end_reason'] for r in pairs}}.items()))}.")
    rec = [r for r in fails if r["B"]["spoken_final"]["fulfilled"]]
    if rec:
        given = [all(x["key"] in r["B"]["user_restated"] for x in r["A"]["spoken"]["slots"] if x["status"] != "correct") for r in rec]
        L.append(f"Recovered episodes in which the simulated user had itself said again every parameter the first response missed or got "
                 f"wrong: {_rate(given)} (the rest the agent got right after a nudge, a confirmation or a question answered without the value).")
    for name, rs in (("A ok", [r for r in pairs if r["A"]["spoken"]["fulfilled"]]), ("A fail", fails),
                     ("A fail → B ok", [r for r in fails if r["B"]["spoken_final"]["fulfilled"]])):
        if rs:
            L.append(f"- {name} (n={len(rs)}): user turns {statistics.mean(r['B']['user_turns_after_first'] for r in rs):.2f}, "
                     f"corrections {statistics.mean(r['B']['user_labels'].count('correct') for r in rs):.2f}, "
                     f"answers {statistics.mean(r['B']['user_labels'].count('answer') for r in rs):.2f}, "
                     f"repeats {statistics.mean(r['B']['user_labels'].count('repeat') for r in rs):.2f} per episode")
    L += ["", "### Timing (first user turn; official definitions)", "",
          "| | turn-take rate | interruption rate (Δt<0) | first response ms (median, non-interrupted) | Δt ms (median, all) |", "|---|---|---|---|---|"]
    for cond, k in (("A", "A"), ("B (before the user's 2nd turn)", "B")):
        ts = [r[k]["timing"] for r in pairs]
        L.append(f"| {cond} | {_rate([t['turn_taken'] for t in ts])} | {_rate([t['interrupted'] for t in ts])} | "
                 f"{_med([t['first_response_ms'] for t in ts])} | {_med([t['delta_ms'] for t in ts])} |")
    for name, pred in (("PAUSE", lambda r: "PAUSE" in r["disfluency"]), ("HESITATION", lambda r: "HESITATION" in r["disfluency"]),
                       ("SELF_CORRECTION", lambda r: "SELF_CORRECTION" in r["disfluency"]), ("no tag", lambda r: not r["disfluency"])):
        rs = [r for r in pairs if pred(r)]
        L.append(f"| A · {name} (n={len(rs)}) | {_rate([r['A']['timing']['turn_taken'] for r in rs])} | {_rate([r['A']['timing']['interrupted'] for r in rs])} | "
                 f"{_med([r['A']['timing']['first_response_ms'] for r in rs])} | {_med([r['A']['timing']['delta_ms'] for r in rs])} |")
    L += ["", f"Response quality (official prompt, proxy judge): A {_rate([r['A']['response_quality'] == 1.0 for r in pairs])}, "
              f"B (whole conversation) {_rate([r['B']['response_quality'] == 1.0 for r in pairs])}; recovered (A 0 → B 1) "
              f"{_rate([r['B']['response_quality'] == 1.0 for r in pairs if r['A']['response_quality'] != 1.0])}.",
          f"The agent has no tools: claims a completed action or reports a result it cannot know (LLM judge) — A "
          f"{_rate([r['A'].get('claims_result', False) for r in pairs])}, B {_rate([r['B'].get('claims_result', False) for r in pairs])}. "
          f"Response quality 1 in B when it claims: {_rate([r['B']['response_quality'] == 1.0 for r in pairs if r['B'].get('claims_result')])}, "
          f"when it does not: {_rate([r['B']['response_quality'] == 1.0 for r in pairs if not r['B'].get('claims_result')])}; "
          f"A: {_rate([r['A']['response_quality'] == 1.0 for r in pairs if r['A'].get('claims_result')])} / "
          f"{_rate([r['A']['response_quality'] == 1.0 for r in pairs if not r['A'].get('claims_result')])}.", ""]
    L += ["### Consistency check (A vs B up to the user's second turn)", ""]
    cs = [r["consistency"] for r in pairs]
    L.append(f"Identical agent speech (start, text): {_rate([c['speech'] for c in cs])}; also identical utterance ends (talker audio length): {_rate([c['text'] for c in cs])}; "
             f"identical first-response metrics: {_rate([c['first_response'] for c in cs])}; both: {_rate([c['ok'] for c in cs])}; "
             f"same per-slot outcome (A cut at B's 2nd user turn vs B first): {_rate([c['outcome_equal'] for c in cs])}.")
    # noise floor: the same input twice (A of two seeds — the seed only changes the simulated user) vs A and B of one seed
    by: dict = {}
    for r in pairs:
        by.setdefault(r["recording"], []).append(r)
    aa = [(x, y) for rs in by.values() for i, x in enumerate(rs) for y in rs[i + 1:]]
    if aa:
        L.append(f"Noise floor — A of two seeds (identical input; differences come from the server, e.g. batch-dependent numerics "
                 f"under concurrent sessions): first agent turn same start and text {_rate([_first_turn(x['A']) == _first_turn(y['A']) for x, y in aa])}, "
                 f"same first-response metrics {_rate([x['A']['timing'] == y['A']['timing'] for x, y in aa])}, same fulfilment "
                 f"{_rate([x['A']['spoken']['fulfilled'] == y['A']['spoken']['fulfilled'] for x, y in aa])}. "
                 f"A vs B of one seed: first agent turn {_rate([c['first_turn'] for c in cs])}, first-response metrics "
                 f"{_rate([c['first_response'] for c in cs])}, fulfilment (A vs B first) "
                 f"{_rate([r['A']['spoken']['fulfilled'] == r['B']['spoken_first']['fulfilled'] for r in pairs])}.")
    return "\n".join(L)


def _first_turn(cond: dict):
    return cond.get("first_turn")


async def score_tag(out: Path, judge: Judge, duplex: bool = False) -> tuple[list[dict], dict, dict]:
    A = {e["meta"]["episode_id"].split("/", 1)[1].rsplit("/", 1)[0]: e for e in load(out / "A" / "episodes.jsonl")}
    bp = out / "B" / "episodes.jsonl"
    B = {e["meta"]["episode_id"].split("/", 1)[1].rsplit("/", 1)[0]: e for e in load(bp)} if bp.exists() else {}
    sem = asyncio.Semaphore(32)

    async def one(k):
        async with sem:
            return await (score_pair_spoken if duplex else score_pair)(A[k], B.get(k), judge)

    return list(await asyncio.gather(*(one(k) for k in sorted(A)))), A, B


def kendall_tau(x: list[float], y: list[float]) -> float | None:
    n, c, d = len(x), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (x[i] - x[j]) * (y[i] - y[j])
            c += s > 0
            d += s < 0
    return (c - d) / (n * (n - 1) / 2) if n > 1 else None


async def report(args) -> None:
    root = Path(args.out)
    judge = Judge(OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=300))
    memo_path = root / "judge_memo.json"
    if memo_path.exists():
        judge.memo = json.loads(memo_path.read_text())
    tags = args.tags.split(",") if args.tags else [run_tag(args)]
    scored = {}
    for tag in tags:
        if (root / tag / "A" / "episodes.jsonl").exists():
            scored[tag] = await score_tag(root / tag, judge, duplex=tag.split("-ep")[0] in DUPLEX_AGENTS)
            memo_path.write_text(json.dumps(judge.memo))
    main = tags[0]
    patient_tag = next((t for t in scored if t.startswith(main + "-ep")), None)
    patient = {(r["recording"], r["seed"]): r for r in scored[patient_tag][0]} if patient_tag else None
    md = ["# FD-Bench v3: open loop (A) vs closed loop (B)", ""]
    for tag, (rows, A, B) in scored.items():
        (root / tag / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        if tag.split("-ep")[0] in DUPLEX_AGENTS:
            md += [f"## {tag} · spoken fulfilment (no tools)", "", tables_spoken(rows), ""]
            continue
        for key in ("pass", "lenient"):
            md += [f"## {tag} · {'official judge prompt' if key == 'pass' else 'lenient judge prompt'}", "",
                   tables(rows, key, patient if tag == main else None), ""]
    if sum(t.split("-ep")[0] not in DUPLEX_AGENTS for t in scored) > 1:
        md += ["## Agents: open- vs closed-loop ranking", "", "| agent | judge | A Pass@1 | B final | B strict | A interruption rate | A first response ms |", "|---|---|---|---|---|---|---|"]
        for key in ("pass", "lenient"):
            xs, ys = [], []
            for tag, (rows, _, _) in scored.items():
                pairs = [r for r in rows if "B" in r]
                a = sum(r["A"][key] for r in pairs) / len(pairs)
                bf = sum(r["B"][key + "_final"] for r in pairs) / len(pairs)
                bs = sum(r["B"][key + "_strict"] for r in pairs) / len(pairs)
                xs.append(a), ys.append(bf)
                md.append(f"| {tag} | {key} | {a:.3f} | {bf:.3f} | {bs:.3f} | {_rate([r['A']['timing']['interrupted'] for r in pairs])} | "
                          f"{_med([r['A']['timing']['first_response_ms'] for r in pairs])} |")
            md.append(f"| Kendall τ (A vs B final) | {key} | {kendall_tau(xs, ys)} | | | | |")
    text = "\n".join(md)
    (root / "summary.md").write_text(text)
    print(text)
    if args.pages:
        from interaction_gym.viewer import export_run

        for tag, (rows, A, B) in scored.items():
            byrow = {f"{r['recording']}/s{r['seed']}": r for r in rows}
            eps, notes = [], {}
            for k in sorted(A):
                r = byrow[k]
                if "B" not in r:
                    continue
                for e, cond in ((A[k], "A"), (B.get(k), "B")):
                    eps.append(e)
                    if "spoken" in r["A"]:  # duplex agent: spoken fulfilment
                        sp = r["A"]["spoken"] if cond == "A" else r["B"]["spoken_final"]
                        notes[e["meta"]["episode_id"]] = (
                            f"{r['difficulty']} {','.join(r['disfluency'])} · {cond} fulfilled={sp['fulfilled']} "
                            + " ".join(f"{x['key'].split('.')[1]}={x['status']}" for x in sp["slots"])
                            + (f" · Δt={r['A']['timing']['delta_ms']} · {spoken_category(r) or ''}" if cond == "A" else
                               f" · first={r['B']['spoken_first']['fulfilled']} · user turns {r['B']['user_turns_after_first']} "
                               f"{r['B']['user_labels']} · same first turn as A={r['consistency']['first_turn']}"))
                    elif cond == "A":
                        notes[e["meta"]["episode_id"]] = (f"{r['difficulty']} {','.join(r['disfluency'])} · A pass={r['A']['pass']} "
                                                          f"(lenient {r['A']['lenient']}) · acted before end={r['A']['acted_before_end']} · "
                                                          f"Δt={r['A']['timing']['delta_ms']} · {r['A']['pass_why']}")
                    else:
                        notes[e["meta"]["episode_id"]] = (f"B first={r['B']['pass_first']} final={r['B']['pass_final']} "
                                                          f"(lenient {r['B']['lenient_final']}) · user turns {r['B']['user_turns_after_first']} "
                                                          f"{r['B']['user_labels']} · consistent={r['consistency']['ok']}")
            print("->", export_run(eps, root / tag / "pages", notes, title=f"FD-Bench v3 open vs closed loop · {tag}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--out", default="runs/fdb3_ab")
    r.add_argument("--agent", default="cascaded-official", choices=list(AGENTS) + list(DUPLEX_AGENTS))
    r.add_argument("--a-idle-ms", type=int, default=None, help="condition A ends after this much quiet (default 3000; 8000 for duplex agents)")
    r.add_argument("--seeds", default="0")
    r.add_argument("--conditions", default="A,B")
    r.add_argument("--ids", default="")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--retries", type=int, default=6)
    r.add_argument("--endpoint-ms", type=int, default=800, help="end-of-turn silence of the agent's endpointing")
    r.add_argument("--audio-out", action="store_true", help="duplex agents speak (Thinker + Talker + Code2Wav server); "
                   "default: text-only output timed at speech_cps, Thinker-only server")
    p = sub.add_parser("report")
    p.add_argument("--out", default="runs/fdb3_ab")
    p.add_argument("--agent", default="cascaded-official", choices=list(AGENTS) + list(DUPLEX_AGENTS))
    p.add_argument("--pages", action="store_true")
    p.add_argument("--endpoint-ms", type=int, default=800)
    p.add_argument("--tags", default="", help="agent run folders to score, the main one first (its -epNNNN ablation is used for attribution)")
    a = ap.parse_args()
    if a.cmd == "run" and a.a_idle_ms is None:
        a.a_idle_ms = 8000 if a.agent in DUPLEX_AGENTS else 3000
    if a.cmd == "run":
        AUDIO_OUT = a.audio_out
    asyncio.run(run(a) if a.cmd == "run" else report(a))
