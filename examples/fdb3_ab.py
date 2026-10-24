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
import time
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.cascaded import CascadedAgent, Endpointing, LatencyModel, ToolChat
from interaction_gym.agent_proxy import servers_behind
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent, check_output_mode, duplex_model, duplex_spec
from interaction_gym.benchmarks import benchmark_user, fdb3
from interaction_gym.benchmarks import fdb3_spoken as spoken
from interaction_gym.clients import OpenAIChat, OpenAISpeech, OpenAITranscribe
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.tools import RESULT, ToolWorld
from interaction_gym.traj import episode, save
from interaction_gym.user import ReplayUser, ResponseDelay, TurnTaking, Voice
from interaction_gym.benchmarks.reports.fdb3 import (  # noqa: F401
    ASKS_PROMPT, CATS, CLAIM_PROMPT, LABEL_PROMPT, SPOKEN_CATS, Judge, Lenient, categorize, kendall_tau, score_pair, score_pair_spoken,
    score_tag, spoken_category, tables, tables_spoken)
from interaction_gym.benchmarks.reports.fdb3 import report as _report


async def report(args) -> None:
    """The report (interaction_gym.benchmarks.reports.fdb3) with the local proxy LLM as the judge."""
    await _report(args, OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=300), default_tag=run_tag(args), duplex_names=set(DUPLEX_AGENTS))

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
DOMAINS = {"travel_identity": "a travel agency (flight search and booking, passport and ID document updates)",
           "finance_billing": "a bank (card benefits, currency exchange, autopay and bill payments)",
           "housing_location": "a rental housing agency (apartment search, commute times, search filters)",
           "ecommerce_support": "an online shop (product search, shopping cart, order tracking)"}
DUPLEX_PROMPT = ("You are the voice assistant of {domain}, answering a customer phone line. Callers tell you what they need done. "
                 "Let the caller finish speaking before you reply. Then confirm the request back to them in one or two short sentences, "
                 "repeating every detail they gave (names, dates, places, amounts, numbers and codes) with the final values they settled on: "
                 "if they corrected themselves, use the corrected value. Then say that you are taking care of it. "
                 "Only ask a question if something essential is missing. Your replies are spoken aloud: keep them short and natural.")
DUPLEX_AGENTS = {"minicpmo": DUPLEX_PROMPT, "gander": DUPLEX_PROMPT, "personaplex": DUPLEX_PROMPT}


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


def make_env(task, condition: str, seed: int, duplex: bool = False, a_idle_ms: int = 3000, agent_name: str = "minicpmo") -> Env:
    dspec = duplex_spec(agent_name, SR) if duplex else None
    tools = ToolWorld(fdb3.MockAPIBackend(), fdb3.latency_ms(task))
    if condition == "A":
        user = ReplayUser(voice=Voice())
        max_ms = task.scenario["benchmark"]["user_end_ms"] + 60_000
        if duplex:  # no tools; the model is given a_idle_ms of quiet to start / continue
            return Env({"user": user}, dspec, max_ms=max_ms, end_idle_ms=a_idle_ms)
    else:
        llm = OpenAIChat(LLM_URL, LLM_MODEL, temperature=0.7, seed=seed, max_tokens=200)
        clone = CloneWithFallback(OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR), OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR))
        user = benchmark_user(fdb3.UserSource(llm, USER_MAX_TURNS), voice=Voice(clone, clone=clone),
                       timing=TurnTaking(response_delay=ResponseDelay(), yield_after_ms=None, nudge_after_ms=10_000))
        max_ms = 180_000
    if duplex:
        return Env({"user": user}, dspec, max_ms=max_ms, end_idle_ms=3000)
    return Env({"user": user, "tools": tools}, SPEC, max_ms=max_ms, end_idle_ms=3000)


AUDIO_OUT = False  # MiniCPM-o output: False = text timed at speech_cps (Thinker-only server); --audio-out = the talker's speech


def make_duplex_agent(task, name: str = "minicpmo") -> VllmOmniDuplexAgent:
    return VllmOmniDuplexAgent(duplex_spec(name, SR), AGENT_URL, **{"audio_out": AUDIO_OUT, **duplex_model(name)}, out_sr=SR, clock="input",
                               trace_tokens=True, session={"instructions": duplex_prompt(task)})


def make_agent(name: str, seed: int, cache: dict, endpoint_ms: int = 800) -> CascadedAgent:
    return CascadedAgent(SPEC, OpenAITranscribe(ASR_URL, ASR_MODEL), ToolChat(LLM_URL, LLM_MODEL, native=False, max_tokens=512),
                         OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR), instructions=AGENTS[name], voice="aiden", seed=seed, cache=cache,
                         endpointing=Endpointing(silence_ms=endpoint_ms), latency=LatencyModel())


async def run_one(task, condition: str, seed: int, agent_name: str, cache: dict, endpoint_ms: int = 800, a_idle_ms: int = 3000):
    duplex = agent_name in DUPLEX_AGENTS
    env = make_env(task, condition, seed, duplex, a_idle_ms, agent_name)
    agent = make_duplex_agent(task, agent_name) if duplex else make_agent(agent_name, seed, cache, endpoint_ms)
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
    if duplex and jobs:  # the first session after a server start returns no audio: one throw-away episode per server
        n = servers_behind(AGENT_URL)  # behind the agent proxy, at once: one lands on each server
        warm = await asyncio.gather(*[run_one(recs[0][1], "A", 0, args.agent, {}, a_idle_ms=args.a_idle_ms) for _ in range(n)])
        env, agent, wall = warm[0]
        print(f"warm-up ({n} server{'s' * (n > 1)}): {env.t / 1000:.1f}s sim in {wall:.1f}s wall, agent events {agent.events}", flush=True)
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

# ---------------------------------------------------------------- report: duplex agents (no tools) — spoken fulfilment

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
