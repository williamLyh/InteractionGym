"""Live checks of OpenAIRealtimeAgent against the real API (each a short scripted call; a few cents in all).

    OPENAI_API_KEY=... uv run python examples/openai_realtime_checks.py                 # every check
    OPENAI_API_KEY=... uv run python examples/openai_realtime_checks.py barge tools     # some

The user is scripted (``ReplayUser``: fixed lines at fixed times, voiced by OpenAI's TTS), so each check controls when
the user speaks. A check passes when the adapter did its part — the model's own choices (whether it stops for a
backchannel, what it says) are printed, not judged:

- ``barge``: the user cuts into a long reply; playback stops where the server detected it, the server's item is
  truncated to exactly what was played, and the model answers the interruption.
- ``backchannel``: the user says "mm-hm" during a reply; if the model stopped, the truncation matches what was played.
- ``tools``: a function call goes to the env's tools, its result back to the model, which uses it.
- ``semantic``: ``semantic_vad`` turn detection ends the user's turn and the model answers.
- ``silence``: ``underrun="silence"`` (play slow audio with gaps, as a live client) still plays the reply.
- ``pcmu``: 8 kHz mu-law reply audio is decoded and played in full.

Exits non-zero if a check fails. Results depend on this machine's link to the API (agents/openai_realtime.py).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import warnings

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents.openai_realtime import OpenAIRealtimeAgent
from interaction_gym.clients import Cached, OpenAISpeech
from interaction_gym.tools import FunctionBackend, ToolWorld
from interaction_gym.traj import episode
from interaction_gym.user import ReplayUser, Voice

SR = 24000
MENU = "You are a restaurant phone assistant. When asked about the menu, describe it at length, course by course."


def check_availability(party: int, time: str) -> dict:
    """Check whether a table is free for a party size at a time."""
    return {"available": False, "alternatives": ["9:40 pm"], "note": "fully booked until 9:40 pm"}


async def call(model, name, turns, max_ms, tools=False, end_idle_ms=1500, **kw):
    spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR, obs=("user.speech", "tool.result"))
    nodes = {"user": ReplayUser(turns, voice=Voice(TTS, voice="coral", clone=False))}
    if tools:
        nodes["tools"] = ToolWorld(FunctionBackend({"check_availability": check_availability}, initial_state={}))
    env = Env(nodes, spec, max_ms=max_ms, end_idle_ms=end_idle_ms)
    agent = OpenAIRealtimeAgent(spec, model, **kw)
    obs = await env.reset(Task(id=name, scenario={}))
    done = False
    try:
        while not (env.truncated or (done and not agent.busy)):
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    ep = episode(env, name, agent=agent)
    tr = agent.trace(name)
    print(f"\n== {name}")
    for t in ep["turns"]:
        print(f'  {t["start_time"]:6d}-{t["end_time"]:6d} {t["role"]:5s} {t["text"][:110]!r}' + (" [cut]" if "unsaid" in t else ""))
    for r in tr["responses"]:
        print(f'  response: created {r["created_t"]}, first audio +{r["first_audio_wall_ms"]} ms wall, {r["status"]}, '
              f'played {r["played_ms"]}/{r["received_ms"]} ms' + (f', cut at {r["cut_t"]}' if "cut_t" in r else "")
              + (f', tools {r["tool_calls"]}' if r.get("tool_calls") else ""))
    print(f'  barriers {tr["barriers"]}, input lag (ms) {[x["handled_t"] - x["audio_t"] for x in tr["input_lag"]]}, errors {tr.get("errors", [])}')
    return ep, agent, tr


def agent_turns(ep):
    return [t for t in ep["turns"] if t["role"] == "agent"]


def truncations_match(tr) -> bool:
    """Every cut response that had audio played was truncated to exactly that much; those with none were deleted."""
    ev = tr["client_events"]
    for r in tr["responses"]:
        if "cut_t" not in r or r["received_ms"] == 0:
            continue
        cut = [e for e in ev if e["type"] == "conversation.item.truncate" and e["audio_end_ms"] == r["played_ms"]]
        gone = [e for e in ev if e["type"] == "conversation.item.delete"]
        if not (cut if r["played_ms"] else gone):
            return False
    return True


async def barge(model):
    turns = [{"t": 0, "text": "Hi, can you describe your whole dinner menu for me, every course in detail?"},
             {"t": 11000, "text": "Sorry to cut you off, do you have vegan options?"}]
    ep, agent, tr = await call(model, "barge", turns, 32000, instructions=MENU)
    first, onset = agent_turns(ep)[0], 11000
    return {"stopped within 1.5 s of the barge-in": first["start_time"] < onset < first["end_time"] <= onset + 1500,
            "truncated to what was played": truncations_match(tr) and any(e["type"] == "conversation.item.truncate" for e in tr["client_events"]),
            "answered the interruption": any(t["start_time"] > onset for t in agent_turns(ep)),
            "no errors": not tr.get("errors")}


async def backchannel(model):
    turns = [{"t": 0, "text": "Hi, can you describe your whole dinner menu for me, every course in detail?"},
             {"t": 11000, "text": "Mm-hm."}]
    ep, agent, tr = await call(model, "backchannel", turns, 30000, instructions=MENU)
    stopped = agent_turns(ep)[0]["end_time"] < 12500
    print(f"  (the model {'stopped' if stopped else 'kept talking'} for the backchannel: its behaviour, not judged)")
    return {"truncations match what was played": truncations_match(tr), "no errors": not tr.get("errors")}


async def tools(model):
    turns = [{"t": 0, "text": "Hi, do you have a table for two at seven tonight?"}, {"t": 16000, "text": "Okay, thanks, bye."}]
    ep, agent, tr = await call(model, "tools", turns, 24000, tools=True,
                               instructions="You are a restaurant phone assistant. Always use check_availability before answering about tables.")
    out = [e for e in tr["client_events"] if e["type"] == "conversation.item.create" and e["item"]["type"] == "function_call_output"]
    said = " ".join(t["text"] for t in agent_turns(ep)).lower()
    return {"function call made": any(r.get("tool_calls") for r in tr["responses"]),
            "result returned to the model": bool(out) and any(e["type"] == "response.create" for e in tr["client_events"]),
            "the model used the result": "9:40" in said or "nine forty" in said, "no errors": not tr.get("errors")}


async def semantic(model):
    vad = {"type": "semantic_vad", "eagerness": "auto", "create_response": True, "interrupt_response": True}
    ep, agent, tr = await call(model, "semantic", [{"t": 0, "text": "Hi, what time do you close on Sundays?"}], 16000,
                               end_idle_ms=8000, turn_detection=vad)
    return {"answered": any(t["text"] for t in agent_turns(ep)), "no errors": not tr.get("errors")}


async def silence(model):
    ep, agent, tr = await call(model, "silence", [{"t": 0, "text": "Hi, what time do you open tomorrow?"}], 15000,
                               end_idle_ms=3000, underrun="silence")
    return {"answered": any(t["text"] for t in agent_turns(ep)), "no errors": not tr.get("errors")}


async def pcmu(model):
    ep, agent, tr = await call(model, "pcmu", [{"t": 0, "text": "Hi, what time do you open tomorrow?"}], 15000,
                               end_idle_ms=3000, audio_format="pcmu")
    r = [r for r in tr["responses"] if r["received_ms"]]
    return {"answered": any(t["text"] for t in agent_turns(ep)),
            "played in full": bool(r) and all(x["played_ms"] == x["received_ms"] for x in r if x["status"] == "completed"),
            "no errors": not tr.get("errors")}


CHECKS = {"barge": barge, "backchannel": backchannel, "tools": tools, "semantic": semantic, "silence": silence, "pcmu": pcmu}


async def main(args):
    failed = []
    for name in args.checks or list(CHECKS):
        verdict = await CHECKS[name](args.model)
        for what, ok in verdict.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {what}")
        if not all(verdict.values()):
            failed.append(name)
    print(f"\n{len(failed)} check(s) failed: {failed}" if failed else "\nall checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checks", nargs="*", choices=[*CHECKS, []])
    ap.add_argument("--model", default="gpt-realtime-mini")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message=".*does not take.*")  # hosted TTS ignores seed / language
    TTS = Cached(OpenAISpeech.openai("gpt-4o-mini-tts", sr=SR))
    sys.exit(asyncio.run(main(args)))
