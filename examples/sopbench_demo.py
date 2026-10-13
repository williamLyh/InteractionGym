"""SOPBench, spoken: a customer calls a service line, the agent must follow the SOP before it acts (or declines).

The user is an LLM holding the case's goal and information, which it gives only when asked; the agent is an LLM
that hears the transcript, has SOPBench's assistant prompt and SOP and calls the domain's functions as tools.
SOPBench's verifier scores the calls and the final database after the call. See ``integrations/sopbench.py`` for the
setup and licences (SOPBENCH_PATH).

    IG_LLM_URL=http://localhost:8000/v1 python examples/sopbench_demo.py --domains bank hotel --n 20     # text-level
    python examples/sopbench_demo.py --record runs/sopbench --ids sopbench-bank-get_loan-3 ... --want 3  # voiced
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.cascaded import ToolChat
from interaction_gym.agents.transcript import TranscriptToolAgent
from interaction_gym.clients import OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.integrations import sopbench as sb
from interaction_gym.media import MediaStore
from interaction_gym.tools import RESULT
from interaction_gym.traj import episode, save
from interaction_gym.user import TurnTaking, Voice

LLM_URL, LLM_MODEL = getenv("IG_LLM_URL", "http://localhost:8000/v1"), getenv("IG_LLM_MODEL", "Qwen3.8-27B")
TTS_URL, TTS_MODEL = getenv("IG_TTS_URL", "http://localhost:8001/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CLONE_URL, CLONE_MODEL = getenv("IG_CLONE_URL", "http://localhost:8005/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SR = 24000
SPEC = AgentSpec(chunk_ms=200, obs=("user.speech", RESULT["agent"]), sr=SR)
CHECKS = ("no_tool_call_error", "constraint_not_violated", "database_match", "action_called_correctly", "dirgraph_satisfied")


async def play(task, agent_llm, user_llm, voice=None, tts=None, max_ms=300_000):
    env = Env(sb.nodes(task, user_llm, voice=voice, timing=TurnTaking(respond_after_ms=700)), SPEC, max_ms=max_ms)
    agent = TranscriptToolAgent(SPEC, agent_llm, tts=tts, voice="aiden")
    obs, done = await env.reset(task, seed=0), False
    while not done:
        obs, _, done = await env.step(await agent.act(env.t, obs))
    return env, agent, sb.evaluate(env, task)


def verdict(task, res) -> str:
    want = "do it" if task.criteria["action_should_succeed"] else "decline"
    failed = [c for c in CHECKS if not res[c]]
    return f"should {want} · {'pass' if res['success'] else 'fail: ' + ', '.join(failed)}"


def show(task, env, res) -> None:
    print(f"\n== {task.id}  {verdict(task, res)}  calls {len(res['calls'])}  {env.t / 1000:.0f}s  {'truncated' if env.truncated else 'ended'}")
    for t in episode(env, task.id)["turns"]:
        calls = "  " + "; ".join(f"{c['name']}({json.dumps(c['arguments'], ensure_ascii=False)[:80]})" for c in t.get("tool_calls", [])) if t.get("tool_calls") else ""
        print(f"   {t['start_time'] / 1000:6.1f}s {t['role']:5s} {t['text'][:150]}{calls}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=list(sb.DOMAINS))
    ap.add_argument("--n", type=int, default=10, help="text-level evaluation: n cases per domain, a seeded sample")
    ap.add_argument("--format", default="structured", choices=sb.FORMATS, help="how the SOP is written in the agent's prompt")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--record", type=Path, help="record voiced episodes of --ids into this run directory")
    ap.add_argument("--ids", nargs="*", default=[], help="task ids, in order of preference")
    ap.add_argument("--want", type=int, default=3)
    args = ap.parse_args()
    agent_llm = ToolChat(LLM_URL, LLM_MODEL, native=False, max_tokens=1200, temperature=0)
    user_llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=200, temperature=0)
    if args.record:
        by_id = {t.id: t for d in sb.DOMAINS for t in sb.tasks(d, args.format)}
        args.record.mkdir(parents=True, exist_ok=True)
        media = MediaStore(args.record)
        tts = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)
        episodes, notes = [], {}
        for tid in args.ids:  # keep the first --want that end by themselves after at least one function call
            if len(episodes) >= args.want:
                break
            task = by_id[tid]
            task.scenario["voice"] = "serena" if len(episodes) % 2 == 0 else "ryan"
            voice = Voice(tts, voice=task.scenario["voice"], clone=OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR))
            env, agent, res = await play(task, agent_llm, user_llm, voice=voice, tts=tts)
            show(task, env, res)
            if env.truncated or not res["calls"]:
                print("   skipped (needs: a natural end, at least one function call)")
                continue
            reward = {"total": res["reward"], "parts": {c: float(res[c]) for c in CHECKS}}
            sc = task.scenario["sopbench"]
            ep = episode(env, task.id, media=media, agent=agent, reward=reward,
                         meta={"benchmark": f"SOPBench {sc['domain']} (SOP format: {sc['sop_format']})", "sopbench": res,
                               "policy": sb.policy(task)}, verifier=sb.VERIFIER)
            episodes.append(ep)
            notes[task.id] = (f"SOPBench {sc['domain'].replace('_', ' ')} · {sc['user_goal'].replace('_', ' ')}: the caller gives their details only "
                              f"when asked; the agent must verify the SOP's conditions with function calls before acting · "
                              f"{verdict(task, res)} ({len(res['calls'])} calls)")
        save(episodes, args.record / "episodes.jsonl")
        (args.record / "agent_traces.jsonl").write_text("")
        (args.record / "qa.json").write_text("{}")
        (args.record / "notes.json").write_text(json.dumps(notes, indent=1))
        return
    picked = []
    for d in args.domains:
        ts = sb.tasks(d, args.format)
        picked += random.Random(0).sample(ts, min(args.n, len(ts)))
    sem = asyncio.Semaphore(args.parallel)

    async def one(task):
        async with sem:
            env, _, res = await play(task, agent_llm, user_llm)
            show(task, env, res)
            return task, res

    out = await asyncio.gather(*(one(t) for t in picked))
    tab = collections.defaultdict(lambda: [0, 0])
    for task, res in out:
        for key in (task.scenario["sopbench"]["domain"], "all"):
            for k in (key, f"{key} / should {'do it' if task.criteria['action_should_succeed'] else 'decline'}"):
                tab[k][0] += res["success"]
                tab[k][1] += 1
    print()
    for k in sorted(tab):
        print(f"{k:40s} {tab[k][0]:3d}/{tab[k][1]:<3d} pass")


if __name__ == "__main__":
    asyncio.run(main())
