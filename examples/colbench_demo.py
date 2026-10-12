"""ColBench Backend Programming, spoken: a caller explains a Python function, the agent asks, then submits the code.

The user is ColBench's human simulator (an LLM holding the hidden reference code); the agent is an LLM that hears the
transcript, sees the signature, asks clarification questions and hands in the function with ``submit_code``. The
hidden tests run after the episode (``tests_passed``, ``success``). See ``integrations/colbench.py`` for data and
licences (COLBENCH_PATH, COLBENCH_DATA).

    IG_LLM_URL=http://localhost:8000/v1 python examples/colbench_demo.py --n 20            # text-level evaluation
    python examples/colbench_demo.py --record runs/colbench --ids 1 9 13 --want 2          # voiced example episodes
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.cascaded import ToolChat
from interaction_gym.agents.transcript import TranscriptToolAgent
from interaction_gym.clients import OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.integrations import colbench as cb
from interaction_gym.media import MediaStore
from interaction_gym.tools import RESULT
from interaction_gym.traj import episode, save
from interaction_gym.user import TurnTaking, Voice

LLM_URL, LLM_MODEL = getenv("IG_LLM_URL", "http://localhost:8000/v1"), getenv("IG_LLM_MODEL", "Qwen3.8-27B")
TTS_URL, TTS_MODEL = getenv("IG_TTS_URL", "http://localhost:8001/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CLONE_URL, CLONE_MODEL = getenv("IG_CLONE_URL", "http://localhost:8005/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SR = 24000
SPEC = AgentSpec(chunk_ms=200, obs=("user.speech", RESULT["agent"]), sr=SR)


async def play(task, agent_llm, user_llm, voice=None, tts=None, max_ms=240_000):
    env = Env(cb.nodes(user_llm, voice=voice, timing=TurnTaking(respond_after_ms=700)), SPEC, max_ms=max_ms)
    agent = TranscriptToolAgent(SPEC, agent_llm, tts=tts, voice="aiden")
    obs, done = await env.reset(task, seed=0), False
    while not done:
        obs, _, done = await env.step(await agent.act(env.t, obs))
    return env, agent, cb.evaluate(env, task)


def show(task, env, res) -> None:
    ep = episode(env, task.id)
    print(f"\n== {task.id}  tests {res['tests_passed']:.1f}  success {res['success']}  submitted {res['submitted']}  {env.t / 1000:.0f}s  "
          f"{'truncated' if env.truncated else 'ended'}")
    for t in ep["turns"]:
        print(f"   {t['start_time'] / 1000:6.1f}s {t['role']:5s} {t['text'][:160]}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="text-level evaluation on the first n test tasks")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--record", type=Path, help="record voiced episodes of --ids into this run directory")
    ap.add_argument("--ids", type=int, nargs="*", default=[])
    ap.add_argument("--want", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=10, help="episodes run concurrently in the text-level evaluation")
    args = ap.parse_args()
    agent_llm = ToolChat(LLM_URL, LLM_MODEL, native=False, max_tokens=1200, temperature=0)
    user_llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=200, temperature=0)
    all_tasks = cb.tasks("test", n=1000)
    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        media = MediaStore(args.record)
        tts = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)
        episodes, notes = [], {}
        for i in args.ids:  # candidates in order; keep the first --want that pass, end by themselves and include a question
            if len(episodes) >= args.want:
                break
            task = all_tasks[i]
            task.scenario["voice"] = "serena" if len(episodes) % 2 == 0 else "ryan"
            voice = Voice(tts, voice=task.scenario["voice"], clone=OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR))
            env, agent, res = await play(task, agent_llm, user_llm, voice=voice, tts=tts)
            show(task, env, res)
            asked = sum(t["role"] == "user" and t["text"].strip() != "" for t in episode(env, task.id)["turns"]) >= 2
            if not (res["success"] and not env.truncated and asked):
                print("   skipped (needs: success, a natural end, at least one question answered)")
                continue
            reward = {"total": res["reward"], "parts": {"tests_passed": res["tests_passed"], "success": res["success"], "submitted": res["submitted"]}}
            ep = episode(env, task.id, media=media, agent=agent, reward=reward,
                         meta={"benchmark": "ColBench backend (test)", "submitted_code": cb.submitted_code(env)})
            episodes.append(ep)
            notes[task.id] = (f"ColBench backend programming (test #{i}): the caller explains the function, the agent asks, then "
                              f"submits the code (a tool call, never spoken) · hidden tests {sum(res['tests'])}/{len(res['tests'])}")
        save(episodes, args.record / "episodes.jsonl")
        (args.record / "agent_traces.jsonl").write_text("")
        (args.record / "qa.json").write_text("{}")
        (args.record / "notes.json").write_text(json.dumps(notes, indent=1))
        return
    sem = asyncio.Semaphore(args.parallel)

    async def one(task):
        async with sem:
            env, _, res = await play(task, agent_llm, user_llm)
            show(task, env, res)
            return res

    stats = await asyncio.gather(*(one(t) for t in all_tasks[args.start:args.start + args.n]))
    n = len(stats)
    print(f"\n{n} tasks: tests passed {sum(r['tests_passed'] for r in stats) / n:.3f}, success {sum(r['success'] for r in stats)}/{n}, "
          f"submitted {sum(r['submitted'] for r in stats)}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
