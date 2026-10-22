"""An OpenAI Realtime model as the agent, against an LLM-driven user — everything on the OpenAI API by default.

    OPENAI_API_KEY=... uv run python examples/openai_realtime_demo.py --html runs/openai_realtime.html

The user's words come from a chat model (``--user-llm``, default gpt-5.4-mini) and its voice from OpenAI's TTS (a
preset voice: hosted TTS cannot clone); ``IG_LLM_URL`` / ``IG_TTS_URL`` point either at a local server instead. The
agent is ``OpenAIRealtimeAgent`` (``--model``, default gpt-realtime-mini). The episode runs in wall time: the agent's
microphone is real-time paced, and the env holds still (unseen by the API) while the user simulator computes
(agents/openai_realtime.py). Each run costs money: the API's usage is printed at the end.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.openai_realtime import OpenAIRealtimeAgent
from interaction_gym.clients import OPENAI_URL, Cached, OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode
from interaction_gym.user import OPENAI_TTS_VOICES, LLMInterrupt, LLMSource, UserSim, Voice
from interaction_gym.viewer import export_html

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for `examples.*`
from examples.user_modes import TASK  # noqa: E402

SR = 24000


async def run(args) -> tuple[Env, OpenAIRealtimeAgent, dict]:
    llm = OpenAIChat(getenv("IG_LLM_URL", OPENAI_URL), args.user_llm, max_tokens=80)
    tts = Cached(OpenAISpeech(getenv("IG_TTS_URL", OPENAI_URL), args.user_tts, sr=SR))
    user = UserSim(LLMSource(llm), Voice(tts, voices=OPENAI_TTS_VOICES, voice="coral", clone=False), interrupt=LLMInterrupt(llm))
    spec = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR, obs=("user.speech", "tool.result"))
    env = Env({"user": user}, spec, max_ms=args.max_s * 1000)
    agent = OpenAIRealtimeAgent(spec, args.model, voice=args.voice,
                                instructions=args.instructions or "You are the phone assistant of a busy restaurant. "
                                "Help the caller with their booking. Keep replies short and natural.")
    t0 = time.perf_counter()
    obs = await env.reset(TASK, seed=args.seed)
    done = False
    try:
        while not (env.truncated or (done and not agent.busy)):
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    return env, agent, {"wall_s": round(time.perf_counter() - t0, 1), "sim_s": env.t / 1000,
                        "user_llm": llm.usage, "user_tts": tts.inner.usage, "agent": agent.usage}


async def main(args):
    env, agent, stats = await run(args)
    media = MediaStore(Path(args.html).parent)
    ep = episode(env, "restaurant-booking/openai-realtime", media=media, agent=agent)
    for t in ep["turns"]:
        print(f'{t["start_time"]:6d}-{t["end_time"]:6d} {t["role"]:5s} {t["text"]!r}' + (f'  [cut, unsaid {t["unsaid"]!r}]' if "unsaid" in t else ""))
    for r in agent.trace(ep["id"] if "id" in ep else "e")["responses"]:
        print(f'  response created {r["created_t"]} ms, first audio after {r["first_audio_wall_ms"]} ms wall, starts {r["start_t"]}, '
              f'{r["status"]}, played {r["played_ms"]}/{r["received_ms"]} ms')
    print(stats)
    print("duplex:", ep["eval"]["duplex"])
    print(f"-> {export_html([ep], args.html)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-realtime-mini")
    ap.add_argument("--voice", default="marin")
    ap.add_argument("--user-llm", default=getenv("IG_LLM_MODEL", "gpt-5.4-mini"))
    ap.add_argument("--user-tts", default=getenv("IG_TTS_MODEL", "gpt-4o-mini-tts"))
    ap.add_argument("--instructions", default=None)
    ap.add_argument("--max-s", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--html", default="runs/openai_realtime.html")
    asyncio.run(main(ap.parse_args()))
