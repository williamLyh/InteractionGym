"""A simulated user backed by real models: a text LLM writes what the user says, a TTS voices it.

The services (vLLM + vLLM-Omni) run on a GPU host; forward them first, e.g.
    ssh -N -L 8000:localhost:8000 -L 8001:localhost:8001 <gpu-host>

    uv run python examples/live_user.py --html runs/live_user.html

Endpoints and models can be overridden with IG_LLM_URL / IG_LLM_MODEL / IG_TTS_URL /
IG_TTS_MODEL / IG_TTS_VOICE.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents import CannedAgent
from interaction_gym.clients import Cached, OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode
from interaction_gym.user import LLMInterrupt, LLMSource, UserSim, Voice
from interaction_gym.viewer import export_html

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for `examples.*`
from examples.user_modes import AGENT_REPLIES, TASK  # noqa: E402

LLM_URL = getenv("IG_LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = getenv("IG_LLM_MODEL", "Qwen3.8-27B")
TTS_URL = getenv("IG_TTS_URL", "http://localhost:8001/v1")
TTS_MODEL = getenv("IG_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
TTS_VOICE = getenv("IG_TTS_VOICE", "vivian")
# the user's turns after the first are cloned from it (Qwen3-TTS Base): one voice, pace and manner for the whole call
CLONE_URL = getenv("IG_CLONE_URL", "http://localhost:8005/v1")
CLONE_MODEL = getenv("IG_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")


class Counting:
    """Counts calls and wall time spent in a client."""

    def __init__(self, inner):
        self.inner, self.calls, self.seconds = inner, 0, 0.0

    async def chat(self, messages, **kw):
        t = time.perf_counter()
        try:
            return await self.inner.chat(messages, **kw)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - t

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None, **kw):
        t = time.perf_counter()
        try:
            return await self.inner.synth(text, voice, instructions, ref_audio, ref_text, language, **kw)
        finally:
            self.calls += 1
            self.seconds += time.perf_counter() - t


async def run(seed: int = 0):
    llm = Counting(OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=80))
    tts = Counting(Cached(OpenAISpeech(TTS_URL, TTS_MODEL, sr=24000)))
    clone = Counting(OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=24000))
    user = UserSim(LLMSource(llm), Voice(tts, voice=TTS_VOICE, clone=clone), interrupt=LLMInterrupt(llm))
    spec = AgentSpec(chunk_ms=200, sr=24000)
    env = Env({"user": user}, spec, max_ms=90_000)
    agent = CannedAgent(AGENT_REPLIES, spec, min_words=2)
    t0 = time.perf_counter()
    obs = await env.reset(TASK, seed=seed)
    done = False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs))
    wall = time.perf_counter() - t0
    return env, {"llm_calls": llm.calls, "llm_s": round(llm.seconds, 2), "tts_calls": tts.calls,
                 "tts_s": round(tts.seconds, 2), "wall_s": round(wall, 2), "sim_s": env.t / 1000}


async def main(html: str):
    env, stats = await run()
    media = MediaStore(Path(html).parent)
    ep = episode(env, "restaurant-booking/live-user", media=media)
    for t in ep["turns"]:
        print(f'{t["start_time"]:6d}-{t["end_time"]:6d} {t["role"]:5s} {t["text"]!r}' + (f'  [cut, unsaid {t["unsaid"]!r}]' if "unsaid" in t else ""))
    print(stats)  # run cost, printed only: not part of the episode
    print("duplex:", ep["eval"]["duplex"])
    print(f"-> {export_html([ep], html)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="runs/live_user.html")
    asyncio.run(main(ap.parse_args().html))
