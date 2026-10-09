"""A full-duplex model as the agent: MiniCPM-o 4.5 on vLLM-Omni, talking to the LLM + TTS user.

Everything runs on the GPU host (the user-sim LLM on :8000, TTS on :8001, MiniCPM-o on :8010):

    PYTHONPATH=src:. python examples/minicpmo_agent.py --html runs/minicpmo_agent.html

The agent hears the mixed microphone every 200 ms and speaks whenever the model decides to. By default its output is
text only, timed at ``speech_cps`` (a Thinker-only server, the reference deployment's default); ``--audio-out`` plays
the talker's speech (Thinker + Talker + Code2Wav). With audio, the first session after the server starts returns no
audio; warm it up once first.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent
from interaction_gym.clients import Cached, OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, save
from interaction_gym.user import LLMInterrupt, LLMSource, UserSim, Voice
from interaction_gym.viewer import export_agent_trace_html, export_html

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the repository root, for `examples.*`
from examples.user_modes import TASK  # noqa: E402

LLM_URL = getenv("IG_LLM_URL", "http://localhost:8000/v1")
TTS_URL = getenv("IG_TTS_URL", "http://localhost:8001/v1")
CLONE_URL = getenv("IG_CLONE_URL", "http://localhost:8005/v1")  # the user's voice: cloned from its first turn
AGENT_URL = getenv("IG_AGENT_URL", "ws://127.0.0.1:8010/v1/realtime?duplex=1")
REF_AUDIO = getenv("IG_AGENT_REF_AUDIO", str(Path(getenv("IG_MODEL_DIR", "models")) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
INSTRUCTIONS = "You are the phone booking assistant of Trattoria Roma, an Italian restaurant. Help callers book a table: ask for the date, time and number of people, then confirm the booking. Keep replies short."


async def run(seed: int = 0, clock: str = "realtime", trace_tokens: bool = False, audio_out: bool = False):
    llm = OpenAIChat(LLM_URL, "Qwen3.8-27B", max_tokens=80)
    user = UserSim(LLMSource(llm), Voice(Cached(OpenAISpeech(TTS_URL, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", sr=24000)), voice="vivian",
                                              clone=OpenAISpeech(CLONE_URL, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", sr=24000)),
                   interrupt=LLMInterrupt(llm))
    spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=24000)
    env = Env({"user": user}, spec, max_ms=90_000)
    agent = VllmOmniDuplexAgent(spec, AGENT_URL, ref_audio=REF_AUDIO, clock=clock, trace_tokens=trace_tokens, audio_out=audio_out,
                                session={"instructions": INSTRUCTIONS})
    obs = await env.reset(TASK, seed=seed)
    wall = time.monotonic()
    done = False
    try:
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    print(f"clock={clock}: {env.t / 1000:.1f} s of conversation in {time.monotonic() - wall:.1f} s wall")
    print("server events:", agent.events)
    return env, agent


async def main(html: str, clock: str, trace_tokens: bool, audio_out: bool):
    env, agent = await run(clock=clock, trace_tokens=trace_tokens, audio_out=audio_out)
    media = MediaStore(Path(html).parent)
    ep = episode(env, "restaurant-booking/minicpmo-agent", media=media,
                 agent=agent.describe())
    for t in ep["turns"]:
        print(f'{t["start_time"]:6d}-{t["end_time"]:6d} {t["role"]:5s} {t["text"]!r}' + (f'  [unsaid {t["unsaid"]!r}]' if "unsaid" in t else ""))
    print("duplex:", ep["eval"]["duplex"], "end:", ep["meta"]["end_reason"])
    save([ep], Path(html).with_suffix(".jsonl"))
    print(f"-> {export_html([ep], html)}")
    trace = agent.trace(ep["meta"]["episode_id"])
    if trace:  # the agent server's own view: a separate record and a page of its own (docs/AGENT_TRACE.md)
        save([trace], Path(html).with_suffix(".agent_traces.jsonl"))
        print(f"-> {export_agent_trace_html(ep, trace, Path(html).with_suffix('.agent.html'))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="runs/minicpmo_agent.html")
    ap.add_argument("--clock", default="realtime", choices=["realtime", "input"], help="input = lockstep (needs the server extension)")
    ap.add_argument("--trace-tokens", action="store_true", help="record the model's per-unit token sequences (needs the server's token trace)")
    ap.add_argument("--audio-out", action="store_true", help="the talker's speech (Thinker + Talker + Code2Wav server); "
                    "default: text only, timed by the speaking rate (Thinker-only server)")
    a = ap.parse_args()
    asyncio.run(main(a.html, a.clock, a.trace_tokens, a.audio_out))
