"""A suite of real conversations with MiniCPM-o 4.5 (vLLM-Omni duplex) as the agent, each user with its own
profile (gender, age, accent, speaking style → voice, TTS instructions and the user LLM's persona).

Runs on the GPU host (user-sim LLM :8000, TTS :8001, clone TTS :8005, MiniCPM-o :8010 with the lockstep + token-trace patches):

    PYTHONPATH=src:. python examples/minicpmo_suite.py --out runs/suite              # Thinker-only server, text output
    PYTHONPATH=src:. python examples/minicpmo_suite.py --out runs/suite --audio-out  # full audio deployment

Agent output: text only, timed at ``speech_cps``, against a Thinker-only server (the default), or with ``--audio-out``
the talker's speech (Thinker + Talker + Code2Wav); the audio run adds ``booking-alex-text``, the same scenario
text-only on the same server.

Writes ``<out>/episodes.jsonl`` (trajectories), ``<out>/agent_traces.jsonl`` (the agent server's per-unit token
trace, docs/AGENT_TRACE.md) and a browsable run (``index.html``, ``env.html``, ``*.agent.html``).
With ``--audio-out``: the first MiniCPM-o session after a server restart returns no audio; warm it up once first.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from array import array
from pathlib import Path

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent
from interaction_gym.audio import Audio
from interaction_gym.benchmarks import benchmark_user
from interaction_gym.clients import Cached, OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.soundscape import Soundscape
from interaction_gym.traj import episode, save
from interaction_gym.user import QWEN3_TTS_VOICES, LLMInterrupt, LLMSource, ReplayUser, ResponseDelay, TurnTaking, Voice
from interaction_gym.viewer import export_run

LLM_URL = getenv("IG_LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = getenv("IG_LLM_MODEL", "Qwen3.8-27B")
TTS_URL = getenv("IG_TTS_URL", "http://localhost:8001/v1")
AGENT_URL = getenv("IG_AGENT_URL", "ws://127.0.0.1:8010/v1/realtime?duplex=1")
AGENT_MODEL = getenv("IG_AGENT_MODEL", "openbmb/MiniCPM-o-4_5")
REF_AUDIO = getenv("IG_AGENT_REF_AUDIO", str(Path(getenv("IG_MODEL_DIR", "models")) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
# voice cloning (Qwen3-TTS Base): every user turn after the first is cloned from it, so a user keeps one voice
CLONE_URL = getenv("IG_CLONE_URL", "http://localhost:8005/v1")
CLONE_MODEL = getenv("IG_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
SR = 24000

RESTAURANT = "You are the phone booking assistant of Trattoria Roma, an Italian restaurant. Help callers book a table: ask for the date, time and number of people, then confirm the booking. Keep replies short."
CLINIC = "You are the receptionist of Greenfield Medical Clinic. Help callers book an appointment with a doctor: ask what it is about, offer a time, and confirm it. Speak clearly and keep replies short."
SHOP = "You are the customer service assistant of an online electronics shop. Help callers with orders and returns. Keep replies short."

SCENARIOS = {
    "booking-alex": dict(
        agent=RESTAURANT,
        persona="You are calling a restaurant to book a table.",
        profile={"name": "Alex Chen", "gender": "female", "age": 34, "language": "en", "occupation": "product manager",
                 "speaking_style": "friendly and quick, a little rushed", "traits": "busy, to the point"},
        goal="Book a table for two people tonight at around 7pm. Correct the agent if it gets a detail wrong.",
        first_turn="Hi, I'd like to book a table for tonight."),
    "clinic-frank": dict(
        agent=CLINIC,
        persona="You are calling your local clinic.",
        profile={"name": "Frank Li", "gender": "male", "age": 72, "language": "en", "occupation": "retired teacher",
                 "speaking_style": "slow and deliberate, with pauses", "traits": "polite, a little hard of hearing, asks to repeat things"},
        goal="Book an appointment with a doctor about your knee pain, preferably on a weekday morning.",
        first_turn="Hello? Is this the clinic? I'd like to see a doctor."),
    "return-jake": dict(
        agent=SHOP,
        persona="You are calling an online shop about a broken purchase.",
        profile={"name": "Jake Miller", "gender": "male", "age": 24, "language": "en", "accent": "american",
                 "speaking_style": "energetic, talks fast", "traits": "impatient, interrupts when the agent rambles"},
        goal="Return a pair of wireless headphones that stopped charging after a week (order 4471) and get a refund.",
        first_turn="Hey, I need to return some headphones, they're already broken.",
        turn_taking={"yield_after_ms": None}),  # never stops when talked over
    "booking-mina-cafe": dict(
        agent=RESTAURANT,
        persona="You are calling a restaurant from a noisy cafe.",
        profile={"name": "Mina Park", "gender": "female", "age": 45, "language": "en", "native_language": "ko", "occupation": "architect",
                 "speaking_style": "warm, measured", "traits": "speaks English with a Korean accent, double-checks details"},
        goal="Book a table for four on Saturday at 8pm, by the window if possible.",
        first_turn="Hello, I would like to make a reservation for Saturday, please.",
        background=-12),
}


def cafe_noise(seconds: int = 120, sr: int = SR, seed: int = 0) -> Audio:
    rng = random.Random(seed)
    return Audio(array("h", (int(rng.gauss(0, 900)) for _ in range(seconds * sr))), sr)


async def run(name: str, sc: dict, *, clock: str = "input", audio_out: bool = False):
    llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=80)
    voice = Voice(Cached(OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)), voices=QWEN3_TTS_VOICES, clone=OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR))
    # an evaluation user: no background floor and no random noise events unless the scenario sets a background
    # itself (booking-mina-cafe's cafe noise is part of what that scenario tests)
    user = benchmark_user(LLMSource(llm), voice=voice, interrupt=LLMInterrupt(llm), timing=TurnTaking(response_delay=ResponseDelay()),  # human-like reply gaps
                          soundscape=Soundscape(background="background" in sc))
    task = Task(id=name, scenario={k: sc[k] for k in ("persona", "profile", "first_turn", "turn_taking", "background") if k in sc} | {"instructions": sc["goal"]})
    spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)
    env = Env({"user": user}, spec, max_ms=120_000)  # the user lays the scenario's background, if it has one
    agent = VllmOmniDuplexAgent(spec, AGENT_URL, model=AGENT_MODEL, ref_audio=REF_AUDIO, clock=clock, audio_out=audio_out, trace_tokens=clock == "input",
                                session={"instructions": sc["agent"]})
    obs = await env.reset(task, seed=0)
    wall, done = time.monotonic(), False
    try:
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    return env, agent, time.monotonic() - wall


async def offline(media_root: Path, audio_out: bool = False):
    """The exactness case: pre-timed user turns with real TTS audio (a barge-in, a backchannel, a noise)."""
    tts = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)
    sched = [(0, "Hi, I'd like to book a table for tonight.", None), (9000, "Around seven pm, for two people please.", None),
             (15000, "Sorry, actually make it eight instead.", None), (23000, "mm-hmm", "backchannel"),
             (27500, "", "noise"), (33000, "Great, thank you. Bye!", None)]
    turns = []
    for t, text, kind in sched:
        audio = cafe_noise(1, seed=7)[: SR // 2] if kind == "noise" else await tts.synth(text, "serena")
        turns.append({"t": t, "text": text, "audio": audio, "kind": kind})
    spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)
    env = Env({"user": ReplayUser(turns, voice=Voice(tts, voice="serena", clone=False))}, spec, max_ms=60_000)
    agent = VllmOmniDuplexAgent(spec, AGENT_URL, model=AGENT_MODEL, ref_audio=REF_AUDIO, clock="input", audio_out=audio_out, trace_tokens=True,
                                session={"instructions": RESTAURANT})
    obs = await env.reset(Task(id="offline-replay", scenario={"persona": "Pre-recorded caller (offline replay)."}), seed=0)
    wall, done = time.monotonic(), False
    try:
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    return env, agent, time.monotonic() - wall


async def main(out: Path, audio_out: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    media = MediaStore(out)
    episodes, traces, notes = [], [], {}

    def record(eid, env, agent, wall, note, **agent_meta):
        traces.append(agent.trace(eid))
        ep = episode(env, eid, media=media, agent={**agent.describe(), **agent_meta}, meta={"wall_s": round(wall, 1)})
        episodes.append(ep)
        notes[eid] = f"{note} · {env.t / 1000:.1f}s in {wall:.1f}s wall"
        print(f"{eid:28s} {env.t / 1000:5.1f}s sim {wall:5.1f}s wall  duplex={list(ep['eval']['duplex'].values())}", flush=True)

    for name, sc in SCENARIOS.items():
        env, agent, wall = await run(name, sc, audio_out=audio_out)
        u = episode(env, "x")["meta"]["user"]
        record(name, env, agent, wall, f"{u['profile']['name']} ({u['profile']['gender']}, {u['profile']['age']}) · voice {u['voice']['speaker']}"
               + (" · never yields" if sc.get("turn_taking") else "") + (f" · background {sc['background']} dB" if "background" in sc else ""))
    env, agent, wall = await run("booking-alex", SCENARIOS["booking-alex"], clock="realtime", audio_out=audio_out)
    record("booking-alex-realtime", env, agent, wall, "same as booking-alex, realtime clock")
    if audio_out:  # an audio server also serves text-only sessions (a Thinker-only one serves only those)
        env, agent, wall = await run("booking-alex", SCENARIOS["booking-alex"], audio_out=False)
        record("booking-alex-text", env, agent, wall, "same as booking-alex, text-only output timed by speaking rate")
    env, agent, wall = await offline(out, audio_out)
    record("offline-replay", env, agent, wall, "pre-timed user turns with real TTS: barge-in at 15 s, backchannel, noise")

    save(episodes, out / "episodes.jsonl")
    save([t for t in traces if t], out / "agent_traces.jsonl")
    print("->", export_run(episodes, out, notes, title="MiniCPM-o suite", traces=traces))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/suite")
    ap.add_argument("--audio-out", action="store_true", help="MiniCPM-o speaks (Thinker + Talker + Code2Wav server); "
                    "default: text-only output timed at speech_cps, Thinker-only server")
    a = ap.parse_args()
    asyncio.run(main(Path(a.out), a.audio_out))
