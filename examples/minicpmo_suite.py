"""A suite of real conversations with MiniCPM-o 4.5 (vLLM-Omni duplex) as the agent, each user with its own
profile (gender, age, accent, speaking style → voice, TTS instructions and the user LLM's persona).

Runs on the GPU host (user-sim LLM :8000, TTS :8001, clone TTS :8005, MiniCPM-o :8010 with the lockstep + token-trace patches):

    PYTHONPATH=src:. python examples/minicpmo_suite.py --out runs/suite              # Thinker-only server, text output
    PYTHONPATH=src:. python examples/minicpmo_suite.py --out runs/suite --audio-out  # full audio deployment
    PYTHONPATH=src:. python examples/minicpmo_suite.py --out runs/chat --audio-out --set chat --seeds 0 1 2 --chunk-ms 1000
                                                 # casual chat + general knowledge, realistic users, 3 takes each

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
import json
import random
from importlib import resources
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
from interaction_gym.user import PERSONA, QWEN3_TTS_VOICES, LLMInterrupt, LLMSource, ReplayUser, ResponseDelay, TurnTaking, UserSim, Voice
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

# --set chat: casual conversation and everyday general knowledge with a general assistant (not a service desk). These
# users run with the env's realism on (``realistic``): persona-driven listening, pauses, random noise events and asides
# from their surroundings, and the surroundings' recorded ambience (sound bank: $IG_SOUNDBANK, DEMAND + MUSAN). The
# personas are written so that their manner shows in behaviour (profile levels: backchanneling, hesitancy, surroundings)
# rather than in what the user says about themselves.
# the scenarios: interaction_gym/data/minicpmo_suite.json (`ig run minicpmo_suite` runs the same file)
_SETS = json.loads((resources.files("interaction_gym") / "data" / "minicpmo_suite.json").read_text())
SCENARIOS, CHAT_SCENARIOS = _SETS["default"], _SETS["chat"]
RESTAURANT = SCENARIOS["booking-alex"]["agent"]
SETS = {"default": "SCENARIOS", "chat": "CHAT_SCENARIOS"}


def cafe_noise(seconds: int = 120, sr: int = SR, seed: int = 0) -> Audio:
    rng = random.Random(seed)
    return Audio(array("h", (int(rng.gauss(0, 900)) for _ in range(seconds * sr))), sr)


async def run(name: str, sc: dict, *, clock: str = "input", audio_out: bool = False, seed: int = 0, chunk_ms: int = 200):
    llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=80)
    voice = Voice(Cached(OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)), voices=QWEN3_TTS_VOICES, clone=OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR))
    # an evaluation user: no background floor and no random noise events unless the scenario sets a background
    # itself (booking-mina-cafe's cafe noise is part of what that scenario tests)
    if sc.get("realistic"):  # the env's realism: persona-driven events, asides and the surroundings' recorded ambience
        user = UserSim(LLMSource(llm), voice, interrupt=LLMInterrupt(llm), timing=TurnTaking(response_delay=ResponseDelay()),
                       behaviors=PERSONA, soundscape=Soundscape())
    else:
        user = benchmark_user(LLMSource(llm), voice=voice, interrupt=LLMInterrupt(llm), timing=TurnTaking(response_delay=ResponseDelay()),  # human-like reply gaps
                              soundscape=Soundscape(background="background" in sc))
    task = Task(id=name, scenario={k: sc[k] for k in ("persona", "profile", "first_turn", "turn_taking", "background", "voice") if k in sc} | {"instructions": sc["goal"]})
    spec = AgentSpec(chunk_ms=chunk_ms, audio="user.audio", sr=SR)
    env = Env({"user": user}, spec, max_ms=sc.get("max_ms", 120_000))  # the user lays the scenario's background, if it has one
    agent = VllmOmniDuplexAgent(spec, AGENT_URL, model=AGENT_MODEL, ref_audio=REF_AUDIO, clock=clock, audio_out=audio_out, trace_tokens=clock == "input",
                                session={"instructions": sc["agent"]})
    obs = await env.reset(task, seed=seed)
    wall, done = time.monotonic(), False
    try:
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    return env, agent, time.monotonic() - wall


async def offline(media_root: Path, audio_out: bool = False, chunk_ms: int = 200):
    """The exactness case: pre-timed user turns with real TTS audio (a barge-in, a backchannel, a noise)."""
    tts = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)
    sched = [(0, "Hi, I'd like to book a table for tonight.", None), (9000, "Around seven pm, for two people please.", None),
             (15000, "Sorry, actually make it eight instead.", None), (23000, "mm-hmm", "backchannel"),
             (27500, "", "noise"), (33000, "Great, thank you. Bye!", None)]
    turns = []
    for t, text, kind in sched:
        audio = cafe_noise(1, seed=7)[: SR // 2] if kind == "noise" else await tts.synth(text, "serena")
        turns.append({"t": t, "text": text, "audio": audio, "kind": kind})
    spec = AgentSpec(chunk_ms=chunk_ms, audio="user.audio", sr=SR)
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


async def main(out: Path, audio_out: bool = False, seeds: tuple = (0,), chunk_ms: int = 200, which: str = "default", only: tuple = ()):
    out.mkdir(parents=True, exist_ok=True)
    media = MediaStore(out)
    episodes, traces, notes = [], [], {}

    def record(eid, env, agent, wall, note, **agent_meta):
        traces.append(agent.trace(eid))
        ep = episode(env, eid, media=media, agent={**agent.describe(), **agent_meta}, meta={"wall_s": round(wall, 1)})
        episodes.append(ep)
        notes[eid] = f"{note} · {env.t / 1000:.1f}s in {wall:.1f}s wall"
        print(f"{eid:28s} {env.t / 1000:5.1f}s sim {wall:5.1f}s wall  duplex={list(ep['eval']['duplex'].values())}", flush=True)

    multi = len(seeds) > 1  # several takes per scenario: episode ids get a -s<seed> suffix
    sfx = lambda s: f"-s{s}" if multi else ""  # noqa: E731
    if which == "chat":  # the chat set only: one take per scenario per seed, all with the realistic user
        for seed in seeds:
            for name, sc in CHAT_SCENARIOS.items():
                if only and name not in only:
                    continue
                env, agent, wall = await run(name, sc, audio_out=audio_out, seed=seed, chunk_ms=chunk_ms)
                u = episode(env, "x")["meta"]["user"]
                p = u["profile"]
                record(name + sfx(seed), env, agent, wall, f"{p['name']} ({p['gender']}, {p['age']}) · voice {u['voice']['speaker']} · "
                       f"{p.get('surroundings', '')} surroundings")
        save(episodes, out / "episodes.jsonl")
        save([t for t in traces if t], out / "agent_traces.jsonl")
        print("->", export_run(episodes, out, notes, title="MiniCPM-o suite: chat", traces=traces))
        return
    for seed in seeds:
        for name, sc in SCENARIOS.items():
            env, agent, wall = await run(name, sc, audio_out=audio_out, seed=seed, chunk_ms=chunk_ms)
            u = episode(env, "x")["meta"]["user"]
            record(name + sfx(seed), env, agent, wall, f"{u['profile']['name']} ({u['profile']['gender']}, {u['profile']['age']}) · voice {u['voice']['speaker']}"
                   + (" · never yields" if sc.get("turn_taking") else "") + (f" · background {sc['background']} dB" if "background" in sc else ""))
        env, agent, wall = await run("booking-alex", SCENARIOS["booking-alex"], clock="realtime", audio_out=audio_out, seed=seed, chunk_ms=chunk_ms)
        record("booking-alex-realtime" + sfx(seed), env, agent, wall, "same as booking-alex, realtime clock")
        if audio_out:  # an audio server also serves text-only sessions (a Thinker-only one serves only those)
            env, agent, wall = await run("booking-alex", SCENARIOS["booking-alex"], audio_out=False, seed=seed, chunk_ms=chunk_ms)
            record("booking-alex-text" + sfx(seed), env, agent, wall, "same as booking-alex, text-only output timed by speaking rate")
        env, agent, wall = await offline(out, audio_out, chunk_ms)
        record("offline-replay" + sfx(seed), env, agent, wall, "pre-timed user turns with real TTS: barge-in at 15 s, backchannel, noise")

    save(episodes, out / "episodes.jsonl")
    save([t for t in traces if t], out / "agent_traces.jsonl")
    print("->", export_run(episodes, out, notes, title="MiniCPM-o suite", traces=traces))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/suite")
    ap.add_argument("--audio-out", action="store_true", help="MiniCPM-o speaks (Thinker + Talker + Code2Wav server); "
                    "default: text-only output timed at speech_cps, Thinker-only server")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="one take per scenario per seed (ids get -s<seed> with several)")
    ap.add_argument("--chunk-ms", type=int, default=200, help="agent step; MiniCPM-o decides once per 1 s unit, so 200 or 1000")
    ap.add_argument("--set", choices=sorted(SETS), default="default", dest="which",
                    help="default: the service-call scenarios plus the clock / output / replay variants; chat: casual "
                         "conversation and general knowledge (CHAT_SCENARIOS) with the env's realism on")
    ap.add_argument("--only", nargs="+", default=[], help="--set chat: run only these scenarios")
    a = ap.parse_args()
    asyncio.run(main(Path(a.out), a.audio_out, tuple(a.seeds), a.chunk_ms, a.which, tuple(a.only)))
