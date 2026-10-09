"""Supply probes: short closed-loop bursts against running services, one concurrency level at a time.

Each level runs ``c`` workers that send requests back to back for ``seconds``; the per-stream time
per unit of work at that level becomes one point of the service's ``Curve``. Probing a balanced
endpoint with ``replicas`` replicas behind it, per-replica concurrency is ``c / replicas``.

Keep bursts short on shared hosts: the defaults are ~10 s per level, and the duplex agent is probed
with few sessions unless asked for more.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from array import array
from statistics import mean, median

from ..audio import Audio
from ..clients import OpenAISpeech, _post
from .profile import Curve

FILLER = ("The caller and the assistant talk about a booking: the date, the time, how many people, "
          "and whether a table by the window is free. ")


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


async def _closed_loop(c: int, seconds: float, once) -> list[tuple[float, float]]:
    """c workers calling ``once()`` (-> units of work) back to back; [(latency, units)] of calls started in the window."""
    deadline = time.perf_counter() + seconds
    out: list[tuple[float, float]] = []
    errors: list[BaseException] = []

    async def worker():
        while time.perf_counter() < deadline:
            t = time.perf_counter()
            try:
                units = await once()
            except Exception as e:  # noqa: BLE001 — a failing service shows up as an error count
                errors.append(e)
                if len(errors) > 3 * c:
                    raise
                continue
            out.append((time.perf_counter() - t, units))

    await asyncio.gather(*(worker() for _ in range(c)))
    if not out and errors:
        raise errors[0]
    return out


def _level(c: int, replicas: int, samples: list[tuple[float, float]], per_unit: bool) -> tuple[tuple[float, float], dict]:
    lat = [s[0] for s in samples]
    units = [s[1] for s in samples]
    ut = sum(lat) / sum(units) if per_unit else mean(lat)
    stats = {"c": c, "n": len(samples), "lat_p50": round(median(lat), 3), "lat_p95": round(_pct(lat, 0.95), 3),
             "unit_time": round(ut, 4), "throughput_per_replica": round(c / replicas / ut, 3)}
    return (c / replicas, ut), stats


async def probe_llm(url: str, model: str, *, prompt_tokens: int = 700, completion_tokens: int = 25,
                    levels=(1, 2, 4, 8, 16), seconds: float = 10.0, replicas: int = 1, api_key: str | None = None,
                    log=print) -> Curve:
    """Chat completions shaped like the user simulator's (prompt / completion sizes from the demand profile)."""
    endpoint = url.rstrip("/") + "/chat/completions"
    filler = (FILLER * (prompt_tokens * 4 // len(FILLER) + 1))[: prompt_tokens * 4]
    seen: dict[str, list[int]] = {"prompt": [], "completion": []}

    async def once():
        msgs = [{"role": "system", "content": "You are a caller on the phone. " + filler},
                {"role": "user", "content": f"[{random.random():.6f}] What do you say next?"}]
        payload = {"model": model, "messages": msgs, "max_tokens": completion_tokens, "min_tokens": completion_tokens,
                   "temperature": 0.7}
        data = json.loads(await asyncio.to_thread(_post, endpoint, payload, api_key))
        u = data.get("usage") or {}
        seen["prompt"].append(u.get("prompt_tokens", 0))
        seen["completion"].append(u.get("completion_tokens", 0))
        return 1.0

    points, stats = [], []
    for c in levels:
        pt, st = _level(c, replicas, await _closed_loop(c, seconds, once), per_unit=False)
        points.append(pt)
        stats.append(st)
        log(f"  llm   c={c:3d}  n={st['n']:4d}  p50={st['lat_p50']:.2f}s  p95={st['lat_p95']:.2f}s  {st['throughput_per_replica']:.2f} calls/s/replica")
    meta = {"url": url, "model": model, "replicas": replicas, "seconds": seconds, "levels": stats,
            "prompt_tokens": round(mean(seen["prompt"]), 1) if seen["prompt"] else None,
            "completion_tokens": round(mean(seen["completion"]), 1) if seen["completion"] else None}
    return Curve("call", points, meta=meta)


def _words_for(audio_s: float, wps: float = 2.6) -> str:
    words = (FILLER.split() * 50)[: max(3, round(audio_s * wps))]
    return " ".join(words)


async def probe_tts(url: str, model: str, *, audio_s: float = 3.0, voice: str = "vivian", levels=(1, 2, 4, 8),
                    seconds: float = 10.0, replicas: int = 1, ref_audio: Audio | None = None, ref_text: str | None = None,
                    log=print, name: str = "tts") -> Curve:
    """TTS (or, with ``ref_audio``, voice cloning) of utterances ~``audio_s`` long; unit = seconds of audio."""
    tts = OpenAISpeech(url, model, sr=24000)
    text = _words_for(audio_s)

    async def once():
        a = await tts.synth(text + random.choice([".", "!", "?"]), voice, ref_audio=ref_audio, ref_text=ref_text)
        return a.dur_ms / 1000

    points, stats = [], []
    for c in levels:
        pt, st = _level(c, replicas, await _closed_loop(c, seconds, once), per_unit=True)
        points.append(pt)
        stats.append(st)
        log(f"  {name:5s} c={c:3d}  n={st['n']:4d}  p50={st['lat_p50']:.2f}s  RTF={st['unit_time']:.3f}  "
            f"{st['throughput_per_replica']:.2f} audio-s/s/replica")
    return Curve("audio_s", points, meta={"url": url, "model": model, "replicas": replicas, "seconds": seconds,
                                          "audio_s_target": audio_s, "clone": ref_audio is not None, "levels": stats})


def speechlike(ms: int, sr: int = 16000, seed: int = 0) -> Audio:
    """Amplitude-modulated noise with syllable-rate bursts: not speech, but loud and bursty like it."""
    rng = random.Random(seed)
    n = round(ms * sr / 1000)
    return Audio(array("h", (int(rng.gauss(0, 3000) * (0.5 + 0.5 * math.sin(2 * math.pi * 4 * i / sr)) ** 2) for i in range(n))), sr)


async def probe_agent(url: str, *, model: str = "openbmb/MiniCPM-o-4_5", ref_audio: str | None = None, levels=(1, 2),
                      sim_s: float = 20.0, turns: list[Audio] | None = None, max_sessions: int = 4, chunk_ms: int = 200,
                      audio_out: bool = False, log=print) -> Curve:
    """k lockstep duplex sessions at once, each fed ``sim_s`` of pre-timed user turns (recorded/TTS
    audio, else speech-like noise) with gaps for the agent to answer; unit = simulated seconds.
    ``audio_out=False`` (default, the evaluation default): text-only sessions, for a Thinker-only server;
    True: the agent speaks (full Thinker + Talker + Code2Wav deployment)."""
    from ..agents.vllm_omni import VllmOmniDuplexAgent
    from ..core import AgentSpec, Env, Task
    from ..user import ReplayUser, Voice
    from .meter import Meter, MeteredAgent

    sr = 24000
    clips = [a.resample(sr) for a in (turns or [speechlike(2500, sr, s) for s in range(3)])]
    sched, t = [], 0
    while t < sim_s * 1000 - 3000:
        a = clips[len(sched) % len(clips)]
        sched.append({"t": t, "text": "(probe)", "audio": a})
        t += a.dur_ms + 5000  # 5 s for the agent to answer

    async def session(i: int, meter: Meter):
        spec = AgentSpec(chunk_ms=chunk_ms, audio="user.audio", sr=sr)
        env = Env({"user": ReplayUser(sched, voice=Voice())}, spec, max_ms=round(sim_s * 1000))
        agent = MeteredAgent(VllmOmniDuplexAgent(spec, url, model=model, ref_audio=ref_audio, clock="input", audio_out=audio_out,
                                                 session={"instructions": "You are a helpful phone assistant. Keep replies short."}),
                             meter, f"s{i}")
        tok = meter.start_episode(f"s{i}")
        obs = await env.reset(Task(id=f"probe-{i}", scenario={"persona": "probe"}), seed=i)
        done = False
        try:
            while not done:
                obs, _, done = await env.step(await agent.act(env.t, obs))
        finally:
            await agent.close()
            meter.end_episode(env.t, tok)

    points, stats = [], []
    for k in [k for k in levels if k <= max_sessions]:
        meter = Meter()
        t0 = time.perf_counter()
        await asyncio.gather(*(session(i, meter) for i in range(k)))
        per = []
        for i in range(k):
            evs = [e for e in meter.events if e.service == f"s{i}"][1:]  # the first step includes connecting
            per.append(sum(e.end - e.start for e in evs) / sum(e.units for e in evs))
        ut = mean(per)
        points.append((k, ut))
        stats.append({"c": k, "unit_time": round(ut, 4), "wall_s": round(time.perf_counter() - t0, 1),
                      "throughput_per_replica": round(k / ut, 3)})
        log(f"  agent k={k:3d}  {1 / ut:.2f}x real time per session  {k / ut:.2f} sim-s/s/replica")
    return Curve("sim_s", points, max_concurrency=max_sessions,
                 meta={"url": url, "model": model, "sim_s": sim_s, "chunk_ms": chunk_ms, "levels": stats,
                       "input": "recorded" if turns else "speech-like noise"})


def supply_from_meter(meter, replicas: dict[str, int] | None = None, units: dict[str, str] | None = None) -> dict[str, Curve]:
    """One point per service from a live run: per-stream time per unit at the concurrency observed.
    Cheap and in-situ, but a single point — the solver extrapolates it flat (treats it as saturated)."""
    from .profile import UNITS

    units = {**UNITS, **(units or {})}
    replicas = replicas or {}
    out = {}
    for s in meter.services():
        evs = [e for e in meter.events if e.service == s]
        busy = sum(e.end - e.start for e in evs)
        work = sum(e.units for e in evs)
        if not work:
            continue
        per_unit = units.get(s) != "call"
        ut = busy / work if per_unit else mean(e.end - e.start for e in evs)
        c = max(1.0, meter.concurrency(s)["mean"]) / replicas.get(s, 1)
        out[s] = Curve(units.get(s, "call"), [(c, ut)], source="in-situ", meta={"events": len(evs)})
    return out
