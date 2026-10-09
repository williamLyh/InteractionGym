#!/usr/bin/env python3
"""Check the input-clocked ("lockstep") duplex protocol against the MiniCPM-o 4.5 server.

With ``session.update`` ``extra_body.clock = "input"`` the server acknowledges every
``input_audio_buffer.append`` with one ``input_audio_buffer.processed`` event, sent only after
every output caused by that append. This script sends 200 ms appends, each only after the previous
append's ``processed`` arrived, and checks:

  (a) no input -> no time: stream a question until the model is speaking, then send nothing for
      --pause-s of wall time: no event may arrive and unit_end_ms must not move; the answer then
      resumes with further input.
  (b) faster than real time: >= 30 s of input (question, silence, second question, silence) as fast
      as acks allow; report input seconds vs wall seconds and keyword-check both answers.
  (c) determinism: (b) twice more, once with random wall-clock sleeps (0-2 s) between appends; the
      sequence of (unit decisions, transcript text, audio delta durations, and the input position at
      which each output appeared = the processed.audio_end_ms it preceded) must be identical.
  (d0) a session without clock=input gets no acks and still answers at real-time pace; the full
      non-lockstep regression is minicpmo_realtime.py (4/4 expected).

Run next to the server (any Python with the script's imports):  python scripts/lockstep_check.py [--only a,b,c]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
import websockets
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
IN_SR, OUT_SR, CHUNK_MS = 16000, 24000, 200
STEP = IN_SR * CHUNK_MS // 1000  # samples per append

Q1 = ("What is the capital of France?", ["paris"])
Q2 = ("What is two plus three?", ["five", "5"])


def tts_16k(url: str, model: str, text: str, voice: str) -> np.ndarray:
    """User speech as 16 kHz int16, cached on disk so every run sends identical bytes."""
    cache = ROOT / "logs" / "duplex_tests" / "tts_cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16] + ".pcm16k")
    if not path.exists():
        r = requests.post(url, json={"model": model, "input": text, "voice": voice, "response_format": "pcm", "language": "English"}, timeout=120)
        r.raise_for_status()
        x = np.frombuffer(r.content, dtype=np.int16).astype(np.float32)
        path.write_bytes(np.clip(np.round(resample_poly(x, 2, 3)), -32768, 32767).astype(np.int16).tobytes())
    return np.frombuffer(path.read_bytes(), dtype=np.int16)


def chunks(pcm: np.ndarray) -> list[bytes]:
    pcm = np.pad(pcm, (0, (-len(pcm)) % STEP))
    return [pcm[i : i + STEP].tobytes() for i in range(0, len(pcm), STEP)]


def silence_chunks(seconds: float) -> list[bytes]:
    return [bytes(STEP * 2)] * int(round(seconds * 1000 / CHUNK_MS))


def session_payload(ref_audio: str, lockstep: bool = True) -> dict:
    extra = {"auto_response": True, "force_listen_count": 0}
    if lockstep:
        extra["clock"] = "input"
    return {
        "model": "openbmb/MiniCPM-o-4_5",
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "audio": {"input": {"sample_rate_hz": IN_SR}, "output": {"sample_rate_hz": OUT_SR}},
        "turn_detection": None,
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(Path(ref_audio).read_bytes()).decode(),
        "extra_body": extra,
    }


class LockstepSession:
    """One input-clocked session: send an append, wait for its ack, record everything in between."""

    def __init__(self, ws):
        self.ws = ws
        self.events: list[dict] = []  # every server event, in arrival order, with t (wall s)
        self.acks: list[dict] = []
        self.sent = 0
        self.t0 = time.monotonic()
        self._ack_cond = asyncio.Condition()
        self.error: dict | None = None

    async def reader(self):
        async for raw in self.ws:
            ev = json.loads(raw)
            ev["_t"] = time.monotonic() - self.t0
            self.events.append(ev)
            if ev.get("type") == "error":
                self.error = ev
            if ev.get("type") == "input_audio_buffer.processed":
                async with self._ack_cond:
                    self.acks.append(ev)
                    self._ack_cond.notify_all()

    async def append(self, pcm: bytes, timeout: float = 120.0) -> dict:
        await self.ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode(), "format": "pcm16", "sample_rate_hz": IN_SR}))
        self.sent += 1
        async with self._ack_cond:
            await asyncio.wait_for(self._ack_cond.wait_for(lambda: len(self.acks) >= self.sent), timeout)
        return self.acks[self.sent - 1]


async def open_session(url: str, ref_audio: str):
    ws = await websockets.connect(url, max_size=None)
    await ws.send(json.dumps({"type": "session.update", "session": session_payload(ref_audio)}))
    s = LockstepSession(ws)
    task = asyncio.create_task(s.reader())
    return s, task


def signature(events: list[dict]) -> list[tuple]:
    """What the model did, keyed by input position (the audio_end_ms of the ack each output preceded)."""
    sig, pending = [], []
    for ev in events:
        t = ev.get("type")
        if t == "input_audio_buffer.processed":
            for item in pending:
                sig.append((ev["audio_end_ms"], *item))
            pending = []
            for u in ev.get("units") or []:
                sig.append((ev["audio_end_ms"], "unit", u.get("end_ms"), u.get("decision")))
        elif t == "response.output_audio.delta":
            pending.append(("audio_ms", round(len(base64.b64decode(ev.get("delta", ""))) / 2 / OUT_SR * 1000, 1)))
        elif t == "response.output_audio_transcript.delta":
            pending.append(("text", ev.get("delta")))
        elif t in ("response.created", "response.done"):
            pending.append((t, ((ev.get("response") or {}).get("status"))))
    for item in pending:
        sig.append((None, *item))
    return sig


def answers(events: list[dict]) -> list[str]:
    out, cur = [], None
    for ev in events:
        t = ev.get("type")
        if t == "response.created":
            cur = ""
        elif t == "response.output_audio_transcript.delta" and cur is not None:
            cur += ev.get("delta") or ""
        elif t == "response.done" and cur is not None:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def contains(text: str, keys: list[str]) -> bool:
    said = re.sub(r"\s+", "", text.lower())
    return any(k.lower().replace(" ", "") in said for k in keys)


# ----------------------------------------------------------------------------- (a)
async def check_a(a, q1: np.ndarray) -> bool:
    print("\n=== (a) no input -> no time")
    s, rx = await open_session(a.url, a.ref_audio)
    try:
        plan = chunks(q1) + silence_chunks(20.0)
        n_first_audio = None
        for i, pcm in enumerate(plan):
            ack = await s.append(pcm)
            if n_first_audio is None and any(e.get("type") == "response.output_audio.delta" for e in s.events):
                n_first_audio = i
            if n_first_audio is not None and i >= n_first_audio + 1:
                break
        assert n_first_audio is not None, "the model never started speaking"
        created = next((e for e in s.events if e.get("type") == "session.created"), {})
        print(f"    session idle_timeout_s = {(created.get('session') or {}).get('idle_timeout_s')}")
        n_events, unit_end = len(s.events), ack["unit_end_ms"]
        print(f"    model speaking; paused after {ack['audio_end_ms']} ms of input (unit_end_ms={unit_end}, {n_events} events so far)")
        print(f"    sending nothing for {a.pause_s:.0f} s of wall time ...")
        await asyncio.sleep(a.pause_s)
        during = s.events[n_events:]
        print(f"    events received during the pause: {len(during)} {[e.get('type') for e in during][:5]}")
        # Resume: the model picks up exactly where it stopped.
        acks_after = []
        for pcm in plan[s.sent : s.sent + 10]:
            acks_after.append(await s.append(pcm))
        resumed_audio = sum(1 for e in s.events[n_events:] if e.get("type") == "response.output_audio.delta")
        print(f"    after resuming: 10 appends -> unit_end_ms {unit_end} -> {acks_after[-1]['unit_end_ms']}, {resumed_audio} more audio deltas")
        ok = not during and acks_after[-1]["unit_end_ms"] > unit_end and s.error is None
        print(f"    [{'PASS' if ok else 'FAIL'}] no events and no unit progress without input; progress resumes with input")
        return ok
    finally:
        await s.ws.send(json.dumps({"type": "session.close"}))
        rx.cancel()
        await s.ws.close()


# ----------------------------------------------------------------------------- (d0)
async def check_plain(a, q1: np.ndarray) -> bool:
    """A session without clock=input: no acks, and the model still answers under real-time pacing."""
    print("\n=== (d0) plain (non-lockstep) session, real-time pacing")
    ws = await websockets.connect(a.url, max_size=None)
    await ws.send(json.dumps({"type": "session.update", "session": session_payload(a.ref_audio, lockstep=False)}))
    events: list[dict] = []

    async def reader():
        async for raw in ws:
            events.append(json.loads(raw))

    rx = asyncio.create_task(reader())
    plan = chunks(q1) + silence_chunks(8.0)
    t0 = time.monotonic()
    for i, pcm in enumerate(plan):
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode(), "format": "pcm16", "sample_rate_hz": IN_SR}))
        await asyncio.sleep(max(0.0, t0 + (i + 1) * CHUNK_MS / 1000 - time.monotonic()))
    await asyncio.sleep(1.0)
    await ws.send(json.dumps({"type": "session.close"}))
    await asyncio.sleep(0.3)
    rx.cancel()
    await ws.close()
    acks = sum(1 for e in events if e.get("type") == "input_audio_buffer.processed")
    reply = " ".join(answers(events))
    ok = acks == 0 and contains(reply, Q1[1])
    print(f"    processed events: {acks}; reply: {reply.strip()!r}")
    print(f"    [{'PASS' if ok else 'FAIL'}] no acks without clock=input; answer correct")
    return ok


# ----------------------------------------------------------------------------- (b)/(c)
async def run_b(a, plan: list[bytes], label: str, jitter: float = 0.0, seed: int = 0) -> dict:
    s, rx = await open_session(a.url, a.ref_audio)
    rng = random.Random(seed)
    sleep_total = 0.0
    try:
        t_start = time.monotonic()
        ack = None
        for pcm in plan:
            ack = await s.append(pcm)
            if jitter:
                d = rng.uniform(0, jitter)
                sleep_total += d
                await asyncio.sleep(d)
        wall = time.monotonic() - t_start
    finally:
        await s.ws.send(json.dumps({"type": "session.close"}))
        await asyncio.sleep(0.3)
        rx.cancel()
        await s.ws.close()
    input_s = len(plan) * CHUNK_MS / 1000
    replies = answers(s.events)
    units = [u for e in s.acks for u in (e.get("units") or [])]
    r = {
        "label": label, "input_s": input_s, "wall_s": wall, "sleep_s": sleep_total, "replies": replies,
        "sig": signature(s.events), "acks": len(s.acks), "appends": len(plan), "unit_end_ms": ack["unit_end_ms"],
        "audio_end_ms": ack["audio_end_ms"], "units": units, "error": s.error,
        "timed_out": sum(1 for u in units if u.get("timed_out")),
    }
    dec = "".join("S" if u.get("decision") == "speak" else "." if u.get("decision") == "listen" else "?" for u in units)
    print(f"--- {label}: {input_s:.1f} s input in {wall:.2f} s wall ({input_s / wall:.2f}x real time"
          + (f"; incl. {sleep_total:.1f} s of random sleeps; {input_s / max(1e-9, wall - sleep_total):.2f}x excluding them" if jitter else "") + ")")
    print(f"    acks {len(s.acks)}/{len(plan)}, audio_end_ms={ack['audio_end_ms']}, unit_end_ms={ack['unit_end_ms']}, units={len(units)} timed_out={r['timed_out']}")
    print(f"    unit decisions: {dec}")
    for i, text in enumerate(replies):
        print(f"    reply {i + 1}: {text.strip()!r}")
    if s.error:
        print(f"    ERROR: {s.error}")
    return r


def check_replies(r: dict) -> bool:
    rs = r["replies"]
    ok1 = any(contains(t, Q1[1]) for t in rs[:1])
    ok2 = any(contains(t, Q2[1]) for t in rs[1:])
    return ok1 and ok2


def diff_sig(x: list, y: list) -> str:
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return f"first difference at item {i}: {p} vs {q}"
    return f"lengths differ: {len(x)} vs {len(y)}" if len(x) != len(y) else "identical"


async def main(a):
    q1 = tts_16k(a.tts_url, a.tts_model, Q1[0], a.voice)
    q2 = tts_16k(a.tts_url, a.tts_model, Q2[0], a.voice)
    only = set(a.only.split(","))
    results = {}
    out_dir = ROOT / "logs" / "lockstep"
    out_dir.mkdir(parents=True, exist_ok=True)
    if "a" in only:
        results["a"] = await check_a(a, q1)
    if "d0" in only:
        results["d0"] = await check_plain(a, q1)
    plan = chunks(q1) + silence_chunks(a.gap_s) + chunks(q2) + silence_chunks(a.gap_s)
    runs = []
    if "b" in only or "c" in only:
        print(f"\n=== (b) faster than real time: {len(plan)} appends x {CHUNK_MS} ms = {len(plan) * CHUNK_MS / 1000:.1f} s of input")
        runs.append(await run_b(a, plan, "run 1 (no sleeps)"))
        ok_speed = runs[0]["input_s"] / runs[0]["wall_s"] > 1.0
        ok_ans = check_replies(runs[0])
        ok_acks = runs[0]["acks"] == runs[0]["appends"] and runs[0]["timed_out"] == 0
        print(f"    [{'PASS' if ok_speed else 'FAIL'}] faster than real time: {runs[0]['input_s'] / runs[0]['wall_s']:.2f}x")
        print(f"    [{'PASS' if ok_ans else 'FAIL'}] answers correct (Q1 -> {Q1[1]}, Q2 -> {Q2[1]})")
        print(f"    [{'PASS' if ok_acks else 'FAIL'}] one ack per append, no unit timed out")
        results["b"] = ok_speed and ok_ans and ok_acks
    if "c" in only:
        print("\n=== (c) determinism")
        runs.append(await run_b(a, plan, "run 2 (no sleeps)"))
        runs.append(await run_b(a, plan, f"run 3 (random 0-{a.jitter_s:g} s sleeps)", jitter=a.jitter_s, seed=a.seed))
        base = runs[0]["sig"]
        ok = True
        for r in runs[1:]:
            same = r["sig"] == base
            ok = ok and same
            print(f"    {r['label']} vs run 1: {diff_sig(base, r['sig'])} ({len(r['sig'])} items)")
        print(f"    [{'PASS' if ok else 'FAIL'}] identical (unit decisions, transcript, audio durations, input position) across runs")
        results["c"] = ok
    for r in runs:
        (out_dir / (re.sub(r"\W+", "_", r["label"]).strip("_") + ".json")).write_text(json.dumps({k: v for k, v in r.items()}, indent=1, default=str))
    print("\nsummary: " + ", ".join(f"({k}) {'PASS' if v else 'FAIL'}" for k, v in results.items()) + f"   (logs in {out_dir})")
    return all(results.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8010/v1/realtime?duplex=1")
    ap.add_argument("--ref-audio", default=str(Path(os.environ.get("IG_MODELS_DIR", os.environ.get("DIG_MODELS_DIR", ROOT / "models"))) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
    ap.add_argument("--tts-url", default="http://127.0.0.1:8001/v1/audio/speech")
    ap.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--voice", default="ryan")
    ap.add_argument("--gap-s", type=float, default=14.0, help="silence after each question")
    ap.add_argument("--pause-s", type=float, default=12.0)
    ap.add_argument("--jitter-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="a,b,c,d0")
    sys.exit(0 if asyncio.run(main(ap.parse_args())) else 1)
