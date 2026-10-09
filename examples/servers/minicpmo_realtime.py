"""Talk to the MiniCPM-o 4.5 full-duplex server like a realtime client, and check its answers.

The server (vLLM-Omni, ``vllm serve ... --omni``) speaks a full-duplex dialect of the OpenAI
Realtime API on ``ws://HOST:8010/v1/realtime?duplex=1``: the client streams 16 kHz PCM16 with
``input_audio_buffer.append`` and the *model itself* decides, once per 1 s of input, whether to
listen or speak (``turn_detection`` is off; no client-side VAD or ``response.create``). Speech comes
back as 24 kHz PCM16 in ``response.output_audio.delta`` with a transcript alongside.

For each question: synthesize it with the user-sim TTS, stream it in 200 ms chunks at real-time
pace followed by silence, and record the reply. A question passes when the reply's transcript
contains an expected keyword. Writes, per question, the agent's audio and a stereo
``*_dialog.wav`` (left = user, right = agent, as a listener would have heard it).

Run on the GPU host (needs websockets, numpy, scipy, requests), e.g.
    python minicpmo_realtime.py --ref-audio <MiniCPM-o-4_5>/assets/system_ref_audio.wav --out /tmp/minicpmo_demo
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import time
import wave
from pathlib import Path

import numpy as np
import requests
import websockets
from scipy.signal import resample_poly

IN_SR, OUT_SR, CHUNK_MS = 16000, 24000, 200

QUESTIONS = [  # (text, language, any of these in the reply = correct)
    ("What is the capital of France?", "English", ["paris"]),
    ("What is two plus three?", "English", ["five", "5"]),
    ("How many days are there in a week?", "English", ["seven", "7"]),
    ("中国的首都是哪个城市？", "Chinese", ["北京", "beijing"]),
]


def tts(url: str, model: str, text: str, language: str, voice: str) -> np.ndarray:
    """User speech as 16 kHz int16 (the TTS returns 24 kHz PCM16)."""
    r = requests.post(url, json={"model": model, "input": text, "voice": voice, "response_format": "pcm", "language": language}, timeout=120)
    r.raise_for_status()
    x = np.frombuffer(r.content, dtype=np.int16).astype(np.float32)
    return np.clip(resample_poly(x, 2, 3), -32768, 32767).astype(np.int16)


def wav_bytes(pcm: np.ndarray, sr: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


async def ask(url: str, ref_audio: str, user: np.ndarray, tail_s: float, max_s: float) -> dict:
    """One session: stream ``user`` + silence at real-time pace; collect the reply."""
    session = {
        "model": "openbmb/MiniCPM-o-4_5",
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "audio": {"input": {"sample_rate_hz": IN_SR}, "output": {"sample_rate_hz": OUT_SR}},
        "turn_detection": None,  # the model decides when to speak
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(Path(ref_audio).read_bytes()).decode(),  # its voice
        "extra_body": {"auto_response": True, "force_listen_count": 0},
    }
    step = IN_SR * CHUNK_MS // 1000
    silence_after = len(user) // step + (len(user) % step > 0)
    chunks = [user[i : i + step] for i in range(0, len(user), step)]
    chunks[-1] = np.pad(chunks[-1], (0, step - len(chunks[-1])))
    out = {"deltas": [], "transcript": "", "done": None, "events": {}}

    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        t0 = time.monotonic()
        stop = asyncio.Event()

        async def receive():
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type", "")
                out["events"][t] = out["events"].get(t, 0) + 1
                if t == "response.output_audio.delta":
                    out["deltas"].append((time.monotonic() - t0, np.frombuffer(base64.b64decode(ev["delta"]), dtype=np.int16)))
                elif t == "response.output_audio_transcript.done":
                    out["transcript"] += ev.get("transcript", "")
                elif t == "response.done":
                    out["done"] = ev["response"].get("status_details")
                    stop.set()
                elif t == "error":
                    raise RuntimeError(ev)

        rx = asyncio.create_task(receive())
        n, tail_end = 0, None
        while True:  # one chunk per CHUNK_MS of wall time, like a live microphone
            if rx.done():
                rx.result()
            if stop.is_set() and tail_end is None:
                tail_end = n + int(tail_s * 1000 / CHUNK_MS)
            if (tail_end is not None and n >= tail_end) or n * CHUNK_MS >= max_s * 1000:
                break
            pcm = chunks[n] if n < len(chunks) else np.zeros(step, np.int16)
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm.tobytes()).decode(), "format": "pcm16", "sample_rate_hz": IN_SR}))
            n += 1
            await asyncio.sleep(max(0.0, t0 + n * CHUNK_MS / 1000 - time.monotonic()))
        await ws.send(json.dumps({"type": "session.close"}))
        rx.cancel()
    out["user_end_s"] = silence_after * CHUNK_MS / 1000
    out["sent_s"] = n * CHUNK_MS / 1000
    return out


def dialog(user: np.ndarray, deltas: list, total_s: float) -> np.ndarray:
    """Stereo track as heard live: each agent delta plays when it arrives (or after the previous one)."""
    n = int(total_s * OUT_SR) + sum(len(d) for _, d in deltas)
    left = np.zeros(n, np.int16)
    u = resample_poly(user.astype(np.float32), 3, 2).astype(np.int16)
    left[: len(u)] = u
    right, pos = np.zeros(n, np.int16), 0
    for t, d in deltas:
        pos = max(pos, int(t * OUT_SR))
        right[pos : pos + len(d)] = d
        pos += len(d)
    end = max(len(u), pos)
    return np.stack([left[:end], right[:end]], axis=1).reshape(-1)


async def main(a):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    passed = 0
    for i, (text, lang, keys) in enumerate(QUESTIONS):
        user = tts(a.tts_url, a.tts_model, text, lang, a.voice)
        r = await ask(a.url, a.ref_audio, user, a.tail_s, a.max_s)
        agent = np.concatenate([d for _, d in r["deltas"]]) if r["deltas"] else np.zeros(0, np.int16)
        (out_dir / f"q{i}_agent.wav").write_bytes(wav_bytes(agent, OUT_SR))
        (out_dir / f"q{i}_dialog.wav").write_bytes(wav_bytes(dialog(user, r["deltas"], r["sent_s"]), OUT_SR, channels=2))
        said = re.sub(r"\s+", "", r["transcript"].lower())  # the model's text sometimes splits words ("sc atters")
        ok = any(k.lower().replace(" ", "") in said for k in keys)
        passed += ok
        first = r["deltas"][0][0] - r["user_end_s"] if r["deltas"] else None
        print(f"[{'PASS' if ok else 'FAIL'}] Q{i}: {text}")
        print(f"       reply ({len(agent) / OUT_SR:.1f}s audio, first audio {first:+.2f}s after the user stopped)" if first is not None else "       reply: no audio")
        print(f"       {r['transcript'].strip()!r}   end={r['done']}")
    print(f"\n{passed}/{len(QUESTIONS)} correct · audio in {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8010/v1/realtime?duplex=1")
    ap.add_argument("--ref-audio", default=str(Path(os.environ.get("IG_MODEL_DIR", os.environ.get("DIG_MODEL_DIR", "models"))) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
    ap.add_argument("--tts-url", default="http://127.0.0.1:8001/v1/audio/speech")
    ap.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--voice", default="ryan")
    ap.add_argument("--tail-s", type=float, default=2.0, help="silence to keep sending after the reply ends")
    ap.add_argument("--max-s", type=float, default=40.0, help="give up on a session after this much input")
    ap.add_argument("--out", default="/tmp/minicpmo_demo")
    asyncio.run(main(ap.parse_args()))
