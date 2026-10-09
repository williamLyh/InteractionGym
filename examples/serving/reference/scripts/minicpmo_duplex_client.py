#!/usr/bin/env python3
"""Raw-WebSocket test client for the MiniCPM-o 4.5 full-duplex server (vLLM-Omni DuplexOmni).

Protocol: OpenAI-Realtime dialect on ws://127.0.0.1:8010/v1/realtime?duplex=1 (alias /v1/duplex).
Input: 16 kHz mono PCM16 in input_audio_buffer.append events. Output: 24 kHz PCM16 in
response.output_audio.delta. The model decides listen/speak natively once per 1 s unit.

Scenarios (all sessions use greedy Thinker decoding + seed 42 from the deploy config):
  basic      user question (TTS) -> silence; record everything; realtime pacing by default
  bargein    question -> once the agent has produced >= --barge-after-ms of audio, send a
             second utterance (realtime pacing), then silence; record whether/when it stops
  pause      same chunk sequence sent three ways: fast (no sleeps), fast + 2 s wall-clock
             pauses after chunks --pause-after, and realtime pacing; diff the event streams
Every received event is logged (JSONL) with t_s (since session start), input audio sent so far,
and a compact summary (base64 stripped; audio deltas -> duration ms + sha1).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import requests
import websockets
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
MODEL = "openbmb/MiniCPM-o-4_5"
TTS_URL = "http://127.0.0.1:8001/v1/audio/speech"
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
IN_SR = 16000
CHUNK_MS = 200
CHUNK_BYTES = IN_SR * CHUNK_MS // 1000 * 2  # 6400 bytes of PCM16


# ----------------------------------------------------------------------------- audio helpers
def tts_16k(text: str, voice: str = "ryan", cache_dir: Path | None = None) -> bytes:
    """Local Qwen3-TTS (24 kHz int16 PCM) -> 16 kHz int16 PCM bytes. Cached on disk."""
    cache_dir = cache_dir or ROOT / "logs" / "duplex_tests" / "tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{voice}|{text}".encode()).hexdigest()[:16]
    path = cache_dir / f"{key}.pcm16k"
    if path.exists():
        return path.read_bytes()
    r = requests.post(
        TTS_URL,
        json={"model": TTS_MODEL, "input": text, "voice": voice, "response_format": "pcm", "language": "English"},
        timeout=120,
    )
    r.raise_for_status()
    x24 = np.frombuffer(r.content, dtype=np.int16).astype(np.float32)
    x16 = resample_poly(x24, 2, 3)
    pcm = np.clip(np.round(x16), -32768, 32767).astype(np.int16).tobytes()
    path.write_bytes(pcm)
    return pcm


def silence(ms: int) -> bytes:
    return bytes(IN_SR * ms // 1000 * 2)


def chunks_of(pcm: bytes) -> list[bytes]:
    pad = (-len(pcm)) % CHUNK_BYTES
    pcm = pcm + bytes(pad)
    return [pcm[i : i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)]


def wav_data_url(path: Path) -> str:
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()


def write_wav(path: Path, pcm: bytes, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


# ----------------------------------------------------------------------------- session
def session_payload(args) -> dict:
    ref = Path(args.ref_audio)
    s = {
        "model": MODEL,
        "modalities": ["audio", "text"],
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "sample_rate_hz": IN_SR,
        "audio": {"input": {"sample_rate_hz": IN_SR}, "output": {"sample_rate_hz": 24000}},
        "turn_detection": None,
        "ref_audio": wav_data_url(ref),
        "overlap_policy": args.overlap_policy,
        "playback_commit_policy": "ack_only",
        "idle_timeout_s": 600,
        "extra_body": {"auto_response": True, "force_listen_count": 0},
    }
    if args.instructions:
        s["instructions"] = args.instructions
    if args.temperature is not None:
        s["temperature"] = args.temperature
    return s


def summarize(ev: dict) -> dict:
    """Compact, base64-free view of one server event."""
    t = ev.get("type")
    out: dict = {"type": t}
    for k in ("response_id", "item_id", "reason", "server_event_seq", "epoch", "action", "policy", "transcript", "text"):
        if k in ev and not isinstance(ev[k], (dict, list)):
            out[k] = ev[k]
    vo = ((ev.get("metadata") or {}).get("vllm_omni")
          or (((ev.get("response") or {}).get("metadata") or {}).get("vllm_omni")) or {})
    s0 = (vo.get("stage_metrics") or {}).get("0") or {}
    if "num_tokens_in" in s0:
        out["s0_tokens_in"] = s0.get("num_tokens_in")  # Stage-0 prompt length = model position marker
        out["s0_tokens_out"] = s0.get("num_tokens_out")
    if t in ("response.output_audio.delta", "response.audio.delta"):
        raw = base64.b64decode(ev.get("delta", ""))
        sr = int(ev.get("sample_rate_hz") or 24000)
        out["audio_ms"] = round(len(raw) / 2 / sr * 1000, 1)
        out["sr"] = sr
        out["sha1"] = hashlib.sha1(raw).hexdigest()[:10]
        md = ev.get("metadata") or {}
        for k in ("end_of_turn", "audio_duration_ms", "model_speak", "audio_text_marks"):
            if k in md:
                out[k] = md[k]
    elif t and t.endswith(".delta"):
        out["delta"] = ev.get("delta")
    if t in ("response.done", "response.listen", "response.created"):
        r = ev.get("response") or {}
        out["status"] = r.get("status")
        if r.get("status_details"):
            out["status_details"] = r.get("status_details")
        md = r.get("metadata") or {}
        for k in ("reason", "model_listen", "buffering", "committed"):
            if k in md:
                out[k] = md[k]
        if md.get("playback"):
            out["playback"] = md["playback"]
    if t == "overlap.decision":
        out["metadata"] = ev.get("metadata")
    if t == "error":
        out["error"] = ev.get("error")
    if t in ("session.created", "session.updated"):
        s = ev.get("session") or {}
        out["session_id"] = s.get("id")
        out["capabilities"] = s.get("capabilities") if t == "session.created" else None
    return out


class Run:
    def __init__(self, name: str, args, out_dir: Path):
        self.name = name
        self.args = args
        self.out_dir = out_dir
        self.events: list[dict] = []
        self.t0 = time.monotonic()
        self.sent_ms = 0
        self.audio_out = bytearray()
        self.agent_audio_ms = 0.0
        self.first_audio_t: float | None = None
        self.ws = None
        self.raw_log = None
        self.raw_full = None

    def t(self) -> float:
        return round(time.monotonic() - self.t0, 3)

    async def reader(self):
        async for raw in self.ws:
            if isinstance(raw, (bytes, bytearray)):
                continue
            ev = json.loads(raw)
            if self.raw_full is not None:
                stripped = {k: (f"<{len(v)} b64 chars>" if k in ("delta", "audio") and isinstance(v, str) else v)
                            for k, v in ev.items()}
                self.raw_full.write(json.dumps({"t_s": self.t(), **stripped}, ensure_ascii=False) + "\n")
            s = summarize(ev)
            rec = {"t_s": self.t(), "input_sent_ms": self.sent_ms, **s}
            self.events.append(rec)
            self.raw_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if s["type"] in ("response.output_audio.delta", "response.audio.delta"):
                raw_audio = base64.b64decode(ev.get("delta", ""))
                self.audio_out += raw_audio
                self.agent_audio_ms += s["audio_ms"]
                if self.first_audio_t is None:
                    self.first_audio_t = rec["t_s"]
            seq = ev.get("server_event_seq")
            if isinstance(seq, int):
                try:
                    await self.ws.send(json.dumps({"type": "session.event_ack", "server_event_seq": seq}))
                except Exception:
                    pass
            if self.args.verbose and s["type"] not in ("session.heartbeat_ack",):
                print(f"[{self.name} {rec['t_s']:7.3f}s in={self.sent_ms:6d}ms] {json.dumps(s, ensure_ascii=False)[:220]}")
            if s["type"] == "session.closed":
                return

    async def send_chunk(self, pcm: bytes):
        self.sent_ms += len(pcm) * 1000 // (IN_SR * 2)
        await self.ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                    "format": "pcm16",
                    "sample_rate_hz": IN_SR,
                    "duration_ms": len(pcm) * 1000 // (IN_SR * 2),
                    "audio_end_ms": self.sent_ms,
                }
            )
        )

    async def __aenter__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_log = open(self.out_dir / f"{self.name}.jsonl", "w")
        self.raw_full = open(self.out_dir / f"{self.name}.raw.jsonl", "w") if self.args.dump_raw else None
        url = f"ws://127.0.0.1:{self.args.port}{self.args.path}"
        sep = "&" if "?" in url else "?"
        url += f"{sep}model={MODEL}&autostart=0"
        self.ws = await websockets.connect(url, max_size=64 << 20)
        self.t0 = time.monotonic()
        self.reader_task = asyncio.create_task(self.reader())
        await self.ws.send(json.dumps({"type": "session.update", "session": session_payload(self.args)}))
        for _ in range(600):
            if any(e["type"] == "session.created" for e in self.events):
                break
            if any(e["type"] == "error" for e in self.events):
                raise RuntimeError(f"handshake error: {self.events}")
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("no session.created")
        return self

    async def __aexit__(self, *exc):
        try:
            await self.ws.send(json.dumps({"type": "session.close"}))
            await asyncio.wait_for(self.reader_task, 20)
        except Exception:
            pass
        await self.ws.close()
        self.raw_log.close()
        if self.raw_full is not None:
            self.raw_full.close()
        if self.audio_out:
            write_wav(self.out_dir / f"{self.name}_agent.wav", bytes(self.audio_out), 24000)


async def stream(run: Run, chunks: list[bytes], mode: str, pauses: dict[int, float] | None = None, on_chunk=None):
    """mode: 'realtime' (sleep chunk duration after each send) or 'fast' (no sleeps)."""
    pauses = pauses or {}
    start = time.monotonic()
    for i, c in enumerate(chunks):
        await run.send_chunk(c)
        if on_chunk is not None:
            await on_chunk(i)
        if i in pauses:
            await asyncio.sleep(pauses[i])
        if mode == "realtime":
            # absolute schedule (no drift), offset by any pauses taken so far
            extra = sum(v for k, v in pauses.items() if k <= i)
            target = start + (i + 1) * CHUNK_MS / 1000 + extra
            await asyncio.sleep(max(0.0, target - time.monotonic()))


async def drain(run: Run, quiet_s: float, max_s: float):
    """Wait until no event arrived for quiet_s (or max_s elapsed)."""
    t_end = time.monotonic() + max_s
    last_n = -1
    last_change = time.monotonic()
    while time.monotonic() < t_end:
        n = len(run.events)
        if n != last_n:
            last_n, last_change = n, time.monotonic()
        elif time.monotonic() - last_change > quiet_s:
            return
        await asyncio.sleep(0.1)


# ----------------------------------------------------------------------------- reports
def content_view(events: list[dict], with_listen: bool = True) -> list[tuple]:
    """Timing-free view used for determinism diffs."""
    view = []
    for e in events:
        t = e["type"]
        if not with_listen and t == "response.listen":
            continue
        if t in ("session.created", "session.updated", "session.closed", "rate_limits.updated", "playback.acknowledged"):
            continue
        if t in ("response.output_audio.delta", "response.audio.delta"):
            view.append((t, e.get("audio_ms"), e.get("end_of_turn"), e.get("s0_tokens_in"), e.get("sha1")))
        elif t.endswith("transcript.delta") or t.endswith("text.delta"):
            view.append((t, e.get("delta")))
        elif t == "response.listen":
            view.append((t, e.get("status"), e.get("reason"), e.get("s0_tokens_in")))
        elif t == "response.done":
            view.append((t, e.get("status"), e.get("reason")))
        else:
            view.append((t,))
    return view


def print_summary(run: Run, speech_end_ms: int | None = None):
    ev = run.events
    text = "".join(e.get("delta") or "" for e in ev if e["type"] == "response.output_audio_transcript.delta")
    counts: dict[str, int] = {}
    for e in ev:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    print(f"\n=== {run.name}: {len(ev)} events; agent audio {run.agent_audio_ms:.0f} ms; transcript: {text!r}")
    print("    counts:", json.dumps(counts))
    if speech_end_ms is not None:
        first = next((e for e in ev if e["type"] == "response.output_audio.delta"), None)
        if first:
            print(f"    first audio delta at t={first['t_s']}s, input sent so far={first['input_sent_ms']} ms "
                  f"(user speech ended at {speech_end_ms} ms of input)")


# ----------------------------------------------------------------------------- scenarios
Q1 = "Hi there. Can you tell me, in two or three sentences, why the sky looks blue during the day?"
Q2 = "Sorry to interrupt, but stop. What is two plus three?"


async def scenario_basic(args, out_dir):
    q = tts_16k(Q1, "ryan")
    chunks = chunks_of(silence(1000) + q + silence(args.tail_ms))
    speech_end = 1000 + len(q) * 1000 // (IN_SR * 2)
    async with Run("basic_" + args.mode, args, out_dir) as run:
        t_send0 = time.monotonic()
        await stream(run, chunks, args.mode)
        t_send = time.monotonic() - t_send0
        await drain(run, quiet_s=4, max_s=60)
    print_summary(run, speech_end)
    audio_ev = [e for e in run.events if e["type"] == "response.output_audio.delta"]
    if audio_ev:
        span = audio_ev[-1]["t_s"] - audio_ev[0]["t_s"]
        print(f"    send phase {t_send:.2f}s for {len(chunks)*CHUNK_MS} ms of input; "
              f"agent audio {run.agent_audio_ms:.0f} ms delivered over {span:.2f}s wall "
              f"(first->last delta) => {run.agent_audio_ms/1000/max(span,1e-6):.2f}x realtime")
        # latency: wall time from the moment the input chunk containing end-of-speech was sent
        print(f"    user speech end at input {speech_end} ms; realtime-equivalent wall t={speech_end/1000:.2f}s; "
              f"first audio at t={audio_ev[0]['t_s']}s => latency ~{audio_ev[0]['t_s'] - speech_end/1000:.2f}s")


async def scenario_twoturn(args, out_dir):
    """Q1, long silence, Q2, silence -- checks that a session keeps answering after the first response."""
    q1 = tts_16k(Q1, "ryan")
    q2 = tts_16k("Thanks. And what is two plus three?", "ryan")
    pcm = silence(1000) + q1 + silence(args.gap_ms) + q2 + silence(args.tail_ms)
    async with Run("twoturn_" + args.mode, args, out_dir) as run:
        await stream(run, chunks_of(pcm), args.mode)
        await drain(run, quiet_s=5, max_s=90)
    print_summary(run)
    q2_start = 1000 + len(q1) * 1000 // (IN_SR * 2) + args.gap_ms
    print(f"    Q2 occupies input {q2_start}..{q2_start + len(q2) * 1000 // (IN_SR * 2)} ms")
    for e in run.events:
        if e["type"] in ("response.created", "response.done"):
            print(f"      t={e['t_s']:7.3f} in={e['input_sent_ms']:6d} {e['type']} {e.get('status')} {e.get('status_details')}")
    seq = []
    for e in run.events:
        k = {"response.listen": "L", "response.output_audio.delta": "A", "response.done": "D|"}.get(e["type"])
        if k:
            seq.append(k)
    print("    per-unit decision sequence (L=listen, A=audio unit, D=response.done):", "".join(seq))


async def scenario_starve(args, out_dir):
    """Send Q1 + --starve-after-ms of silence, then send NOTHING for --starve-s wall-clock seconds.
    An input-driven server can only emit output for audio units it has received; anything emitted
    beyond that during the starvation window is server-initiated (wall-clock) generation."""
    q = tts_16k(Q1, "ryan")
    pcm = silence(1000) + q + silence(args.starve_after_ms)
    if args.starve_with_q2:
        pcm += tts_16k(Q2, "ryan")  # user talks over the agent, then input stops
    pre = chunks_of(pcm)
    async with Run("starve" + ("_q2" if args.starve_with_q2 else ""), args, out_dir) as run:
        await stream(run, pre, args.mode)
        t_stop = run.t()
        sent = run.sent_ms
        await asyncio.sleep(args.starve_s)
        during = [e for e in run.events if e["t_s"] >= t_stop]
        a_ms = sum(e.get("audio_ms", 0) for e in during if e["type"] == "response.output_audio.delta")
        print(f"--- stopped sending at t={t_stop}s after {sent} ms of input ({sent // 1000} full 1 s units); "
              f"waited {args.starve_s}s with no input")
        for e in during:
            if e["type"] in ("response.output_audio.delta", "response.listen", "response.created", "response.done"):
                print(f"      t={e['t_s']:7.3f} {e['type']} {e.get('audio_ms', '')} {e.get('status', '')}")
        print(f"    agent audio emitted during the no-input window: {a_ms:.0f} ms")
        await stream(run, chunks_of(silence(args.tail_ms)), args.mode)
        await drain(run, quiet_s=5, max_s=60)
    print_summary(run)


async def scenario_bargein(args, out_dir):
    q1 = tts_16k(Q1, "ryan")
    q2 = tts_16k(Q2, "ryan")
    pre = chunks_of(silence(1000) + q1)
    async with Run("bargein", args, out_dir) as run:
        await stream(run, pre, "realtime")
        # keep sending silence (realtime) until agent has produced >= barge_after_ms of audio
        t_limit = time.monotonic() + 30
        sil = silence(CHUNK_MS)
        while run.agent_audio_ms < args.barge_after_ms and time.monotonic() < t_limit:
            await stream(run, [sil], "realtime")
        barge_t = run.t()
        barge_in_ms = run.sent_ms
        audio_at_barge = run.agent_audio_ms
        print(f"--- barge-in: sending Q2 at t={barge_t}s (input {barge_in_ms} ms), agent audio so far {audio_at_barge:.0f} ms")
        await stream(run, chunks_of(q2), "realtime")
        q2_end_t = run.t()
        await stream(run, chunks_of(silence(args.tail_ms)), "realtime")
        await drain(run, quiet_s=4, max_s=40)
    print_summary(run)
    ev = run.events
    after = [e for e in ev if e["t_s"] >= barge_t]
    print(f"    Q2 sent t={barge_t}..{q2_end_t}s")
    for e in after:
        if e["type"] in ("response.done", "response.created", "response.listen", "overlap.decision",
                         "response.speak", "output_audio_buffer.cleared", "error", "turn.event"):
            print(f"      t={e['t_s']:7.3f} in={e['input_sent_ms']:6d} {e['type']} "
                  f"{ {k: v for k, v in e.items() if k in ('response_id','status','status_details','reason','action')} }")
    # audio of the first response after barge start
    rid1 = next((e.get("response_id") for e in ev if e["type"] == "response.created"), None)
    a1_after = sum(e.get("audio_ms", 0) for e in after if e["type"] == "response.output_audio.delta" and e.get("response_id") == rid1)
    print(f"    first response {rid1}: agent audio generated after barge start = {a1_after:.0f} ms")


async def scenario_pause(args, out_dir):
    q = tts_16k(Q1, "ryan")
    chunks = chunks_of(silence(1000) + q + silence(args.tail_ms))
    pause_idx = [int(x) for x in args.pause_after.split(",")]
    variants = [("fast", "fast", {}), ("fast_pause", "fast", {i: args.pause_s for i in pause_idx})]
    if not args.skip_realtime:
        variants.append(("realtime", "realtime", {}))
    views = {}
    for rep in range(args.repeats):
        for name, mode, pauses in variants:
            rn = f"pause_{name}_r{rep}"
            async with Run(rn, args, out_dir) as run:
                await stream(run, chunks, mode, pauses)
                await drain(run, quiet_s=5, max_s=90)
            print_summary(run)
            views[rn] = {"full": content_view(run.events), "no_listen": content_view(run.events, with_listen=False)}
    names = list(views)
    base = names[0]
    import difflib
    for kind in ("no_listen", "full"):
        for n in names[1:]:
            a_v, b_v = views[base][kind], views[n][kind]
            nosha = lambda v: [x[:4] if x[0] == "response.output_audio.delta" else x for x in v]
            same = a_v == b_v
            same_nosha = nosha(a_v) == nosha(b_v)
            print(f"\n### [{kind}] {n} vs {base}: {'IDENTICAL' if same else 'DIFFERENT'}"
                  f"{'' if same else (' (identical ignoring audio sha1)' if same_nosha else '')}")
            if not same_nosha:
                a = [json.dumps(x, ensure_ascii=False) for x in nosha(a_v)]
                b = [json.dumps(x, ensure_ascii=False) for x in nosha(b_v)]
                for line in list(difflib.unified_diff(a, b, base, n, lineterm="", n=1))[:args.diff_lines]:
                    print("   ", line)
    (out_dir / "pause_views.json").write_text(json.dumps(views, ensure_ascii=False, indent=0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scenario", choices=["basic", "bargein", "pause", "twoturn", "starve"])
    p.add_argument("--starve-after-ms", type=int, default=2000)
    p.add_argument("--starve-s", type=float, default=20.0)
    p.add_argument("--starve-with-q2", action="store_true")
    p.add_argument("--gap-ms", type=int, default=15000, help="twoturn: silence between Q1 and Q2")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--path", default="/v1/realtime?duplex=1")
    p.add_argument("--mode", choices=["realtime", "fast"], default="realtime")
    p.add_argument("--tail-ms", type=int, default=12000, help="silence streamed after the utterance")
    p.add_argument("--barge-after-ms", type=float, default=2000)
    p.add_argument("--pause-after", default="5,12")
    p.add_argument("--pause-s", type=float, default=2.0)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--diff-lines", type=int, default=60)
    p.add_argument("--skip-realtime", action="store_true")
    p.add_argument("--overlap-policy", default="listen_only")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--instructions", default=None)
    p.add_argument("--ref-audio", default=str(Path(os.environ.get("IG_MODELS_DIR", os.environ.get("DIG_MODELS_DIR", ROOT / "models"))) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
    p.add_argument("--out", default=str(ROOT / "logs/duplex_tests"))
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--dump-raw", action="store_true", help="also write full server events (base64 stripped)")
    args = p.parse_args()
    out_dir = Path(args.out)
    fn = {"basic": scenario_basic, "bargein": scenario_bargein, "pause": scenario_pause,
          "twoturn": scenario_twoturn, "starve": scenario_starve}[args.scenario]
    asyncio.run(fn(args, out_dir))


if __name__ == "__main__":
    main()
