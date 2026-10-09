#!/usr/bin/env python3
"""Check the opt-in per-unit token trace (session.extra_body.trace_tokens) on the MiniCPM-o 4.5 server.

One input-clocked session (clock="input" + trace_tokens=true): a TTS question then silence, in 200 ms
appends sent one ack at a time. Dumps every debug.unit_tokens event to logs/token_trace/trace.json,
prints a compact layout (runs collapsed, e.g. <unk>x10) of the prompt unit, the first units and the
first speak unit, and checks:
  - one unit event per unit acknowledged (sum of processed.units), plus one prompt unit (-1)
  - unit_index 0..n-1 in order, end_ms equal to the ack's units end_ms (multiples of 1000 here)
  - decisions identical to the lockstep acks', and every unit event precedes the ack covering it

Run next to the server (any Python with the script's imports):  python scripts/token_trace_check.py
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lockstep_check import (  # noqa: E402
    IN_SR, LockstepSession, Q1, ROOT, chunks, session_payload, silence_chunks, tts_16k,
)


def collapse(pairs: list[list], width: int = 160) -> str:
    """'<unit> <unk>x10 </unit>' style: consecutive identical tokens are collapsed into one 'piecexN'."""
    out, prev, n = [], None, 0
    for tok_id, text in pairs:
        key = (tok_id, text)
        if key == prev:
            n += 1
            continue
        if prev is not None:
            out.append(f"{prev[1]}" + (f"x{n}" if n > 1 else ""))
        prev, n = key, 1
    if prev is not None:
        out.append(f"{prev[1]}" + (f"x{n}" if n > 1 else ""))
    s = " ".join(out)
    return s


def show(ev: dict) -> None:
    print(f"  unit {ev['unit_index']:>3}  end_ms={ev['end_ms']:<6} decision={ev['decision']}  special_ids={ev['special_ids']}")
    for st in ev["stages"]:
        for key in ("input", "output"):
            if key in st:
                print(f"    {st['stage']:>7}.{key:<6} ({len(st[key]):>4}) {collapse(st[key])}")


async def main(a) -> bool:
    q = tts_16k(a.tts_url, a.tts_model, Q1[0], a.voice)
    payload = session_payload(a.ref_audio, lockstep=not a.no_lockstep)
    payload["extra_body"]["trace_tokens"] = True
    if a.no_lockstep:
        return await run_realtime(a, payload, q)
    ws = await websockets.connect(a.url, max_size=None)
    await ws.send(json.dumps({"type": "session.update", "session": payload}))
    s = LockstepSession(ws)
    rx = asyncio.create_task(s.reader())
    try:
        for pcm in chunks(q) + silence_chunks(a.tail_s):
            await s.append(pcm)
    finally:
        await ws.send(json.dumps({"type": "session.close"}))
        await asyncio.sleep(0.3)
        rx.cancel()
        await ws.close()
    traces = [e for e in s.events if e.get("type") == "debug.unit_tokens"]
    out_dir = ROOT / "logs" / "token_trace"
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in e.items() if k != "_t"} for e in traces]
    (out_dir / "trace.json").write_text(json.dumps(clean, indent=1, ensure_ascii=False))

    # Lockstep view: units from the acks, and the position of each trace relative to the acks.
    ack_units, unit_ack_pos, trace_pos = [], [], {}
    n_acks = 0
    for e in s.events:
        if e.get("type") == "input_audio_buffer.processed":
            for u in e.get("units") or []:
                ack_units.append(u)
                unit_ack_pos.append(n_acks)
            n_acks += 1
        elif e.get("type") == "debug.unit_tokens":
            trace_pos[e["unit_index"]] = n_acks
    units = [e for e in traces if e["unit_index"] >= 0]
    prompts = [e for e in traces if e["unit_index"] == -1]

    print(f"session: {s.sent} appends, {len(s.acks)} acks, {len(ack_units)} units acknowledged, "
          f"{len(units)} unit traces + {len(prompts)} prompt trace  -> {out_dir / 'trace.json'}")
    if prompts:
        print("\n--- prompt (unit -1)")
        show(prompts[0])
    print("\n--- first units")
    for ev in units[: a.first]:
        show(ev)
    speak = next((ev for ev in units if ev["decision"] == "speak"), None)
    if speak is not None and speak["unit_index"] >= a.first:
        print("\n--- first speak unit")
        show(speak)
    speaks = [ev for ev in units if ev["decision"] == "speak"]
    if len(speaks) > 1 and speaks[-1]["unit_index"] >= a.first:
        print("\n--- last speak unit (turn end)")
        show(speaks[-1])

    checks = {
        "one prompt trace": len(prompts) == 1,
        "unit traces == acknowledged units": len(units) == len(ack_units),
        "unit_index 0..n-1 in order": [e["unit_index"] for e in units] == list(range(len(units))),
        "end_ms == ack end_ms": [e["end_ms"] for e in units] == [u["end_ms"] for u in ack_units],
        "end_ms multiples of 1000": all(e["end_ms"] % 1000 == 0 for e in units),
        "decisions == ack decisions": [e["decision"] for e in units] == [u["decision"] for u in ack_units],
        "each trace precedes its ack": all(trace_pos.get(i, 1 << 30) <= unit_ack_pos[i] for i in range(len(ack_units))),
        "speak units carry a talker stage": all(
            any(st["stage"] == "talker" for st in e["stages"]) for e in units if e["decision"] == "speak"
        ),
    }
    print()
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return all(checks.values())


async def run_realtime(a, payload: dict, q) -> bool:
    """Trace without clock=input: real-time pacing, just count unit traces vs. a plausible unit count."""
    import time

    ws = await websockets.connect(a.url, max_size=None)
    await ws.send(json.dumps({"type": "session.update", "session": payload}))
    events: list[dict] = []

    async def reader():
        async for raw in ws:
            events.append(json.loads(raw))

    rx = asyncio.create_task(reader())
    plan = chunks(q) + silence_chunks(a.tail_s)
    t0 = time.monotonic()
    for i, pcm in enumerate(plan):
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode(), "format": "pcm16", "sample_rate_hz": IN_SR}))
        await asyncio.sleep(max(0.0, t0 + (i + 1) * 0.2 - time.monotonic()))
    await asyncio.sleep(2.0)
    await ws.send(json.dumps({"type": "session.close"}))
    await asyncio.sleep(0.3)
    rx.cancel()
    await ws.close()
    traces = [e for e in events if e.get("type") == "debug.unit_tokens"]
    acks = sum(1 for e in events if e.get("type") == "input_audio_buffer.processed")
    decisions = "".join("S" if e["decision"] == "speak" else "." if e["decision"] == "listen" else "P" for e in traces)
    print(f"no-lockstep: {len(plan)} appends ({len(plan) * 0.2:.1f} s), {acks} acks, {len(traces)} traces: {decisions}")
    ok = acks == 0 and len(traces) >= int(len(plan) * 0.2) - 1 and traces[0]["unit_index"] == -1
    print(f"[{'PASS' if ok else 'FAIL'}] traces without clock=input (no acks; ~one trace per second of input + prompt)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8010/v1/realtime?duplex=1")
    ap.add_argument("--ref-audio", default=str(Path(os.environ.get("IG_MODELS_DIR", os.environ.get("DIG_MODELS_DIR", ROOT / "models"))) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
    ap.add_argument("--tts-url", default="http://127.0.0.1:8001/v1/audio/speech")
    ap.add_argument("--tts-model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--voice", default="ryan")
    ap.add_argument("--tail-s", type=float, default=14.0)
    ap.add_argument("--first", type=int, default=6)
    ap.add_argument("--no-lockstep", action="store_true", help="trace a real-time paced session without clock=input (prototype builds only: the patched server refuses it)")
    sys.exit(0 if asyncio.run(main(ap.parse_args())) else 1)
