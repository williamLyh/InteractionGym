"""Voice A/B, judge stage: a local audio LLM (Qwen3-Omni-30B-A3B-Instruct on vLLM) listens to each episode's user turns
(concat.wav: the caller's turns in order, 500 ms gaps) and (1) scores each variant on its own, (2) compares pairs of
variants on the same episode, both presentation orders (position bias is measured, and cancels out).
Resumable: results are appended to <out>/absolute.jsonl and <out>/pairwise.jsonl; done keys are skipped."""
import argparse, asyncio, base64, io, itertools, json, random, re, wave
from pathlib import Path
import urllib.request

import numpy as np
import librosa

SYSTEM = ("You are an expert listener who rates synthetic speech for a voice-AI benchmark. You listen carefully and "
          "judge only how the speech sounds, never the content of what is said.")
CONTEXT = ("{what} everything one simulated caller says during a single phone call to a voice assistant: the caller's "
           "turns in order, joined with short gaps. Very short turns (\"mm-hmm\", \"okay\", \"right\") are backchannels said "
           "while listening. The words are fixed in advance; judge only HOW they are spoken. A good simulated caller "
           "sounds like ONE real person on an ordinary phone call: the same voice throughout, a steady natural pace (no turn "
           "suddenly much slower or faster than the others), steady loudness, and an everyday manner (no turn suddenly "
           "over-emotional, theatrical or acted). Persona differences (some callers are brisk, some calm) are fine as long "
           "as they are consistent and natural.")
ABS = CONTEXT + """

Rate each from 1 (very bad) to 10 (perfect):
- naturalness: sounds like a real person on a phone call, not a TTS demo or an actor
- pace_consistency: speaking rate is consistent across turns
- emotion_consistency: emotional tone is consistent; no sudden over-emotional turn
- loudness_consistency: loudness is consistent across turns
- voice_consistency: clearly the same speaker throughout
- not_overacted: 10 = no over-acting at all
- overall: how good this is as a simulated caller

Reply with JSON only: {"naturalness": n, "pace_consistency": n, "emotion_consistency": n, "loudness_consistency": n, "voice_consistency": n, "not_overacted": n, "overall": n, "reason": "<2-3 sentences, naming the specific turns or moments you mean>"}"""
PAIR = CONTEXT.replace("{what}", "Clip A and Clip B each contain") + """
Both clips have the same words and the same intended caller; they differ only in how the speech was synthesized.

Which clip is the better simulated caller: more natural, and more consistent in voice, pace, emotion and loudness across turns, with less over-acting? Listen to both fully before deciding; the order in which they are played says nothing about quality.

Reply with JSON only: {"winner": "A" | "B" | "tie", "reason": "<2-3 sentences comparing the two, naming specific turns or moments>"}"""

CLONE = ["C0", "V3", "V4", "V5"]
PAIRS = list(itertools.combinations(CLONE, 2)) + [(c, "V0") for c in CLONE] + [("V1", "V0"), ("V2", "V0"), ("V1", "V2"), ("V2", "V4"), ("V1", "V3")]


def b64wav(path: Path) -> str:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    y = (np.clip(librosa.resample(x, orig_sr=sr, target_sr=16000), -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(y.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else None


async def ask(a, content):
    payload = {"model": a.model, "temperature": 0.0, "max_tokens": 400,
               "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    for attempt in range(6):
        try:
            r = await asyncio.to_thread(post, a.url + "/chat/completions", payload)
            text = r["choices"][0]["message"]["content"] or ""
            js = parse(text)
            if js is not None:
                return js, text
        except Exception as e:
            text = repr(e)
        await asyncio.sleep(10 * (attempt + 1))
    return None, text


def done_keys(f):
    return {json.loads(l)["key"] for l in open(f)} if f.exists() else set()


async def main(a):
    syn, out = Path(a.synth), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = json.loads((syn / "episodes.json").read_text())
    cache = {}

    def clip(v, e):
        if (v, e) not in cache:
            cache[(v, e)] = b64wav(syn / v / e / "concat.wav")
        return cache[(v, e)]

    sem = asyncio.Semaphore(a.concurrency)
    lock = asyncio.Lock()
    fa, fp = out / "absolute.jsonl", out / "pairwise.jsonl"
    da, dp = done_keys(fa), done_keys(fp)
    variants = sorted({v for p in PAIRS for v in p})

    async def absolute(v, e):
        key = f"{v}|{e}"
        if key in da:
            return
        async with sem:
            content = [{"type": "audio_url", "audio_url": {"url": clip(v, e)}},
                       {"type": "text", "text": ABS.replace("{what}", "This clip contains")}]
            js, raw = await ask(a, content)
        async with lock:
            with open(fa, "a") as f:
                f.write(json.dumps({"key": key, "variant": v, "episode": e, "scores": js, "raw": raw}) + "\n")

    async def pairwise(x, y, e, order):
        key = f"{x}|{y}|{e}|{order}"
        if key in dp:
            return
        first, second = (x, y) if order == 0 else (y, x)
        async with sem:
            content = [{"type": "text", "text": "Clip A:"}, {"type": "audio_url", "audio_url": {"url": clip(first, e)}},
                       {"type": "text", "text": "Clip B:"}, {"type": "audio_url", "audio_url": {"url": clip(second, e)}},
                       {"type": "text", "text": PAIR}]
            js, raw = await ask(a, content)
        w = (js or {}).get("winner")
        winner = first if w == "A" else second if w == "B" else "tie" if w == "tie" else None
        async with lock:
            with open(fp, "a") as f:
                f.write(json.dumps({"key": key, "pair": [x, y], "episode": e, "A": first, "B": second, "verdict": w,
                                    "winner": winner, "reason": (js or {}).get("reason"), "raw": raw}) + "\n")

    rng = random.Random(0)
    jobs = [absolute(v, e) for v in variants for e in eps]
    pj = [pairwise(x, y, e, o) for x, y in PAIRS for e in eps for o in (0, 1)]
    rng.shuffle(pj)  # interleave pairs and orders
    await asyncio.gather(*jobs, *pj)
    print("JUDGE DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="voice_ab_work/synth")
    ap.add_argument("--out", default="voice_ab_work/judge")
    ap.add_argument("--url", default="http://127.0.0.1:8230/v1")
    ap.add_argument("--model", default="Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--concurrency", type=int, default=48)
    asyncio.run(main(ap.parse_args()))
