"""Voice A/B, latency stage: per-turn synthesis latency of CustomVoice vs Base-clone, alone (concurrency 1, one
replica) and under load (concurrency 16 through the proxies), on the same texts. Writes <out>/latency.json."""
import argparse, asyncio, json, statistics, time
from pathlib import Path

from interaction_gym.audio import Audio
from interaction_gym.clients import OpenAISpeech

SR = 24000


async def timed(coro):
    t0 = time.perf_counter()
    au = await coro
    return time.perf_counter() - t0, au.dur_ms / 1000


def summ(xs):
    lat = [x[0] for x in xs]
    rtf = [x[0] / max(0.2, x[1]) for x in xs]
    return {"n": len(xs), "mean_s": round(statistics.mean(lat), 3), "p50_s": round(statistics.median(lat), 3),
            "p90_s": round(sorted(lat)[int(0.9 * len(lat)) - 1], 3), "rtf_mean": round(statistics.mean(rtf), 3)}


async def main(a):
    syn = Path(a.synth)
    metas = [json.loads(p.read_text()) for p in sorted((syn / "V4").glob("*/meta.json"))][:12]
    items = []  # (ref audio, ref text, speaker, text)
    for m in metas:
        d = syn / "V4" / m["episode_id"]
        ref = next(r for r in m["turns"] if r["kind"] is None and len(r["text"].split()) >= 3)
        for r in m["turns"]:
            if r["kind"] is None and r["i"] != ref["i"]:
                items.append((Audio.read_wav(d / f"{ref['i']:02d}.wav"), ref["text"], m["speaker"], r["text"]))
    items = items[: a.n]
    out = {}
    for label, cv, cl in (("alone", a.tts1, a.clone1), ("load16", a.tts, a.clone)):
        tts = OpenAISpeech(cv, "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", sr=SR)
        clone = OpenAISpeech(cl, "Qwen/Qwen3-TTS-12Hz-1.7B-Base", sr=SR)
        conc = 1 if label == "alone" else 16
        for kind in ("customvoice", "clone"):
            sem = asyncio.Semaphore(conc)

            async def one(it):
                async with sem:
                    if kind == "clone":
                        return await timed(clone.synth(it[3], it[2], ref_audio=it[0], ref_text=it[1], language="English"))
                    return await timed(tts.synth(it[3], it[2], "Natural, relaxed everyday phone-call voice.", language="English"))
            await one(items[0])  # warm
            t0 = time.perf_counter()
            res = await asyncio.gather(*(one(it) for it in items))
            wall = time.perf_counter() - t0
            out[f"{kind}_{label}"] = summ(res) | {"wall_s": round(wall, 1), "turns_per_s": round(len(res) / wall, 2),
                                                 "audio_s_per_s": round(sum(r[1] for r in res) / wall, 2)}
            print(kind, label, out[f"{kind}_{label}"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="voice_ab_work/synth")
    ap.add_argument("--out", default="voice_ab_work/latency.json")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--tts1", default="http://127.0.0.1:8201/v1")
    ap.add_argument("--clone1", default="http://127.0.0.1:8211/v1")
    ap.add_argument("--tts", default="http://127.0.0.1:8200/v1")
    ap.add_argument("--clone", default="http://127.0.0.1:8210/v1")
    asyncio.run(main(ap.parse_args()))
