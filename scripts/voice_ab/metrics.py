"""Voice A/B, objective metrics per variant x episode (resumable: one JSON per variant/episode in <out>/metrics/).

Per turn: active speech rate (words / speech span, normal turns with >= 3 words), active-speech level, peak,
trailing silence, ASR transcript (Qwen3-ASR via vLLM) for WER, DNSMOS OVRL/SIG (MOS predictor), ECAPA speaker embedding.
Per episode: spreads of rate / level, mean pairwise speaker similarity, WER, mean MOS, latency."""
import argparse, asyncio, itertools, json, math, re, statistics
from pathlib import Path

import numpy as np
import torch
import jiwer

from interaction_gym.audio import Audio, active_level, speech_span
from interaction_gym.clients import OpenAITranscribe

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def norm(s):
    s = s.lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def to16(au: Audio) -> torch.Tensor:
    x = np.frombuffer(au.samples.tobytes(), dtype=np.int16).astype(np.float32) / 32768
    import librosa
    return torch.from_numpy(librosa.resample(x, orig_sr=au.sr, target_sr=16000))


class Models:
    def __init__(self, hub):
        from speechbrain.inference.speaker import EncoderClassifier
        self.ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=f"{hub}/ecapa",
                                                    run_opts={"device": DEV})
        import onnxruntime as ort  # DNSMOS P.835 (microsoft/DNS-Challenge sig_bak_ovr.onnx); UTMOS weights were unreachable
        self.dnsmos = ort.InferenceSession(f"{hub}/sig_bak_ovr.onnx", providers=["CPUExecutionProvider"])

    @torch.no_grad()
    def emb(self, x):
        return self.ecapa.encode_batch(x[None].to(DEV)).squeeze().cpu().numpy()

    def mos(self, x):
        """DNSMOS (OVRL, SIG): the clip repeated to the model's 9.01 s window, non-personalized polynomial fit."""
        a = x.numpy().astype(np.float32)
        n = int(9.01 * 16000)
        while len(a) < n:
            a = np.concatenate([a, a])
        a = a[:n]
        sig, bak, ovr = self.dnsmos.run(None, {"input_1": a[None]})[0][0]
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535]); p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        return float(p_ovr(ovr)), float(p_sig(sig))


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def spread(xs):
    return {"std": round(statistics.pstdev(xs), 3), "range": round(max(xs) - min(xs), 3), "mean": round(statistics.mean(xs), 3),
            "cv": round(statistics.pstdev(xs) / statistics.mean(xs), 3) if statistics.mean(xs) else None} if len(xs) >= 2 else None


async def one(d: Path, models, asr, out: Path):
    meta = json.loads((d / "meta.json").read_text())
    f = out / meta["variant"] / f"{meta['episode_id']}.json"
    if f.exists():
        return
    rows = []
    texts = []
    for r in meta["turns"]:
        au = Audio.read_wav(d / f"{r['i']:02d}.wav")
        span = speech_span(au)
        lv = active_level(au)
        peak = max((abs(x) for x in au.samples), default=0)
        row = {"i": r["i"], "kind": r["kind"], "words": len(r["text"].split()), "dur_s": au.dur_ms / 1000,
               "level_db": None if lv is None else round(lv, 2),
               "peak_db": round(20 * math.log10(peak / 32768), 2) if peak else None,
               "lead_s": None if span is None else round(span[0] / au.sr, 3),
               "trail_s": None if span is None else round((len(au) - span[1]) / au.sr, 3),
               "latency_s": r["latency_s"], "cloned": r["cloned"]}
        speech_s = (span[1] - span[0]) / au.sr if span else 0
        if r["kind"] is None and row["words"] >= 3 and speech_s > 0.3:
            row["rate_wps"] = round(row["words"] / speech_s, 3)
        if r["kind"] is None:
            x = to16(au)
            if len(x) > 8000:
                o, sg = models.mos(x)
                row["mos"], row["mos_sig"] = round(o, 3), round(sg, 3)
                row["_emb"] = models.emb(x)
            texts.append((row, r["text"], au))
        rows.append(row)
    hyps = await asyncio.gather(*(asr.transcribe(au, "en") for _, _, au in texts), return_exceptions=True)
    refs, hs = [], []
    for (row, ref, _), h in zip(texts, hyps):
        if isinstance(h, Exception):
            raise h
        h = re.sub(r"^.*<asr_text>", "", h)
        row["asr"] = h
        refs.append(norm(ref)); hs.append(norm(h))
    pairs = [(a, b) for a, b in zip(refs, hs) if a]
    wer = jiwer.wer([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
    embs = [r.pop("_emb") for r in rows if "_emb" in r]
    sims = [cos(a, b) for a, b in itertools.combinations(embs, 2)]
    first = [cos(embs[0], e) for e in embs[1:]]
    rates = [r["rate_wps"] for r in rows if "rate_wps" in r]
    lv_norm = [r["level_db"] for r in rows if r["kind"] is None and r["level_db"] is not None]
    lv_all = [r["level_db"] for r in rows if r["level_db"] is not None]
    peaks = [r["peak_db"] for r in rows if r["peak_db"] is not None]
    lat = [r["latency_s"] for r in rows]
    lat_clone = [r["latency_s"] for r in rows if r["cloned"]]
    ep = {"episode_id": meta["episode_id"], "variant": meta["variant"], "speaker": meta["speaker"],
          "style": (meta.get("profile") or {}).get("speaking_style"),
          "rate": spread(rates), "level_normal": spread(lv_norm), "level_all": spread(lv_all), "peak": spread(peaks),
          "trail_s_mean": round(statistics.mean(r["trail_s"] for r in rows if r["trail_s"] is not None), 3),
          "spk_sim_pairwise": round(statistics.mean(sims), 4) if sims else None,
          "spk_sim_min": round(min(sims), 4) if sims else None,
          "spk_sim_to_first": round(statistics.mean(first), 4) if first else None,
          "wer": None if wer is None else round(wer, 4), "n_words": sum(len(a.split()) for a, _ in pairs),
          "mos": round(statistics.mean(r["mos"] for r in rows if "mos" in r), 3) if any("mos" in r for r in rows) else None,
          "mos_sig": round(statistics.mean(r["mos_sig"] for r in rows if "mos" in r), 3) if any("mos" in r for r in rows) else None,
          "latency_mean_s": round(statistics.mean(lat), 3), "latency_cloned_mean_s": round(statistics.mean(lat_clone), 3) if lat_clone else None,
          "turns": rows}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(ep, indent=1))
    print("metrics", meta["variant"], meta["episode_id"], flush=True)


async def main(a):
    models = Models(a.hub)
    asr = OpenAITranscribe(a.asr, a.asr_model)
    for d in sorted(Path(a.synth).glob("*/*/meta.json")):
        await one(d.parent, models, asr, Path(a.out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", default="voice_ab_work/synth")
    ap.add_argument("--out", default="voice_ab_work/metrics")
    ap.add_argument("--hub", default="voice_ab_work/hub")
    ap.add_argument("--asr", default="http://127.0.0.1:8220/v1")
    ap.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    asyncio.run(main(ap.parse_args()))
