"""Add third-party speech recordings (the noise label "speech", ``Behaviors.third_party_per_min``) to a sound bank.

    python scripts/add_speech_events.py --bank DIR --clip-bank CLIPS/manifest.jsonl            # HumDial's real third-party speech
    python scripts/add_speech_events.py --bank DIR --musan musan.tar.gz --musan-n 300             # + read / broadcast speech from MUSAN

Writes ``DIR/events/speech/*.wav`` (mono 16 kHz 16-bit) and ``DIR/events/speech/SOURCES.json``. From a clip bank
(``benchmarks.clip_bank``): its real (not synthetic) ``noise`` clips, material split only (HumDial: people talking to the
caller or near them; Apache-2.0). From MUSAN (CC BY 4.0; ``musan/speech``: LibriVox audiobooks and US government
hearings, public domain): ``--musan-n`` windows of 4-9 s with speech in them, drawn with a fixed seed, streamed from
the archive (nothing else is extracted). The env sets each event's level; clips are stored as recorded.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import tarfile
import wave
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from interaction_gym.audio import Audio  # noqa: E402
from interaction_gym.benchmarks.clip_bank import speech_events  # noqa: E402
from interaction_gym.soundscape import active_rms  # noqa: E402


def musan_windows(tar_path: str, out: Path, n: int, seed: int = 0, per_file: int = 2) -> list[dict]:
    rng = random.Random(seed)
    recs = []
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not (m.isfile() and "/speech/" in m.name and m.name.endswith(".wav")):
                continue
            if rng.random() > 0.35:  # a spread over the corpus, not its first files
                continue
            with wave.open(io.BytesIO(tf.extractfile(m).read())) as w:
                sr, ch, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
            pcm = array("h")
            pcm.frombytes(raw)
            a = Audio(pcm[::ch] if ch > 1 else pcm, sr)
            if a.sr != 16000:
                a = a.resample(16000)
            for _ in range(per_file):
                dur = rng.randint(4000, 9000) * 16
                if len(a.samples) <= dur:
                    break
                i0 = rng.randrange(len(a.samples) - dur)
                clip = Audio(a.samples[i0:i0 + dur], 16000)
                if active_rms(clip.samples, 16000) < 300:  # (near) silence
                    continue
                name = f"musan-{Path(m.name).stem}-{i0 // 16}.wav"
                clip.write_wav(out / name)
                recs.append({"file": name, "source": m.name, "start_ms": i0 // 16, "dataset": "MUSAN speech", "license": "CC BY 4.0"})
            if len(recs) >= n:
                break
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--clip-bank")
    ap.add_argument("--musan")
    ap.add_argument("--musan-n", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.bank) / "events" / "speech"
    out.mkdir(parents=True, exist_ok=True)
    src = {}
    if a.clip_bank:
        src["clip_bank"] = {"manifest": a.clip_bank, "n": speech_events(a.clip_bank, a.bank), "license": "per clip (HumDial: Apache-2.0)"}
    if a.musan:
        recs = musan_windows(a.musan, out, a.musan_n)
        src["musan"] = {"archive": a.musan, "n": len(recs), "clips": recs}
    (out / "SOURCES.json").write_text(json.dumps(src, indent=1))
    print({k: v["n"] for k, v in src.items()}, "->", out)


if __name__ == "__main__":
    main()
