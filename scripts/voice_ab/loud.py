"""Independent loudness check (BS.1770 integrated LUFS per turn, pyloudnorm): the leveling targets an active-RMS level,
so the metrics-stage level spread is circular for leveled variants; LUFS is a different measure. Adds per-episode
LUFS spread to metrics/<v>/<ep>.json (key lufs). Turns >= 0.5 s."""
import glob, json, statistics as S, wave
from pathlib import Path
import numpy as np, pyloudnorm as pyln

for f in sorted(glob.glob("metrics/*/*.json")):
    e = json.load(open(f))
    d = Path("synth") / e["variant"] / e["episode_id"]
    vals = []
    for r in e["turns"]:
        with wave.open(str(d / ("%02d.wav" % r["i"]))) as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
        if len(x) / sr >= 0.5:
            l = pyln.Meter(sr, block_size=0.4).integrated_loudness(x)
            if np.isfinite(l):
                r["lufs"] = round(float(l), 2); vals.append(l)
    e["lufs"] = {"std": round(S.pstdev(vals), 3), "range": round(max(vals) - min(vals), 3), "mean": round(S.mean(vals), 2)} if len(vals) >= 2 else None
    json.dump(e, open(f, "w"), indent=1)
print("LOUD DONE")
