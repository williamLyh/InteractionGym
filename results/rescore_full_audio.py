"""Re-score a finished run's eval.scores with the current latency curve, next to the curve of its time.

    PYTHONPATH=src:. python results/rescore_full_audio.py --runs <runs dir> --out <file.json>

For every ``episodes.jsonl`` under ``--runs`` (one per benchmark / agent / condition or subset), the mean per-turn
score of ``eval.scores`` recomputed from the saved turns twice: with ``eval.latency_score`` (logistic, mid 950 ms,
width 100 ms) and with the log-normal curve used until 2026-10-09 (1 at 200 ms, sigma 1 in log time; faster than
200 ms full for a stop). Only latency-graded outcomes change; the outcome rates and every official metric do not.
Used for the full-audio appendix (results/BENCHMARKS_full_audio.md).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from interaction_gym import eval as ev


def old_curve(ms: float, best_ms: float = 200.0, sigma: float = 1.0, faster_is_fine: bool = False) -> float:
    if ms <= 0:
        return 1.0 if faster_is_fine else 0.0
    if faster_is_fine and ms <= best_ms:
        return 1.0
    return math.exp(-(math.log(ms / best_ms) ** 2) / (2 * sigma**2))


def rescore(events: list[dict]) -> tuple[list[float], list[float]]:
    new, old = [], []
    for e in events:
        if e.get("score") is None:
            continue
        new.append(e["score"])
        if e.get("outcome") == "responded":
            old.append(old_curve(e["latency_ms"]))
        elif e.get("outcome") == "yielded" and e.get("expects") == "yield":
            old.append(old_curve(e["latency_ms"], faster_is_fine=True))
        else:
            old.append(e["score"])
    return new, old


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, res = Path(a.runs), {}
    for d, _, fs in sorted(os.walk(root, followlinks=True)):
        if "episodes.jsonl" not in fs or "/media" in d:
            continue
        new, old, n_eps = [], [], 0
        for line in (Path(d) / "episodes.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            ep = json.loads(line)
            if (ep.get("eval", {}).get("benchmark") or {}).get("clean") or ep["meta"]["episode_id"].endswith("-clean"):
                continue  # FD-Bench v1.5 clean inputs (judge only)
            n_eps += 1
            x, y = rescore(ev.scores(ep["turns"], end_ms=ep["meta"]["duration_ms"])["events"])
            new += x
            old += y
        if new:
            res[str(Path(d).relative_to(root))] = {"episodes": n_eps, "scored_turns": len(new), "mean_new": round(sum(new) / len(new), 4),
                                                   "mean_old": round(sum(old) / len(old), 4)}
            print(Path(d).relative_to(root), res[str(Path(d).relative_to(root))], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
