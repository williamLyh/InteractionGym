"""Voice A/B, report: judge pairwise wins (both orders), absolute scores, objective metrics, latency -> report.md + report.json."""
import argparse, collections, json, statistics
from pathlib import Path

DESC = {"V0": "current: CustomVoice + persona style, no post, no clone", "V1": "V0 + trim/loudness", "V2": "V1 + neutral instruction",
        "C0": "clone (persona-style ref), no post", "V3": "V1 + clone (persona-style ref)", "V4": "V1 + clone (neutral ref)",
        "V5": "V4 + one TTS seed per episode"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def main(a):
    root = Path(a.root)
    pw = [json.loads(l) for l in open(root / "judge/pairwise.jsonl")]
    ab = [json.loads(l) for l in open(root / "judge/absolute.jsonl")]
    variants = sorted({r["variant"] for r in ab}, key=lambda v: list(DESC).index(v))
    # pairwise
    wins, games = collections.Counter(), collections.Counter()
    pair_tab = collections.defaultdict(lambda: [0, 0, 0, 0])  # x wins, y wins, ties, failed
    pos = collections.Counter()
    consistent = collections.Counter()
    by_ep = collections.defaultdict(dict)
    for r in pw:
        x, y = r["pair"]
        t = pair_tab[(x, y)]
        if r["winner"] is None:
            t[3] += 1; continue
        games[x] += 1; games[y] += 1
        if r["winner"] == "tie":
            t[2] += 1; wins[x] += 0.5; wins[y] += 0.5
        else:
            wins[r["winner"]] += 1; t[0 if r["winner"] == x else 1] += 1
        pos[r["verdict"]] += 1
        by_ep[(x, y, r["episode"])][r["A"]] = r["winner"]
    for k, d in by_ep.items():
        if len(d) == 2:
            consistent["both"] += 1
            consistent["same" if len(set(d.values())) == 1 else "flip"] += 1
    # absolute
    keys = ["naturalness", "pace_consistency", "emotion_consistency", "loudness_consistency", "voice_consistency", "not_overacted", "overall"]
    absd = {v: {k: mean([(r["scores"] or {}).get(k) for r in ab if r["variant"] == v and isinstance((r["scores"] or {}).get(k), (int, float))]) for k in keys} for v in variants}
    # metrics
    met = {}
    for v in variants:
        eps = [json.loads(p.read_text()) for p in sorted((root / "metrics" / v).glob("*.json"))]
        tot_w = sum(e["n_words"] for e in eps)
        met[v] = {
            "rate_cv": mean([(e["rate"] or {}).get("cv") for e in eps]),
            "rate_range_wps": mean([(e["rate"] or {}).get("range") for e in eps]),
            "rate_mean_wps": mean([(e["rate"] or {}).get("mean") for e in eps]),
            "level_std_db": mean([(e["level_all"] or {}).get("std") for e in eps]),
            "level_range_db": mean([(e["level_all"] or {}).get("range") for e in eps]),
            "lufs_std": mean([(e.get("lufs") or {}).get("std") for e in eps]),
            "lufs_range": mean([(e.get("lufs") or {}).get("range") for e in eps]),
            "peak_range_db": mean([(e["peak"] or {}).get("range") for e in eps]),
            "trail_s": mean([e["trail_s_mean"] for e in eps]),
            "spk_sim": mean([e["spk_sim_pairwise"] for e in eps]),
            "spk_sim_min": mean([e["spk_sim_min"] for e in eps]),
            "wer": round(sum((e["wer"] or 0) * e["n_words"] for e in eps) / tot_w, 4) if tot_w else None,
            "mos": mean([e["mos"] for e in eps]), "mos_sig": mean([e.get("mos_sig") for e in eps]),
            "latency_turn_s": mean([e["latency_mean_s"] for e in eps]),
            "n_eps": len(eps)}
    lat = json.loads((root / "latency.json").read_text()) if (root / "latency.json").exists() else {}
    rank = sorted(variants, key=lambda v: -(wins[v] / games[v] if games[v] else 0))
    rep = {"desc": DESC, "wins": {v: [wins[v], games[v], round(wins[v] / games[v], 3) if games[v] else None] for v in variants},
           "pairs": {f"{x} vs {y}": t for (x, y), t in pair_tab.items()}, "position": dict(pos),
           "order_consistency": dict(consistent), "absolute": absd, "metrics": met, "latency": lat, "rank": rank}
    (root / "report.json").write_text(json.dumps(rep, indent=1))
    L = ["# Voice A/B report", "", f"Episodes: {met[variants[0]]['n_eps']}; pairwise judgments: {len(pw)}; absolute: {len(ab)}", "",
         "| variant | what | pairwise win rate (wins/games) | judge overall | pace | emotion | loudness | voice | not overacted | rate CV | rate range w/s | LUFS std | LUFS range | active-RMS std dB | trail s | spk sim | WER | DNSMOS ovrl/sig | latency/turn s (load) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in rank:
        w, g, r = rep["wins"][v]; s = absd[v]; m = met[v]
        L.append(f"| {v} | {DESC[v]} | {r} ({w}/{g}) | {s['overall']} | {s['pace_consistency']} | {s['emotion_consistency']} | {s['loudness_consistency']} | "
                 f"{s['voice_consistency']} | {s['not_overacted']} | {m['rate_cv']} | {m['rate_range_wps']} | {m['lufs_std']} | {m['lufs_range']} | {m['level_std_db']} | "
                 f"{m['trail_s']} | {m['spk_sim']} | {m['wer']} | {m['mos']}/{m['mos_sig']} | {m['latency_turn_s']} |")
    L += ["", "## Pairs (first wins / second wins / ties / failed)", ""] + [f"- {k}: {t}" for k, t in rep["pairs"].items()]
    L += ["", f"Position: {dict(pos)}; order consistency over episode-pairs: {dict(consistent)}", "", "## Latency", "", "```", json.dumps(lat, indent=1), "```"]
    (root / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="voice_ab_work")
    main(ap.parse_args())
