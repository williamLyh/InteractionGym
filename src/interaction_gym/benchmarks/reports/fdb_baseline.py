"""Full-Duplex-Bench v1.0 / v1.5, Easy Turn and HumDial-FDBench report: the official per-subset metrics
(``full_duplex_bench.official_metrics``), the rule scores' summary and outcomes, optional LLM judges (v1.0
interruption ratings, v1.5 behaviour tags; verdicts cached in ``per_sample.json``), ``summary.json`` and a browsable
run. ``report(args, judge)``: ``args`` has ``data``, ``bench``, ``out``, ``repo``, ``gt``, ``judge_model``, ``rejudge``,
``pages``; ``judge`` is a chat client or None."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .. import easy_turn as et
from .. import full_duplex_bench as fdb
from .. import humdial as hd
from ...eval import duplex, scores
from ...media import MediaStore
from ...traj import load
from ...viewer import export_run


BENCHES = {"fdb": ("Full-Duplex-Bench", fdb.SUBSETS), "easy_turn": ("Easy Turn", et.SUBSETS), "humdial": ("HumDial-FDBench", hd.SUBSETS)}

def _subsets(args) -> dict:
    return BENCHES[args.bench][1]

def _is_fdb(subset: str) -> bool:
    return subset in fdb.SUBSETS

def _eid(subset: str, sid: str, clean: bool) -> str:
    if subset in et.SUBSETS:
        return f"et-{et.SUBSETS[subset]}-{sid}"
    lang_cat = "-".join(hd.SUBSETS[subset]) if subset in hd.SUBSETS else None
    base = f"fdb-{subset.replace('/', '-')}-{sid}" if lang_cat is None else f"hd-{lang_cat}-{sid}"
    return base + ("-clean" if clean else "")

async def report(args, judge=None) -> None:
    out = Path(args.out)
    media = MediaStore(out)
    gt = json.loads(Path(args.gt).read_text()) if args.gt and Path(args.gt).exists() else None
    summary, pages = {}, []
    summary_prev = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for subset in _subsets(args):
        path = out / subset / "episodes.jsonl"
        if not path.exists():
            continue
        eps = load(path)
        for e in eps:  # re-score: episodes recorded before eval.scores knew the episode end (censoring)
            e["eval"]["duplex"] = duplex(e["turns"], end_ms=e["meta"]["duration_ms"])
            e["eval"]["scores"] = scores(e["turns"], end_ms=e["meta"]["duration_ms"])
        noisy = [e for e in eps if not fdb.bench_meta(e).get("clean")]
        clean = {fdb.bench_meta(e)["sample_id"]: e for e in eps if fdb.bench_meta(e).get("clean")}
        ratings = behaviours = None
        cached = json.loads((out / subset / "per_sample.json").read_text()) if (out / subset / "per_sample.json").exists() and not args.rejudge else {}
        if judge is None and any("rating" in v for v in cached.values()):  # reuse earlier judge verdicts
            ratings = {k: v["rating"] for k, v in cached.items() if v.get("rating") is not None}
        if judge is None and any("behaviour" in v for v in cached.values()):
            behaviours = {k: v["behaviour"] for k, v in cached.items() if v.get("behaviour")}
        if not _is_fdb(subset):
            m = {"subset": subset, "n": len(noisy)}
        if judge is not None and _is_fdb(subset) and fdb.SUBSETS[subset][1] == "user_interruption" and fdb.SUBSETS[subset][0] == "1.0":
            rs = await asyncio.gather(*(fdb.judge_interruption(e, judge, media) for e in noisy))
            ratings = {e["meta"]["episode_id"]: r for e, r in zip(noisy, rs) if r is not None}
        if judge is not None and _is_fdb(subset) and fdb.SUBSETS[subset][0] == "1.5" and clean:
            instr = fdb.behaviour_instruction(args.repo)
            pairs = [(e, clean[fdb.bench_meta(e)["sample_id"]]) for e in noisy if fdb.bench_meta(e)["sample_id"] in clean]
            tags = await asyncio.gather(*(fdb.judge_behaviour(e, c, judge, instr, media) for e, c in pairs))
            behaviours = {e["meta"]["episode_id"]: t or "C_UNKNOWN" for (e, _), t in zip(pairs, tags)}
        if _is_fdb(subset):
            m = fdb.official_metrics(noisy, media, gt_distribution=gt, ratings=ratings, behaviours=behaviours)
        per = m.pop("per_sample", None) or {e["meta"]["episode_id"]: {"text": [t.get("text") for t in e["turns"]],
                                                                    "scores": e["eval"]["scores"]["by_expectation"]} for e in noisy}
        by: dict[str, list[float]] = {}
        totals, outcomes = [], {}
        for e in noisy:
            sc = e["eval"]["scores"]
            if sc["total"] is not None:
                totals.append(sc["total"])
            for k, v in sc["by_expectation"].items():
                by.setdefault(k, []).append(v)
            for ev in sc["events"]:
                o = outcomes.setdefault(ev["expects"], {})
                o[ev["outcome"]] = o.get(ev["outcome"], 0) + 1
        m["scores"] = {"total": round(sum(totals) / len(totals), 4) if totals else None,
                       "by_expectation": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
                       "outcomes": {k: dict(sorted(v.items())) for k, v in sorted(outcomes.items())}}
        m["wall_s"] = round(sum(e["meta"].get("wall_s", 0) for e in eps), 1)
        m["sim_s"] = round(sum(e["meta"]["duration_ms"] for e in eps) / 1000, 1)
        if ratings is not None or behaviours is not None:
            m["judge"] = args.judge_model or (summary_prev.get(subset, {}).get("judge") if judge is None else None)
        summary[subset] = m
        (out / subset / "per_sample.json").write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items()} | ({"rating": ratings.get(k)} if ratings else {}) | ({"behaviour": behaviours.get(k)} if behaviours else {})
             for k, v in per.items()}, indent=1))
        pages += noisy[: args.pages] + [clean[fdb.bench_meta(e)["sample_id"]] for e in noisy[: args.pages] if fdb.bench_meta(e)["sample_id"] in clean]
        print(json.dumps({subset: {k: v for k, v in m.items()}}, indent=1), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    traces = []
    for subset in _subsets(args):
        p = out / subset / "agent_traces.jsonl"
        if p.exists():
            want = {e["meta"]["episode_id"] for e in pages}
            traces += [t for t in load(p) if t["episode_id"] in want]
    notes = {e["meta"]["episode_id"]: f"{fdb.bench_meta(e)['subset']} #{fdb.bench_meta(e)['sample_id']}"
             + (" (clean)" if fdb.bench_meta(e).get("clean") else "") + " · " + json.dumps(
                 {k: v for k, v in e["eval"].get("benchmark", {}).items() if not isinstance(v, (list, str))}) for e in pages}
    title = BENCHES[args.bench][0]
    print("->", export_run(pages, out, notes, title=f"{title} · MiniCPM-o 4.5", traces=traces))
