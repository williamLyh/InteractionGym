"""Aggregate the cross-benchmark runs into one JSON of headline numbers with 95% CIs (results/data/summary.json).

    PYTHONPATH=src:. python results/aggregate.py --runs <runs dir> --out results/data/summary.json

Expects the run folders written by the runners (after each runner's ``report``): ``fdb3_ab`` (examples/fdb3_ab.py),
``fdb2_ab`` (examples/fdb2_ab.py, reports ``rows_180.json`` and ``rows_120.json``), ``audiomc_ab``
(examples/audiomc_ab.py), and ``turn_metrics.json`` (results/turn_metrics.py: pass by user turn). The release results
are these three open- vs closed-loop benchmarks; ``--only fdb,et,hd`` still aggregates the open-loop-only runs
(``fdb_minicpmo45`` / ``et_minicpmo45`` / ``hd_minicpmo45``, examples/fdb_baseline.py) of the full-audio appendix. Official-metric values are recomputed from the reports' per-sample files; duplex timing
from the episodes (``eval.timing_counts``, collisions kept apart). CIs: percentile bootstrap (2,000 resamples) over
samples, or over tasks / recordings when a task has several seeds; A/B differences are paired. ``score_A`` /
``score_B``: eval.scores (one score per user turn) recomputed from the saved turns with the env checkout in use, so a
later scoring change only needs a re-aggregation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from interaction_gym.eval import scores, timing_counts, timing_summary

B = 2000


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _clusters(items, key):
    cl: dict = {}
    for it in items:
        cl.setdefault(key(it), []).append(it)
    return list(cl.values())


def boot(items, stat, key=None, seed=0):
    """stat(list of items) -> float | None; bootstrap over clusters (key) or items. Returns {est, lo, hi, n}."""
    items = list(items)
    est = stat(items) if items else None
    out = {"est": _r(est), "lo": None, "hi": None, "n": len(items)}
    if est is None or len(items) < 2:
        return out
    groups = _clusters(items, key) if key else [[x] for x in items]
    rng, vals = random.Random(seed), []
    for _ in range(B):
        s = [x for _ in groups for x in groups[rng.randrange(len(groups))]]
        v = stat(s)
        if v is not None:
            vals.append(v)
    vals.sort()
    if vals:
        out["lo"], out["hi"] = _r(vals[int(0.025 * (len(vals) - 1))]), _r(vals[int(0.975 * (len(vals) - 1))])
    return out


def _r(x):
    return None if x is None else round(float(x), 4)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2 if xs else None


def m_of(field):
    return lambda rows: mean([r.get(field) for r in rows])


# ---------------------------------------------------------------- duplex timing (eval.scores), any episode list

OPEN_INTENDED = lambda u: u.get("expects") in ("yield", "interrupt", "wait")  # noqa: E731  (benchmark overlap events are by design)


def ep_timing(ep: dict, end_ms: int | None = None, intended=None) -> dict:
    end = end_ms or ep["meta"]["duration_ms"]
    turns = [t for t in ep["turns"] if t["start_time"] < end]
    turns = [dict(t, end_time=min(t["end_time"], end)) for t in turns]
    return timing_counts(turns, scores(turns, end_ms=end)["events"], intended=intended, end_ms=end)


def timing_block(counts: list[dict], key=None) -> dict:
    """counts: list of (cluster id, timing_counts)."""
    if not counts:
        return {}
    items = counts
    k = (lambda it: it[0]) if key else None
    f = lambda name: (lambda its: timing_summary([c for _, c in its])[name])  # noqa: E731
    s = timing_summary([c for _, c in counts])
    out = {"episodes": s["episodes"], "user_turns": s["user_turns"]}
    for name in ("turn_take_rate", "turn_take_rate_clean", "latency_median_ms", "latency_p90_ms", "yield_rate",
                 "yield_latency_median_ms", "cut_in_rate", "collision_share", "collisions_per_episode", "collision_yield_rate",
                 "nondirected_ok_rate"):
        out[name] = boot(items, f(name), key=k) if s.get(name) is not None else None
    out["barge_ins"], out["collisions"], out["nondirected"] = s["barge_ins"], s["collisions"], s["nondirected"]
    return out


# ---------------------------------------------------------------- open-loop benchmarks (fdb_baseline runs)

def events_of(ep: dict) -> list[dict]:
    return scores(ep["turns"], end_ms=ep["meta"]["duration_ms"])["events"]


def score_block(eps: list[dict], key=None) -> dict:
    """eval.scores (one rule-based score per user turn, recomputed from the turns): mean over the scored user turns,
    pooled over episodes; bootstrap over episodes or over clusters (key: episode -> cluster id)."""
    for e in eps:
        if "_ev" not in e:
            e["_ev"] = events_of(e)
    stat = lambda es: mean([ev["score"] for e in es for ev in e["_ev"] if ev.get("score") is not None])  # noqa: E731
    return boot(eps, stat, key=key)


def outcome_rate(eps, expects, good, bad):
    """Share of events of `expects` with an outcome in `good` among good + bad (per-episode pooling)."""
    def stat(es):
        g = sum(1 for e in es for ev in e["_ev"] if ev["expects"] == expects and ev.get("outcome") in good)
        b = sum(1 for e in es for ev in e["_ev"] if ev["expects"] == expects and ev.get("outcome") in bad)
        return g / (g + b) if g + b else None
    return boot(eps, stat)


def openloop(root: Path, bench: str) -> dict:
    out = {}
    for sub in sorted(p.parent for p in root.rglob("episodes.jsonl")):
        name = str(sub.relative_to(root))
        eps = [e for e in load(sub / "episodes.jsonl") if not (e["eval"].get("benchmark") or {}).get("clean")
               and not e["meta"]["episode_id"].endswith("-clean")]
        for e in eps:
            e["_ev"] = events_of(e)
        per = json.loads((sub / "per_sample.json").read_text()) if (sub / "per_sample.json").exists() else {}
        m = {"n": len(eps)}
        if bench == "fdb" and per:
            vals = list(per.values())
            for k in ("TOR", "latency", "rating", "freq", "JSD", "stop", "resp"):
                if any(k in v for v in vals):
                    if k == "latency":
                        xs = [v for v in vals if v.get("TOR") == 1 and v.get("latency") is not None]
                    else:
                        xs = [v for v in vals if v.get(k) is not None]
                    m[k] = boot(xs, lambda its, k=k: mean([v[k] for v in its]))
            if any("behaviour" in v for v in vals):
                xs = [v for v in vals if v.get("behaviour")]
                for c in ("C_RESPOND", "C_RESUME"):
                    m[c] = boot(xs, lambda its, c=c: mean([v["behaviour"] == c for v in its]))
        # eval.scores: one event per user turn; the share of turns of each expectation handled as expected
        m["respond"] = outcome_rate(eps, "respond", {"responded"}, {"no_response", "talked_over"})
        m["yield"] = outcome_rate(eps, "yield", {"yielded"}, {"kept_talking", "talked_over", "no_response", "took_floor"})
        m["wait"] = outcome_rate(eps, "wait", {"waited", "checked_in"}, {"took_floor", "kept_talking"})
        m["ignore"] = outcome_rate(eps, "ignore", {"ignored"}, {"replied", "stopped"})
        m["score"] = boot(eps, lambda es: mean([ev["score"] for e in es for ev in e["_ev"] if "score" in ev]))
        m["outcomes"] = {}
        for e in eps:
            for ev in e["_ev"]:
                o = m["outcomes"].setdefault(ev["expects"], {})
                o[ev.get("outcome")] = o.get(ev.get("outcome"), 0) + 1
        m["timing"] = timing_block([(e["meta"]["episode_id"], ep_timing(e, intended=OPEN_INTENDED)) for e in eps])
        out[name] = m
    return out


# ---------------------------------------------------------------- FD-Bench v3

def fdb3(root: Path) -> dict:
    out = {}
    for tag_dir in sorted(p for p in root.iterdir() if (p / "rows.json").exists()):
        rows = json.loads((tag_dir / "rows.json").read_text())
        rec = lambda r: r["recording"]  # noqa: E731
        m: dict = {"n": len(rows), "seeds": sorted({r["seed"] for r in rows})}
        duplex = "spoken" in json.dumps(rows[0]["A"])[:20000]
        if duplex:
            A = lambda r: r["A"]["spoken"]["fulfilled"]  # noqa: E731
            Bf = lambda r: r["B"]["spoken_first"]["fulfilled"]  # noqa: E731
            Bl = lambda r: r["B"]["spoken_final"]["fulfilled"]  # noqa: E731
            m["fulfilled_A"] = boot(rows, lambda rs: mean([A(r) for r in rs]), rec)
            m["fulfilled_B_first"] = boot(rows, lambda rs: mean([Bf(r) for r in rs]), rec)
            m["fulfilled_B_final"] = boot(rows, lambda rs: mean([Bl(r) for r in rs]), rec)
            m["fulfilled_B_final_minus_A"] = boot(rows, lambda rs: mean([Bl(r) - A(r) for r in rs]), rec)
            m["slot_score_A"] = boot(rows, lambda rs: mean([r["A"]["spoken"]["outcome"] for r in rs]), rec)
            m["slot_score_B_final"] = boot(rows, lambda rs: mean([r["B"]["spoken_final"]["outcome"] for r in rs]), rec)
            fails = [r for r in rows if not A(r)]
            m["recovered"] = boot(fails, lambda rs: mean([Bl(r) for r in rs]), rec)
            m["consistency_first_text"] = mean([r["consistency"].get("first_turn") for r in rows])
            m["consistency_first_response"] = mean([r["consistency"].get("first_response") for r in rows])
            m["consistency_outcome"] = mean([r["consistency"].get("outcome_equal") for r in rows])
            m["consistency_speech"] = mean([r["consistency"].get("speech") for r in rows])
            m["consistency_fulfilled"] = mean([A(r) == Bf(r) for r in rows])
            m["broken"] = boot([r for r in rows if A(r)], lambda rs: mean([not Bl(r) for r in rs]), rec)
            m["user_turns_after_first_B"] = boot(rows, lambda rs: mean([r["B"].get("user_turns_after_first") for r in rs]), rec)
            m["claims_result_A"] = mean([r["A"].get("claims_result") for r in rows if "claims_result" in r["A"]])
            m["claims_result_B"] = mean([r["B"].get("claims_result") for r in rows if "claims_result" in r["B"]])
        else:
            for j in ("", "lenient_"):
                pa = (lambda r: r["A"]["pass"]) if not j else (lambda r: r["A"]["lenient"])
                pfin = (lambda r: r["B"]["pass_final"]) if not j else (lambda r: r["B"]["lenient_final"])
                pstr = (lambda r: r["B"]["pass_strict"]) if not j else (lambda r: r["B"]["lenient_strict"])
                pfirst = (lambda r: r["B"]["pass_first"]) if not j else (lambda r: r["B"]["lenient_first"])
                m[f"{j}pass_A"] = boot(rows, lambda rs, f=pa: mean([f(r) for r in rs]), rec)
                m[f"{j}pass_B_first"] = boot(rows, lambda rs, f=pfirst: mean([f(r) for r in rs]), rec)
                m[f"{j}pass_B_final"] = boot(rows, lambda rs, f=pfin: mean([f(r) for r in rs]), rec)
                m[f"{j}pass_B_strict"] = boot(rows, lambda rs, f=pstr: mean([f(r) for r in rs]), rec)
                m[f"{j}pass_B_final_minus_A"] = boot(rows, lambda rs, a=pa, b=pfin: mean([b(r) - a(r) for r in rs]), rec)
                fails = [r for r in rows if not pa(r)]
                m[f"{j}recovered"] = boot(fails, lambda rs, f=pfin: mean([f(r) for r in rs]), rec)
            c = [r["consistency"] for r in rows]
            m["consistency_calls"] = mean([x.get("calls") for x in c])
            m["consistency_speech"] = mean([x.get("speech") for x in c])
            m["consistency_first_response"] = mean([x.get("first_response") for x in c])
        for cond in ("A", "B"):
            t = [r[cond]["timing"] for r in rows]
            m[f"official_timing_{cond}"] = {
                "turn_take": boot(t, lambda ts: mean([x["turn_taken"] for x in ts])),
                "interrupted": boot(t, lambda ts: mean([bool(x.get("interrupted")) for x in ts])),
                "first_response_ms_median": boot([x for x in t if x.get("first_response_ms") is not None and not x.get("interrupted")],
                                                 lambda ts: median([x["first_response_ms"] for x in ts])),
            }
            m[f"response_quality_{cond}"] = boot(rows, lambda rs, c=cond: mean([r[c].get("response_quality") for r in rs]), rec)
            eps = load(tag_dir / cond / "episodes.jsonl")
            m[f"score_{cond}"] = score_block(eps, key=lambda e: e["meta"].get("recording", e["meta"]["episode_id"]))
            m[f"timing_{cond}"] = timing_block([(e["meta"].get("recording", e["meta"]["episode_id"]), ep_timing(e)) for e in eps], key=True)
        out[tag_dir.name] = m
    return out


# ---------------------------------------------------------------- FD-Bench v2

def fdb2(root: Path) -> dict:
    out = {}
    for name in ("180", "120"):
        p = root / f"rows_{name}.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        res = {}
        for agent in sorted({r["agent"] for r in rows}):
            R = [r for r in rows if r["agent"] == agent]
            by = {(r["task_id"], r["seed"], r["cond"]): r for r in R}
            pairs = [(by[(t, s, "A")], by[(t, s, "B")]) for (t, s, c) in by if c == "A" and (t, s, "B") in by]
            key = lambda pr: pr[0]["task_id"]  # noqa: E731
            m = {"pairs": len(pairs), "seeds": sorted({r["seed"] for r in R})}
            for f in ("tt", "if", "task", "tt_r", "if_r", "task_r"):
                for ci, cond in enumerate("AB"):
                    m[f"{f}_{cond}"] = boot(pairs, lambda ps, f=f, ci=ci: mean([p[ci].get(f) for p in ps]), key)
                m[f"{f}_B_minus_A"] = boot([p for p in pairs if p[0].get(f) is not None and p[1].get(f) is not None],
                                           lambda ps, f=f: mean([p[1][f] - p[0][f] for p in ps]), key)
            st = lambda r: _stage_score(r)  # noqa: E731
            for ci, cond in enumerate("AB"):
                m[f"stage_score_{cond}"] = boot(pairs, lambda ps, ci=ci: mean([st(p[ci]) for p in ps]), key)
                m[f"stages_completed_{cond}"] = boot(pairs, lambda ps, ci=ci: mean([_stages(p[ci], "completed") for p in ps]), key)
                m[f"reached_end_{cond}"] = boot(pairs, lambda ps, ci=ci: mean([bool(p[ci].get("reached_end")) for p in ps]), key)
                m[f"timing_{cond}"] = timing_block([(p[ci]["task_id"], p[ci]["timing"]) for p in pairs], key=True)
                if name == "180":
                    m[f"score_{cond}"] = score_block(load(root / agent / cond / "episodes.jsonl"), key=lambda e: e["meta"].get("task_id"))
                v = [x for p in pairs for x in (p[ci].get("validity") or [])]
                m[f"validity_{cond}"] = {"turns": len(v), "invalid": mean([x["invalid"] for x in v]),
                                         "timing": mean([x["timing"] for x in v]), "content": mean([x["content"] for x in v]),
                                         "invalid_llm": mean([x["invalid_llm"] for x in v])}
            m["consistency_text"] = mean([p[1].get("consistent_text") for p in pairs])
            m["consistency_onset"] = mean([p[1].get("consistent_onset") for p in pairs])
            m["stage_score_B_minus_A"] = boot(pairs, lambda ps: mean([st(p[1]) - st(p[0]) for p in ps if st(p[0]) is not None and st(p[1]) is not None]), key)
            m["llm_invalid_excess"] = boot(pairs, lambda ps: mean([_vshare(p[0]) - _vshare(p[1]) for p in ps
                                                                  if _vshare(p[0]) is not None and _vshare(p[1]) is not None]), key)
            res[agent] = m
        if len(res) > 1:
            res["between_agents"] = _between(rows)
        out[f"window_{name}s"] = res
    return out


def _between(rows) -> dict:
    """Paired differences between agents on the same (task, seed), per condition."""
    by = {(r["agent"], r["task_id"], r["seed"], r["cond"]): r for r in rows}
    out = {}
    for a, b in (("minicpmo", "cascaded"), ("minicpmo-confirm", "minicpmo"), ("minicpmo-confirm", "cascaded")):
        for cond in "AB":
            ks = [(t, s) for (ag, t, s, c) in by if ag == a and c == cond and (b, t, s, cond) in by]
            pairs = [(by[(a, t, s, cond)], by[(b, t, s, cond)]) for t, s in ks]
            d = {}
            for f in ("tt", "if_r", "task_r"):
                P = [p for p in pairs if p[0].get(f) is not None and p[1].get(f) is not None]
                d[f] = boot(P, lambda ps, f=f: mean([p[0][f] - p[1][f] for p in ps]), lambda p: p[0]["task_id"])
            P = [p for p in pairs if _stage_score(p[0]) is not None and _stage_score(p[1]) is not None]
            d["stage_score"] = boot(P, lambda ps: mean([_stage_score(p[0]) - _stage_score(p[1]) for p in ps]), lambda p: p[0]["task_id"])
            out[f"{a} - {b} ({cond})"] = d
    return out


def _stage_list(r):
    st = r.get("stages") or {}
    return (st.get("stages") or {}) if isinstance(st, dict) else {}


def _stage_score(r):
    v = {"yes": 1.0, "partial": 0.5, "no": 0.0}
    xs = [v[s["agent"]] for s in _stage_list(r).values() if s.get("reached") and s.get("agent") in v]
    return mean(xs)


def _stages(r, what):
    return sum(1 for s in _stage_list(r).values() if s.get(what))


def _vshare(r):
    v = r.get("validity") or []
    return mean([x["invalid_llm"] for x in v]) if v else None


# ---------------------------------------------------------------- Audio MultiChallenge

def audiomc(root: Path) -> dict:
    rows = json.loads((root / "rows.json").read_text())
    out = {}
    for agent in sorted({r["agent"] for r in rows}):
        R = [r for r in rows if r["agent"] == agent]
        by = {(r["task_id"], r["seed"], r["cond"]): r for r in R}
        pairs = [(by[(t, s, "A")], by[(t, s, "B")]) for (t, s, c) in by if c == "A" and (t, s, "B") in by]
        m = {"pairs": len(pairs), "seeds": sorted({p[0]["seed"] for p in pairs})}
        tk = lambda p: p[0]["task_id"]  # noqa: E731  (bootstrap over conversations: several seeds per conversation)
        axes = ["all"] + sorted({p[0]["axis"] for p in pairs})
        for ax in axes:
            P = [p for p in pairs if ax == "all" or p[0]["axis"] == ax]
            d = {"n": len(P)}
            for f in ("rubric_rate", "all_pass"):
                for ci, cond in enumerate("AB"):
                    d[f"{f}_{cond}"] = boot(P, lambda ps, f=f, ci=ci: mean([float(p[ci][f]) for p in ps if p[ci][f] is not None]), tk)
                d[f"{f}_B_minus_A"] = boot(P, lambda ps, f=f: mean([float(p[1][f]) - float(p[0][f]) for p in ps
                                                                    if p[0][f] is not None and p[1][f] is not None]), tk)
            m[ax] = d
        for ci, cond in enumerate("AB"):
            m[f"timing_{cond}"] = timing_block([(p[ci]["task_id"], p[ci]["timing"]) for p in pairs], key=True)
            m[f"score_{cond}"] = score_block(load(root / agent / cond / "episodes.jsonl"))
            v = [x for p in pairs for x in (p[ci].get("validity") or [])]
            m[f"validity_{cond}"] = {"turns": len(v), "invalid": mean([x["invalid"] for x in v]),
                                     "invalid_llm": mean([x["invalid_llm"] for x in v]), "content": mean([x["content"] for x in v])}
        m["llm_invalid_excess"] = boot(pairs, lambda ps: mean([_vshare(p[0]) - _vshare(p[1]) for p in ps
                                                              if _vshare(p[0]) is not None and _vshare(p[1]) is not None]), tk)
        m["first_onset_equal"] = mean([_onset(p[0]) == _onset(p[1]) for p in pairs])
        m["first_text_equal"] = mean([_first_text(p[0]) == _first_text(p[1]) for p in pairs])
        m["consistent_before_u1"] = mean([p[1].get("consistent", p[0].get("consistent")) for p in pairs])
        out[agent] = m
    return out


def _first_text(r):
    s = r.get("speech_before_u1") or []
    return s[0][1] if s and len(s[0]) > 1 else None


def _onset(r):
    s = r.get("speech_before_u1") or []
    return s[0][0] if s else None


# ---------------------------------------------------------------- pass by user turn (results/turn_metrics.py)

def turns(path: Path) -> dict:
    """Curves by user turn N (definitions in results/turn_metrics.py): FD-Bench v3 pass@N-turn of B (carried forward
    after a conversation's last user turn) with A (= pass@1turn of the open loop) as the reference; FD-Bench v2
    goals completed / stage score by examiner line N for A and B. Bootstrap over recordings / tasks."""
    d = json.loads(path.read_text())
    out: dict = {}
    rows = [r for r in d.get("fdb3", []) if r.get("B")]
    if rows:
        n_max = max(len(r["B"]) for r in rows)
        at = lambda r, n: r["B"][min(n, len(r["B"])) - 1]  # noqa: E731
        rec = lambda r: r["recording"]  # noqa: E731
        f3 = {"n": len(rows), "A": boot(rows, lambda rs: mean([r["A"] for r in rs]), rec), "by_n": []}
        for n in range(1, n_max + 1):
            f3["by_n"].append({"N": n, "pass_B": boot(rows, lambda rs, n=n: mean([at(r, n) for r in rs]), rec),
                               "pass_B_minus_A": boot(rows, lambda rs, n=n: mean([at(r, n) - r["A"] for r in rs]), rec),
                               "still_going": mean([len(r["B"]) >= n for r in rows])})
        f3["final"] = boot(rows, lambda rs: mean([r["B"][-1] for r in rs]), rec)
        f3["user_turns"] = {"median": median([r["n_users"] for r in rows]), "max": max(r["n_users"] for r in rows)}
        out["fdb3"] = f3
    rows2 = [r for r in d.get("fdb2", []) if r["by_n"]]
    if rows2:
        f2 = {}
        for agent in sorted({r["agent"] for r in rows2}):
            g = {}
            for cond in "AB":
                R = [r for r in rows2 if r["agent"] == agent and r["cond"] == cond]
                if not R:
                    continue
                n_max = max(len(r["by_n"]) for r in R)
                at = lambda r, n: r["by_n"][min(n, len(r["by_n"])) - 1]  # noqa: E731
                key = lambda r: r["task_id"]  # noqa: E731
                g[cond] = {"n": len(R), "by_n": [
                    {"N": n, "completed": boot(R, lambda rs, n=n: mean([(at(r, n) or {}).get("completed") for r in rs]), key),
                     "all4": boot(R, lambda rs, n=n: mean([None if at(r, n) is None else at(r, n)["completed"] == 4 for r in rs]), key),
                     "stage_score": boot(R, lambda rs, n=n: mean([(at(r, n) or {}).get("stage_score") for r in rs]), key),
                     "still_going": mean([len(r["by_n"]) >= n for r in R])} for n in range(1, n_max + 1)]}
            f2[agent] = g
        out["fdb2"] = f2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="results/data/summary.json")
    ap.add_argument("--only", default="fdb3,fdb2,audiomc,turns", help="comma list of: fdb3,fdb2,audiomc,turns (release); fdb,et,hd (open-loop-only runs)")
    ap.add_argument("--fresh", action="store_true", help="start a new summary instead of updating --out")
    a = ap.parse_args()
    runs, only = Path(a.runs), set(a.only.split(",")) if a.only else None
    outp = Path(a.out)
    res = json.loads(outp.read_text()) if outp.exists() and not a.fresh else {}
    jobs = {"fdb": lambda: openloop(runs / "fdb_minicpmo45", "fdb"), "et": lambda: openloop(runs / "et_minicpmo45", "et"),
            "hd": lambda: openloop(runs / "hd_minicpmo45", "hd"), "fdb3": lambda: fdb3(runs / "fdb3_ab"),
            "fdb2": lambda: fdb2(runs / "fdb2_ab"), "audiomc": lambda: audiomc(runs / "audiomc_ab"),
            "turns": lambda: turns(runs / "turn_metrics.json")}
    for k, f in jobs.items():
        if only and k not in only:
            continue
        try:
            res[k] = f()
            print("ok", k, flush=True)
        except FileNotFoundError as e:
            print("skip", k, e, flush=True)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
