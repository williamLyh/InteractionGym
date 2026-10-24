"""Post-hoc reports of the benchmark runs: the pieces the A/B runners' ``report`` commands share — a memoized judge
on a text LLM, open-loop user-turn validity (``turn_validity``), duplex timing tables, bootstrap confidence intervals.
Each benchmark's report is a module here (``audiomc``, ``fdb2``, ``fdb3``) and a ``report/*`` plugin (``ig report``);
``examples/*_ab.py report`` call the same functions.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from ... import turn_validity as tv
from ...eval import timing_counts, timing_summary


class Judge:
    def __init__(self, llm_, cache_path: Path):
        self.llm, self.path = llm_, cache_path
        self.memo = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        self.new = 0

    async def ask(self, prompt: str, max_tokens: int = 1500) -> str:
        if prompt not in self.memo:
            for attempt in range(4):
                try:
                    self.memo[prompt] = await self.llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=max_tokens)
                    self.new += 1
                    if self.new % 500 == 0:  # long reports: do not lose the cache to a crash
                        self.flush()
                    break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(20)
            else:
                return ""
        return self.memo[prompt]

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.memo))
        tmp.replace(self.path)

def _strip(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    return raw

def _speech(ep, before=None):
    """Agent speech that is complete before ``before`` (a turn still running there can be cut differently once the
    two conditions diverge at the examiner's 2nd line, so it is left out)."""
    return [(t["start_time"], t["text"]) for t in sorted(ep["turns"], key=lambda t: t["start_time"])
            if t["role"] != "user" and t["text"] and t["end_time"] > t["start_time"] and (before is None or t["end_time"] <= before)]

def _users(ep):
    return [t for t in sorted(ep["turns"], key=lambda t: t["start_time"]) if t["role"] == "user" and t["text"]]

def _dialogue_upto(ep, t_cut):
    lines = []
    for t in sorted(ep["turns"], key=lambda t: t["start_time"]):
        if t["start_time"] >= t_cut or not t["text"] or t["end_time"] <= t["start_time"]:
            continue
        lines.append(("CALLER: " if t["role"] == "user" else "ASSISTANT: ") + t["text"])
    return "\n".join(lines) or "(nothing yet)"

def overlap_stats(ep) -> dict:
    """Examiner lines that start while the agent is talking (talked over / collisions) and agent turns started
    while the examiner talks (cut-ins), from the turn spans."""
    us, ag = _users(ep), [t for t in ep["turns"] if t["role"] != "user" and t["end_time"] > t["start_time"]]
    u_into_a = sum(any(a["start_time"] < u["start_time"] < a["end_time"] for a in ag) for u in us[1:])
    a_into_u = sum(any(u["start_time"] < a["start_time"] < u["end_time"] for u in us) for a in ag)
    return {"examiner_over_agent": u_into_a, "agent_over_examiner": a_into_u, "examiner_lines": len(us), "agent_turns": len(ag)}

async def validity(ep, judge: Judge, max_ms: int | None = None, context: str = "") -> list[dict]:
    """Per checked user turn (after the first): open-loop user-turn validity (``turn_validity``: timing rule + LLM
    content / ignored question / stale), with the turn id and start."""
    rules, out = tv.rule_flags(ep["turns"], end_ms=max_ms), []
    for u in tv.user_turns(ep["turns"], end_ms=max_ms):
        raw = await judge.ask(tv.prompt(ep["turns"], u, context=context), max_tokens=250)
        out.append({"turn": u["id"], "t": u["start_time"], **tv.combine(rules[u["id"]], tv.parse(_strip(raw) if raw else raw))})
    return out

def timing(ep) -> dict:
    return timing_counts(ep["turns"], (ep["eval"].get("scores") or {}).get("events"), end_ms=ep["meta"]["duration_ms"])

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def _fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"

def boot_ci(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    diffs = [d for d in diffs if d is not None]
    if len(diffs) < 3:
        return None, None
    ms = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]

def _pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"

def _ms(x):
    return "—" if x is None else f"{x / 1000:.2f}"

def timing_table(rows, agents, title="## Duplex timing (eval.scores events; open-loop collisions kept apart)") -> list[str]:
    """Turn-take rate (FD-Bench-style take-over after a user turn), response latency, yield on intended barge-ins,
    agent cut-ins on non-collision turns, and collisions (unintended overlaps of a replayed line) with the agent's
    behaviour on them, which are not counted in its yield / cut-in scores."""
    L = ["", title, "",
         "Collision = a user line that starts while the agent talks although it was not meant to cut in (the static script "
         "always waited for the reference assistant). Turn-take rate and latency in brackets: collision turns left out.", "",
         "| agent | cond | episodes | user turns | turn-take rate [clean] | latency median / p90 s [clean median] | intended barge-ins | yield rate | "
         "agent cut-in rate | collisions / episode | collision share of user turns | agent yielded on collisions | agent cut in during collisions | non-directed ignored |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        for cond in ("A", "B"):
            R = [r["timing"] for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("timing")]
            if not R:
                continue
            s = timing_summary(R)
            L.append(f"| {ag} | {cond} | {s['episodes']} | {s['user_turns']} | {_pct(s['turn_take_rate'])} [{_pct(s['turn_take_rate_clean'])}] | "
                     f"{_ms(s['latency_median_ms'])} / {_ms(s['latency_p90_ms'])} [{_ms(s['latency_median_clean_ms'])}] | {s['barge_ins']} | "
                     f"{_pct(s['yield_rate'])} | {_pct(s['cut_in_rate'])} | {_fmt(s['collisions_per_episode'])} | {_pct(s['collision_share'])} | "
                     f"{_pct(s['collision_yield_rate'])} | {_pct(s['collision_cut_in_rate'])} | "
                     f"{_pct(s['nondirected_ok_rate']) if s['nondirected'] else 'none'} |")
    return L

def validity_table(rows, agents, title="## Open-loop user-turn failure rate (turn_validity; B = noise floor of the LLM checker)") -> list[str]:
    """Invalid user turns (after the first) by type, A (replayed) vs B (live user: the checker's noise floor), and the
    excess A − B with a paired bootstrap over conversations."""
    L = ["", title, "",
         "Per user turn after the first: timing = starts inside the agent's speech without meaning to cut in (rule); "
         "content = responds to something the agent never said; ignored = leaves the agent's open question unaddressed "
         "(rule: a question was asked, and LLM); stale = re-gives / re-asks something already settled (LLM). "
         "LLM-only = content, ignored or stale.", "",
         "| agent | cond | turns | invalid (any) | timing | content | ignored question | stale | LLM-only | conversations with ≥1 invalid | ≥1 LLM-only |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        S = {}
        for cond in ("A", "B"):
            R = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("validity") is not None]
            if not R:
                continue
            s = S[cond] = tv.summarize([r["validity"] for r in R])
            L.append(f"| {ag} | {cond} | {s['turns']} | {_pct(s['invalid'])} | {_pct(s['timing'])} | {_pct(s['content'])} | "
                     f"{_pct(s['ignored_question'])} | {_pct(s['stale'])} | {_pct(s['invalid_llm'])} | {_pct(s['episodes_with_invalid'])} | "
                     f"{_pct(s['episodes_with_invalid_llm'])} |")
        if len(S) == 2:
            ex = {k: S["A"][k] - S["B"][k] for k in ("invalid", "timing", "content", "ignored_question", "stale", "invalid_llm",
                                                      "episodes_with_invalid", "episodes_with_invalid_llm")}
            # paired bootstrap over (task, seed) of the per-conversation invalid-turn rate difference
            by = {(r["task_id"], r["seed"], r["cond"]): r for r in rows if r["agent"] == ag and r.get("validity")}
            d = [_mean([f["invalid_llm"] for f in by[(t, s_, "A")]["validity"]]) - _mean([f["invalid_llm"] for f in by[(t, s_, "B")]["validity"]])
                 for (t, s_, c) in by if c == "A" and (t, s_, "B") in by and by[(t, s_, "A")]["validity"] and by[(t, s_, "B")]["validity"]]
            lo, hi = boot_ci(d)
            L.append(f"| {ag} | **A − B** | | **{100 * ex['invalid']:+.0f}** | {100 * ex['timing']:+.0f} | {100 * ex['content']:+.0f} | "
                     f"{100 * ex['ignored_question']:+.0f} | {100 * ex['stale']:+.0f} | {100 * ex['invalid_llm']:+.0f} "
                     f"(per conversation {100 * _mean(d):+.0f} [{_fmt(None if lo is None else 100 * lo, 0)}, {_fmt(None if hi is None else 100 * hi, 0)}]) | "
                     f"{100 * ex['episodes_with_invalid']:+.0f} | {100 * ex['episodes_with_invalid_llm']:+.0f} |")
    return L
