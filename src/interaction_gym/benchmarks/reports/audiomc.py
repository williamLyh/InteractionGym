"""Audio MultiChallenge report: the official rubric judge per item on a proxy LLM, open-loop validity, duplex timing,
A = B consistency up to the user's second turn; tables with bootstrap CIs. ``report(args, judge_llm)``: ``args`` has
``out``, ``data``, ``concurrency``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .. import audiomc
from . import Judge, _strip, boot_ci, timing, timing_table, validity, validity_table


async def report(args, judge_llm) -> None:
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max(32, args.concurrency)))  # LLM calls run in threads
    out = Path(args.out)
    tasks = {t.id: t for t in audiomc.load(args.data)}
    judge = Judge(judge_llm, out / "judge_cache.json")
    eps = []
    for p in sorted(out.glob("*/*/episodes.jsonl")):
        eps += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    sem = asyncio.Semaphore(args.concurrency)

    async def one(ep):
        task = tasks[ep["meta"]["task_id"]]
        conv = audiomc.history(ep)
        async with sem:
            verdicts = []
            for item in task.criteria["rubric"]:
                raw = await judge.ask(audiomc.judge_prompt(conv, item), max_tokens=800)
                try:
                    verdicts.append(bool(json.loads(_strip(raw)).get("criteria_met")))
                except Exception:  # noqa: BLE001
                    verdicts.append(None)
            val = await validity(ep, judge)
        users = [t for t in sorted(ep["turns"], key=lambda t: t["start_time"]) if t["role"] == "user" and t["text"]]
        script = task.scenario["script"]
        return {"id": ep["meta"]["episode_id"], "task_id": task.id, "axis": task.criteria["axis"], "agent": ep["meta"]["agent_name"],
                "cond": ep["meta"]["condition"], "seed": ep["meta"].get("seed"), "verdicts": verdicts,
                "rubric_rate": sum(v for v in verdicts if v is not None) / max(1, sum(v is not None for v in verdicts)),
                "all_pass": all(v for v in verdicts if v is not None) and any(v is not None for v in verdicts),
                "final_reply": audiomc.final_reply(ep), "user_turns": len(users), "script_turns": len(script),
                "complete": len(users) >= len(script), "duration_ms": ep["meta"]["duration_ms"], "end_reason": ep["meta"]["end_reason"],
                "timing": timing(ep), "validity": val,
                "u1_ms": users[1]["start_time"] if len(users) > 1 else None,
                "speech_before_u1": [(t["start_time"], t["text"]) for t in sorted(ep["turns"], key=lambda t: t["start_time"])
                                     if t["role"] != "user" and t["text"] and len(users) > 1 and t["start_time"] < users[1]["start_time"]]}

    rows = await asyncio.gather(*(one(ep) for ep in eps))
    judge.flush()
    by = {(r["agent"], r["task_id"], r["seed"], r["cond"]): r for r in rows}
    for r in rows:
        a = by.get((r["agent"], r["task_id"], r["seed"], "A"))
        if r["cond"] == "B" and a:
            cut = min(x for x in (a["u1_ms"], r["u1_ms"], 10**9) if x is not None)
            r["consistent"] = [x for x in a["speech_before_u1"] if x[0] < cut] == [x for x in r["speech_before_u1"] if x[0] < cut]
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    agents = sorted({r["agent"] for r in rows})
    L = ["# Audio MultiChallenge: open loop (recorded turns) vs closed loop (script replanned to the agent)", "",
         "Judge: official prompt, per rubric item, local proxy LLM; history = the conversation as it happened.", "",
         "| agent | axis | n pairs | rubric pass A | rubric pass B | B−A [95% CI] | all items A | all items B | B−A [95% CI] | complete A / B |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        for axis in (None, *audiomc.AXES):
            P = [(by[(ag, r["task_id"], r["seed"], "A")], r) for r in rows if r["agent"] == ag and r["cond"] == "B"
                 and (axis is None or r["axis"] == axis) and (ag, r["task_id"], r["seed"], "A") in by]
            if not P:
                continue
            m = lambda xs: sum(xs) / len(xs)  # noqa: E731
            d1 = [b["rubric_rate"] - a["rubric_rate"] for a, b in P]
            d2 = [float(b["all_pass"]) - float(a["all_pass"]) for a, b in P]
            c1, c2 = boot_ci(d1), boot_ci(d2)
            L.append(f"| {ag} | {axis or 'all'} | {len(P)} | {m([a['rubric_rate'] for a, _ in P]):.3f} | {m([b['rubric_rate'] for _, b in P]):.3f} | "
                     f"{m(d1):+.3f} [{c1[0]:+.3f}, {c1[1]:+.3f}] | {m([a['all_pass'] for a, _ in P]):.3f} | {m([b['all_pass'] for _, b in P]):.3f} | "
                     f"{m(d2):+.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}] | {m([a['complete'] for a, _ in P]):.2f} / {m([b['complete'] for _, b in P]):.2f} |"
                     if c1[0] is not None else f"| {ag} | {axis or 'all'} | {len(P)} | (too few) |")
    L += ["", "Consistency (agent speech before the user's 2nd turn, A vs B):"]
    for ag in agents:
        C = [r for r in rows if r["agent"] == ag and "consistent" in r]
        if C:
            L.append(f"- {ag}: {sum(r['consistent'] for r in C)}/{len(C)}")
    L += timing_table(rows, agents) + validity_table(rows, agents)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
