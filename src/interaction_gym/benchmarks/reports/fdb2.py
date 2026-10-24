"""Full-Duplex-Bench v2 report: the official judge (TT / IF / task) per conversation on a proxy LLM, examiner-line
coherence, open-loop validity, stage analysis and cap diagnosis, A = B consistency; tables with paired bootstrap CIs.
``report(args, judge_llm)``: ``args`` has ``out``, ``data``, ``concurrency``, ``window_ms``, ``name``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from .. import fdb2
from . import Judge, _dialogue_upto, _fmt, _mean, _pct, _speech, _strip, _users, boot_ci, overlap_stats, timing, timing_table, validity, validity_table


COHERENCE_PROMPT = """Below is a spoken conversation between a CALLER and an ASSISTANT, up to the caller's next line.

{dialogue}

The caller's next line is:
CALLER: "{line}"

Check strictly whether this line fits what the ASSISTANT ACTUALLY SAID right before it (not what a typical assistant would have said):
1. What did the assistant's last turn ask or say? (If it asked a question, which one?)
2. Does the caller's line answer that question / react to that statement? A line that answers a DIFFERENT question (e.g. gives a coffee size when the assistant asked how the eggs should be cooked), confirms or thanks for something the assistant never said or offered, supplies information nobody asked for while ignoring the question that was asked, or acts as if a step happened that did not, does NOT fit.
3. Volunteering extra details is fine only if the caller ALSO handles what the assistant asked.

Respond with ONLY a JSON object: {{"assistant_last": "...", "fits": true/false, "explanation": "brief reason"}}"""

async def score_ep(ep, task, judge: Judge, window_ms: int | None = None) -> dict:
    """One conversation's row, scored on its first ``window_ms`` (default: the run's own cap)."""
    max_ms = window_ms or ep["meta"].get("max_ms") or fdb2.MAX_MS
    raw = await judge.ask(fdb2.judge_prompt(task, ep))
    j = fdb2.parse_judgement(_strip(raw))
    row = {"id": ep["meta"]["episode_id"], "task_id": task.id, "split": task.criteria["split"], "seed": ep["meta"].get("seed"),
           "cond": ep["meta"]["condition"], "agent": ep["meta"]["agent_name"], "duration_ms": ep["meta"]["duration_ms"],
           "end_reason": ep["meta"]["end_reason"], "reached_end": fdb2.reached_end(ep, max_ms), **fdb2.episode_scores(j, task.criteria["split"]),
           "judge_parsed": bool(j["events"]) or j["task"] is not None, "scores_total": (ep["eval"].get("scores") or {}).get("total"),
           "by_expectation": (ep["eval"].get("scores") or {}).get("by_expectation"), **overlap_stats(ep), "max_ms": max_ms,
           "run_max_ms": ep["meta"].get("max_ms") or fdb2.MAX_MS}
    if max_ms != fdb2.MAX_MS:  # the official judge reads the official 120 s window; a longer run is judged on all of it
        raw = await judge.ask(fdb2.judge_prompt(task, ep, max_ms=max_ms))
        row.update(fdb2.episode_scores(fdb2.parse_judgement(_strip(raw)), task.criteria["split"]))
    # coherence of each examiner line after the first, given what the agent actually said before it
    us, coh = _users(ep), []
    for u in us[1:]:
        if u["start_time"] >= max_ms:
            break
        r = await judge.ask(COHERENCE_PROMPT.format(dialogue=_dialogue_upto(ep, u["start_time"]), line=u["text"]), max_tokens=300)
        try:
            coh.append(bool(json.loads(_strip(r)).get("fits")))
        except Exception:  # noqa: BLE001
            pass
    row["coherent"] = coh
    row["u1_ms"] = us[1]["start_time"] if len(us) > 1 else None
    # duplex timing (collisions apart), open-loop user-turn validity, stage analysis + pacing, reached-stage judge
    row["timing"] = timing(ep)
    row["validity"] = await validity(ep, judge, max_ms)
    st = fdb2.parse_stages(await judge.ask(fdb2.stage_prompt(task, ep, max_ms), max_tokens=700))
    row["stages"], row["pacing"] = st, fdb2.pacing(ep, max_ms)
    row["timeout"] = not row["reached_end"]
    row["cap_cause"] = fdb2.cap_cause(st, row["pacing"]) if st and row["timeout"] else None
    if st is not None and row["timeout"]:
        jr = fdb2.parse_judgement(_strip(await judge.ask(fdb2.judge_prompt(task, ep, stages=st["reached"], max_ms=max_ms))))
        r = fdb2.episode_scores(jr, task.criteria["split"])
    else:
        r = {k: row[k] for k in ("tt", "if", "task")}
    row.update(tt_r=r["tt"], if_r=r["if"], task_r=r["task"])
    log = ep["meta"].get("examiner_log") or []
    row["forced_end"] = any(x.get("forced_end") for x in log)
    row["close_reason"] = next((x.get("close") for x in log if x.get("close")), None)
    return row

async def report(args, judge_llm) -> None:
    from concurrent.futures import ThreadPoolExecutor

    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max(32, args.concurrency)))  # LLM calls run in threads
    out = Path(args.out)
    tasks = {t.id: t for t in fdb2.load(args.data)}
    judge = Judge(judge_llm, out / "judge_cache.json")
    agents = [p.name for p in sorted(out.iterdir()) if (p / "A").exists() or (p / "B").exists()]
    rows, sem = [], asyncio.Semaphore(args.concurrency)
    eps_by = {}
    for ag in agents:
        for cond in ("A", "B"):
            p = out / ag / cond / "episodes.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if line.strip():
                    ep = json.loads(line)
                    eps_by[ep["meta"]["episode_id"]] = ep

    async def one(ep):
        async with sem:
            return await score_ep(ep, tasks[ep["meta"]["task_id"]], judge, args.window_ms or None)

    rows = await asyncio.gather(*(one(ep) for ep in eps_by.values()))
    judge.flush()
    # consistency: agent speech before the examiner's 2nd line identical in A and B
    by = {(r["agent"], r["task_id"], r["seed"], r["cond"]): r for r in rows}
    for r in rows:
        if r["cond"] != "B" or (r["agent"], r["task_id"], r["seed"], "A") not in by:
            continue
        a = by[(r["agent"], r["task_id"], r["seed"], "A")]
        ea, eb = eps_by[a["id"]], eps_by[r["id"]]
        cut = min(x for x in (a["u1_ms"], r["u1_ms"], 10**9) if x is not None)
        sa, sb = _speech(ea, cut), _speech(eb, cut)
        r["consistent_text"] = sa == sb
        first = lambda ep: min((t["start_time"] for t in ep["turns"] if t["role"] != "user" and t["text"]), default=None)  # noqa: E731
        r["consistent_onset"] = first(ea) == first(eb)
    (out / f"rows{args.name}.json").write_text(json.dumps(rows, indent=1))
    lines = ["# FD-Bench v2: open loop (static script) vs closed loop (live examiner)", "",
             "Judge: official prompt on a local proxy LLM; TT / IF = mean over the agent's turn-taking events (per conversation, "
             "then averaged); event-weighted (official aggregation) in brackets; task = task-specific score (1–5).", ""]
    hdr = "| agent | split | n A/B | TT A | TT B | IF A | IF B | IF B−A [95% CI] | task A | task B | task B−A [95% CI] | examiner reached end (B) | examiner lines A / B | incoherent examiner lines A / B | examiner over agent A / B |"
    lines += [hdr, "|" + "---|" * hdr.count("|")[:-1] if False else "|" + "---|" * (hdr.count("|") - 1)]
    for ag in agents:
        for split in (None, *fdb2.SPLITS):
            R = [r for r in rows if r["agent"] == ag and (split is None or r["split"] == split)]
            A, B = [r for r in R if r["cond"] == "A"], [r for r in R if r["cond"] == "B"]
            pa = {(r["task_id"], r["seed"]): r for r in A}
            pairs = [(pa[(r["task_id"], r["seed"])], r) for r in B if (r["task_id"], r["seed"]) in pa]

            def ew(rs, k):  # event-weighted
                num = sum((r[k] or 0) * r["n_events"] for r in rs if r[k] is not None)
                den = sum(r["n_events"] for r in rs if r[k] is not None)
                return num / den if den else None

            dif = lambda k: [(b[k] - a[k]) if a[k] is not None and b[k] is not None else None for a, b in pairs]  # noqa: E731
            ci_if, ci_task = boot_ci(dif("if")), boot_ci(dif("task"))
            inco = lambda rs: _mean([1 - _mean(r["coherent"]) for r in rs if r["coherent"]])  # noqa: E731
            lines.append(f"| {ag} | {split or 'all'} | {len(A)}/{len(B)} | {_fmt(_mean([r['tt'] for r in A]))} [{_fmt(ew(A, 'tt'))}] | "
                         f"{_fmt(_mean([r['tt'] for r in B]))} [{_fmt(ew(B, 'tt'))}] | {_fmt(_mean([r['if'] for r in A]))} | {_fmt(_mean([r['if'] for r in B]))} | "
                         f"{_fmt(_mean(dif('if')))} [{_fmt(ci_if[0])}, {_fmt(ci_if[1])}] | {_fmt(_mean([r['task'] for r in A]))} | "
                         f"{_fmt(_mean([r['task'] for r in B]))} | {_fmt(_mean(dif('task')))} [{_fmt(ci_task[0])}, {_fmt(ci_task[1])}] | "
                         f"{_fmt(_mean([r['reached_end'] for r in B]))} | {_fmt(_mean([r['examiner_lines'] for r in A]), 1)} / {_fmt(_mean([r['examiner_lines'] for r in B]), 1)} | "
                         f"{_fmt(inco(A))} / {_fmt(inco(B))} | {_fmt(_mean([r['examiner_over_agent'] for r in A]))} / {_fmt(_mean([r['examiner_over_agent'] for r in B]))} |")
    lines += ["", "## Consistency (agent speech before the examiner's 2nd line, A vs B)", ""]
    for ag in agents:
        Bs = [r for r in rows if r["agent"] == ag and "consistent_text" in r]
        if Bs:
            lines.append(f"- {ag}: identical text {sum(r['consistent_text'] for r in Bs)}/{len(Bs)}, same first onset "
                         f"{sum(r['consistent_onset'] for r in Bs)}/{len(Bs)}")
    lines += ["", "## Ranking (all splits)", ""]
    for k in ("tt", "if", "task"):
        for cond in ("A", "B"):
            ms = {ag: _mean([r[k] for r in rows if r["agent"] == ag and r["cond"] == cond]) for ag in agents}
            order = sorted((a for a in ms if ms[a] is not None), key=lambda a: -ms[a])
            lines.append(f"- {k} {cond}: " + " > ".join(f"{a} {ms[a]:.2f}" for a in order))
    lines += timing_table(rows, agents) + validity_table(rows, agents) + stage_tables(rows, agents)
    unparsed = sum(not r["judge_parsed"] for r in rows)
    lines += ["", f"Judge replies not parsed: {unparsed}/{len(rows)}"]
    (out / f"summary{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

def stage_tables(rows, agents) -> list[str]:
    """Stage-based rescoring (time cap is not a failure: the agent is scored on the stages reached) and the diagnosis of
    conversations that hit the cap without the end phrase."""
    from collections import Counter

    L = ["", "## Stages reached and stage-based scores (the time cap is not a failure)", "",
         "Stage analysis: one LLM call per conversation (T = 0) marks each goal T1–T4 reached / completed / the agent's "
         "behaviour on it (yes 1, partial 0.5, no 0). Stage score = mean over reached goals. IF_r / TT_r / task_r = the "
         "official judge with only the reached goals listed and told the call was cut by the time limit (= IF / TT / task "
         "when the examiner reached its end phrase).", "",
         "| agent | cond | n | ended (end phrase) | timeout | forced end (closing rule) | stages reached | stages completed | all 4 completed | "
         "stage score | IF_r | TT_r | task_r | IF (official) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ag in agents:
        for cond in ("A", "B"):
            R = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("stages")]
            if not R:
                continue
            L.append(f"| {ag} | {cond} | {len(R)} | {_pct(_mean([r['reached_end'] for r in R]))} | {_pct(_mean([r['timeout'] for r in R]))} | "
                     f"{_pct(_mean([r.get('forced_end', False) for r in R]))} | {_fmt(_mean([len(r['stages']['reached']) for r in R]))} | "
                     f"{_fmt(_mean([len(r['stages']['completed']) for r in R]))} | {_pct(_mean([len(r['stages']['completed']) == 4 for r in R]))} | "
                     f"{_fmt(_mean([r['stages']['stage_score'] for r in R]))} | {_fmt(_mean([r['if_r'] for r in R]))} | "
                     f"{_fmt(_mean([r['tt_r'] for r in R]))} | {_fmt(_mean([r['task_r'] for r in R]))} | {_fmt(_mean([r['if'] for r in R]))} |")
    L += ["", "Paired B − A [95% CI] (same task and seed):", ""]
    for ag in agents:
        by = {(r["task_id"], r["seed"], r["cond"]): r for r in rows if r["agent"] == ag and r.get("stages")}
        P = [(by[(t, s_, "A")], by[(t, s_, "B")]) for (t, s_, c) in by if c == "B" and (t, s_, "A") in by]
        if len(P) < 3:
            continue
        parts = []
        for k, f in (("stage score", lambda r: r["stages"]["stage_score"]), ("IF_r", lambda r: r["if_r"]),
                     ("stages completed", lambda r: len(r["stages"]["completed"]))):
            d = [None if f(a) is None or f(b) is None else f(b) - f(a) for a, b in P]
            lo, hi = boot_ci(d)
            parts.append(f"{k} {_mean(d):+.2f} [{_fmt(lo)}, {_fmt(hi)}]")
        L.append(f"- {ag} (n = {len(P)}): " + "; ".join(parts))
    L += ["", "## Cap diagnosis: conversations without the end phrase", "",
          "Cause (first match): agent_failed_stage = a reached goal the examiner retried ≥ 2 times and the agent did not "
          "complete (or failed); pleasantry_loop; examiner_drift; all_done_no_end = every goal completed, no end phrase; "
          "slow_pacing = the agent talked ≥ 60 % of the window; slow_other.", "",
          "| agent | cond | timeouts | max stage reached T0/T1/T2/T3/T4 | agent_failed_stage | pleasantry_loop | examiner_drift | all_done_no_end | "
          "slow_pacing | slow_other | examiner lines | agent talk share | agent turn s | gap before examiner s |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    causes = ("agent_failed_stage", "pleasantry_loop", "examiner_drift", "all_done_no_end", "slow_pacing", "slow_other")
    for ag in agents:
        for cond in ("A", "B"):
            R = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("stages") and r["timeout"]]
            E = [r for r in rows if r["agent"] == ag and r["cond"] == cond and r.get("stages") and not r["timeout"]]
            if not R:
                continue
            mx, cc = Counter(r["stages"]["max_reached"] for r in R), Counter(r["cap_cause"] for r in R)
            pm = lambda k, rs: _fmt(_mean([r["pacing"][k] for r in rs]), 2)  # noqa: E731
            L.append(f"| {ag} | {cond} | {len(R)} | " + "/".join(_pct(mx[i] / len(R)) for i in range(5)) + " | "
                     + " | ".join(_pct(cc[c] / len(R)) for c in causes)
                     + f" | {pm('examiner_lines', R)} (ended: {pm('examiner_lines', E)}) | {pm('agent_share', R)} ({pm('agent_share', E)}) | "
                     f"{pm('agent_turn_s', R)} ({pm('agent_turn_s', E)}) | {pm('gap_s', R)} ({pm('gap_s', E)}) |")
    return L
