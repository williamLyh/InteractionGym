"""Turn-indexed pass metrics for the open- vs closed-loop benchmarks (results/BENCHMARKS.md, "Pass by user turn").

    PYTHONPATH=extras/fdbench/src:src:examples:. python results/turn_metrics.py --runs <runs> --out <runs>/turn_metrics.json

Definitions (N = 1, 2, 3, ...; a conversation's user turns are numbered from 1 in start order):

- **cut(N)**: the start of user turn N + 1, or the end of the conversation if it has at most N user turns. "The
  conversation up to the agent's response after the N-th user turn" is everything that starts before cut(N).
- **FD-Bench v3, pass@N-turn** (MiniCPM-o, no tools): spoken fulfilment (``benchmarks.fdb3_spoken.outcome``: every
  parameter of the request confirmed in the agent's speech with its final value, rules + proxy-LLM fallback) over
  the agent turns that start before cut(N). pass@1turn is what the open loop measures: in A the recording is the only
  user turn, so A's score is pass@1turn of A (and B's pass@1turn should equal it: the two conditions are identical
  until the user's second turn). A conversation that ended before turn N + 1 keeps its final value (carried forward).
- **FD-Bench v2, stages@N-turn**: the stage analysis of the report (``fdb2.stage_prompt`` / ``parse_stages``, one
  proxy-LLM call at T = 0) on the transcript up to cut(N), N counting the examiner's lines: number of staged goals
  T1-T4 completed, and the stage score (mean agent score over the reached goals). Computed for A (the replayed
  script's lines) and B (the live examiner), up to ``--fdb2-max-n`` lines and the whole conversation.
- **Audio MultiChallenge**: not turn-indexed. Its rubric grades the reply to the last user turn only, and A and B
  have the same number of user turns, so a per-turn curve has nothing to grade before the last turn.

Writes one record per conversation (and condition) to ``--out``; results/aggregate.py turns them into the curves.
Judge calls are memoized in the reports' caches (fdb3_ab/judge_memo.json, fdb2_ab/judge_cache.json), so a rerun is cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIMIT = 0  # --limit
sys.path[:0] = [str(HERE.parent / "examples"), str(HERE.parent / "src")]


def load(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def cuts(users: list[dict], n_max: int) -> list[int | None]:
    """cut(N) for N = 1..min(n_max, len(users)): the start of user turn N + 1, None (= the whole conversation) after
    the last user turn."""
    return [users[n]["start_time"] if n < len(users) else None for n in range(1, min(n_max, len(users)) + 1)]


async def fdb3_rows(root: Path, judge_url: str, judge_model: str, n_max: int) -> list[dict]:
    import fdb3_ab
    from interaction_gym.benchmarks import fdb3
    from interaction_gym.benchmarks import fdb3_spoken as spoken
    from interaction_gym.clients import OpenAIChat

    out = root / "fdb3_ab"
    tag = "minicpmo"
    if not (out / tag / "A" / "episodes.jsonl").exists():
        return []
    judge = fdb3_ab.Judge(OpenAIChat(judge_url, judge_model, max_tokens=300))
    memo = out / "judge_memo.json"
    if memo.exists():
        judge.memo = json.loads(memo.read_text())
    key = lambda e: e["meta"]["episode_id"].split("/", 1)[1].rsplit("/", 1)[0]  # noqa: E731
    A = {key(e): e for e in load(out / tag / "A" / "episodes.jsonl")}
    B = {key(e): e for e in load(out / tag / "B" / "episodes.jsonl")}
    sem = asyncio.Semaphore(48)

    async def one(k):
        a, b = A[k], B.get(k)
        async with sem:
            slots = spoken.slots_of(a["meta"]["task"])
            bench = a["meta"]["task"]["scenario"]["benchmark"]
            row = {"key": k, "recording": bench["recording"], "seed": a["meta"].get("seed"), "difficulty": bench["difficulty"],
                   "A": (await spoken.outcome(a, judge, slots=slots))["fulfilled"]}
            if b is not None:
                users = fdb3.user_turns(b)
                row["n_users"] = len(users)
                row["B"] = [(await spoken.outcome(b, judge, before_ms=c, slots=slots))["fulfilled"] for c in cuts(users, n_max)]
            return row

    rows = await asyncio.gather(*(one(k) for k in sorted(A)[: LIMIT or None]))
    memo.write_text(json.dumps(judge.memo))
    return list(rows)


async def fdb2_rows(root: Path, data: str, judge_url: str, judge_model: str, n_max: int) -> list[dict]:
    import fdb2_ab
    from interaction_gym.benchmarks import fdb2

    out = root / "fdb2_ab"
    agents = [p.name for p in sorted(out.iterdir()) if (p / "A").exists() or (p / "B").exists()] if out.exists() else []
    if not agents:
        return []
    tasks = {t.id: t for t in fdb2.load(data)}
    judge = fdb2_ab.Judge(fdb2_ab.llm(0, 0.0, 1500), out / "judge_cache.json")
    sem = asyncio.Semaphore(48)

    async def one(ep):
        task = tasks[ep["meta"]["task_id"]]
        users = fdb2_ab._users(ep)
        cap = ep["meta"].get("max_ms") or fdb2.MAX_MS
        per = []
        async with sem:
            for c in cuts(users, n_max) + ([None] if len(users) > n_max else []):
                st = fdb2.parse_stages(await judge.ask(fdb2.stage_prompt(task, ep, min(c, cap) if c is not None else cap), max_tokens=700))
                per.append(None if st is None else {"completed": len(st["completed"]), "reached": len(st["reached"]), "stage_score": st["stage_score"]})
        return {"id": ep["meta"]["episode_id"], "agent": ep["meta"]["agent_name"], "task_id": task.id, "split": task.criteria["split"],
                "seed": ep["meta"].get("seed"), "cond": ep["meta"]["condition"], "n_users": len(users), "by_n": per,
                "final_is_extra": len(users) > n_max}

    eps = [e for ag in agents if ag == "minicpmo" for c in ("A", "B") for e in load(out / ag / c / "episodes.jsonl")[: LIMIT or None]]
    rows = await asyncio.gather(*(one(e) for e in eps))
    judge.flush()
    return list(rows)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fdb2-data", required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--judge-model", default="Qwen/Qwen3.8-27B-FP8")
    ap.add_argument("--fdb3-max-n", type=int, default=12)
    ap.add_argument("--fdb2-max-n", type=int, default=10)
    ap.add_argument("--only", default="fdb3,fdb2")
    ap.add_argument("--limit", type=int, default=0, help="at most this many conversations per benchmark (a quick check)")
    args = ap.parse_args()
    import os

    os.environ.setdefault("IG_LLM_URL", args.judge_url)
    os.environ.setdefault("IG_LLM_MODEL", args.judge_model)
    root = Path(args.runs)
    global LIMIT
    LIMIT = args.limit
    only = set(args.only.split(","))
    res = {}
    if "fdb3" in only:
        res["fdb3"] = await fdb3_rows(root, args.judge_url, args.judge_model, args.fdb3_max_n)
        print("fdb3 rows", len(res["fdb3"]), flush=True)
    if "fdb2" in only:
        res["fdb2"] = await fdb2_rows(root, args.fdb2_data, args.judge_url, args.judge_model, args.fdb2_max_n)
        print("fdb2 rows", len(res["fdb2"]), flush=True)
    res["max_n"] = {"fdb3": args.fdb3_max_n, "fdb2": args.fdb2_max_n}
    Path(args.out).write_text(json.dumps(res))


if __name__ == "__main__":
    asyncio.run(main())
