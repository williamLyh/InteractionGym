"""The spoken SOP service calls as plugins (``examples/sop_calls.py`` as a composition): the SOP package serves three
seams — ``task/sop_calls`` (the scenarios: a caller, the business's SOP and its case; per-episode ``meta.policy`` and
the verifier), the controlled role's instructions (``agent/* instructions_by_task:
interaction_gym.sop_task:agent_instructions``, the SOP rendered in a format), ``scorer/sop_judge`` (an LLM reports
what happened, rules on the SOP tree score it → ``eval.reward``, ``meta.sop_judge``); plus ``sink/sop_pages`` (the
run's notes, judge results and viewer pages) and ``report/sop_rejudge`` (score a run's episodes again)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...core import Task
from ... import sop_task as st
from ..registry import PluginError, plugin


@dataclass
class SopCallsConfig:
    path: str | None = None  # a scenarios file (default: the shipped interaction_gym/data/sop_calls.json)
    only: list | None = None
    format: str = "structured"  # how the SOP is rendered for the agent (sop.formats()) and recorded as meta.policy
    max_ms: int = 180000  # the cap of scenarios that set none


class SopCalls:
    def __init__(self, cfg: SopCallsConfig):
        from ...sop import formats

        if cfg.format not in formats():
            raise PluginError(f"task/sop_calls: unknown SOP format {cfg.format!r}; available: {', '.join(sorted(formats()))}")
        self.cfg = cfg
        self.scenarios = st.scenarios(cfg.path, cfg.only)
        if cfg.only and set(cfg.only) - set(self.scenarios):
            raise PluginError(f"task/sop_calls: unknown scenarios {sorted(set(cfg.only) - set(self.scenarios))}; known: {', '.join(st.scenarios(cfg.path))}")

    def tasks(self) -> list[Task]:
        return [st.task_of(name, sc) for name, sc in self.scenarios.items()]

    def prepare(self, task: Task, ep) -> Task:
        sc = self.scenarios.get(task.id)
        if sc is None:
            return task
        ep.env["max_ms"] = sc.get("max_ms", self.cfg.max_ms)
        ep.meta["policy"] = st.policy(task, self.cfg.format)
        ep.verifier = st.VERIFIER
        return task


@plugin("task", "sop_calls", config=SopCallsConfig)
def sop_calls(ctx, cfg: SopCallsConfig) -> SopCalls:
    """Service calls with an SOP: a caller with a request, the business's SOP and the case's ground truth."""
    return SopCalls(cfg)


@dataclass
class SopJudgeConfig:
    judge: str  # row id of a backend/chat (a reasoning-capable model; long outputs)


@plugin("scorer", "sop_judge", config=SopJudgeConfig, refs=("judge",))
def sop_judge(ctx, cfg: SopJudgeConfig):
    """``sop_task.judge``: an LLM reports the decision, what was established and the details; rules on the SOP tree
    score it. Writes ``eval.reward`` and ``meta.sop_judge`` (an error leaves the reward uncomputed, the episode kept)."""
    llm = ctx[cfg.judge]

    async def score(ep: dict, env, agent):
        task = Task(**{k: ep["meta"]["task"].get(k) or {} for k in ("scenario", "initial_state", "criteria")}, id=ep["meta"]["task"]["id"])
        try:
            res = await st.judge(ep, task, llm)
        except Exception as e:  # noqa: BLE001 — keep the episode; it is scored as not computed
            res = {"reward": {"total": None, "parts": {}}, "error": f"{e.__class__.__name__}: {e}"}
        ep["eval"]["reward"] = res["reward"]
        ep["meta"]["sop_judge"] = {k: v for k, v in res.items() if k != "reward"}
        return None
    return score


def note_of(ep: dict) -> str:
    """One line per episode for the run's pages: the SOP, the requested action, what should happen, the decision, the reward."""
    task = Task(**{k: ep["meta"]["task"].get(k) or {} for k in ("scenario", "initial_state", "criteria")}, id=ep["meta"]["task"]["id"])
    j, sc = ep["meta"].get("sop_judge", {}), task.scenario
    want = "do it" if st.should_do(task) else "decline"
    rw = ep["eval"].get("reward") or {}
    return (f"SOP {sc['sop']['name']} · {sc['sop_case']['action']} · should {want} · decision {j.get('decision')} · "
            f"reward {rw.get('total')} {json.dumps(rw.get('parts', {}))} · {ep['meta']['duration_ms'] / 1000:.0f}s"
            + (" · truncated" if ep["meta"].get("end_reason") == "max_duration" else ""))


@dataclass
class SopPagesConfig:
    title: str = "SOP service calls"


class SopPages:
    out: Path = Path(".")

    def __init__(self, cfg: SopPagesConfig):
        self.cfg, self.episodes, self.traces, self.notes, self.results = cfg, [], [], {}, {}

    def write(self, ep, env, agent) -> None:
        eid = ep["meta"]["episode_id"]
        self.episodes.append(ep)
        trace = getattr(agent, "trace", None)
        tr = trace(eid) if callable(trace) else None
        if tr:
            self.traces.append(tr)
        self.notes[eid] = note_of(ep)
        self.results[eid] = {**ep["meta"].get("sop_judge", {}), "reward": ep["eval"].get("reward")}

    def close(self) -> None:
        if not self.episodes:
            return
        from ...viewer import export_run

        (self.out / "notes.json").write_text(json.dumps(self.notes, indent=1))
        (self.out / "judge.json").write_text(json.dumps(self.results, indent=1, ensure_ascii=False))
        (self.out / "qa.json").write_text("{}")
        export_run(self.episodes, self.out, self.notes, title=self.cfg.title, traces=self.traces)


@plugin("sink", "sop_pages", config=SopPagesConfig)
def sop_pages(ctx, cfg: SopPagesConfig) -> SopPages:
    """The run's notes.json, judge.json and viewer pages (``viewer.export_run``), as the script wrote them."""
    return SopPages(cfg)


class SopRejudge:
    def __init__(self, llm):
        self.llm = llm

    async def run(self, out: Path, tag: str = "", log=print) -> None:
        import asyncio

        from ...traj import load, save

        path = Path(out) / "episodes.jsonl"
        if not path.exists():
            raise PluginError(f"report/sop_rejudge: no {path}")
        eps = load(path)
        tasks = [Task(**{k: e["meta"]["task"].get(k) or {} for k in ("scenario", "initial_state", "criteria")}, id=e["meta"]["task"]["id"]) for e in eps]

        async def one(e, t):
            try:
                res = await st.judge(e, t, self.llm)
            except Exception as ex:  # noqa: BLE001
                res = {"reward": {"total": None, "parts": {}}, "error": f"{ex.__class__.__name__}: {ex}"}
            e["eval"]["reward"] = res["reward"]
            e["meta"]["sop_judge"] = {k: v for k, v in res.items() if k != "reward"}
            return res

        res = await asyncio.gather(*(one(e, t) for e, t in zip(eps, tasks)))
        for e, r in zip(eps, res):
            log(f"{e['meta']['episode_id']:22s} decision {r.get('decision')} reward {r['reward']['total']} {json.dumps(r['reward']['parts'])} {r.get('error', '')}")
        save(eps, path)
        (Path(out) / "judge.json").write_text(json.dumps({e["meta"]["episode_id"]: r for e, r in zip(eps, res)}, indent=1, ensure_ascii=False))


@plugin("report", "sop_rejudge", config=SopJudgeConfig, refs=("judge",))
def sop_rejudge(ctx, cfg: SopJudgeConfig) -> SopRejudge:
    """Score a run's episodes again in place (after a judge change): ``ig report sop_calls --out DIR``."""
    return SopRejudge(ctx[cfg.judge])
