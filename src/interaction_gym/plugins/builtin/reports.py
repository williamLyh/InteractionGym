"""Reports: post-hoc scoring of a run's episodes (judges, tables, pages), run by ``ig report <bundle> --out DIR``.
Each returns an object with ``async run(out, tag, log)``; the judge is a ``backend/chat`` row named by ``judge``
(a local proxy LLM, or the hosted API). They call ``interaction_gym.benchmarks.reports.*``, the same functions the
``examples/*_ab.py report`` commands run, on the layout the driver writes (``<out>/<tag>/<arm>/episodes.jsonl``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from ..registry import PluginError, plugin


class Report:
    kind = ""

    async def run(self, out: Path, tag: str, log=print) -> None:  # pragma: no cover - interface
        raise NotImplementedError


def _task_data(ctx, row: str) -> str:
    cfg = ctx.resolved.configs.get(row)
    data = getattr(cfg, "data", None)
    if data is None:
        raise PluginError(f"report: row {row!r} is not a task row with a `data` field")
    return data


@dataclass
class AudiomcReportConfig:
    judge: str  # row id of a backend/chat
    tasks: str = "tasks"  # the task/audiomc row (its data holds the rubric)
    concurrency: int = 32


class AudiomcReport(Report):
    def __init__(self, judge, data: str, cfg: AudiomcReportConfig):
        self.judge, self.data, self.cfg = judge, data, cfg

    async def run(self, out: Path, tag: str = "", log=print) -> None:
        from ...benchmarks.reports.audiomc import report

        await report(SimpleNamespace(out=str(out), data=self.data, concurrency=self.cfg.concurrency), self.judge)


@plugin("report", "audiomc", config=AudiomcReportConfig, refs=("judge", "tasks"))
def audiomc_report(ctx, cfg: AudiomcReportConfig) -> AudiomcReport:
    """Audio MultiChallenge: rubric judge per item, open-loop validity, timing, A = B consistency → rows.json, summary.md."""
    return AudiomcReport(ctx[cfg.judge], _task_data(ctx, cfg.tasks), cfg)


@dataclass
class Fdb2ReportConfig:
    judge: str
    tasks: str = "tasks"  # the task/fdb2 row
    concurrency: int = 32
    window_ms: int = 0  # score every conversation on its first window_ms (0: its run's cap)
    name: str = ""  # suffix of the rows / summary file names


class Fdb2Report(Report):
    def __init__(self, judge, data: str, cfg: Fdb2ReportConfig):
        self.judge, self.data, self.cfg = judge, data, cfg

    async def run(self, out: Path, tag: str = "", log=print) -> None:
        from ...benchmarks.reports.fdb2 import report

        await report(SimpleNamespace(out=str(out), data=self.data, concurrency=self.cfg.concurrency, window_ms=self.cfg.window_ms, name=self.cfg.name), self.judge)


@plugin("report", "fdb2", config=Fdb2ReportConfig, refs=("judge", "tasks"))
def fdb2_report(ctx, cfg: Fdb2ReportConfig) -> Fdb2Report:
    """FD-Bench v2: the official judge (TT / IF / task), coherence, validity, stages, cap diagnosis → rows.json, summary.md."""
    return Fdb2Report(ctx[cfg.judge], _task_data(ctx, cfg.tasks), cfg)


@dataclass
class Fdb3ReportConfig:
    judge: str
    tags: list | None = None  # run folders to score, the main one first (default: the run's tag)
    duplex: list | None = None  # agent names (the tag before "-ep") scored as spoken fulfilment (default: by the agent row's plugin)
    pages: bool = False  # viewer pages per tag


class Fdb3Report(Report):
    def __init__(self, judge, cfg: Fdb3ReportConfig, duplex_default: bool):
        self.judge, self.cfg, self.duplex_default = judge, cfg, duplex_default

    async def run(self, out: Path, tag: str = "", log=print) -> None:
        from ...benchmarks.reports.fdb3 import report

        tags = self.cfg.tags or ([tag] if tag else [])
        if not tags:
            raise PluginError("report/fdb3: give --tag (the run folder) or tags in the report row")
        duplex = set(self.cfg.duplex) if self.cfg.duplex is not None else ({t.split("-ep")[0] for t in tags} if self.duplex_default else set())
        await report(SimpleNamespace(out=str(out), tags=",".join(tags), pages=self.cfg.pages), self.judge, default_tag=tags[0], duplex_names=duplex)


@plugin("report", "fdb3", config=Fdb3ReportConfig, refs=("judge",))
def fdb3_report(ctx, cfg: Fdb3ReportConfig) -> Fdb3Report:
    """FD-Bench v3: Pass@1 (official / lenient), why open-loop runs failed, user effort, timing, consistency; spoken
    fulfilment for duplex agents → rows.json per tag, summary.md, optional pages."""
    agent_row = ctx.resolved.provides.get("agent/vllm_omni") or ctx.resolved.provides.get("agent/openai_realtime")
    duplex_default = agent_row is not None and ctx.resolved.plugins.get(agent_row) is not None
    return Fdb3Report(ctx[cfg.judge], cfg, duplex_default)
