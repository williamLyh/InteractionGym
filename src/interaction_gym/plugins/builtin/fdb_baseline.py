"""Full-Duplex-Bench v1.0 / v1.5, Easy Turn and HumDial-FDBench as plugins (``examples/fdb_baseline.py`` as a
composition): every sample is replayed exactly as the benchmark streams it — the agent's microphone carries
``input.wav`` sample for sample, the episode lasts exactly its length and never ends early on silence
(``full_duplex_bench.make_env``) — with no system prompt, in lockstep.

- ``task/fdb_samples``: one task per (subset, sample, clean variant), loaded when its episode starts (``prepare``: the
  sample's turns and background to the role, its length to the kernel limits);
- ``role/fdb_sample``: the replayed benchmark speaker (``ReplayUser`` on the loaded sample's turns);
- ``scorer/fdb_sample_metrics``: the benchmark's per-sample metrics (``eval.benchmark``);
- ``report/fdb_baseline``: the official per-subset metrics, optional judges, ``summary.json`` and pages (``ig report``).

The driver groups the output by subset (``group_by: benchmark.subset`` → ``<out>/<subset>/episodes.jsonl``), which is
what the report reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from ...core import Task
from ...user import ReplayUser
from ..registry import PluginError, plugin


@dataclass
class FdbSamplesConfig:
    data: str  # the unpacked benchmark root
    bench: str = "fdb"  # fdb | easy_turn | humdial
    subsets: list | None = None  # default: every subset of the bench
    limit: int | None = None  # samples per subset
    spread: bool = False  # with limit: evenly spaced samples instead of the first ones
    clean: bool = False  # v1.5 subsets: also the clean_input.wav variant (for the behaviour judge)
    sr: int = 16000
    exact: bool = True  # everything outside the turns as a background track: the microphone carries input.wav sample for sample


class FdbSamples:
    def __init__(self, cfg: FdbSamplesConfig):
        from ...benchmarks import easy_turn as et
        from ...benchmarks import full_duplex_bench as fdb
        from ...benchmarks import humdial as hd
        from ...benchmarks.reports.fdb_baseline import BENCHES

        if cfg.bench not in BENCHES:
            raise PluginError(f"task/fdb_samples: bench is one of {', '.join(BENCHES)}, got {cfg.bench!r}")
        self.cfg, self.fdb, self.et, self.hd = cfg, fdb, et, hd
        self.subsets = list(cfg.subsets or BENCHES[cfg.bench][1])
        unknown = [s for s in self.subsets if s not in BENCHES[cfg.bench][1]]
        if unknown:
            raise PluginError(f"task/fdb_samples: unknown subsets {unknown} for bench {cfg.bench!r}")
        self.et_cache: dict = {}

    def is_fdb(self, subset: str) -> bool:
        return subset in self.fdb.SUBSETS

    def ids(self, subset: str) -> list[str]:
        """The subset's sample ids (``limit`` / ``spread`` applied per subset); a subset absent from ``data`` has none."""
        root = Path(self.cfg.data)
        try:
            ids = self.fdb._ids(root / subset) if self.is_fdb(subset) else self.et.ids(root, subset) if subset in self.et.SUBSETS else self.hd.ids(root, subset)
        except FileNotFoundError:
            return []
        if self.cfg.spread and self.cfg.limit and len(ids) > self.cfg.limit:  # evenly spaced, so a small run covers all speakers
            step = len(ids) / self.cfg.limit
            return [ids[int(k * step)] for k in range(self.cfg.limit)]
        return ids[: self.cfg.limit]

    def eid(self, subset: str, sid: str, clean: bool) -> str:
        from ...benchmarks.reports.fdb_baseline import _eid

        return _eid(subset, sid, clean)

    def tasks(self) -> list[Task]:
        out = []
        for subset in self.subsets:
            variants = [False, True] if self.cfg.clean and self.is_fdb(subset) and self.fdb.SUBSETS[subset][0] == "1.5" else [False]
            for sid in self.ids(subset):
                for clean in variants:
                    # a placeholder until the episode starts: loading a sample reads its audio
                    out.append(Task(self.eid(subset, sid, clean), {"benchmark": {"subset": subset, "sample_id": sid, "clean": clean, "bench": self.cfg.bench}}))
        return out

    def load(self, subset: str, sid: str, clean: bool):
        root = Path(self.cfg.data)
        if self.is_fdb(subset):
            return self.fdb.load_sample(root, subset, sid, sr=self.cfg.sr, exact=self.cfg.exact, clean=clean)
        if subset in self.et.SUBSETS:
            return self.et.load_sample(root, subset, sid, sr=self.cfg.sr, _cache=self.et_cache)
        return self.hd.load_sample(root, subset, sid, sr=self.cfg.sr, exact=self.cfg.exact, clean=clean)

    def prepare(self, task: Task, ep) -> Task:
        b = task.scenario.get("benchmark", {})
        if "bench" not in b or ep.data.get("sample") is not None:
            return task
        sample = self.load(b["subset"], b["sample_id"], b["clean"])
        ep.data["sample"] = sample
        horizon = sample.duration_ms
        ep.env.update(max_ms=horizon, end_idle_ms=horizon + 1, background=list(sample.background))  # exactly input.wav long
        return sample.task


@plugin("task", "fdb_samples", config=FdbSamplesConfig)
def fdb_samples(ctx, cfg: FdbSamplesConfig) -> FdbSamples:
    """Full-Duplex-Bench v1.0 / v1.5, Easy Turn or HumDial-FDBench samples, replayed exactly as released."""
    return FdbSamples(cfg)


@dataclass
class FdbSampleRoleConfig:
    stream: str = "user.speech"


@plugin("role", "fdb_sample", config=FdbSampleRoleConfig)
def fdb_sample(ctx, cfg: FdbSampleRoleConfig):
    """The replayed benchmark speaker: the sample's turns at their recorded times (open loop)."""
    def make(ep):
        sample = (ep.data.get("sample") if ep is not None else None)
        if sample is None:
            raise PluginError("role/fdb_sample needs a task/fdb_samples row: the episode's sample was not loaded")
        return ReplayUser(sample.turns, stream=cfg.stream)
    return make


@plugin("scorer", "fdb_sample_metrics")
def fdb_sample_metrics(ctx, kw: dict):
    """The benchmark's per-sample metrics for a Full-Duplex-Bench episode (``full_duplex_bench.sample_metrics``; the
    agent's word times from its audio when the run keeps media); ``{}`` for Easy Turn and HumDial."""
    from ...benchmarks import full_duplex_bench as fdb

    def score(ep, env, agent) -> dict:
        b = fdb.bench_meta(ep)
        if b.get("subset") not in fdb.SUBSETS:
            return {}
        media = None
        if ep["meta"].get("media_root"):
            from ...media import MediaStore

            media = MediaStore(ep["meta"]["media_root"])
        return fdb.sample_metrics(ep, media)
    return score


@dataclass
class FdbReportConfig:
    tasks: str = "tasks"  # the task/fdb_samples row (its data and bench)
    judge: str | None = None  # row id of a backend/chat for the v1.0 interruption ratings and v1.5 behaviour tags
    judge_name: str = ""  # recorded as the judge in summary.json
    repo: str = ""  # a checkout of the official repo (the behaviour judge's instructions)
    gt: str = ""  # icc_gt_distribution.json from the official repo
    rejudge: bool = False  # without a judge, earlier verdicts in per_sample.json are reused unless set
    pages: int = 3  # episodes per subset in the browsable run


class FdbReport:
    def __init__(self, data: str, bench: str, judge, cfg: FdbReportConfig):
        self.data, self.bench, self.judge, self.cfg = data, bench, judge, cfg

    async def run(self, out: Path, tag: str = "", log=print) -> None:
        from ...benchmarks.reports.fdb_baseline import report

        args = SimpleNamespace(data=self.data, bench=self.bench, out=str(out), repo=self.cfg.repo, gt=self.cfg.gt,
                               judge_model=self.cfg.judge_name or (self.cfg.judge or ""), rejudge=self.cfg.rejudge, pages=self.cfg.pages)
        await report(args, self.judge)


@plugin("report", "fdb_baseline", config=FdbReportConfig, refs=("tasks", "judge"))
def fdb_report(ctx, cfg: FdbReportConfig) -> FdbReport:
    """Official per-subset metrics (+ eval.scores summary, optional LLM judges) → summary.json, per_sample.json, pages."""
    tcfg = ctx.resolved.configs.get(cfg.tasks)
    if tcfg is None or not hasattr(tcfg, "bench"):
        raise PluginError(f"report/fdb_baseline: {cfg.tasks!r} is not a task/fdb_samples row")
    return FdbReport(tcfg.data, tcfg.bench, ctx.get(cfg.judge), cfg)
