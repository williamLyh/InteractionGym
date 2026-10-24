"""Task sources: where a run's episodes come from."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...core import Task
from ..registry import PluginError, plugin


@dataclass
class InlineConfig:
    tasks: list  # [{"id", "scenario", "initial_state", "criteria"}]


@plugin("task", "inline", config=InlineConfig)
def inline(ctx, cfg: InlineConfig) -> list:
    """Tasks written in the configuration itself."""
    return list(cfg.tasks)


@dataclass
class AudiomcConfig:
    data: str  # the extracted dataset root (index.json + wav), see benchmarks.audiomc.extract
    per_axis: int | None = 25  # conversations per axis (every k-th in index order); None: all
    axes: list | None = None  # subset of benchmarks.audiomc.AXES
    ids: list | None = None


@plugin("task", "audiomc", config=AudiomcConfig)
def audiomc(ctx, cfg: AudiomcConfig) -> list[Task]:
    """Audio MultiChallenge conversations (docs/BENCHMARKS.md §13): recorded user turns + rubric."""
    from ...benchmarks import audiomc as amc

    return amc.load(cfg.data, axes=cfg.axes, only=cfg.ids, per_axis=cfg.per_axis)


@dataclass
class JsonlConfig:
    path: str
    ids: list | None = None  # only these task ids
    limit: int = 0


@plugin("task", "jsonl", config=JsonlConfig)
def jsonl(ctx, cfg: JsonlConfig) -> list[Task]:
    """One task per line: ``{"id", "scenario", "initial_state", "criteria"}``."""
    p = Path(cfg.path)
    if not p.exists():
        raise PluginError(f"task/jsonl: no file {p}")
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            if cfg.ids and d.get("id") not in cfg.ids:
                continue
            out.append(Task(d.get("id", f"task-{len(out)}"), d.get("scenario", {}), d.get("initial_state", {}), d.get("criteria", {})))
            if cfg.limit and len(out) >= cfg.limit:
                break
    return out
