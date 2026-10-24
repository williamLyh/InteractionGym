"""Sinks: where finished episodes go. Paths are relative to the run's output directory (``ig run --out``), which the
driver sets on every sink as ``sink.out`` before the first episode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ...traj import save
from ..registry import plugin


class Sink:
    out: Path = Path(".")

    def write(self, ep: dict, env, agent) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass


@dataclass
class JsonlSinkConfig:
    path: str = "episodes.jsonl"


class JsonlSink(Sink):
    def __init__(self, cfg: JsonlSinkConfig):
        self.cfg = cfg

    @property
    def path(self) -> Path:
        return self.out / self.cfg.path

    def done(self) -> set[str]:
        """Episode ids already in the file (a resumed run skips them)."""
        if not self.path.exists():
            return set()
        return {json.loads(line)["meta"]["episode_id"] for line in self.path.read_text().splitlines() if line.strip()}

    def write(self, ep, env, agent) -> None:
        save([ep], self.path, append=True)


@plugin("sink", "jsonl", config=JsonlSinkConfig)
def jsonl(ctx, cfg: JsonlSinkConfig) -> JsonlSink:
    """Appends each episode to a JSON-lines file (docs/FORMAT.md); a rerun with the same output resumes from it."""
    return JsonlSink(cfg)


@dataclass
class HtmlSinkConfig:
    path: str = "index.html"


class HtmlSink(Sink):
    def __init__(self, cfg: HtmlSinkConfig):
        self.cfg = cfg
        self.episodes: list[dict] = []

    def write(self, ep, env, agent) -> None:
        self.episodes.append(ep)

    def close(self) -> None:
        if self.episodes:
            from ...viewer import export_html

            export_html(self.episodes, self.out / self.cfg.path)


@plugin("sink", "html", config=HtmlSinkConfig)
def html(ctx, cfg: HtmlSinkConfig) -> HtmlSink:
    """Writes the viewer page of the run's episodes when the run ends."""
    return HtmlSink(cfg)


@dataclass
class TracesSinkConfig:
    path: str = "agent_traces.jsonl"


class TracesSink(Sink):
    """The agent adapter's own trace of each episode (docs/AGENT_TRACE.md), when it has one."""

    def __init__(self, cfg: TracesSinkConfig):
        self.cfg = cfg

    def write(self, ep, env, agent) -> None:
        trace = getattr(agent, "trace", None)
        tr = trace(ep["meta"]["episode_id"]) if callable(trace) else None
        if tr:
            save([tr], self.out / self.cfg.path, append=True)


@plugin("sink", "agent_traces", config=TracesSinkConfig)
def agent_traces(ctx, cfg: TracesSinkConfig) -> TracesSink:
    """Appends ``agent.trace(episode_id)`` (unit-level internals of a server adapter) to a JSON-lines file."""
    return TracesSink(cfg)


class StdoutSink(Sink):
    def write(self, ep, env, agent) -> None:
        print(f"== {ep['meta']['episode_id']}  reward {ep['eval']['reward']['total']}  scores {ep['eval']['scores']['total']}")
        for t in ep["turns"]:
            cut = " [cut]" if "unsaid" in t else ""
            kind = f" ({t['kind']})" if t.get("kind") else ""
            print(f"  {t['start_time']:6d}-{t['end_time']:6d} {t['role']:6s}{kind} {t['text'][:100]!r}{cut}")


@plugin("sink", "stdout")
def stdout(ctx, kw: dict) -> StdoutSink:
    """Prints each episode's turns."""
    return StdoutSink()
