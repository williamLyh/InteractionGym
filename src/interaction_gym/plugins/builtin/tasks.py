"""Task sources: where a run's episodes come from."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
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
class Fdb2Config:
    data: str  # prompts_staged_200.json
    splits: list | None = None
    ids: list | None = None
    scripts: str | None = None  # the reference scripts' folder (default <out>/scripts), shared by every agent's run
    max_turns: int = 12  # examiner lines per script / live episode
    max_ms: int = 120000  # the official cap (both arms); meta.max_ms of the live arm
    a_max_ms: int | None = None  # the replayed arm's cap (default max_ms): lines starting later are dropped
    arms: dict = field(default_factory=lambda: {"A": "replay", "B": "live"})  # arm name -> "replay" | "live"
    examiner_llm: str = "examiner_llm"  # rows used by `ig prepare`: the examiner and the reference assistant
    assistant_llm: str = "reference_llm"
    tts: str = "tts"  # the examiner's voice: `voice` on tts, later lines cloned from the first (clone)
    clone: str | None = "clone_tts"
    voice: str = "serena"
    retries: int = 4  # `ig prepare`: attempts per script
    retry_wait_s: float = 20.0


class Fdb2Tasks:
    """Full-Duplex-Bench v2 as a task source: ``tasks()`` loads the examiner tasks; ``build`` (``ig prepare``) writes
    one agent-free reference script per (task, seed) — examiner / assistant lines by two LLMs, the examiner's lines
    rendered once and scheduled (``fdb2.make_script`` / ``schedule``) — to ``<scripts>/<task>/s<seed>/script.json``;
    ``prepare`` gives each episode its arm's view: the replayed arm speaks the script's lines at their times (capped
    at ``a_max_ms``; the episode ends 15 s after the script), the live arm starts from the script's first line."""

    out: Path = Path(".")

    def __init__(self, ctx, cfg: Fdb2Config):
        from ...benchmarks import fdb2

        self.cfg, self.fdb2 = cfg, fdb2
        self.examiner_llm, self.assistant_llm = ctx[cfg.examiner_llm], ctx[cfg.assistant_llm]
        self.tts, self.clone = ctx[cfg.tts], ctx.get(cfg.clone)

    def tasks(self) -> list[Task]:
        return self.fdb2.load(self.cfg.data, splits=self.cfg.splits, only=self.cfg.ids)

    def script_dir(self, task_id: str, seed: int) -> Path:
        return (Path(self.cfg.scripts) if self.cfg.scripts else Path(self.out) / "scripts") / task_id / f"s{seed}"

    @property
    def a_max_ms(self) -> int:
        return self.cfg.a_max_ms or self.cfg.max_ms

    def prepare(self, task: Task, ep) -> Task:
        mode = self.cfg.arms.get(ep.arm) if ep.arm else None
        if mode is None:
            return task
        if mode not in ("replay", "live"):
            raise PluginError(f"task/fdb2: arm {ep.arm!r} is {mode!r}; arms map names to 'replay' or 'live'")
        d = self.script_dir(task.id, ep.seed)
        if not (d / "script.json").exists():
            raise PluginError(f"task/fdb2: no reference script for {task.id} s{ep.seed} ({d}); run `ig prepare` first")
        rec = json.loads((d / "script.json").read_text())
        t = copy.deepcopy(task)
        ex = [x for x in rec["lines"] if x["role"] == "examiner"]
        if mode == "replay":
            kept = [x for x in ex if x["t"] < self.a_max_ms]
            t.scenario["turns"] = [{"t": x["t"], "text": x["text"], "audio": str(d / x["audio"]), "final": i == len(kept) - 1} for i, x in enumerate(kept)]
            ep.env["max_ms"] = min(rec["script_end_ms"] + 15_000, self.a_max_ms)
            ep.meta.update(max_ms=self.a_max_ms, closing=False)
        else:
            t.scenario["first_turn"] = {"text": ex[0]["text"], "audio": str(d / ex[0]["audio"])}
            ep.meta.update(max_ms=self.cfg.max_ms)
        return t

    async def build_one(self, task: Task, seed: int) -> dict:
        from ...user import UserTurn, Voice
        from ..registry import CURRENT_EPISODE, EpisodeContext

        d = self.script_dir(task.id, seed)
        if (d / "script.json").exists():
            return json.loads((d / "script.json").read_text())
        token = CURRENT_EPISODE.set(EpisodeContext(task, seed, None, {}, f"prepare/{task.id}/s{seed}"))
        try:
            lines = await self.fdb2.make_script(task, self.examiner_llm, self.assistant_llm, self.cfg.max_turns)
            d.mkdir(parents=True, exist_ok=True)
            voice = Voice(self.tts, voice=self.cfg.voice, clone=self.clone)
            durs, k, ref = [], 0, None
            for x in lines:
                if x["role"] != "examiner":
                    continue
                seg = await voice.render(f"e{k}", 0, UserTurn(x["text"]), task, ref)
                ref = voice.reference(seg, ref)
                seg.data.write_wav(d / f"e{k:02d}.wav")
                x["audio"] = f"e{k:02d}.wav"
                durs.append(seg.data.dur_ms)
                k += 1
        finally:
            CURRENT_EPISODE.reset(token)
        starts = self.fdb2.schedule(lines, durs)
        ex = [x for x in lines if x["role"] == "examiner"]
        for x, t0, dur in zip(ex, starts, durs):
            x["t"], x["dur"] = t0, dur
        rec = {"task": task.id, "seed": seed, "lines": lines, "reached_end": self.fdb2._end(ex[-1]["text"]) if ex else False,
               "script_end_ms": (starts[-1] + durs[-1]) if ex else 0}
        (d / "script.json").write_text(json.dumps(rec, indent=1))
        return rec

    async def build(self, tasks: list[Task], seeds, *, concurrency: int = 4, log=print) -> int:
        import asyncio
        import random

        sem, n = asyncio.Semaphore(concurrency), [0]

        async def one(task, seed):
            async with sem:
                for attempt in range(max(1, self.cfg.retries)):
                    try:
                        rec = await self.build_one(task, seed)
                        break
                    except Exception as ex:  # noqa: BLE001 — a service restarting: wait and retry
                        log(f"  {task.id} s{seed} attempt {attempt + 1}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                        if attempt + 1 >= max(1, self.cfg.retries):
                            return
                        await asyncio.sleep(self.cfg.retry_wait_s + random.random() * self.cfg.retry_wait_s)
                n[0] += 1
                ex = [x for x in rec["lines"] if x["role"] == "examiner"]
                log(f"  {n[0]} {task.id} s{seed}: {len(ex)} examiner lines, end={rec['reached_end']}, {rec['script_end_ms'] / 1000:.0f}s"
                    + (f" | {ex[0]['text'][:80]}" if ex else ""), flush=True)

        await asyncio.gather(*(one(t, s) for s in seeds for t in tasks))
        return n[0]


@plugin("task", "fdb2", config=Fdb2Config, refs=("examiner_llm", "assistant_llm", "tts", "clone"))
def fdb2(ctx, cfg: Fdb2Config) -> Fdb2Tasks:
    """Full-Duplex-Bench v2 (docs/BENCHMARKS.md §12): examiner tasks; reference scripts by `ig prepare`; per-arm views."""
    return Fdb2Tasks(ctx, cfg)


@dataclass
class Fdb3Config:
    data: str  # the recordings' root: <id>/metadata.json + input.wav|opus|flac
    ids: list | None = None
    sr: int = 16000
    arms: dict = field(default_factory=lambda: {"A": "replay", "B": "live"})  # arm name -> "replay" | "live"
    replay_tail_ms: int = 60000  # the replayed arm's cap: the request's end + this
    live_max_ms: int = 180000  # the live arm's cap


class Fdb3Tasks:
    """Full-Duplex-Bench v3 as a task source: one task per recording (``fdb3.load``); the replayed arm ends
    ``replay_tail_ms`` after the recorded request, the live arm at ``live_max_ms``; ``meta.recording``."""

    def __init__(self, cfg: Fdb3Config):
        from ...benchmarks import fdb3

        self.cfg, self.fdb3 = cfg, fdb3

    def tasks(self) -> list[Task]:
        out = []
        for _rec, task in self.fdb3.load(self.cfg.data, only=self.cfg.ids, sr=self.cfg.sr):
            task.scenario["profile"] = {"language": "en"}
            out.append(task)
        return out

    def prepare(self, task: Task, ep) -> Task:
        ep.meta["recording"] = task.id
        mode = self.cfg.arms.get(ep.arm) if ep.arm else None
        if mode == "replay":
            ep.env["max_ms"] = task.scenario["benchmark"]["user_end_ms"] + self.cfg.replay_tail_ms
        elif mode == "live":
            ep.env["max_ms"] = self.cfg.live_max_ms
        elif mode is not None:
            raise PluginError(f"task/fdb3: arm {ep.arm!r} is {mode!r}; arms map names to 'replay' or 'live'")
        return task


@plugin("task", "fdb3", config=Fdb3Config)
def fdb3(ctx, cfg: Fdb3Config) -> Fdb3Tasks:
    """Full-Duplex-Bench v3 (docs/BENCHMARKS.md §11): recorded tool-use requests, open vs closed loop."""
    return Fdb3Tasks(cfg)


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
