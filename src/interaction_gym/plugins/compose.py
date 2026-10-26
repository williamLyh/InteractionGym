"""Build a run from a resolved composition: each row's factory gets a ``Context`` (the objects of the rows built so
far, by row id and by service key) and its parsed config, and what it returns is sorted by kind into a ``Run``.

The run is what a driver executes: ``make_env(ep)`` gives a fresh ``Env`` over the simulated nodes for one episode,
``agent(ep)`` the external policy for the controlled role, ``tasks`` the episodes to run, ``scorers`` / ``sinks`` what
happens to each finished episode. Nothing here knows what the plugins do; the kinds are the contract
(plugins/registry.py).

A ``role`` / ``node`` plugin returns a ``Node`` (shared by every episode: its state must live in ``init_state``'s
state, as the kernel requires) or a factory ``(EpisodeContext) -> Node`` when the node keeps per-episode state on
itself (a benchmark examiner with a log, a closing tracker). A ``task`` plugin returns tasks, or a *task source* with
``tasks()`` and optionally ``prepare(task, ep) -> Task`` (the episode's view of a task: an arm's turns, a first line,
``ep.env`` / ``ep.meta`` overrides) and ``build(...)`` (a preparation stage, ``ig prepare``).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from ..core import AgentSpec, Env, Node, Task
from .config import Resolved
from .registry import Controlled, EpisodeContext, PluginError


class Context:
    """What a plugin factory sees: the objects built before it (``ctx["llm"]`` by row id or service key), the
    resolved composition and the registry of the run."""

    def __init__(self, resolved: Resolved):
        self.resolved = resolved
        self.objects: dict[str, Any] = {}  # row id -> object
        self.services: dict[str, str] = dict(resolved.provides)  # key -> row id

    def __getitem__(self, key: str) -> Any:
        rid = self.services.get(key)
        if rid is None or rid not in self.objects:
            built = ", ".join(self.objects) or "(none yet)"
            raise PluginError(f"no service {key!r} is available here (built so far: {built}); add it to the plugin's requires")
        return self.objects[rid]

    def get(self, key: str | None, default: Any = None) -> Any:
        if key is None:
            return default
        rid = self.services.get(key)
        return self.objects.get(rid, default) if rid is not None else default

    def __contains__(self, key: str) -> bool:
        rid = self.services.get(key)
        return rid is not None and rid in self.objects


@dataclass
class EnvConfig:
    max_ms: int | None = None
    end_idle_ms: int = 3000
    log: tuple[str, ...] = ("*",)
    peek: bool = False


def is_factory(obj: Any) -> bool:
    """A per-episode factory rather than a ready object (a Node, a source with ``next``)."""
    return callable(obj) and not isinstance(obj, Node) and not hasattr(obj, "next")


@dataclass
class Run:
    resolved: Resolved
    ctx: Context
    nodes: dict[str, Any] = field(default_factory=dict)  # row id -> Node | (EpisodeContext) -> Node
    controlled: Controlled | None = None
    controlled_id: str | None = None
    agent_factory: Any = None  # (EpisodeContext) -> agent
    agent_id: str | None = None

    @property
    def agent_label(self) -> str | None:
        """The run's label of its agent (the agent row's ``name``, as the scripts' ``--agent`` names; else the row id):
        the A/B run folder, ``meta.agent_name`` / ``meta.agent_base``, the reports' default tag."""
        return getattr(self.agent_factory, "name", None) or self.agent_id
    tasks: list[Task] = field(default_factory=list)
    task_sources: list[Any] = field(default_factory=list)  # objects with tasks() and optionally prepare / build
    scorers: dict[str, Any] = field(default_factory=dict)  # row id -> (ep, env, agent) -> dict
    sinks: dict[str, Any] = field(default_factory=dict)  # row id -> object with write / close
    reports: dict[str, Any] = field(default_factory=dict)  # row id -> object with async run(out, tag, log)
    driver: Any = None
    driver_id: str | None = None
    env_config: EnvConfig = field(default_factory=EnvConfig)

    @property
    def spec(self) -> AgentSpec:
        return self.controlled.spec if self.controlled is not None else AgentSpec()

    def make_env(self, ep: EpisodeContext | None = None) -> Env:
        """A fresh ``Env`` for one episode: shared nodes as they are, factories called with ``ep``; ``ep.env``
        overrides the kernel limits."""
        c, over = self.env_config, (ep.env if ep is not None else {})
        nodes = {k: (n(ep) if is_factory(n) else n) for k, n in self.nodes.items()}
        return Env(nodes, self.spec, log=tuple(c.log), max_ms=over.get("max_ms", c.max_ms),
                   end_idle_ms=over.get("end_idle_ms", c.end_idle_ms), peek=c.peek, background=over.get("background", ()))

    def agent(self, ep: EpisodeContext):
        """The external policy for one episode (``ep``: task, seed, arm, shared cache)."""
        if self.agent_factory is None:
            raise PluginError("the composition has no agent row (kind agent): nothing plays the controlled role")
        return self.agent_factory(ep)

    def prepare_task(self, task: Task, ep: EpisodeContext) -> Task:
        """The episode's view of a task: every task source's ``prepare`` in turn (sources that do not know the task
        return it unchanged)."""
        for src in self.task_sources:
            prepare = getattr(src, "prepare", None)
            if callable(prepare):
                task = prepare(task, ep)
        return task

    def meta(self) -> dict:
        """What every episode of this run records in ``meta``."""
        m = {"config": self.resolved.meta()}
        if self.controlled is not None:
            m["controlled"] = self.controlled.name
        return m


def compose(resolved: Resolved) -> Run:
    ctx = Context(resolved)
    run = Run(resolved, ctx)
    for row in resolved.rows:
        p = resolved.plugins[row.id]
        try:
            obj = p.factory(ctx, resolved.configs[row.id])
        except PluginError:
            raise
        except Exception as e:
            raise PluginError(f"row {row.id!r} ({p.key}) failed to build: {type(e).__name__}: {e}") from e
        ctx.objects[row.id] = obj
        _place(run, row.id, p.kind, obj)
    if run.controlled is None and run.agent_factory is not None:
        raise PluginError(f"agent row {run.agent_id!r} has no controlled role to play: add a row of plugin role/controlled")
    # nodes in the order the layers wrote them (not the build order): the kernel deals each node its rng in this order,
    # so it decides the trajectory and must match the scripts' {"user": ..., "tools": ...}
    run.nodes = dict(sorted(run.nodes.items(), key=lambda kv: resolved.rank.get(kv[0], 0)))
    return run


def _place(run: Run, rid: str, kind: str, obj: Any) -> None:
    if kind == "kernel":
        if isinstance(obj, EnvConfig):
            run.env_config = obj
    elif kind in ("role", "node"):
        if isinstance(obj, Controlled):
            if run.controlled is not None:
                raise PluginError(f"two controlled roles ({run.controlled_id!r}, {rid!r}); one external policy per run for now")
            run.controlled, run.controlled_id = obj, rid
        elif isinstance(obj, Node) or is_factory(obj):
            run.nodes[rid] = obj
        else:
            raise PluginError(f"row {rid!r}: a {kind} plugin must return a Node, a factory (EpisodeContext) -> Node, or Controlled, got {type(obj).__name__}")
    elif kind == "agent":
        if not callable(obj):
            raise PluginError(f"row {rid!r}: an agent plugin must return a factory (EpisodeContext) -> agent")
        if run.agent_factory is not None:
            raise PluginError(f"two agent rows ({run.agent_id!r}, {rid!r}); disable one")
        run.agent_factory, run.agent_id = obj, rid
    elif kind == "task":
        if callable(getattr(obj, "tasks", None)):
            run.task_sources.append(obj)
            run.tasks.extend(_tasks(rid, obj.tasks()))
        else:
            run.tasks.extend(_tasks(rid, obj))
    elif kind == "scorer":
        if not callable(obj):
            raise PluginError(f"row {rid!r}: a scorer plugin must return a callable (episode, env, agent) -> dict")
        run.scorers[rid] = obj
    elif kind == "sink":
        if not callable(getattr(obj, "write", None)):
            raise PluginError(f"row {rid!r}: a sink plugin must return an object with write(ep, env, agent)")
        run.sinks[rid] = obj
    elif kind == "report":
        if not inspect.iscoroutinefunction(getattr(obj, "run", None)):
            raise PluginError(f"row {rid!r}: a report plugin must return an object with async run(out, tag, log)")
        run.reports[rid] = obj
    elif kind == "driver":
        if not inspect.iscoroutinefunction(getattr(obj, "run", None)):
            raise PluginError(f"row {rid!r}: a driver plugin must return an object with async run(run, **options)")
        if run.driver is not None:
            raise PluginError(f"two driver rows ({run.driver_id!r}, {rid!r}); disable one")
        run.driver, run.driver_id = obj, rid
    # backend / source / tool / behaviour / latency / mixer: used by the rows that require them, nothing to place


def _tasks(rid: str, obj: Any) -> list[Task]:
    items = list(obj() if callable(obj) and not isinstance(obj, (list, tuple)) else obj)
    out = []
    for i, t in enumerate(items):
        if isinstance(t, Task):
            out.append(t)
        elif isinstance(t, dict):
            extra = set(t) - {"id", "scenario", "initial_state", "criteria"}
            if extra:
                raise PluginError(f"row {rid!r}: task #{i} has unknown fields {sorted(extra)}")
            out.append(Task(t.get("id", f"{rid}-{i}"), dict(t.get("scenario", {})), dict(t.get("initial_state", {})), dict(t.get("criteria", {}))))
        else:
            raise PluginError(f"row {rid!r}: a task plugin yields Task objects or dicts, got {type(t).__name__}")
    return out
