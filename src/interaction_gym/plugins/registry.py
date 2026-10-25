"""The plugin registry: every swappable part of a run is a named plugin of one kind.

A plugin is a factory ``(ctx, config) -> object`` registered under ``"<kind>/<name>"`` with the ``@plugin`` decorator
(docs/ARCHITECTURE.md). It declares the services it ``requires`` (row ids or service keys of other rows: they are
built first and reachable through ``ctx``), optionally the service key it ``provides``, and a ``Config`` dataclass
that its configuration is validated against when the composition is resolved — unknown keys and missing required
fields are errors at load time, not surprises inside an episode. A plugin without a ``Config`` takes its
configuration as keyword arguments unchanged (a thin wrapper around an existing class, whose own signature is then
the contract).

Kinds (``KINDS``): what the composition expects back from the factory.

- ``kernel``: configuration of the kernel itself (``kernel/env``)
- ``role``: a participant: a ``Node`` (simulated / replayed), or ``Controlled`` — the role an external policy plays
- ``source``: what a simulated role says (a script, an LLM, a benchmark's replanned script); a role names it
- ``node``: any other node of the episode (a tool world, a soundscape, a watcher)
- ``behaviour``: something a simulated role does besides its turns (barge-in, backchannel, noise, ...)
- ``backend``: a model client: LLM / TTS / ASR / judge
- ``tool``: a ``ToolBackend`` (the tools a world exposes)
- ``agent``: how an external policy is attached to the controlled role: ``(EpisodeContext) -> agent``
- ``task``: where the episodes' tasks come from: an iterable of ``Task``
- ``scorer``: ``(episode, env, agent) -> dict`` recorded under ``eval.<row id>``
- ``sink``: receives finished episodes (``write(ep, env, agent)``, ``close()``): files, viewer pages
- ``report``: post-hoc scoring of a whole run from its files (judges, tables, pages): ``async run(out, tag, log)``, by ``ig report``
- ``driver``: how episodes are organised (an evaluation runner, an RL rollout, a live session)
- ``latency``: where in simulated time an output lands; ``mixer``: how concurrent segments of one stream combine

Third-party packages register plugins through the entry point group ``interaction_gym.plugins``: each entry point
names a module that registers its plugins when imported.
"""

from __future__ import annotations

import contextvars
import dataclasses
import difflib
import importlib
import types
import typing
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

KINDS = ("kernel", "role", "source", "node", "behaviour", "backend", "tool", "agent", "task", "scorer", "sink", "report", "driver", "latency", "mixer")
ENTRY_POINT_GROUP = "interaction_gym.plugins"
BUILTIN = "interaction_gym.plugins.builtin"


class PluginError(Exception):
    """A plugin or a configuration row is wrong; the message names the row and what to fix."""


@dataclass(frozen=True)
class Plugin:
    kind: str
    name: str
    factory: Callable[..., Any]
    config: type | None = None  # a dataclass; None: the config dict is passed as keyword arguments
    requires: tuple[str, ...] = ()
    provides: str | None = None
    doc: str = ""
    # config fields that name other rows (``"voice.tts"``, ``"backends[]"`` for each item of a list) → the literal
    # values also allowed there (``{"source": ("script",)}``). A string there must be a row id or one of the literals;
    # the rows named are built first. This is how a plugin's dependencies on configured rows become explicit.
    refs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.kind}/{self.name}"


def config_refs(p: Plugin, raw: dict | None) -> list[tuple[str, str]]:
    """``(path, value)`` for every string a ``refs`` field holds in ``raw`` (the row's config)."""
    out = []
    for path in p.refs:
        values = [raw or {}]
        for part in path.split("."):
            each = part.endswith("[]")
            key = part[:-2] if each else part
            nxt = []
            for v in values:
                if isinstance(v, dict) and key in v:
                    nxt.extend(v[key] if each and isinstance(v[key], (list, tuple)) else [v[key]])
            values = nxt
        out.extend((path, v) for v in values if isinstance(v, str))
    return out


class Registry:
    def __init__(self):
        self._plugins: dict[str, Plugin] = {}
        self._loaded = False

    def add(self, p: Plugin) -> Plugin:
        if p.kind not in KINDS:
            raise PluginError(f"plugin {p.key}: unknown kind {p.kind!r}; kinds are {', '.join(KINDS)}")
        if p.key in self._plugins and self._plugins[p.key].factory is not p.factory:
            raise PluginError(f"plugin {p.key} is already registered (by {self._plugins[p.key].factory.__module__})")
        if p.config is not None and not dataclasses.is_dataclass(p.config):
            raise PluginError(f"plugin {p.key}: Config must be a dataclass, got {p.config!r}")
        self._plugins[p.key] = p
        return p

    def load(self) -> None:
        """Import the built-in plugins and every ``interaction_gym.plugins`` entry point (once)."""
        if self._loaded:
            return
        self._loaded = True
        importlib.import_module(BUILTIN)
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            try:
                ep.load()
            except Exception as e:  # noqa: BLE001 — one broken plugin package must not take the others down
                warnings.warn(f"plugin entry point {ep.name} ({ep.value}) failed to load: {e}", stacklevel=2)

    def get(self, key: str) -> Plugin:
        self.load()
        if key in self._plugins:
            return self._plugins[key]
        near = difflib.get_close_matches(key, self._plugins, n=3, cutoff=0.5)
        kind = key.split("/", 1)[0]
        same_kind = sorted(k for k in self._plugins if k.startswith(kind + "/"))
        hint = f"; did you mean {', '.join(near)}?" if near else (f"; {kind} plugins: {', '.join(same_kind)}" if same_kind else "")
        raise PluginError(f"unknown plugin {key!r}{hint}")

    def all(self, kind: str | None = None) -> list[Plugin]:
        self.load()
        return [p for p in self._plugins.values() if kind is None or p.kind == kind]


REGISTRY = Registry()


def plugin(kind: str, name: str, *, requires: Iterable[str] = (), provides: str | None = None, config: type | None = None,
           refs: dict[str, Iterable[str]] | Iterable[str] = (), registry: Registry = REGISTRY):
    """Register ``fn(ctx, config) -> object`` as plugin ``kind/name``.

        @plugin("agent", "canned", requires=("controlled",), config=CannedConfig)
        def canned(ctx, cfg): ...

        @plugin("role", "user", config=UserConfig, refs={"source": ("script",), "voice.tts": ()})
        def user(ctx, cfg): ...   # ctx[cfg.voice["tts"]] is built before this row
    """
    refs = {k: tuple(v) for k, v in (refs.items() if isinstance(refs, dict) else ((r, ()) for r in refs))}

    def deco(fn):
        registry.add(Plugin(kind, name, fn, config, tuple(requires), provides, (fn.__doc__ or "").strip(), refs))
        return fn
    return deco


# ---------------------------------------------------------------- configuration validation

_SCALARS = {int: (int,), float: (int, float), str: (str,), bool: (bool,)}


def parse_config(p: Plugin, raw: dict | None, where: str) -> Any:
    """``raw`` as the plugin's ``Config`` (or the dict itself for a plugin without one). Unknown keys, missing
    required fields and scalar type mismatches raise ``PluginError`` naming ``where`` (the row)."""
    raw = dict(raw or {})
    if not isinstance(raw, dict):
        raise PluginError(f"{where}: config must be a mapping, got {type(raw).__name__}")
    if p.config is None:
        return raw
    fields = {f.name: f for f in dataclasses.fields(p.config)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise PluginError(f"{where}: {p.key} does not take {', '.join(unknown)}; its fields are {', '.join(fields)}")
    missing = [n for n, f in fields.items() if n not in raw and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING]
    if missing:
        raise PluginError(f"{where}: {p.key} needs {', '.join(missing)}")
    hints = typing.get_type_hints(p.config)
    for n, v in raw.items():
        want, optional = _scalar(hints.get(n))
        if want is None or (v is None and optional):
            continue
        if not isinstance(v, _SCALARS[want]) or (want is not bool and isinstance(v, bool)):
            raise PluginError(f"{where}: {p.key}.{n} must be {want.__name__}{' or null' if optional else ''}, got {v!r}")
    return p.config(**raw)


def _scalar(hint) -> tuple[type | None, bool]:
    """``(scalar type, allows None)`` for ``int``, ``str | None`` and the like; ``(None, ...)`` for anything else."""
    if hint in _SCALARS:
        return hint, False
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        args = typing.get_args(hint)
        rest = [a for a in args if a is not type(None)]
        if len(rest) == 1 and rest[0] in _SCALARS:
            return rest[0], len(rest) < len(args)
    return None, False


def config_fields(p: Plugin) -> list[dict]:
    """The plugin's configuration fields with defaults, for ``ig plugins``."""
    if p.config is None:
        return []
    out = []
    for f in dataclasses.fields(p.config):
        d = f.default if f.default is not dataclasses.MISSING else (f.default_factory() if f.default_factory is not dataclasses.MISSING else "<required>")
        out.append({"name": f.name, "type": f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type)), "default": d})
    return out


@dataclass
class Controlled:
    """What a ``role`` plugin returns for the role an external policy plays: its session spec, and the name the
    trajectory records for it."""

    spec: Any  # AgentSpec
    name: str = "agent"
    extra: dict = field(default_factory=dict)


@dataclass
class EpisodeContext:
    """What an agent factory and episode-aware backends get to know about the episode being built: the task, the
    seed, the arm (condition) of an A/B run, and a ``cache`` shared by the arms of one (task, seed) pair (a cascaded
    agent memoizes its model calls in it, so arms with the same prefix make the same calls)."""

    task: Any
    seed: int = 0
    arm: str | None = None
    cache: dict = field(default_factory=dict)
    episode_id: str = ""
    env: dict = field(default_factory=dict)  # per-episode Env overrides a task source sets (``max_ms``, ``end_idle_ms``)
    meta: dict = field(default_factory=dict)  # extra ``meta`` fields for the trajectory
    verifier: dict | None = None  # ``eval.verifier``: how the task's own checker computes the reward (traj.episode)
    data: dict = field(default_factory=dict)  # objects a task source hands to the roles of this episode (a loaded sample)


CURRENT_EPISODE: contextvars.ContextVar[EpisodeContext | None] = contextvars.ContextVar("ig_episode", default=None)
