"""Tools: ``tool/*`` rows are ``ToolBackend``s (what can be called), ``node/tool_world`` is the node that executes
calls through them with simulated latency and tells the controlled role what it may call (``mode``)."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from ...tools import FunctionBackend, ToolWorld
from ..registry import PluginError, plugin


def _import(path: str, where: str):
    """``package.module:attr`` → the attribute."""
    if ":" not in path:
        raise PluginError(f"{where}: give functions as 'package.module:name', got {path!r}")
    mod, attr = path.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as e:
        raise PluginError(f"{where}: cannot import {path!r}: {e}") from None


@dataclass
class FunctionsConfig:
    functions: dict | list | str  # {"tool name": "module:function"}, ["module:function", ...] or "module:DICT"
    name: str = "tools"
    description: str = ""
    instructions: str = ""
    initial_state: dict | None = None
    context: str | None = None  # "module:function(state) -> dict" revealed to the agent up front


@plugin("tool", "functions", config=FunctionsConfig)
def functions(ctx, cfg: FunctionsConfig) -> FunctionBackend:
    """Tools from Python functions (``FunctionBackend``); a function taking ``state`` keeps a mutable world."""
    where = "tool/functions"
    if isinstance(cfg.functions, str):
        fns = _import(cfg.functions, where)
    elif isinstance(cfg.functions, dict):
        fns = {n: _import(p, where) for n, p in cfg.functions.items()}
    else:
        fns = {p.split(":", 1)[1]: _import(p, where) for p in cfg.functions}
    return FunctionBackend(fns, cfg.initial_state, cfg.instructions, name=cfg.name, description=cfg.description,
                           context=_import(cfg.context, where) if cfg.context else None)


@dataclass
class ToolWorldConfig:
    backends: list = field(default_factory=list)  # row ids of tool/* rows
    latency_ms: int = 0
    mode: str = "all"  # "all" | "discover"
    callers: tuple[str, ...] = ("agent", "user")
    screen: str | None = None
    serial: tuple[str, ...] = ()


@plugin("node", "tool_world", config=ToolWorldConfig, refs=("backends[]",))
def tool_world(ctx, cfg: ToolWorldConfig) -> ToolWorld:
    """Executes tool calls through the named backends; results land ``latency_ms`` later."""
    if not cfg.backends:
        raise PluginError("node/tool_world: name at least one tool row in backends")
    backends = {rid: ctx[rid] for rid in cfg.backends}
    return ToolWorld(backends if len(backends) > 1 else next(iter(backends.values())), cfg.latency_ms, callers=tuple(cfg.callers),
                     mode=cfg.mode, screen=cfg.screen, serial=cfg.serial)
