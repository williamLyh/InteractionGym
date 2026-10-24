"""Everything above the kernel is a plugin, composed from layered YAML (docs/ARCHITECTURE.md).

    from interaction_gym.plugins import Composition, compose, plugin

    comp = Composition.bundle("minimal")          # built-in bundle, or a path
    comp.apply(open("my.yml").read(), "my.yml")   # a patch layer
    comp.set("env.max_ms=30000")                  # one key
    run = compose(comp.resolve())                 # validated, requirements first
    await run.driver.run(run, out="runs/x", seeds=[0])

Command line: ``ig run <bundle> [--patch f.yml] [--set row.key=v] [--dump-config]``, ``ig plugins``.
"""

from .compose import Context, EnvConfig, Run, compose
from .config import Composition, Resolved, Row
from .registry import CURRENT_EPISODE, KINDS, REGISTRY, Controlled, EpisodeContext, Plugin, PluginError, Registry, plugin

__all__ = ["CURRENT_EPISODE", "KINDS", "REGISTRY", "Composition", "Context", "Controlled", "EnvConfig", "EpisodeContext", "Plugin",
           "PluginError", "Registry", "Resolved", "Row", "Run", "compose", "plugin"]
