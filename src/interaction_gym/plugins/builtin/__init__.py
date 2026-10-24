"""The built-in plugins: thin wrappers that register the existing classes under plugin names. Importing this package
registers them all (the registry does it on first use)."""

from . import agents, backends, drivers, kernel, roles, scorers, sinks, sources, tasks, tools  # noqa: F401
