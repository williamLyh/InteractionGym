"""The built-in plugins: thin wrappers that register the existing classes under plugin names. Importing this package
registers them all (the registry does it on first use)."""

from . import agents, backends, behaviours, drivers, kernel, reports, roles, scorers, sinks, sop, sources, tasks, tools  # noqa: F401
