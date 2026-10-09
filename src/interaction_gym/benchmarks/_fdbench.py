"""Lazy access to the optional Full-Duplex-Bench component (``interaction-gym-fdbench``, CC BY-NC 4.0).

The metric rules, prompts, mock APIs and tool schemas ported from Full-Duplex-Bench are non-commercial, so they are
not part of this Apache-2.0 package: they live in the separate distribution ``interaction-gym-fdbench``
(``extras/fdbench`` in the repository, THIRD_PARTY.md). ``benchmarks.full_duplex_bench``, ``fdb2`` and ``fdb3``
import without it and load it on first use; a helpful ``ImportError`` says how to install it.
"""

from __future__ import annotations

import importlib
from types import ModuleType

PACKAGE = "interaction_gym_fdbench"
DISTRIBUTION = "interaction-gym-fdbench"
HINT = (f"this needs the optional Full-Duplex-Bench component `{DISTRIBUTION}` (CC BY-NC 4.0: non-commercial use "
        "only), which is not part of the Apache-2.0 package. Install it with `uv sync --extra fdbench` in the "
        "repository, or `pip install ./extras/fdbench` (THIRD_PARTY.md).")


def load(part: str) -> ModuleType:
    """``interaction_gym_fdbench.<part>`` (``v1`` / ``v2`` / ``v3``), or an ``ImportError`` with ``HINT``."""
    try:
        return importlib.import_module(f"{PACKAGE}.{part}")
    except ModuleNotFoundError as e:
        if e.name in (PACKAGE, f"{PACKAGE}.{part}"):
            raise ImportError(f"{PACKAGE}.{part}: {HINT}") from e
        raise


class Lazy:
    """A stand-in for ``interaction_gym_fdbench.<part>`` that imports it on first attribute access."""

    def __init__(self, part: str):
        self._part = part

    def __getattr__(self, name: str):
        return getattr(load(self._part), name)


def reexport(module: str, part: str, names: set[str], aliases: dict[str, str] | None = None):
    """A module-level ``__getattr__`` (PEP 562) that serves ``names`` (and ``aliases``: local name → upstream name)
    from the optional component, so ``from ...full_duplex_bench import take_turn`` keeps working."""
    aliases = aliases or {}

    def __getattr__(name: str):
        if name in names or name in aliases:
            return getattr(load(part), aliases.get(name, name))
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return __getattr__
