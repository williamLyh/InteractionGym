"""Environment variables: the ``IG_*`` names, with the pre-rename ``DIG_*`` names as a deprecated fallback.

    from interaction_gym.envvars import getenv
    LLM_URL = getenv("IG_LLM_URL", "http://localhost:8000/v1")

``getenv("IG_X")`` returns ``$IG_X`` if it is set, else ``$DIG_X`` (warning once per process that ``DIG_*`` is
deprecated), else the default.
"""

from __future__ import annotations

import os
import warnings

PREFIX, OLD_PREFIX = "IG_", "DIG_"
_warned = False


def getenv(name: str, default: str | None = None) -> str | None:
    """``$IG_<name>``, else the deprecated ``$DIG_<name>``, else ``default``. ``name`` may omit the ``IG_`` prefix."""
    global _warned
    key = name if name.startswith(PREFIX) else PREFIX + name
    if key in os.environ:
        return os.environ[key]
    old = OLD_PREFIX + key[len(PREFIX):]
    if old in os.environ:
        if not _warned:
            _warned = True
            warnings.warn(f"environment variable {old} is deprecated, use {key} (DIG_* -> IG_*)", FutureWarning, stacklevel=2)
        return os.environ[old]
    return default
