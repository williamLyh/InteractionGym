"""``kernel/env``: the kernel's own knobs for a run (the ``Env`` constructor's)."""

from __future__ import annotations

from ..compose import EnvConfig
from ..registry import plugin


@plugin("kernel", "env", config=EnvConfig)
def env(ctx, cfg: EnvConfig) -> EnvConfig:
    """Episode limits: ``max_ms`` (None: no cap), ``end_idle_ms`` (quiet after the last activity that ends the
    episode), ``log`` (stream patterns kept in the log), ``peek`` (raw frames to the agent: debugging only)."""
    return EnvConfig(cfg.max_ms, cfg.end_idle_ms, tuple(cfg.log), cfg.peek)
