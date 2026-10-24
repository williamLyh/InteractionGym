"""Scorers: read a finished episode, return a dict recorded under ``eval.<row id>``. ``traj.episode`` already
computes ``eval.scores`` (one expectation per user turn) and ``eval.duplex``; these add to it."""

from __future__ import annotations

from dataclasses import dataclass

from ...eval import timing_counts, timing_summary
from ...interaction import metrics
from ..registry import plugin


@dataclass
class InteractionConfig:
    nudge_gap_ms: int = 4000
    repeat_overlap: float = 0.6


@plugin("scorer", "interaction", config=InteractionConfig)
def interaction(ctx, cfg: InteractionConfig):
    """Conversation-level counts: nudges, repeats, turns (``interaction.metrics``)."""
    return lambda ep, env, agent: metrics(ep, cfg.nudge_gap_ms, cfg.repeat_overlap)


@plugin("scorer", "timing")
def timing(ctx, kw: dict):
    """Duplex timing summary of the episode (``eval.timing_counts`` / ``timing_summary``)."""
    return lambda ep, env, agent: timing_summary([timing_counts(ep["turns"], ep["eval"]["scores"]["events"], **kw)])
