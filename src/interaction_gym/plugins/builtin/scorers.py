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


@plugin("scorer", "fdb3_quick")
def fdb3_quick(ctx, kw: dict):
    """Full-Duplex-Bench v3's per-episode facts for its report (``eval.<row id>``, the runner's ``eval.benchmark``):
    the agent's tool calls, the user's second turn, timing against the recorded request's end, user turns."""
    from ...benchmarks import fdb3

    def quick(ep, env, agent) -> dict:
        b = ep["meta"]["task"]["scenario"]["benchmark"]
        users = fdb3.user_turns(ep)
        u1 = users[1]["start_time"] if len(users) > 1 else None
        return {"calls": fdb3.agent_calls(ep), "u1_ms": u1, "timing": fdb3.timing(ep, b["user_end_ms"], before_ms=u1), "user_turns": len(users)}
    return quick


@plugin("scorer", "timing")
def timing(ctx, kw: dict):
    """Duplex timing summary of the episode (``eval.timing_counts`` / ``timing_summary``)."""
    return lambda ep, env, agent: timing_summary([timing_counts(ep["turns"], ep["eval"]["scores"]["events"], **kw)])
