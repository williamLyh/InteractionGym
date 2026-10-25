"""τ-bench (τ²-bench's tasks, tools and official evaluators) as plugins — ``examples/tau_mock.py`` as a composition
and the tool seam with an external provider: ``task/tau`` (a domain's tasks), ``tool/tau`` (its tools and database
through ``integrations.tau.TauBackend``, on ``node/tool_world``), ``agent/tau_oracle`` (replays the task's gold
actions: the plumbing check), ``scorer/tau`` (the official reward → ``eval.reward``). Needs the ``tau`` extra and
the domain data (README, ``$TAU2_DATA_DIR``)."""

from __future__ import annotations

from dataclasses import dataclass

from ..registry import PluginError, plugin


def _tau():
    try:
        from ...integrations import tau
    except ImportError as e:  # the optional extra
        raise PluginError(f"τ-bench plugins need the `tau` extra (uv sync --extra tau) and its data: {e}") from None
    return tau


@dataclass
class TauTasksConfig:
    domain: str = "mock"  # airline | retail | telecom | mock
    task_set: str | None = None
    ids: list | None = None


@plugin("task", "tau", config=TauTasksConfig)
def tau_tasks(ctx, cfg: TauTasksConfig) -> list:
    """A τ-bench domain's tasks (persona, instructions, initial state, evaluation criteria)."""
    tasks = _tau().tasks(cfg.domain, cfg.task_set)
    return [t for t in tasks if not cfg.ids or t.id in cfg.ids or t.scenario["tau_id"] in cfg.ids]


@dataclass
class TauToolConfig:
    domain: str = "mock"


@plugin("tool", "tau", config=TauToolConfig)
def tau_tool(ctx, cfg: TauToolConfig):
    """The domain's tools and database (``TauBackend``): the agent's and the user's, on a shared world."""
    return _tau().TauBackend(cfg.domain)


@plugin("agent", "tau_oracle", requires=("controlled",))
def tau_oracle(ctx, kw: dict):
    """Replays the task's gold agent actions after the user's first turn, then says what the task expects to be
    communicated — checks the env and the evaluation plumbing, not a model."""
    spec, tau = ctx["controlled"].spec, _tau()
    return lambda ep: tau.OracleAgent(ep.task, spec, **kw)


@plugin("scorer", "tau")
def tau_scorer(ctx, kw: dict):
    """The official τ-bench reward (``integrations.tau.evaluate``) → ``eval.reward`` (total, breakdown by reward
    type); ``eval.<row>`` keeps the parts per evaluator, whether NL assertions were skipped and the termination."""
    tau = _tau()

    def score(ep, env, agent) -> dict:
        res = tau.evaluate(env.log, env.sim.task, env.truncated)
        ep["eval"]["reward"] = {"total": res["reward"], "parts": res["breakdown"]}
        return {k: res[k] for k in ("parts", "nl_skipped", "termination")}
    return score
