"""τ-bench / τ-Voice adapter (Sierra tau2-bench, MIT).

Plugs τ-bench domains into InteractionGym's tool interface; our Sim replaces its orchestrator.

- ``tasks`` / ``load_task``: τ-bench tasks → our ``Task`` (JSON-clean; the τ task is
  looked up again by id when needed).
- ``TauBackend``: a ``ToolBackend`` holding one τ-bench ``Environment`` per episode; its
  ``evaluate`` rebuilds a τ-bench message trajectory from our log and runs the official
  DB / ENV_ASSERTION / ACTION / COMMUNICATE evaluators under the task's reward basis
  (NL assertions need an LLM judge and are reported as skipped).
- ``OracleAgent``: replays a task's gold actions; used to check the plumbing.
"""

from __future__ import annotations

import math
import os
from functools import cache
from pathlib import Path


def _find_tau2_data() -> None:
    """tau2 reads its domain data from ``$TAU2_DATA_DIR`` or its own source tree; a git / wheel install has no
    ``data/``. Point it at a checkout in ``third_party/tau2-bench`` (current directory or repository root)."""
    if os.environ.get("TAU2_DATA_DIR"):
        return
    for root in (Path.cwd(), Path(__file__).resolve().parents[3]):
        d = root / "third_party" / "tau2-bench" / "data"
        if (d / "tau2" / "domains").is_dir():
            os.environ["TAU2_DATA_DIR"] = str(d)
            return


_find_tau2_data()

from loguru import logger

logger.disable("tau2")  # τ-bench logs every tool call at DEBUG; re-enable with logger.enable("tau2")

from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.data_model.message import ToolCall as TauToolCall
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import RewardType
from tau2.data_model.tasks import Task as TauTask
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.registry import registry

from ..core import AgentSpec, Chunk, Frame, Segment, Task
from ..eval import turns
from ..tools import CALL, RESULT, ToolBackend, ToolCall, ToolResult, ToolSpec, tool_log

REQUESTOR = {"agent": "assistant", "user": "user"}

# ---------------------------------------------------------------- tasks


@cache
def _tau_tasks(domain: str, task_set: str | None = None) -> dict[str, TauTask]:
    return {t.id: t for t in registry.get_tasks_loader(task_set or domain)()}


def to_task(domain: str, t: TauTask, task_set: str | None = None) -> Task:
    sc = t.user_scenario
    return Task(
        id=f"{domain}/{t.id}",
        scenario={"domain": domain, "task_set": task_set, "tau_id": t.id, "persona": sc.persona or "", "instructions": str(sc.instructions)},
        initial_state=t.initial_state.model_dump(mode="json") if t.initial_state else {},
        criteria=t.evaluation_criteria.model_dump(mode="json") if t.evaluation_criteria else {},
    )


def tasks(domain: str, task_set: str | None = None) -> list[Task]:
    return [to_task(domain, t, task_set) for t in _tau_tasks(domain, task_set).values()]


def load_task(domain: str, task_id: str, task_set: str | None = None) -> Task:
    return to_task(domain, _tau_tasks(domain, task_set)[task_id], task_set)


def tau_task(task: Task) -> TauTask:
    sc = task.scenario
    return _tau_tasks(sc["domain"], sc.get("task_set"))[sc["tau_id"]]


# ---------------------------------------------------------------- backend


class TauBackend(ToolBackend):
    def __init__(self, domain: str):
        self.domain = self.name = domain
        self.description = f"τ-bench {domain} customer-service backend"
        self._probe = registry.get_env_constructor(domain)()  # for tool schemas and policy only

    def tools(self, caller="agent"):
        tools = self._probe.get_tools() if caller == "agent" else (self._probe.get_user_tools() if self._probe.user_tools else [])
        out = []
        for tool in tools:
            fn = tool.openai_schema["function"]
            out.append(ToolSpec(fn["name"], fn.get("description", ""), fn.get("parameters", {})))
        return out

    def instructions(self, caller="agent"):
        return self._probe.get_policy() if caller == "agent" else ""

    def reset(self, task, rng):
        tt = tau_task(task)
        env = registry.get_env_constructor(self.domain)()
        init = tt.initial_state
        env.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=(init.message_history or []) if init else [],
        )
        return env

    async def call(self, env, caller, call):
        res = env.get_response(TauToolCall(id=call.id, name=call.name, arguments=call.arguments, requestor=REQUESTOR[caller]))
        return ToolResult(res.id, call.name, res.content, res.error)

    def evaluate(self, task, log, truncated: bool = False):
        return evaluate(log, task, truncated)


# ---------------------------------------------------------------- evaluation


def to_messages(log: list[Frame], user: str = "user.speech", agent: str = "policy.speech") -> list:
    """Linearize our log into τ-bench messages. Speech uses what was actually said; each tool call is
    followed directly by its result (the state change happened at call time)."""
    events = []  # (time, order, messages)
    for i, turn in enumerate(turns(log, user)):
        if turn.actual.transcript:
            events.append((turn.actual.t0, i, [UserMessage(role="user", content=turn.actual.transcript)]))
    for i, turn in enumerate(turns(log, agent)):
        if turn.actual.transcript:
            events.append((turn.actual.t0, i, [AssistantMessage(role="assistant", content=turn.actual.transcript)]))
    for i, (t, caller, call, res) in enumerate(tool_log(log)):
        if res is None:
            continue
        req = REQUESTOR[caller]
        tc = TauToolCall(id=call.id, name=call.name, arguments=call.arguments, requestor=req)
        msg = (UserMessage if caller == "user" else AssistantMessage)(role=req, tool_calls=[tc])
        events.append((t, len(log) + i, [msg, ToolMessage(id=call.id, role="tool", content=res.content, requestor=req, error=res.error)]))
    return [m for _, _, ms in sorted(events, key=lambda e: (e[0], e[1])) for m in ms]


def termination(truncated: bool) -> TerminationReason:
    # conversations end by themselves once the user stops talking; only hitting max_ms is premature
    return TerminationReason.MAX_STEPS if truncated else TerminationReason.USER_STOP


_PARTS = (
    ("env", EvaluationType.ENV, {RewardType.DB, RewardType.ENV_ASSERTION}),
    ("action", EvaluationType.ACTION, {RewardType.ACTION}),
    ("communicate", EvaluationType.COMMUNICATE, {RewardType.COMMUNICATE}),
)


def evaluate(log: list[Frame], task: Task, truncated: bool = False) -> dict:
    """Official τ-bench reward (without NL assertions).

    Returns ``{"reward", "parts", "breakdown", "nl_skipped", "termination"}``: ``parts`` is the reward
    per evaluator, ``breakdown`` per reward type (DB, ENV_ASSERTION, ACTION, COMMUNICATE). ``reward``
    is None when the task's basis includes NL assertions, which need an LLM judge.
    """
    tt = tau_task(task)
    history = list(tt.initial_state.message_history or []) if tt.initial_state else []
    reason = termination(truncated)
    run = SimulationRun(
        id="interaction_gym", task_id=tt.id, start_time="", end_time="", duration=0.0, termination_reason=reason, messages=history + to_messages(log)
    )
    if tt.evaluation_criteria is None:
        return {"reward": 1.0, "parts": {}, "breakdown": {}, "nl_skipped": False, "termination": reason.value}
    basis = set(tt.evaluation_criteria.reward_basis)
    domain = task.scenario["domain"]
    parts, breakdown = {}, {}
    for name, kind, bases in _PARTS:
        if basis & bases:
            info = evaluate_simulation(run, tt, kind, solo_mode=False, domain=domain)
            parts[name] = info.reward
            breakdown.update({k.value: v for k, v in (info.reward_breakdown or {}).items()})
    nl = RewardType.NL_ASSERTION in basis
    reward = None if nl else math.prod(parts.values())
    return {"reward": reward, "parts": parts, "breakdown": breakdown, "nl_skipped": nl, "termination": reason.value}


# ---------------------------------------------------------------- oracle agent


class OracleAgent:
    """Replays the task's gold agent actions after the user's first turn, then says the
    information the task expects to be communicated. Checks env + evaluation plumbing."""

    def __init__(self, task: Task, spec: AgentSpec = AgentSpec(obs=("user.speech", RESULT["agent"])), wps: float = 3.4):
        crit = tau_task(task).evaluation_criteria
        self.actions = [a for a in (crit.actions or []) if a.requestor == "assistant"] if crit else []
        self.say = " ".join(crit.communicate_info or []) if crit and crit.communicate_info else "All done, is there anything else?"
        self.spec, self.wps = spec, wps
        self.user_done_at: int | None = None
        self.pending: set[str] = set()
        self.next_action = 0
        self.spoke = False

    def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        for f in obs:
            if isinstance(f.data, Chunk) and f.data.last and self.user_done_at is None:
                self.user_done_at = f.data.t0 + f.data.dur
            if isinstance(f.data, ToolResult):
                self.pending.discard(f.data.id)
        if self.user_done_at is None or t - self.user_done_at < 300:
            return []
        if self.next_action < len(self.actions):
            a = self.actions[self.next_action]
            self.next_action += 1
            call = ToolCall(f"call_{self.next_action}", a.name, a.arguments)
            self.pending.add(call.id)
            return [Frame(CALL["agent"], t, call)]
        if not self.pending and not self.spoke:
            self.spoke = True
            return [Frame(self.spec.out, t, Segment("a0", t, round(len(self.say.split()) / self.wps * 1000), self.say))]
        return []
