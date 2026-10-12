"""SOPBench adapter (Li et al., arXiv 2503.08669; github.com/Leezekun/SOPBench; code MIT, data CC BY 4.0).

SOPBench tests whether an agent follows a standard operating procedure (SOP) in customer service: seven domains
(bank, DMV, healthcare, library, online market, hotel, university), each a Python system whose functions the agent
calls, and an SOP that says which conditions must be verified, with which functions and in which order, before a
service action may run. Each test case is one user goal, an initial database and the user's own information; the
database is set up so that the SOP's conditions do or do not hold (``action_should_succeed``), so about two thirds of
the cases are requests the agent has to decline. Its verifier replays the agent's calls against a strict version of
the system: a case passes only if no call errored, no constraint was violated (each response equals the strict
system's), the final database matches the oracle's, the target action ran exactly when it should, and every call was
preceded by the verifications its directed action graph requires.

Here it is spoken:

- **The user** is our ``LLMSource`` with the case's goal and information as the goal text: it says what it wants and
  gives each piece of information only when asked (in SOPBench the user hands over everything in its first
  message, and again whenever the agent stops calling tools). It hangs up once the agent has done it or declined.
- **The agent** gets SOPBench's own assistant prompt (role, core principles, the SOP) plus a note that this is a
  phone call, and the domain's functions as tools (``SOPBenchBackend``; ``exit_conversation`` is left out: the call
  ends when the user hangs up). The SOP is rendered in one of ``FORMATS``: SOPBench's ``structured`` text (default),
  its flat ``list`` text, or ``structured+order``, which also spells out the verification order from the case's
  action graph (SOPBench's ``--action_order`` ablation).
- **Scoring** (``evaluate``) is SOPBench's ``evaluator_function_directed_graph`` on the episode's calls and final
  database; ``reward`` = its ``success``.

Like SOPBench's default ``prompt`` mode, the environment does not enforce the SOP: an action whose conditions do not
hold still runs, and the verifier catches it.

Put a checkout at ``third_party/sopbench`` (``git clone https://github.com/Leezekun/SOPBench third_party/sopbench``) or
point ``SOPBENCH_PATH`` at one; only its ``env/`` package and ``data/`` are used (standard library only).
"""

from __future__ import annotations

import ast
import copy
import json
import os
import sys
from functools import cache
from pathlib import Path

from ..core import Task
from ..tools import ToolBackend, ToolResult, ToolSpec, ToolWorld, tool_log

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = ("bank", "dmv", "healthcare", "library", "online_market", "hotel", "university")
FORMATS = ("structured", "list", "structured+order")
_FMT = {"structured": ("structured", False), "list": ("old", False), "structured+order": ("structured", True)}

SPOKEN_NOTE = (
    "\n\n\n### This is a phone call\n"
    "You are talking with the user on the phone: what you write is spoken to them, and you hear what they say. Speak "
    "briefly and naturally, one or two short sentences; no lists, markdown or JSON. The user has not given you their "
    "details yet: ask for what you need, one or two items at a time. Function calls are not spoken. When the request "
    "is done, or you have to decline it, tell the user in one or two sentences."
)


def path() -> Path:
    for root in (os.environ.get("SOPBENCH_PATH"), ROOT / "third_party" / "sopbench"):
        if root and (Path(root) / "env" / "variables.py").is_file():
            return Path(root)
    raise FileNotFoundError("SOPBench not found: clone github.com/Leezekun/SOPBench into third_party/sopbench or set SOPBENCH_PATH")


@cache
def _sop():
    """SOPBench's ``env`` package (its top-level name is ``env``; imported from the checkout)."""
    root = str(path())
    if root not in sys.path:
        sys.path.insert(0, root)
    from env import evaluator, task, variables  # noqa: F401
    return task, evaluator, variables


@cache
def _defaults(domain: str):
    """(innate dependencies, default dependencies, their descriptions) as SOPBench's runner builds them
    (``default_constraint_option="full"``, structured and list formats)."""
    task, _, _ = _sop()
    out = {}
    for fmt in ("structured", "old"):
        out[fmt] = task.task_default_dep_full(domain, "full", fmt, dependency_verb_dep_orig=True)
    return out


@cache
def records(domain: str) -> tuple[tuple[str, int, dict], ...]:
    """(user goal, index within the goal, case) for every test case of ``domain``, in file order."""
    data = json.loads((path() / "data" / f"{domain}_tasks.json").read_text())  # hotel / university cases carry no
    return tuple((goal, i, {**rec, "user_goal": goal}) for goal, cases in data.items() for i, rec in enumerate(cases))  # user_goal


def _known(user_known: dict) -> str:
    """The user's information as a list. SOPBench's "identification" is what logs the user in: a string is the account
    password, a dict the driver's licence; its own user simply hands over the raw keys, a speaking user needs to know."""
    def fmt(v):
        return json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    out = []
    for k, v in user_known.items():
        name = k.replace("_", " ")
        if k == "identification":
            name = "identification, i.e. your account password" if isinstance(v, str) else "identification, i.e. your driver's licence"
        out.append(f"- {name}: {fmt(v)}")
    return "\n".join(out)


def to_task(domain: str, goal: str, i: int, rec: dict, sop_format: str = "structured") -> Task:
    assert sop_format in FORMATS, sop_format
    goal_text = (f"{rec['user_instruction']}\n\nThe information you have (give each item only when the assistant asks for it, "
                 f"exactly as written here, so that it can be typed in; the assistant may use another word for an item):\n"
                 f"{_known(rec['user_known'])}\n\nThis is all you know. If you are asked for anything else, or about the state of "
                 f"your account or of the system (for example whether you are logged in), say you don't have it or don't know; "
                 f"never make it up. If the assistant says it cannot do it, accept that, thank them and say goodbye.")
    return Task(
        id=f"sopbench-{domain}-{goal}-{i}",
        scenario={
            "persona": f"A customer calling the {domain.replace('_', ' ')} service line.",
            "instructions": goal_text,
            "sopbench": {"domain": domain, "user_goal": goal, "index": i, "sop_format": sop_format,
                         "action_should_succeed": rec["action_should_succeed"]},
        },
        initial_state={"sopbench_case": rec},
        criteria={"action_should_succeed": rec["action_should_succeed"], "user_goal": goal},
    )


def tasks(domain: str, sop_format: str = "structured", goals: list[str] | None = None) -> list[Task]:
    return [to_task(domain, g, i, r, sop_format) for g, i, r in records(domain) if goals is None or g in goals]


def _literal(s: str):
    """A function response as SOPBench's evaluator reads it back from the transcript (its ``try_eval``)."""
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


class SOPBenchBackend(ToolBackend):
    """One case's domain system behind the tool interface. The SOP and tool list depend on the case (the target
    action's constraints are the case's own), so a backend is built per task (``tool_world(task)``)."""

    def __init__(self, task: Task):
        self.task = task
        sc = task.scenario["sopbench"]
        self.domain, self.rec = sc["domain"], task.initial_state["sopbench_case"]
        self.name = f"sopbench_{self.domain}"
        self.description = f"SOPBench {self.domain.replace('_', ' ')} system"
        _, assistant = self._init()
        self.prompt = assistant["instructions"] + SPOKEN_NOTE
        self.specs = [ToolSpec(t["function"]["name"], t["function"].get("description", ""),
                               t["function"].get("parameters") or {"type": "object", "properties": {}}) for t in assistant["tools"]]

    def _init(self):
        task, _, _ = _sop()
        fmt, order = _FMT[self.task.scenario["sopbench"]["sop_format"]]
        innate, dep_full, dep_descr = _defaults(self.domain)[fmt]
        system, _, assistant, _ = task.task_initializer(self.domain, copy.deepcopy(self.rec), innate, dep_full, dep_descr, None, "prompt",
                                                        False, fmt, abl_order=order)
        return system, assistant

    def tools(self, caller="agent"):
        return list(self.specs) if caller == "agent" else []

    def instructions(self, caller="agent"):
        return self.prompt if caller == "agent" else ""

    def reset(self, task, rng):
        system, _ = self._init()
        return {"system": system, "calls": []}

    def fork(self, state, n):
        return [copy.deepcopy(state) for _ in range(n)]

    async def call(self, state, caller, call):
        if call.name not in {s.name for s in self.specs}:
            return ToolResult(call.id, call.name, f"Error: Tool {call.name} not found.", error=True)
        fn = getattr(state["system"], call.name, None)
        if fn is None:
            return ToolResult(call.id, call.name, f"Error: Tool {call.name} not found.", error=True)
        try:  # the response as in SOPBench's released trajectories: the whole (success, value) tuple, which its evaluator
            raw = fn(**call.arguments)  # reads (its current runner keeps only the value, and then "succeeded" is never seen)
        except Exception as e:  # noqa: BLE001
            raw = f"{e.__class__.__name__}: {e}"
        content = str(raw)
        state["calls"].append({"tool_name": call.name, "arguments": dict(call.arguments), "content": _literal(content)})
        return ToolResult(call.id, call.name, content)


def tool_world(task: Task, latency_ms: int = 300) -> ToolWorld:
    return ToolWorld(SOPBenchBackend(task), latency_ms, callers=("agent",))


def nodes(task: Task, llm, voice=None, **user_kwargs):
    """``{"user": UserSim(LLMSource), "tools": ToolWorld}`` for one case. ``voice`` (a ``Voice``) makes the user speak."""
    from ..user import LLMSource, UserSim
    user = UserSim(LLMSource(llm), voice, **user_kwargs) if voice is not None else UserSim(LLMSource(llm), **user_kwargs)
    return {"user": user, "tools": tool_world(task)}


def evaluate(env, task: Task, node: str = "tools") -> dict:
    """SOPBench's verdict on the episode: its evaluator's fields (``success``, ``constraint_not_violated``,
    ``database_match``, ``action_called_correctly``, ``dirgraph_satisfied``, ...) plus ``reward`` (= success) and the
    calls as the evaluator saw them."""
    _, evaluator, _ = _sop()
    sim = getattr(env, "sim", env)
    st = next(iter(sim.states[node]["services"].values()))
    domain, rec = task.scenario["sopbench"]["domain"], task.initial_state["sopbench_case"]
    msgs = [{"sender": f.stream} for f in sim.log if f.stream in ("user.speech", "policy.speech") and getattr(f.data, "text", "")]
    res = evaluator.evaluator_function_directed_graph(domain, copy.deepcopy(rec), msgs + [{}], copy.deepcopy(st["calls"]),
                                                      {"final_database": st["system"].data}, "full")
    res = {k: v for k, v in res.items() if isinstance(v, (bool, int, float, str))}
    t_goal = next((t for t, caller, c, r in tool_log(sim.log) if caller == "agent" and c.name == rec["user_goal"]), None)
    return {**res, "reward": float(res["success"]), "calls": [{"name": c["tool_name"], "arguments": c["arguments"]} for c in st["calls"]],
            "target_call_time": t_goal}


def replay(task: Task, calls: list[tuple[str, dict]]) -> dict:
    """SOPBench's verdict for a fixed list of (function, arguments) calls, without an episode (for checks and tests)."""
    import asyncio
    import random

    from ..tools import ToolCall
    _, evaluator, _ = _sop()
    b = SOPBenchBackend(task)
    st = b.reset(task, random.Random(0))
    for i, (name, args) in enumerate(calls):
        asyncio.run(b.call(st, "agent", ToolCall(f"c{i}", name, args)))
    rec = task.initial_state["sopbench_case"]
    return evaluator.evaluator_function_directed_graph(b.domain, copy.deepcopy(rec), [{"sender": "u"}, {}], copy.deepcopy(st["calls"]),
                                                       {"final_database": st["system"].data}, "full")
