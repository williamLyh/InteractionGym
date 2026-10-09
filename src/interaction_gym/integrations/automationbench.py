"""AutomationBench adapter (Zapier, MIT; arXiv 2604.18934; github.com/zapier/AutomationBench).

AutomationBench tasks are business workflows across 47 simulated SaaS apps (Gmail, Salesforce,
Sheets, Slack, ...): one request, a pre-populated world (policies are often *buried in it*, e.g. an
email in the inbox), and end-state assertions. The whole world is a local Python simulator
(pydantic ``WorldState`` + per-app tool functions); nothing calls Zapier.

- ``tasks`` / ``load_task``: the public tasks → our ``Task``. The request becomes what the
  simulated user says (``scenario["turns"]``); the world goes to ``initial_state``; the assertions
  to ``criteria``. Policies stay inside the world, exactly where the benchmark put them.
- Toolsets (``tool_world`` / ``nodes``):

  ``"api"`` (official leaderboard setting): ``api_search`` (BM25 over ~500 REST endpoint schemas) +
  ``api_fetch`` (method, URL, params, body) + ``base64_encode``; the agent discovers endpoints itself.
  ``"zapier"``: the benchmark's ``search_tools`` / ``execute_tool`` meta-tools over its Zapier actions.
  ``"limited_zapier"``: only the task's own Zapier actions, called directly (one service per app).
  ``"apps"``: every Zapier action, one service per app, behind ToolWorld's ``discover`` mode
  (``list_services`` / ``search_tools`` / ``load_tools``) — our discovery instead of theirs.

  App services share one world (``state_group``), so cross-app workflows and evaluation see one state.
- ``evaluate``: the official rubric (``partial_credit`` with free/negative-assertion handling, and
  ``task_completed_correctly`` = the binary score) on the episode's end-state world.
- ``ReplayAgent`` + ``HANDWRITTEN_SOLUTIONS``: the benchmark ships no reference solutions; these
  call sequences were written by hand to check that reward 1 is reachable through our plumbing.

AutomationBench declares Python >= 3.13 but the parts used here run on 3.12; it is therefore not a
uv dependency. Put a checkout at ``third_party/automationbench`` (or point ``AUTOMATIONBENCH_PATH``
at one); its ``datasets`` / ``verifiers`` dependencies are not needed.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
import re
import sys
import types
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Callable

from ..core import AgentSpec, Chunk, Frame, Node, Segment, Task
from ..tools import CALL, RESULT, ToolBackend, ToolCall, ToolResult, ToolSpec, ToolWorld


def _import_automationbench():
    try:
        import automationbench  # noqa: F401
        return
    except ImportError:
        pass
    here = Path(__file__).resolve().parents[3] / "third_party" / "automationbench"
    for root in (os.environ.get("AUTOMATIONBENCH_PATH"), here):
        if root and (Path(root) / "automationbench" / "__init__.py").is_file():
            sys.path.insert(0, str(root))
            break
    import automationbench  # noqa: F401  (ImportError here means: not installed / not cloned)


_import_automationbench()

from automationbench.schema.world import WorldState  # noqa: E402

DOMAINS = ("sales", "marketing", "operations", "support", "finance", "hr")  # the scored public set
ALL_DOMAINS = DOMAINS + ("simple",)  # "simple": 200 warm-up tasks, not part of the benchmark score
TOOLSETS = ("api", "zapier", "limited_zapier", "apps")

# ---------------------------------------------------------------- tasks


class _Rows(list):
    """Stand-in for ``datasets.Dataset`` while loading tasks (the task modules only call from_list)."""

    @classmethod
    def from_list(cls, rows):
        return cls(rows)


@contextmanager
def _datasets():
    if importlib.util.find_spec("datasets") is not None:
        yield
        return
    shim = types.ModuleType("datasets")
    shim.Dataset = _Rows
    shim.concatenate_datasets = lambda parts: _Rows(r for p in parts for r in p)
    sys.modules["datasets"] = shim
    try:
        yield
    finally:  # the task modules keep their own reference; don't leave a fake package behind
        sys.modules.pop("datasets", None)


def _strip_none(obj):
    """As the official runner does before building the world (and harmless otherwise)."""
    if isinstance(obj, dict):
        return {k: _strip_none(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none(x) for x in obj if x is not None]
    return obj


@cache
def _rows(domain: str) -> tuple[dict, ...]:
    with _datasets():
        from automationbench.domains import DOMAINS as LOADERS

        rows = list(LOADERS[domain]())
    out = []
    for r in rows:
        info = r["info"] if isinstance(r["info"], dict) else json.loads(r["info"])
        out.append({"example_id": r["example_id"], "prompt": [dict(m) for m in r["prompt"]], "info": info})
    return tuple(out)


@cache
def _by_name() -> dict[str, tuple[str, dict]]:
    return {row["info"]["task_name"]: (d, row) for d in ALL_DOMAINS for row in _rows(d)}


@cache
def system_prompt() -> str:
    """The agent's system prompt (the same for every task in this release)."""
    prompts = {m["content"] for d in ALL_DOMAINS for row in _rows(d) for m in row["prompt"] if m["role"] == "system"}
    assert len(prompts) == 1, "tasks with different system prompts: build the backend with instructions=..."
    return prompts.pop()


USER_GOAL = (
    "You are asking a workflow-automation assistant to do the following for you. Say it in your own words if you "
    "like, but keep every name, number, address and identifier exactly as written, and do not add anything: the "
    "assistant has to work it out alone.\n\n{request}"
)


def to_task(domain: str, row: dict) -> Task:
    info = row["info"]
    request = next(m["content"] for m in row["prompt"] if m["role"] == "user")
    return Task(
        id=f"automationbench/{info['task_name']}",
        scenario={
            "benchmark": "automationbench",
            "domain": domain,
            "task_name": info["task_name"],
            "example_id": row["example_id"],
            "request": request,
            "turns": [request],  # a scripted user says the request verbatim (the benchmark's single trigger)
            "first_turn": request,  # an LLM user opens with it too, then only answers
            "instructions": USER_GOAL.format(request=request),
            "zapier_tools": list(info.get("zapier_tools", [])),
        },
        initial_state=_strip_none(info.get("initial_state", {})),
        criteria={"assertions": [_strip_none(a) for a in info.get("assertions", [])]},
    )


def tasks(domain: str | None = None) -> list[Task]:
    """Public tasks of one domain, or of the six scored domains (``"simple"`` must be asked for)."""
    return [to_task(d, row) for d in ([domain] if domain else DOMAINS) for row in _rows(d)]


def load_task(name: str) -> Task:
    """By task name (``"sales.multi_hop_lookup"``) or Task id (``"automationbench/sales.multi_hop_lookup"``)."""
    domain, row = _by_name()[name.removeprefix("automationbench/")]
    return to_task(domain, row)


# ---------------------------------------------------------------- world


def _service_fields() -> list[str]:
    return sorted((f for f in WorldState.model_fields if f != "meta"), key=len, reverse=True)


def service_of(name: str) -> str | None:
    """The WorldState service a tool name or assertion type belongs to (``gmail_send_email`` → ``gmail``)."""
    for field in _service_fields():
        if name == field or name.startswith(field + "_"):
            return field
    return None


def allowed_services(task: Task) -> list[str]:
    """Services the task's workspace is connected to (as ``automationbench.runner.compute_allowed_services``,
    copied because that module imports ``verifiers``): seeded, asserted on, or granted a Zapier tool.
    ``api_fetch`` answers anything else with a 401 "no account connected"."""
    allowed = {k for k in task.initial_state if k != "meta" and k in WorldState.model_fields}
    allowed |= {s for a in task.criteria.get("assertions", []) if (s := service_of(str(a.get("type", ""))))}
    allowed |= {s for t in task.scenario.get("zapier_tools", []) if (s := service_of(t))}
    return sorted(allowed)


def make_world(task: Task) -> WorldState:
    world = WorldState(**copy.deepcopy(task.initial_state))
    world.meta.allowed_services = allowed_services(task)
    return world


# ---------------------------------------------------------------- tool schemas and calls


def _doc_sections(fn: Callable) -> tuple[str, dict[str, str]]:
    """Description (text before the Google-style sections) and per-argument descriptions."""
    doc = inspect.getdoc(fn) or ""
    head = re.split(r"\n\s*(?:Args|Arguments|Parameters|Returns|Raises|Examples?):\s*\n", doc, maxsplit=1)[0].strip()
    args: dict[str, str] = {}
    m = re.search(r"\n\s*(?:Args|Arguments|Parameters):\s*\n(.*?)(?:\n\s*(?:Returns|Raises|Examples?):\s*\n|\Z)", doc, re.S)
    if m:
        name = None
        for line in m.group(1).splitlines():
            hit = re.match(r"\s{0,8}(\w+)(?:\s*\([^)]*\))?:\s*(.*)", line)
            if hit and (name is None or len(line) - len(line.lstrip()) <= 4):
                name = hit.group(1)
                args[name] = hit.group(2).strip()
            elif name and line.strip():
                args[name] = f"{args[name]} {line.strip()}".strip()
    return head or fn.__name__, args


def _schema(fn: Callable) -> dict:
    """JSON schema of a tool function's arguments (``world`` excluded), descriptions from its docstring."""
    from pydantic import create_model

    _, arg_docs = _doc_sections(fn)
    hints = inspect.get_annotations(fn, eval_str=True)
    fields = {}
    for p in inspect.signature(fn).parameters.values():
        if p.name == "world":
            continue
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (hints.get(p.name, Any), default)
    schema = create_model(f"{fn.__name__}_args", **fields).model_json_schema()

    def clean(node):
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    out = {"type": "object", "properties": clean(schema.get("properties", {})), "required": schema.get("required", [])}
    for name, desc in arg_docs.items():
        if name in out["properties"] and desc:
            out["properties"][name]["description"] = desc
    if "$defs" in schema:
        out["$defs"] = clean(schema["$defs"])
    return out


def _spec(fn: Callable, name: str | None = None) -> ToolSpec:
    return ToolSpec(name or fn.__name__, _doc_sections(fn)[0], _schema(fn))


def _invoke(fn: Callable, world: WorldState, call: ToolCall, name: str) -> ToolResult:
    # As the official env: an empty object stands for "no value" (models emit {} for optional args).
    kwargs = {k: v for k, v in call.arguments.items() if not (isinstance(v, dict) and not v)}
    if "world" in inspect.signature(fn).parameters:
        kwargs["world"] = world
    try:
        out = fn(**kwargs)
    except Exception as e:  # tool errors are part of the interaction, not crashes
        return ToolResult(call.id, name, f"Error: {type(e).__name__}: {e}", error=True)
    return ToolResult(call.id, name, out if isinstance(out, str) else json.dumps(out, default=str))


@cache
def _zapier_tools() -> dict[str, Callable]:
    from automationbench.tools import ALL_TOOLS

    return {fn.__name__: fn for fn in ALL_TOOLS}


def _toolset_functions(toolset: str, search_top_k: int | None) -> list[Callable]:
    if toolset == "api":
        from automationbench.tools.api import API_TOOLS

        return list(API_TOOLS)
    if toolset == "zapier":
        from automationbench.tools.zapier import meta
        from automationbench.tools.zapier.meta import execute_tool, make_search_tools, search_tools

        if meta._registry is None and importlib.util.find_spec("agents") is None:
            # Their registry builds parameter schemas with the openai-agents SDK; without it, use ours
            # (same arguments and docstring descriptions; small formatting differences are possible).
            class Registry(meta.ToolRegistry):
                _get_parameter_schema = staticmethod(lambda fn: {k: v for k, v in _schema(fn).items() if k != "$defs"})

            meta._registry = Registry(list(_zapier_tools().values()))
        return [make_search_tools(max_top_k=search_top_k) if search_top_k is not None else search_tools, execute_tool]
    raise ValueError(f"toolset {toolset!r}: use app_backends() for per-app services")


# ---------------------------------------------------------------- backends


class AutomationBenchBackend(ToolBackend):
    """One AutomationBench workspace behind one of the benchmark's own toolsets (``"api"``: official;
    ``"zapier"``: its search_tools / execute_tool meta-tools). Per-episode state is the ``WorldState``.
    Nothing about the world is revealed up front (``context`` is empty): policies are read from it."""

    state_group = "automationbench"

    def __init__(self, toolset: str = "api", *, name: str = "automationbench", instructions: str | None = None,
                 search_top_k: int | None = 20):
        """``search_top_k``: hard cap on the zapier ``search_tools`` top_k (the official CLI's default is 20)."""
        self.toolset, self.name = toolset, name
        self.description = f"AutomationBench workspace: 47 simulated business apps ({toolset} toolset)"
        self.functions = {fn.__name__: fn for fn in _toolset_functions(toolset, search_top_k)}
        self._instructions = system_prompt() if instructions is None else instructions

    def tools(self, caller="agent"):
        return [_spec(fn) for fn in self.functions.values()] if caller == "agent" else []

    def instructions(self, caller="agent"):
        return self._instructions if caller == "agent" else ""

    def reset(self, task, rng):
        return make_world(task)

    async def call(self, world, caller, call):
        fn = self.functions.get(call.name)
        if fn is None:
            return ToolResult(call.id, call.name, f"Error: unknown tool {call.name}", error=True)
        return _invoke(fn, world, call, call.name)


def app_title(app: str) -> str:
    return {"chatgpt": "ChatGPT (OpenAI)", "bamboohr": "BambooHR", "hubspot": "HubSpot", "zoho_desk": "Zoho Desk"}.get(
        app, app.replace("_", " ").title())


class AppBackend(ToolBackend):
    """One app's Zapier actions as a service. All app services of an episode share one ``WorldState``
    (``state_group``); tool names drop the app prefix when ``strip_prefix`` (``gmail__send_email``)."""

    state_group = "automationbench"

    def __init__(self, app: str, functions: list[Callable], strip_prefix: bool = True, instructions: str = ""):
        self.name = app
        self.description = f"{app_title(app)} (simulated): {len(functions)} actions"
        cut = len(app) + 1 if strip_prefix else 0
        self.functions = {fn.__name__[cut:]: fn for fn in functions}
        self._instructions = instructions

    def tools(self, caller="agent"):
        return [_spec(fn, name) for name, fn in self.functions.items()] if caller == "agent" else []

    def instructions(self, caller="agent"):
        return self._instructions if caller == "agent" else ""

    def reset(self, task, rng):
        return make_world(task)

    async def call(self, world, caller, call):
        fn = self.functions.get(call.name)
        if fn is None:
            return ToolResult(call.id, call.name, f"Error: unknown tool {call.name}", error=True)
        return _invoke(fn, world, call, call.name)


def app_backends(tool_names: list[str] | None = None, strip_prefix: bool | None = None) -> dict[str, AppBackend]:
    """Zapier actions (all, or the given names) grouped into one service per app."""
    registry = _zapier_tools()
    names = list(registry) if tool_names is None else list(tool_names)
    unknown = set(names) - set(registry)
    if unknown:
        raise ValueError(f"unknown AutomationBench tools: {sorted(unknown)}")
    groups: dict[str, list[Callable]] = {}
    for n in names:
        groups.setdefault(service_of(n) or n.split("_")[0], []).append(registry[n])
    strip = len(groups) > 1 if strip_prefix is None else strip_prefix
    return {app: AppBackend(app, fns, strip) for app, fns in sorted(groups.items())}


class SystemPrompt(Node):
    """Contributes the benchmark's system prompt to the agent's session. Used with the app toolsets,
    where it belongs to no single service (in discover mode a service's instructions arrive only
    when it is loaded)."""

    def __init__(self, text: str | None = None):
        self.text = system_prompt() if text is None else text

    def session(self, task, state=None):
        return {"instructions": self.text}


def tool_world(task: Task | None = None, toolset: str = "api", latency_ms: int | Callable[[ToolCall], int] = 0,
               mode: str | None = None, **kwargs) -> ToolWorld:
    """The ToolWorld for a toolset. ``"limited_zapier"`` needs the task (its granted tools); ``"apps"``
    defaults to discover mode, the others to ``"all"``. The system prompt is in the session's
    instructions for ``"api"`` / ``"zapier"``; for the app toolsets add a ``SystemPrompt`` node (``nodes``)."""
    assert toolset in TOOLSETS, toolset
    if toolset in ("api", "zapier"):
        return ToolWorld(AutomationBenchBackend(toolset, **kwargs), latency_ms, callers=("agent",), mode=mode or "all")
    if toolset == "limited_zapier":
        assert task is not None, "limited_zapier exposes the task's own tools: pass the task"
        return ToolWorld(app_backends(task.scenario["zapier_tools"]), latency_ms, callers=("agent",), mode=mode or "all")
    return ToolWorld(app_backends(), latency_ms, callers=("agent",), mode=mode or "discover")


def nodes(task: Task | None = None, toolset: str = "api", latency_ms: int | Callable[[ToolCall], int] = 0,
          mode: str | None = None, **kwargs) -> dict[str, Node]:
    """``{"tools": ToolWorld}`` plus, for the app toolsets, ``{"brief": SystemPrompt()}``. Add your user."""
    out: dict[str, Node] = {"tools": tool_world(task, toolset, latency_ms, mode, **kwargs)}
    if toolset in ("limited_zapier", "apps"):
        out["brief"] = SystemPrompt()
    return out


def rest_latency(search_ms: int = 800, read_ms: int = 300, write_ms: int = 600, other_ms: int = 300) -> Callable[[ToolCall], int]:
    """A simulated-latency function: slow searches, faster reads than writes (by HTTP method / action
    name); ``list_services`` / ``load_tools`` etc. take ``other_ms``."""
    reads = ("find", "get", "list", "lookup", "search", "query")

    def latency(call: ToolCall) -> int:
        name = call.name.split("__")[-1]
        if name in ("api_search", "search_tools"):
            return search_ms
        if name == "api_fetch":
            return read_ms if str(call.arguments.get("method", "GET")).upper() == "GET" else write_ms
        if name in ("list_services", "load_tools", "base64_encode"):
            return other_ms
        if name == "execute_tool":
            name = str(call.arguments.get("tool_name", ""))
        return read_ms if any(w in name.split("_") for w in reads) else write_ms

    return latency


# ---------------------------------------------------------------- evaluation


def world_of(env, node: str = "tools") -> WorldState:
    """The episode's (end-state) world, from an ``Env``, a ``Sim`` or a ToolWorld state."""
    sim = getattr(env, "sim", env)
    state = sim.states[node] if hasattr(sim, "states") else env
    worlds = {id(s): s for s in state["services"].values() if isinstance(s, WorldState)}
    assert len(worlds) == 1, f"expected one shared AutomationBench world, found {len(worlds)}"
    return next(iter(worlds.values()))


def evaluate(env_or_world, task: Task, truncated: bool = False, node: str = "tools") -> dict:
    """Official AutomationBench scoring of the end state (``automationbench.rubric``).

    Returns ``{"reward", "partial_credit", "parts", "breakdown", "truncated"}``: ``reward`` is
    ``task_completed_correctly`` (1.0 iff every scored assertion passes — the leaderboard metric, no
    partial credit); ``partial_credit`` is their dense training signal (assertions already true in
    the initial state are excluded unless broken); ``breakdown`` lists every assertion with
    ``passed`` / ``excluded``. Truncation does not change the score (the official harness also
    scores whatever state a run that hit its step limit left behind).

    The world is needed, not just the log: ids in the simulator are random, so a log cannot be
    replayed into the same state.
    """
    from automationbench.rubric import partial_credit, task_completed_correctly

    world = env_or_world if isinstance(env_or_world, WorldState) else world_of(env_or_world, node)
    state = {"info": {"assertions": copy.deepcopy(task.criteria["assertions"])}, "world": world,
             "initial_state": copy.deepcopy(task.initial_state)}
    pc = partial_credit(state)
    passed = task_completed_correctly(state)
    breakdown = state.get("_assertion_results", [])
    return {"reward": passed, "partial_credit": pc, "parts": {"task_completed_correctly": passed, "partial_credit": pc},
            "breakdown": breakdown, "truncated": truncated}


# ---------------------------------------------------------------- scripted agents


class ReplayAgent:
    """After the user's request has been said, makes the given tool calls one at a time (each after
    the previous result), then says ``say``. ``say_first`` is said while the first call runs."""

    def __init__(self, calls: list[tuple[str, dict]], say: str = "Done, the workflow is complete.", say_first: str | None = None,
                 spec: AgentSpec = AgentSpec(obs=("user.speech", RESULT["agent"])), wps: float = 3.4, wait_ms: int = 300):
        self.calls, self.say, self.say_first = list(calls), say, say_first
        self.spec, self.wps, self.wait_ms = spec, wps, wait_ms
        self.user_done_at: int | None = None
        self.pending: str | None = None
        self.n = 0
        self.results: list[ToolResult] = []
        self.spoke = False

    def _speak(self, t: int, text: str, sid: str) -> Frame:
        return Frame(self.spec.out, t, Segment(sid, t, round(len(text.split()) / self.wps * 1000), text))

    def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        for f in obs:
            if isinstance(f.data, Chunk) and f.data.last and self.user_done_at is None:
                self.user_done_at = f.data.t0 + f.data.dur
            if isinstance(f.data, ToolResult) and f.data.id == self.pending:
                self.results.append(f.data)
                self.pending = None
        if self.user_done_at is None or t - self.user_done_at < self.wait_ms or self.pending:
            return []
        out = []
        if self.n < len(self.calls):
            if self.n == 0 and self.say_first:
                out.append(self._speak(t, self.say_first, "a_ack"))
            name, args = self.calls[self.n]
            self.n += 1
            self.pending = f"call_{self.n}"
            out.append(Frame(CALL["agent"], t, ToolCall(self.pending, name, args)))
        elif not self.spoke:
            self.spoke = True
            out.append(self._speak(t, self.say, "a_done"))
        return out


_SF = "https://yourinstance.salesforce.com/services/data/v61.0"
_GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"


def _raw_email(to: str, subject: str, body: str) -> str:
    import base64

    return base64.urlsafe_b64encode(f"To: {to}\r\nSubject: {subject}\r\n\r\n{body}".encode()).decode()


# Hand-written (not from the benchmark, which ships no solutions): (task name, toolset) -> tool calls.
# The "api" ones include discovery and reading the policy email; the sales one skips the sheet lookups (tier, FX).
HANDWRITTEN_SOLUTIONS: dict[tuple[str, str], list[tuple[str, dict]]] = {
    ("simple.email_sf_contact_phone_update", "api"): [
        ("api_search", {"query": "gmail messages list"}),
        ("api_fetch", {"method": "GET", "url": f"{_GMAIL}/messages", "params": json.dumps({"q": "Jordan Lee"})}),
        ("api_fetch", {"method": "GET", "url": f"{_GMAIL}/messages/msg_3001"}),
        ("api_search", {"query": "salesforce contact update"}),
        ("api_fetch", {"method": "GET", "url": f"{_SF}/query", "params": json.dumps({"q": "SELECT Id, Name, Phone FROM Contact WHERE Name = 'Jordan Lee'"})}),
        ("api_fetch", {"method": "PATCH", "url": f"{_SF}/sobjects/Contact/003001", "body": json.dumps({"Phone": "+1-555-0101"})}),
    ],
    ("simple.email_sf_contact_phone_update", "limited_zapier"): [
        ("gmail__find_email", {"query": "Jordan Lee"}),
        ("salesforce__find_records", {"object": "Contact", "searchField": "Name", "searchValue": "Jordan Lee"}),
        ("salesforce__contact_update", {"id": "003001", "phone": "+1-555-0101"}),
    ],
    ("simple.email_sf_contact_phone_update", "apps"): [
        ("search_tools", {"query": "find email gmail"}),
        ("load_tools", {"service": "gmail"}),
        ("gmail__find_email", {"query": "Jordan Lee"}),
        ("load_tools", {"service": "salesforce"}),
        ("salesforce__find_records", {"object": "Contact", "searchField": "Name", "searchValue": "Jordan Lee"}),
        ("salesforce__contact_update", {"id": "003001", "phone": "+1-555-0101"}),
    ],
    ("simple.email_sf_contact_phone_update", "zapier"): [
        ("search_tools", {"query": "salesforce update contact"}),
        ("execute_tool", {"tool_name": "salesforce_contact_update", "arguments": json.dumps({"id": "003001", "phone": "+1-555-0101"})}),
    ],
    ("sales.multi_hop_lookup", "api"): [
        ("api_search", {"query": "gmail messages list"}),
        ("api_fetch", {"method": "GET", "url": f"{_GMAIL}/messages", "params": json.dumps({"q": "routing policy"})}),
        ("api_fetch", {"method": "GET", "url": f"{_GMAIL}/messages/msg_routing_policy"}),
        ("api_fetch", {"method": "GET", "url": f"{_SF}/query",
                       "params": json.dumps({"q": "SELECT Id, Name, Amount, Currency, AccountId FROM Opportunity WHERE Name LIKE '%Meridian%'"})}),
        ("api_fetch", {"method": "PATCH", "url": f"{_SF}/sobjects/Opportunity/006xx000004MER1", "body": json.dumps({"StageName": "Closed Won"})}),
        ("base64_encode", {"text": "To: executive-team@example.com\r\nSubject: Deal Closed Notification\r\n\r\n"
                                   "Meridian Corp - Platform Deal (Enterprise) closed won: $156,000."}),
        ("api_fetch", {"method": "POST", "url": f"{_GMAIL}/messages/send", "body": json.dumps({"raw": _raw_email(
            "executive-team@example.com", "Deal Closed Notification", "Meridian Corp - Platform Deal (Enterprise) closed won: $156,000.")})}),
        ("api_fetch", {"method": "POST", "url": f"{_GMAIL}/messages/send", "body": json.dumps({"raw": _raw_email(
            "support-escalation@example.com", "Deal Closed Notification",
            "Meridian Corp - Platform Deal (Enterprise) closed won: $156,000. The account has open escalations.")})}),
    ],
}
