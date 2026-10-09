"""Tool-use environments behind one interface.

The env only knows ``ToolSpec`` / ``ToolCall`` / ``ToolResult`` and the ``ToolWorld``
node, which routes calls by caller, applies simulated latency and forks with the
episode. Any external tool environment (τ-bench domains, MCP servers, plain Python
functions, a robot skill API, ...) plugs in by implementing a ``ToolBackend`` adapter.

Streams: the agent calls on ``policy.tool_call`` and hears back on ``tool.result``;
a simulated user calls on ``user.tool_call`` and hears back on ``tool.user_result``.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .core import Frame, Node, Task

CALL = {"agent": "policy.tool_call", "user": "user.tool_call"}
RESULT = {"agent": "tool.result", "user": "tool.user_result"}
CALLER = {stream: caller for caller, stream in CALL.items()}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})  # JSON schema
    service: str = ""  # the backend (service) the tool belongs to

    def openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    id: str
    name: str
    content: str  # what the caller reads (JSON text for structured results)
    error: bool = False


class ToolBackend:
    """Adapter base class. Per-episode state comes from ``reset`` and is passed back to
    ``call``; keep everything mutable in it so that episodes fork cleanly. ``name`` and
    ``description`` identify the backend as a service in the discover-mode catalog."""

    name: str = "tools"
    description: str = ""
    state_group: str | None = None  # backends (services) with the same group share one per-episode state

    def tools(self, caller: str = "agent") -> list[ToolSpec]:
        return []

    def instructions(self, caller: str = "agent") -> str:
        """Text the caller needs in its prompt (e.g. a domain policy)."""
        return ""

    def context(self, caller: str, state: Any) -> dict:
        """What this environment chooses to reveal to ``caller`` when its tools are loaded (e.g. an
        account overview). The backend decides: anything secret, or meant to be obtained by calling
        a tool, simply stays out. Default: nothing."""
        return {}

    def reset(self, task: Task, rng) -> Any:
        return None

    async def call(self, state: Any, caller: str, call: ToolCall) -> ToolResult:
        raise NotImplementedError

    def fork(self, state: Any, n: int) -> list[Any]:
        return [copy.deepcopy(state) for _ in range(n)]

    def evaluate(self, task: Task, log: list[Frame], truncated: bool = False) -> dict | None:
        """Optional outcome reward computed by the backend from the episode log."""
        return None


META_TOOLS = [
    ToolSpec("list_services", "List the services whose tools can be loaded, with a short description of each."),
    ToolSpec("search_tools", "Search all services for tools matching a query; returns tool names, services and descriptions.",
             {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}),
    ToolSpec("load_tools", "Load every tool of a service so that it can be called; returns their schemas and the service's instructions.",
             {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}),
]
DISCOVER_INSTRUCTIONS = (
    "Tools are organised in services. Only the services' names and descriptions are known at first: "
    "use list_services or search_tools to find what you need, then load_tools(service) before calling a tool."
)


class ToolWorld(Node):
    """Executes tool calls through one or more backends (services); results arrive after
    ``latency_ms`` (a number, or a function of the call, e.g. slow search vs. instant lookup).

    How the agent learns what it can call (``mode``):

    - ``"all"``: every tool schema is in the session from the start (benchmark setting).
    - ``"discover"``: the session holds only a catalog of services and three meta-tools
      (``list_services``, ``search_tools``, ``load_tools``); a service's tools become callable
      once loaded, which also sends the agent a session update. Simulated users always see
      all of their own tools.

    With several services, tool names are qualified as ``<service>__<tool>``. Services whose
    backends set the same ``state_group`` operate on one shared world (reset once, forked together).
    """

    SESSION = "session"

    def __init__(self, backends: ToolBackend | dict[str, ToolBackend], latency_ms: int | Callable[[ToolCall], int] = 0,
                 callers=("agent", "user"), mode: str = "all"):
        if isinstance(backends, ToolBackend):
            backends = {backends.name: backends}
        assert mode in ("all", "discover"), mode
        self.backends = backends
        self.latency_ms = latency_ms
        self.mode = mode
        self.reads = tuple(CALL[c] for c in callers)
        qualify = len(backends) > 1
        self.specs: dict[str, dict[str, ToolSpec]] = {}  # caller -> exposed name -> spec
        self.route: dict[str, dict[str, tuple[str, str]]] = {}  # caller -> exposed name -> (service, backend's name)
        for caller in ("agent", "user"):
            self.specs[caller], self.route[caller] = {}, {}
            for svc, backend in backends.items():
                for spec in backend.tools(caller):
                    exposed = f"{svc}__{spec.name}" if qualify else spec.name
                    assert exposed not in self.specs[caller], f"duplicate tool {exposed}"
                    self.specs[caller][exposed] = dataclasses.replace(spec, name=exposed, service=svc)
                    self.route[caller][exposed] = (svc, spec.name)

    # ---- what the agent is told at the start of the episode

    def catalog(self) -> list[dict]:
        return [{"service": svc, "description": b.description, "tools": len(b.tools("agent"))} for svc, b in self.backends.items()]

    def instructions(self, services=None) -> str:
        parts = [b.instructions("agent") for svc, b in self.backends.items() if services is None or svc in services]
        return "\n\n".join(p for p in parts if p)

    def context(self, state, services) -> dict:
        """What the given services choose to reveal to the agent (namespaced by service when there are several)."""
        ctx = {svc: self.backends[svc].context("agent", state["services"][svc]) for svc in services}
        ctx = {svc: c for svc, c in ctx.items() if c}
        if len(self.backends) == 1:
            return next(iter(ctx.values()), {})
        return ctx

    def session(self, task: Task, state=None) -> dict:
        """This node's contribution to the agent's initial session. In discover mode a service's
        context is revealed only once it is loaded."""
        if self.mode == "all":
            ctx = self.context(state, self.backends) if state is not None else {}
            return {"mode": "all", "instructions": self.instructions(), "tools": list(self.specs["agent"].values()), "context": ctx}
        return {"mode": "discover", "instructions": DISCOVER_INSTRUCTIONS, "tools": list(META_TOOLS), "services": self.catalog()}

    # ---- episode

    def init_state(self, task, rng):
        services, groups = {}, {}
        for svc, b in self.backends.items():
            if b.state_group is None:
                services[svc] = b.reset(task, rng)
            else:  # one world for the whole group, reset by its first backend
                if b.state_group not in groups:
                    groups[b.state_group] = b.reset(task, rng)
                services[svc] = groups[b.state_group]
        return {"services": services, "loaded": []}

    async def step(self, state, t, inbox):
        out = []
        for f in inbox:
            caller, call = CALLER[f.stream], f.data
            lat = self.latency_ms(call) if callable(self.latency_ms) else self.latency_ms
            if caller == "agent" and self.mode == "discover" and call.name in {m.name for m in META_TOOLS}:
                res, update = self._meta(state, call)
                out.append(Frame(RESULT[caller], t + lat, res))
                if update is not None:
                    out.append(Frame(self.SESSION, t + lat, update))
                continue
            if call.name not in self.route[caller]:
                res = ToolResult(call.id, call.name, f"Error: unknown tool {call.name}", error=True)
            elif caller == "agent" and self.mode == "discover" and self.route[caller][call.name][0] not in state["loaded"]:
                svc = self.route[caller][call.name][0]
                res = ToolResult(call.id, call.name, f"Error: tool {call.name} is not loaded; call load_tools(service={svc!r}) first", error=True)
            else:
                svc, inner = self.route[caller][call.name]
                res = await self.backends[svc].call(state["services"][svc], caller, dataclasses.replace(call, name=inner))
                res = dataclasses.replace(res, name=call.name)
            out.append(Frame(RESULT[caller], t + lat, res))
        return out, None

    def _meta(self, state, call: ToolCall):
        name, args = call.name, call.arguments
        if name == "list_services":
            return ToolResult(call.id, name, json.dumps(self.catalog())), None
        if name == "search_tools":
            words = [w for w in str(args.get("query", "")).lower().replace("_", " ").split() if w]
            scored = []
            for spec in self.specs["agent"].values():
                hay = f"{spec.name.replace('_', ' ')} {spec.description} {spec.service}".lower()
                score = sum(w in hay for w in words)
                if score:
                    scored.append((score, spec))
            hits = [{"service": s.service, "name": s.name, "description": s.description} for _, s in sorted(scored, key=lambda x: -x[0])[:10]]
            return ToolResult(call.id, name, json.dumps(hits)), None
        svc = args.get("service")
        if svc not in self.backends:
            return ToolResult(call.id, name, f"Error: unknown service {svc!r}", error=True), None
        if svc not in state["loaded"]:
            state["loaded"].append(svc)
        tools = [s for s in self.specs["agent"].values() if s.service == svc]
        body = {"service": svc, "instructions": self.backends[svc].instructions("agent"), "tools": [s.openai() for s in tools]}
        own = self.backends[svc].context("agent", state["services"][svc])
        if own:
            body["context"] = own
        ctx = self.context(state, [svc])  # as merged into the session (namespaced when there are several services)
        loaded = [s for s in self.specs["agent"].values() if s.service in state["loaded"]]
        update = {"mode": "discover", "instructions": self.instructions(state["loaded"]), "tools": list(META_TOOLS) + loaded,
                  "services": self.catalog(), "context": ctx}
        return ToolResult(call.id, name, json.dumps(body)), update

    def fork(self, state, n):
        copies = {}  # id(state) -> its n copies: a state shared by several services is forked once and stays shared
        for svc, b in self.backends.items():
            s = state["services"][svc]
            if id(s) not in copies:
                copies[id(s)] = b.fork(s, n)
        return [{"services": {svc: copies[id(state["services"][svc])][i] for svc in self.backends}, "loaded": list(state["loaded"])}
                for i in range(n)]


def tool_log(log: list[Frame]) -> list[tuple[int, str, ToolCall, ToolResult | None]]:
    """(call time, caller, call, result) for every tool call in a log — for backend evaluators."""
    caller_of = {stream: caller for caller, stream in RESULT.items()}
    results = {(caller_of[f.stream], f.data.id): f.data for f in log if f.stream in caller_of}  # ids are per caller
    return [(f.t, CALLER[f.stream], f.data, results.get((CALLER[f.stream], f.data.id))) for f in log if f.stream in CALLER]


# ---------------------------------------------------------------- plain Python functions


def _spec_from_function(name: str, fn: Callable) -> ToolSpec:
    types = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}
    params = [p for p in inspect.signature(fn).parameters.values() if p.name != "state"]
    props = {p.name: {"type": types.get(p.annotation, "string")} for p in params}
    required = [p.name for p in params if p.default is inspect.Parameter.empty]
    return ToolSpec(name, inspect.getdoc(fn) or "", {"type": "object", "properties": props, "required": required})


class FunctionBackend(ToolBackend):
    """Tools from Python functions. A function may take a ``state`` argument (the episode's
    dict, from ``initial_state`` or ``task.initial_state``) to keep a mutable world."""

    def __init__(self, functions: dict[str, Callable], initial_state: dict | None = None, instructions: str = "",
                 name: str = "tools", description: str = "", context: Callable[[dict], dict] | None = None):
        """``context(state)`` returns what to reveal to the agent up front (default: nothing)."""
        self.functions = functions
        self.initial_state = initial_state
        self._instructions = instructions
        self.name, self.description = name, description
        self._context = context

    def context(self, caller, state):
        return self._context(state) if self._context is not None and caller == "agent" else {}

    def tools(self, caller="agent"):
        return [_spec_from_function(name, fn) for name, fn in self.functions.items()]

    def instructions(self, caller="agent"):
        return self._instructions

    def reset(self, task, rng):
        return copy.deepcopy(self.initial_state if self.initial_state is not None else task.initial_state)

    async def call(self, state, caller, call):
        fn = self.functions.get(call.name)
        if fn is None:
            return ToolResult(call.id, call.name, f"Error: unknown tool {call.name}", error=True)
        try:
            kwargs = dict(call.arguments)
            if "state" in inspect.signature(fn).parameters:
                kwargs["state"] = state
            out = fn(**kwargs)
            if inspect.isawaitable(out):
                out = await out
            return ToolResult(call.id, call.name, out if isinstance(out, str) else json.dumps(out, default=str))
        except Exception as e:  # tool errors are part of the interaction, not crashes
            return ToolResult(call.id, call.name, f"Error: {e}", error=True)
