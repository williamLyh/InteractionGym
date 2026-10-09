"""Tool use in InteractionGym, shown in the viewer. No real models: scripted user and agent.

Cases
  functions/slow-tool-with-filler   the agent says "one moment" while a slow search runs
  functions/user-talks-while-waiting the user adds a request while the tool is still pending
  functions/tool-error-and-retry    a call fails, the agent apologizes and retries
  functions/discover-tools          many services: the agent searches, loads a service, then calls it
  tau/retail-oracle                 a τ-bench retail task solved with its gold tool calls
  tau/mock-user-side-tool           a τ-bench task where the *user* calls a tool too

    uv sync --extra tau
    uv run python examples/tools_demo.py --html runs/tools_demo.html
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from interaction_gym import AgentSpec, Chunk, Env, Frame, Segment, Task
from interaction_gym.tools import CALL, RESULT, FunctionBackend, ToolCall, ToolResult, ToolWorld
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode
from interaction_gym.user import ScriptSource, TurnTaking, UserSim
from interaction_gym.viewer import export_html

SPEC = AgentSpec(chunk_ms=200, obs=("user.speech", RESULT["agent"]))

# ---------------------------------------------------------------- a small custom tool world

FLIGHTS = {"BA117": {"to": "New York", "price": 520, "seats": 3}, "VS3": {"to": "New York", "price": 480, "seats": 0}}


def search_flights(state: dict, destination: str) -> list:
    """Search flights to a destination (slow: hits an external system)."""
    return [{"flight": k, **v} for k, v in state["flights"].items() if v["to"] == destination]


def book_flight(state: dict, flight: str, seat: str = "any") -> dict:
    """Book one seat on a flight."""
    f = state["flights"].get(flight)
    if f is None:
        raise ValueError(f"unknown flight {flight}")
    if f["seats"] == 0:
        raise ValueError(f"{flight} is sold out")
    f["seats"] -= 1
    state["bookings"].append({"flight": flight, "seat": seat})
    return {"booked": flight, "seat": seat, "price": f["price"]}


def flight_world(search_ms=2500):
    backend = FunctionBackend({"search_flights": search_flights, "book_flight": book_flight}, {"flights": FLIGHTS, "bookings": []})
    return ToolWorld(backend, lambda call: search_ms if call.name == "search_flights" else 300)


# ---------------------------------------------------------------- scripted agent


class ScriptedToolAgent:
    """Runs a list of steps in order:
    ("wait_user", n)  until the user's n-th turn has ended (+300 ms)
    ("say", text)     start speaking
    ("call", name, arguments)
    ("wait_results",) until every pending tool call has returned
    ("wait_speech",)  until the agent's own speech has ended
    """

    def __init__(self, steps, spec=SPEC, wps=3.4):
        self.steps, self.spec, self.wps = list(steps), spec, wps
        self.user_ends: dict[str, int] = {}
        self.pending: set[str] = set()
        self.speaking_until = 0
        self.n_calls = self.n_says = 0

    def act(self, t, obs):
        for f in obs:
            if isinstance(f.data, Chunk) and f.data.last:
                self.user_ends[f.data.id] = f.data.t0 + f.data.dur
            if isinstance(f.data, ToolResult):
                self.pending.discard(f.data.id)
        out = []
        while self.steps:
            kind, *args = self.steps[0]
            if kind == "wait_user":
                ends = sorted(self.user_ends.values())
                if len(ends) < args[0] or t - ends[args[0] - 1] < 300:
                    break
            elif kind == "wait_results":
                if self.pending:
                    break
            elif kind == "wait_speech":
                if t < self.speaking_until:
                    break
            elif kind == "say":
                seg = Segment(f"a{self.n_says}", t, round(len(args[0].split()) / self.wps * 1000), args[0])
                self.n_says, self.speaking_until = self.n_says + 1, seg.end
                out.append(Frame(self.spec.out, t, seg))
            elif kind == "call":
                self.n_calls += 1
                call = ToolCall(f"call_{self.n_calls}", args[0], args[1])
                self.pending.add(call.id)
                out.append(Frame(CALL["agent"], t, call))
            self.steps.pop(0)
        return out


async def run(nodes, task, agent, inject=None, max_ms=60_000):
    env = Env(nodes, SPEC, max_ms=max_ms)
    obs = await env.reset(task)
    done = False
    while not done:
        action = agent.act(env.t, obs)
        if inject and env.t >= inject[0]:
            action, inject = action + [inject[1]], None
        obs, _, done = await env.step(action)
    return env


def script_user(*turns, **timing):
    return UserSim(ScriptSource(list(turns)), timing=TurnTaking(**timing))


# ---------------------------------------------------------------- cases


async def slow_tool_with_filler():
    user = script_user("Hi, are there any flights to New York tomorrow?", "Great, thank you!")
    agent = ScriptedToolAgent([
        ("wait_user", 1),
        ("call", "search_flights", {"destination": "New York"}),
        ("say", "One moment, let me look that up for you."),
        ("wait_results",), ("wait_speech",),
        ("say", "I found flight BA117 with three seats left for five hundred and twenty dollars."),
    ])
    return await run({"user": user, "tools": flight_world()}, Task(id="flights"), agent), {"search_latency_ms": 2500}


async def user_talks_while_waiting():
    user = script_user("Hi, I need a flight to New York tomorrow.", "Oh, and a window seat if possible.", "Perfect, thanks!", respond_after_ms=600)
    agent = ScriptedToolAgent([
        ("wait_user", 1),
        ("call", "search_flights", {"destination": "New York"}),
        ("say", "Sure, let me check."),
        ("wait_results",), ("wait_user", 2),
        ("call", "book_flight", {"flight": "BA117", "seat": "window"}),
        ("wait_results",),
        ("say", "Done. You are booked on BA117 with a window seat."),
    ])
    return await run({"user": user, "tools": flight_world(search_ms=5000)}, Task(id="flights"), agent), {"search_latency_ms": 5000}


async def tool_error_and_retry():
    user = script_user("Please book me on flight VS3 to New York.", "Okay, that works, thanks.")
    agent = ScriptedToolAgent([
        ("wait_user", 1),
        ("call", "book_flight", {"flight": "VS3"}),
        ("wait_results",),
        ("say", "I'm sorry, VS3 is sold out. Let me book BA117 instead."),
        ("call", "book_flight", {"flight": "BA117"}),
        ("wait_results",), ("wait_speech",),
        ("say", "You are now booked on BA117."),
    ])
    return await run({"user": user, "tools": flight_world()}, Task(id="flights"), agent), {}


def weather(city: str) -> str:
    """Current weather for a city."""
    return f"sunny in {city}"


async def discover_tools():
    flights = FunctionBackend({"search_flights": search_flights, "book_flight": book_flight}, {"flights": FLIGHTS, "bookings": []},
                              instructions="Always confirm the price before booking.", name="flights", description="Search and book flights")
    world = ToolWorld(
        {"flights": flights, "weather": FunctionBackend({"get_weather": weather}, name="weather", description="Weather forecasts")},
        lambda call: {"search_tools": 600, "load_tools": 400, "flights__search_flights": 2500}.get(call.name, 300),
        mode="discover",
    )
    user = script_user("Hi, are there any flights to New York tomorrow?", "Great, thank you!")
    agent = ScriptedToolAgent([
        ("wait_user", 1),
        ("say", "Let me check what I can do for that."),
        ("call", "search_tools", {"query": "flights"}),
        ("wait_results",),
        ("call", "load_tools", {"service": "flights"}),
        ("wait_results",),
        ("call", "flights__search_flights", {"destination": "New York"}),
        ("wait_results",), ("wait_speech",),
        ("say", "I found flight BA117 with three seats left for five hundred and twenty dollars."),
    ])
    return await run({"user": user, "tools": world}, Task(id="flights"), agent), {}


async def tau_cases():
    from interaction_gym.integrations import tau

    out = []
    user = lambda: script_user("Hi, I need some help with my order.", "Great, thanks!")  # noqa: E731

    task = tau.load_task("retail", "1")
    env = await run({"user": user(), "tools": ToolWorld(tau.TauBackend("retail"), 400)}, task, tau.OracleAgent(task, SPEC), max_ms=120_000)
    out.append(("tau/retail-oracle", env, {"tau": tau.evaluate(env.log, task, env.truncated)["breakdown"], "tool_latency_ms": 400}))

    task = tau.load_task("mock", "update_task_with_user_tools")
    dismiss = Frame(CALL["user"], 1200, ToolCall("u1", "dismiss_notification", {"notification_id": "notif_1"}), src="user")
    env = await run({"user": user(), "tools": ToolWorld(tau.TauBackend("mock"), 400)}, task, tau.OracleAgent(task, SPEC), inject=(1200, dismiss))
    out.append(("tau/mock-user-side-tool", env, {"tau": tau.evaluate(env.log, task, env.truncated)["breakdown"], "tool_latency_ms": 400}))
    return out


async def main(html: str):
    cases = []
    for fn in (slow_tool_with_filler, user_talks_while_waiting, tool_error_and_retry, discover_tools):
        env, meta = await fn()
        cases.append((f"functions/{fn.__name__.replace('_', '-')}", env, meta))
    try:
        cases += await tau_cases()
    except ImportError:
        print("tau2 not installed; skipping τ-bench cases (uv sync --extra tau)")
    media = MediaStore(Path(html).parent)
    records = []
    for name, env, meta in cases:
        tau_parts = meta.pop("tau", None)
        reward = {"total": tau_parts.get("DB"), "parts": tau_parts} if tau_parts else None  # τ-bench's checks as reward
        records.append(episode(env, name, media=media, meta=meta, reward=reward))
    print(f"{len(records)} cases -> {export_html(records, html)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default="runs/tools_demo.html")
    asyncio.run(main(ap.parse_args().html))
