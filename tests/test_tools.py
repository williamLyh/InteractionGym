"""The generic tool interface, exercised with FunctionBackend (no external dependencies)."""

import asyncio

from interaction_gym import AgentSpec, Env, Frame, Task
from interaction_gym.tools import CALL, RESULT, FunctionBackend, ToolCall, ToolResult, ToolWorld, tool_log


def add_item(state: dict, name: str, qty: int = 1) -> dict:
    """Add an item to the cart."""
    state["cart"][name] = state["cart"].get(name, 0) + qty
    return state["cart"]


def weather(city: str) -> str:
    """Current weather for a city."""
    return f"sunny in {city}"


def backend():
    return FunctionBackend({"add_item": add_item, "weather": weather}, instructions="Be nice.")


def make_env(latency=0):
    return Env({"tools": ToolWorld(backend(), latency)}, AgentSpec(chunk_ms=100, obs=(RESULT["agent"],)))


def run(coro):
    return asyncio.run(coro)


def test_specs_from_functions():
    specs = {s.name: s for s in backend().tools()}
    assert specs["add_item"].parameters["properties"] == {"name": {"type": "string"}, "qty": {"type": "integer"}}
    assert specs["add_item"].parameters["required"] == ["name"] and "state" not in specs["add_item"].parameters["properties"]
    assert specs["weather"].openai()["function"]["description"] == "Current weather for a city."


def test_calls_results_errors_and_latency():
    async def main():
        env = make_env(latency=lambda call: 900 if call.name == "weather" else 0)
        await env.reset(Task(initial_state={"cart": {}}))
        calls = [ToolCall("1", "add_item", {"name": "tea", "qty": 2}), ToolCall("2", "weather", {"city": "Paris"}), ToolCall("3", "nope")]
        seen = []
        obs, _, _ = await env.step([Frame(CALL["agent"], 0, c) for c in calls])
        seen += obs
        for _ in range(10):
            obs, _, _ = await env.step([])
            seen += obs
        return env, seen

    env, seen = run(main())
    by_id = {f.data.id: f for f in seen}
    assert by_id["1"].data == ToolResult("1", "add_item", '{"tea": 2}') and by_id["1"].t == 0
    assert by_id["2"].t == 900 and by_id["2"].data.content == "sunny in Paris"  # per-tool simulated latency
    assert by_id["3"].data.error
    assert [(c.id, r.id) for _, _, c, r in tool_log(env.log)] == [("1", "1"), ("2", "2"), ("3", "3")]


def test_state_is_per_episode_and_forks():
    async def main():
        env = make_env()
        await env.reset(Task(initial_state={"cart": {}}))
        a, b = env.fork(2)
        await a.step([Frame(CALL["agent"], a.t, ToolCall("1", "add_item", {"name": "tea"}))])
        await b.step([])
        return env, a, b

    env, a, b = run(main())
    carts = [e.sim.states["tools"]["services"]["tools"]["cart"] for e in (env, a, b)]
    assert carts == [{}, {"tea": 1}, {}]
