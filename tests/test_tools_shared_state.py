"""Services with the same ``state_group`` share one per-episode state, also across forks."""

import asyncio

from interaction_gym import AgentSpec, Env, Frame, Task
from interaction_gym.tools import CALL, RESULT, FunctionBackend, ToolCall, ToolWorld


def add(state: dict, item: str) -> int:
    """Add an item."""
    state["items"].append(item)
    return len(state["items"])


def count(state: dict) -> int:
    """Count items."""
    return len(state["items"])


def world(shared: bool):
    a = FunctionBackend({"add": add}, name="a")
    b = FunctionBackend({"count": count}, name="b")
    if shared:
        a.state_group = b.state_group = "office"
    return ToolWorld({"a": a, "b": b})


def run(shared: bool):
    async def main():
        env = Env({"tools": world(shared)}, AgentSpec(obs=(RESULT["agent"],)))
        await env.reset(Task(initial_state={"items": []}))
        x, y = env.fork(2)
        await x.step([Frame(CALL["agent"], x.t, ToolCall("1", "a__add", {"item": "pen"}))])
        await x.step([Frame(CALL["agent"], x.t, ToolCall("2", "b__count", {}))])
        await y.step([Frame(CALL["agent"], y.t, ToolCall("3", "b__count", {}))])
        for e in (x, y):
            await e.step([])
        return env, x, y

    env, x, y = asyncio.run(main())
    counts = lambda e: [f.data.content for f in e.log if f.stream == RESULT["agent"] and f.data.name == "b__count"]  # noqa: E731
    return env, x, y, counts(x), counts(y)


def test_shared_state_group():
    env, x, y, cx, cy = run(shared=True)
    assert cx == ["1"] and cy == ["0"]  # b sees a's write in the same fork; the other fork is untouched
    s = x.sim.states["tools"]["services"]
    assert s["a"] is s["b"] and env.sim.states["tools"]["services"]["a"]["items"] == []


def test_separate_states_by_default():
    *_, cx, cy = run(shared=False)
    assert cx == ["0"] and cy == ["0"]
