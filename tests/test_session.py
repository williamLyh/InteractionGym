"""Telling the agent about its environment: the t=0 session, and the two tool-exposure modes."""

import asyncio
import json

import pytest

from interaction_gym import SESSION, AgentSpec, Env, Frame, Session, Task
from interaction_gym.tools import CALL, RESULT, FunctionBackend, ToolCall, ToolWorld
from interaction_gym.traj import episode
from tests.test_traj import check


def run(coro):
    return asyncio.run(coro)


def add_item(state: dict, name: str) -> dict:
    """Add an item to the cart."""
    state["cart"].append(name)
    return {"cart": state["cart"]}


def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"sunny in {city}"


def services():
    return {
        "shop": FunctionBackend({"add_item": add_item}, {"cart": []}, instructions="Never add more than 3 items.", name="shop",
                                description="Online grocery shop: manage the customer's cart"),
        "weather": FunctionBackend({"get_weather": get_weather}, name="weather", description="Weather forecasts"),
    }


SPEC = AgentSpec(chunk_ms=100, obs=(RESULT["agent"],))
TASK = Task(id="t", scenario={"agent_context": {"caller_phone": "+44 7700 900123"}})


def sessions(frames):
    return [f.data for f in frames if f.stream == SESSION]


def test_all_mode_gives_every_tool_up_front():
    env = Env({"tools": ToolWorld(services(), mode="all")}, SPEC)
    (s,) = sessions(run(env.reset(TASK)))
    assert isinstance(s, Session) and s.mode == "all"
    assert {t.name for t in s.tools} == {"shop__add_item", "weather__get_weather"}  # qualified: two services
    assert {t.service for t in s.tools} == {"shop", "weather"}
    assert "Never add more than 3 items." in s.instructions
    assert s.context == {"caller_phone": "+44 7700 900123"} and s.services == ()
    # nothing about the world's state is in the session
    assert "cart" not in json.dumps([t.parameters for t in s.tools]) and "cart" not in s.instructions.lower()


def test_single_service_keeps_plain_tool_names():
    env = Env({"tools": ToolWorld(services()["shop"])}, SPEC)
    (s,) = sessions(run(env.reset(TASK)))
    assert [t.name for t in s.tools] == ["add_item"]


def test_discover_mode_find_load_then_call():
    lat = lambda call: 400 if call.name in ("search_tools", "load_tools") else 0  # noqa: E731

    async def main():
        env = Env({"tools": ToolWorld(services(), lat, mode="discover")}, SPEC)
        seen = await env.reset(TASK)
        acts = {
            0: [Frame(CALL["agent"], 0, ToolCall("1", "shop__add_item", {"name": "tea"}))],  # not loaded yet
            100: [Frame(CALL["agent"], 100, ToolCall("2", "search_tools", {"query": "cart"}))],
            600: [Frame(CALL["agent"], 600, ToolCall("3", "load_tools", {"service": "shop"}))],
            1100: [Frame(CALL["agent"], 1100, ToolCall("4", "shop__add_item", {"name": "tea"}))],
        }
        while env.t < 1500:
            obs, _, _ = await env.step(acts.get(env.t, []))
            seen += obs
        return env, seen

    env, seen = run(main())
    first, update = sessions(seen)
    assert first.mode == "discover" and {t.name for t in first.tools} == {"list_services", "search_tools", "load_tools"}
    assert {x["service"] for x in first.services} == {"shop", "weather"} and "load_tools" in first.instructions
    results = {f.data.id: f for f in seen if f.stream == RESULT["agent"]}
    assert results["1"].data.error and "load_tools(service='shop')" in results["1"].data.content
    assert json.loads(results["2"].data.content)[0]["name"] == "shop__add_item"
    body = json.loads(results["3"].data.content)
    assert body["instructions"] == "Never add more than 3 items." and body["tools"][0]["function"]["name"] == "shop__add_item"
    upd_frame = next(f for f in seen if f.stream == SESSION and f.data is update)
    assert upd_frame.t == 600 + 400 == results["3"].t  # the session update arrives with the load result
    assert "shop__add_item" in {t.name for t in update.tools} and update.context == first.context  # merged, context kept
    assert not results["4"].data.error and json.loads(results["4"].data.content) == {"cart": ["tea"]}
    ep = check(episode(env, "e"))
    assert ep["meta"]["env"]["tool_schema_mode"] == "discover" and ep["meta"]["env"]["components"] == {"tools": "ToolWorld"}
    calls = [c["name"] for t in ep["turns"] for c in t.get("tool_calls", [])]
    assert calls == ["shop__add_item", "search_tools", "load_tools", "shop__add_item"]


def test_loaded_services_fork_per_branch():
    async def main():
        env = Env({"tools": ToolWorld(services(), mode="discover")}, SPEC)
        await env.reset(TASK)
        a, b = env.fork(2)
        await a.step([Frame(CALL["agent"], 0, ToolCall("1", "load_tools", {"service": "weather"}))])
        await b.step([])
        return [e.sim.states["tools"]["loaded"] for e in (env, a, b)]

    assert run(main()) == [[], ["weather"], []]


def test_tau_and_custom_services_together():
    pytest.importorskip("tau2")
    from interaction_gym.integrations.tau import TauBackend, load_task

    world = ToolWorld({"mock": TauBackend("mock"), "weather": services()["weather"]}, mode="discover")
    env = Env({"tools": world}, SPEC)
    (s,) = sessions(run(env.reset(load_task("mock", "create_task_1"))))
    assert {x["service"] for x in s.services} == {"mock", "weather"}
    assert "mock__create_task" in world.specs["agent"] and "τ-bench" in world.catalog()[0]["description"]


def test_env_context_records_full_tool_schemas_and_updates():
    async def main():
        env = Env({"tools": ToolWorld(services(), 300, mode="discover")}, SPEC)
        await env.reset(TASK)
        await env.step([Frame(CALL["agent"], 0, ToolCall("1", "load_tools", {"service": "shop"}))])
        for _ in range(5):
            await env.step([])
        return env

    ep = check(episode(run(main()), "e"))
    assert "agent" not in ep["meta"]  # tool schemas belong to the environment's context
    env = ep["meta"]["env"]
    assert all("parameters" in t for t in env["tools"])  # full schemas at t=0
    assert {x["service"] for x in env["tool_services"]} == {"shop", "weather"}
    assert env["agent_context"] == {"caller_phone": "+44 7700 900123"}
    (upd,) = env["tool_updates"]
    assert upd["time"] == 300 and upd["instructions"] == "Never add more than 3 items."
    (tool,) = upd["tools"]
    assert tool == {"name": "shop__add_item", "description": "Add an item to the cart.", "service": "shop",
                    "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}


def crm():
    """A backend that reveals an account overview up front but keeps the payment card secret."""
    state = {"customer": {"name": "Alex Chen", "tier": "gold", "card": "4111 1111 1111 1111"}, "cart": []}
    return FunctionBackend({"add_item": add_item}, state, name="shop", description="Online shop",
                           context=lambda st: {"customer": {"name": st["customer"]["name"], "tier": st["customer"]["tier"]}})


def test_tool_env_decides_what_the_agent_sees_up_front():
    env = Env({"tools": ToolWorld(crm())}, SPEC)
    (s,) = sessions(run(env.reset(TASK)))
    assert s.context == {"caller_phone": "+44 7700 900123", "customer": {"name": "Alex Chen", "tier": "gold"}}
    assert "4111" not in json.dumps(s.context)  # the secret never leaves the tool environment
    ep = check(episode(env, "e"))
    assert ep["meta"]["env"]["agent_context"]["customer"] == {"name": "Alex Chen", "tier": "gold"}


def test_discover_mode_reveals_a_service_context_when_it_is_loaded():
    async def main():
        env = Env({"tools": ToolWorld({"shop": crm(), "weather": services()["weather"]}, 200, mode="discover")}, SPEC)
        seen = await env.reset(TASK)
        seen += (await env.step([Frame(CALL["agent"], 0, ToolCall("1", "load_tools", {"service": "shop"}))]))[0]
        for _ in range(3):
            seen += (await env.step([]))[0]
        return env, seen

    env, seen = run(main())
    first, update = sessions(seen)
    assert first.context == {"caller_phone": "+44 7700 900123"}  # nothing from the shop before loading it
    assert update.context == {"caller_phone": "+44 7700 900123", "shop": {"customer": {"name": "Alex Chen", "tier": "gold"}}}
    res = next(f.data for f in seen if f.stream == RESULT["agent"])
    assert json.loads(res.content)["context"] == {"customer": {"name": "Alex Chen", "tier": "gold"}}
    (upd,) = check(episode(env, "e"))["meta"]["env"]["tool_updates"]
    assert upd["context"] == {"shop": {"customer": {"name": "Alex Chen", "tier": "gold"}}}
