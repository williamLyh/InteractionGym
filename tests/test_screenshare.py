"""Screen sharing without a browser: a fake form as the page, scripted LLMs for the user and the agent."""

import asyncio
import json
import random

from interaction_gym import AgentSpec, Env, Frame, Segment, Task
from interaction_gym.core import Chunk
from interaction_gym.integrations import screenshare as ss
from interaction_gym.tools import CALL, RESULT, ToolBackend, ToolResult, ToolWorld
from interaction_gym.traj import episode
from interaction_gym.user import TurnTaking
from interaction_gym.video import SCREEN, VideoFrame

PAGE = """RootWebArea 'Enter Text Task', focused
\t[14] textbox '', clickable
\t[15] button 'Submit', clickable"""
GOAL = 'Enter "Myron" into the text field and press Submit.'


class FakeForm(ToolBackend):
    """A text field and a submit button. Named like the browser backend so that ``bg.evaluate`` reads it."""

    name = "browser"

    def tools(self, caller="agent"):
        from interaction_gym.integrations.browsergym import TOOLS
        return list(TOOLS) if caller == "user" else []

    def reset(self, task, rng):
        return {"values": {}, "reward": 0.0, "terminated": False, "truncated": False, "steps": 0}

    async def call(self, state, caller, call):
        state["steps"] += 1
        if call.name == "fill":
            state["values"][call.arguments["bid"]] = call.arguments["value"]
        if call.name == "click" and call.arguments["bid"] == "15":
            state["terminated"] = True
            state["reward"] = 1.0 if state["values"].get("14") == "Myron" else -1.0
        return ToolResult(call.id, call.name, "ok")

    def render(self, state):
        return json.dumps(state["values"]).encode()


class UserLLM:
    model = "fake-user"

    async def chat(self, msgs, **kw):
        last = msgs[-1]["content"]
        if last.startswith("A caller is filling in"):
            return "yes" if "Myorn" in last else "no"
        agent = next((m["content"] for m in reversed(msgs) if m["role"] == "user" and not m["content"].startswith("(")), "")
        if "has taken it" in last:
            return "Great, thanks! ###STOP###"
        if "look at your screen" in last:
            return "Oh, you're right, let me fix that." if "indeed a mistake" in last else "Looks right to me."
        return "Okay."


class Oracle:
    """Says the typo out loud ``delay_ms`` after first seeing it on the screen (or never: ``delay_ms=None``)."""

    def __init__(self, spec, delay_ms):
        self.spec, self.delay_ms, self.seen, self.said = spec, delay_ms, None, False

    async def act(self, t, obs):
        for f in obs:
            if f.stream == SCREEN and b"Myorn" in f.data.png and self.seen is None:
                self.seen = f.t
        if self.delay_ms is None or self.said or self.seen is None or t < self.seen + self.delay_ms:
            return []
        self.said = True
        return [Frame(self.spec.out, t, Segment("a1", t, 1500, "Careful, you typed Myorn, it should be Myron."))]


def task(kind="typo"):
    p = ss.plan("enter-text", GOAL, PAGE)
    s = ss.slip(p["steps"], kind, random.Random(0))
    return Task(id="t", scenario={"first_turn": "Hi, I need to enter Myron, M Y R O N.", "instructions": ss.USER_INSTRUCTIONS.format(what=p["what"]),
                                  "hands": {"steps": p["steps"], "slip": s, "review_ms": 3000, "char_ms": 200}}, criteria={"slip": s})


def run(delay_ms, kind="typo"):
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN))
    user = ss.ScreenUser(ss.ScreenShareSource(UserLLM()), timing=TurnTaking(respond_after_ms=500))
    world = ToolWorld(FakeForm(), 300, callers=("agent", "user"), screen=SCREEN)
    env = Env({"user": user, "tools": world}, spec, max_ms=60_000)
    agent = Oracle(spec, delay_ms)

    async def go():
        obs, done = await env.reset(task(kind), seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    asyncio.run(go())
    return env


def test_plan_and_slip():
    p = ss.plan("enter-text", GOAL, PAGE)
    assert [s["do"] for s in p["steps"]] == ["type", "click"] and p["steps"][0]["value"] == "Myron" and p["steps"][1]["submit"]
    s = ss.slip(p["steps"], "typo", random.Random(0))
    assert s["intended"] == "Myron" and s["done"] != "Myron" and sorted(s["done"]) == sorted("Myron") and s["done"][0] == "M"
    assert ss.slip(p["steps"], "none", random.Random(0)) is None


def test_caught_slip_is_fixed_before_submit():
    env = run(delay_ms=400)
    res = ss.evaluate(env, task())
    assert res["success"] and res["caught"] and 0 < res["catch_ms"] < 3000, res
    frames = [f for f in env.log if f.stream == SCREEN]
    assert frames[0].t == 0 and all(isinstance(f.data, VideoFrame) for f in frames)
    typed = [f.data.arguments.get("value") for f in env.log if f.stream == CALL["user"] and f.data.name == "fill"]
    assert typed[:5] == ["M", "My", "Myr", "Myro", "Myorn"] or "Myorn" in typed  # one key at a time, the slip among them
    assert typed[-1] == "Myron"
    ep = episode(env, "t")
    assert len(ep["video"]) == len(frames) and ep["video"][0]["t"] == 0


def test_silent_agent_lets_the_slip_through():
    env = run(delay_ms=None)
    res = ss.evaluate(env, task())
    assert not res["success"] and res["submitted"] and not res["caught"] and res["visible_t"] is not None
    # the user paused review_ms before submitting
    assert res["submit_t"] - res["visible_t"] >= 3000


def test_too_late_does_not_count():
    env = run(delay_ms=8000)
    res = ss.evaluate(env, task())
    assert not res["success"] and not res["caught"]


def test_agent_never_sees_the_users_tool_calls():
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN))
    seen = []

    class Spy(Oracle):
        async def act(self, t, obs):
            seen.extend(f.stream for f in obs)
            return await super().act(t, obs)

    user = ss.ScreenUser(ss.ScreenShareSource(UserLLM()), timing=TurnTaking(respond_after_ms=500))
    env = Env({"user": user, "tools": ToolWorld(FakeForm(), 300, callers=("agent", "user"), screen=SCREEN)}, spec, max_ms=30_000)
    agent = Spy(spec, None)

    async def go():
        obs, done = await env.reset(task(), seed=0), False
        seen.extend(f.stream for f in obs)
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    asyncio.run(go())
    assert SCREEN in seen and CALL["user"] not in seen and RESULT["user"] not in seen


def test_user_believes_what_it_meant_until_it_checks():
    p = ss.plan("enter-text", GOAL, PAGE)
    s = ss.slip(p["steps"], "typo", random.Random(0))
    p["steps"][0]["slipped"] = s["done"]
    hands = {"steps": p["steps"], "i": 1, "typed": {"14": s["done"]}, "clicked": {}, "typed_all": True, "submitted": False}
    assert "'Myron'" in ss.screen_note(hands) and s["done"] not in ss.screen_note(hands)
    assert repr(s["done"]) in ss.screen_note(hands, actual=True)


class Nag(Oracle):
    """Claims a typo that is not there, as soon as typing starts."""

    async def act(self, t, obs):
        for f in obs:
            if f.stream == SCREEN and b'"14": "M' in f.data.png and self.seen is None:
                self.seen = f.t
        if self.said or self.seen is None or t < self.seen + 300:
            return []
        self.said = True
        return [Frame(self.spec.out, t, Segment("a1", t, 1500, "Careful, you typed Myorn, it should be Myron."))]


def test_claim_without_a_mistake_is_a_false_alarm():
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN))
    user = ss.ScreenUser(ss.ScreenShareSource(UserLLM()), timing=TurnTaking(respond_after_ms=500))
    env = Env({"user": user, "tools": ToolWorld(FakeForm(), 300, callers=("agent", "user"), screen=SCREEN)}, spec, max_ms=60_000)
    agent = Nag(spec, None)
    t = task("none")

    async def go():
        obs, done = await env.reset(t, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    asyncio.run(go())
    res = ss.evaluate(env, t)
    assert res["success"] and res["false_alarms"] == 1 and not env.sim.states["user"]["hands"]["fixed"]
