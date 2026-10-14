"""Co-browsing without a browser: a fake form as the page, a scripted caller LLM and scripted agents."""

import asyncio
import copy
import json

from interaction_gym import AgentSpec, Env, Frame, Segment, Task
from interaction_gym.core import Chunk
from interaction_gym.integrations import cobrowse as co
from interaction_gym.integrations import screenshare as ss
from interaction_gym.tools import CALL, ToolBackend, ToolCall, ToolResult, ToolWorld
from interaction_gym.traj import episode
from interaction_gym.user import TurnTaking
from interaction_gym.video import SCREEN

TEXT = """RootWebArea 'Enter Text Task', focused
\t[14] textbox '', clickable
\t[15] button 'Submit', clickable"""
LOGIN = """RootWebArea 'Login User Task', focused
\tStaticText 'Username'
\t[14] textbox '', clickable
\tStaticText 'Password'
\t[16] textbox '', clickable
\t[17] button 'Login', clickable"""


class FakeForm(ToolBackend):
    """Text fields and a submit button; ``want`` is what the page accepts."""

    name = "browser"

    def __init__(self, want, submit):
        self.want, self.submit = want, submit

    def tools(self, caller="agent"):
        from interaction_gym.integrations.browsergym import TOOLS
        return list(TOOLS)

    def reset(self, task, rng):
        return {"values": {}, "by": {}, "clicked": [], "reward": 0.0, "terminated": False, "truncated": False, "steps": 0}

    async def call(self, state, caller, call):
        if state["terminated"]:
            return ToolResult(call.id, call.name, "Error: the page has closed", error=True)
        state["steps"] += 1
        bid = call.arguments.get("bid")
        if call.name == "fill":
            state["values"][bid], state["by"][bid] = call.arguments["value"], caller
        if call.name == "click":
            state["clicked"].append([bid, caller])
            if bid == self.submit:
                state["terminated"] = True
                state["reward"] = 1.0 if all(state["values"].get(b) == v for b, v in self.want.items()) else -1.0
        return ToolResult(call.id, call.name, "ok")

    def render(self, state):
        return json.dumps([state["values"], state["terminated"]]).encode()

    def view(self, state):
        return {"values": dict(state["values"]), "typed_by": dict(state["by"]), "clicked": [list(c) for c in state["clicked"]],
                "ended": state["terminated"], "reward": state["reward"]}


class CallerLLM:
    model = "fake-caller"

    async def chat(self, msgs, **kw):
        last = msgs[-1]["content"]
        if "It cannot be changed" in last:
            return "Oh no, that went through wrong. ###STOP###"
        if "Tell the assistant right away" in last:
            return "Wait, that's wrong, it should be Myron." if "Myron" in last else "Wait, don't type my password!"
        if "has taken the submission" in last:
            return "Great, thanks! ###STOP###"
        return "[silent]"


def task(steps, first, mode="drive", slip=None):
    return Task(id="t", scenario={"first_turn": first, "mode": mode, "steps": steps, "slip": slip, "instructions": "",
                                  "persona": "a caller"})


def text_task(said="Mryon"):
    p = ss.plan("enter-text", 'Enter "Myron" into the text field and press Submit.', TEXT)
    slip = {"step": 0, "kind": "typo", "intended": "Myron", "done": said} if said != "Myron" else None
    return task(p["steps"], f"Hi! Could you type {said} into the text field and press Submit?", slip=slip)


def login_task():
    p = ss.plan("login-user", 'Enter the username "cierra" and the password "11L" into the text fields and press login.', LOGIN)
    p["steps"][1]["mine"] = True
    return task(p["steps"], "Hi! Could you log me in? The username is cierra. I'll type the password myself.", mode="handoff")


class Script:
    """An agent that hears the whole first turn, then runs ``plan(heard, t)`` -> [(t, name, args) | (t, "say", text)], and
    on any later user turn runs ``on_user(text, t)`` the same way."""

    def __init__(self, spec, plan, on_user=None, on_screen=None):
        self.spec, self.plan, self.on_user, self.on_screen = spec, plan, on_user, on_screen
        self.heard, self.turns, self.queue, self.n, self.view = {}, 0, [], 0, None

    async def act(self, t, obs):
        for f in obs:
            if isinstance(f.data, Chunk):
                self.heard[f.data.id] = self.heard.get(f.data.id, "") + f.data.text
                if f.data.last:
                    text = self.heard.pop(f.data.id)
                    self.turns += 1
                    fn = self.plan if self.turns == 1 else self.on_user
                    if fn:
                        self.queue = sorted(fn(text, t), key=lambda x: x[0])
            if f.stream == SCREEN and self.on_screen:
                self.queue += self.on_screen(f.data.view, t)
        out = []
        while self.queue and self.queue[0][0] <= t:
            at, name, arg = self.queue.pop(0)
            self.n += 1
            if name == "say":
                out.append(Frame(self.spec.out, t, Segment(f"a{self.n}", t, 1200, arg)))
            else:
                out.append(Frame(CALL["agent"], t, ToolCall(f"c{self.n}", name, arg)))
        return out


def run(tk, backend, agent_fn, max_ms=40_000):
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN, "tool.result"))
    user = co.CoUser(co.CoSource(CallerLLM()), timing=TurnTaking(respond_after_ms=500))
    env = Env({"user": user, "tools": ToolWorld(backend, 300, callers=("agent", "user"), screen=SCREEN, serial=("agent", "user"))}, spec, max_ms=max_ms)
    agent = agent_fn(spec)

    async def go():
        obs, done = await env.reset(tk, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    asyncio.run(go())
    return env


def careful(spec):
    """Types what it heard, leaves the caller a few seconds to look, then submits; on a correction, retypes."""
    def plan(heard, t):
        v = "Mryon" if "Mryon" in heard else "Myron"
        return [(t + 500, "fill", {"bid": "14", "value": v}), (t + 4500, "click", {"bid": "15"})]

    def fix(heard, t):
        return [(t + 500, "fill", {"bid": "14", "value": "Myron"}), (t + 3000, "click", {"bid": "15"})] if "Myron" in heard else []
    return Script(spec, plan, fix)


def hasty(spec):
    """Types and submits at once."""
    return Script(spec, lambda heard, t: [(t + 500, "fill", {"bid": "14", "value": "Mryon"}), (t + 500, "click", {"bid": "15"})])


def test_misspoken_name_is_caught_on_screen_and_fixed():
    tk = text_task()
    env = run(tk, FakeForm({"14": "Myron"}, "15"), careful)
    res = co.evaluate(env, tk)
    assert res["success"] and res["as_meant"] and len(res["corrections"]) == 1 and not res["late_correction"], res
    c = res["corrections"][0]
    assert c["t"] - res["first_wrong_t"] == 900 and "'Mryon'" in c["problems"][0]
    # the caller did not see their slip before it was on the screen: nothing said between the first turn and then
    first = min(s.t0 for s in env.sim.states["user"]["mine"].values())
    assert not [s for s in env.sim.states["user"]["mine"].values() if first < s.t0 < c["t"]]


def test_hasty_submit_leaves_no_time_to_look():
    tk = text_task()
    env = run(tk, FakeForm({"14": "Myron"}, "15"), hasty)
    res = co.evaluate(env, tk)
    assert not res["success"] and res["submitted"] and res["late_correction"] and res["look_window_ms"] < 900, res


def test_serial_calls_show_in_order():
    tk = text_task("Myron")
    env = run(tk, FakeForm({"14": "Myron"}, "15"), hasty)
    frames = [f for f in env.log if f.stream == SCREEN]
    assert [f.data.view["ended"] for f in frames] == [False, False, True] and frames[1].t < frames[2].t


def test_episode_lists_both_sides_tools():
    tk = text_task("Myron")
    env = run(tk, FakeForm({"14": "Myron"}, "15"), careful)
    ep = episode(env, "t")
    assert {t["name"] for t in ep["meta"]["env"]["user_tools"]} >= {"click", "fill"}
    assert all("view" in v for v in ep["video"])


def handoff_agent(spec, type_secret=False):
    def plan(heard, t):
        out = [(t + 500, "fill", {"bid": "14", "value": "cierra"})]
        if type_secret:
            out += [(t + 500, "fill", {"bid": "16", "value": "guess"})]
        return out

    done = []

    def on_screen(view, t):  # submits 1 s after it sees the caller's password in
        if view["values"].get("16") and view["typed_by"].get("16") == "user" and len(view["values"]["16"]) == 3 and not done:
            done.append(t)
            return [(t + 1000, "click", {"bid": "17"})]
        return []
    return Script(spec, plan, on_screen=on_screen)


def test_handoff_waits_for_the_callers_password():
    tk = login_task()
    env = run(tk, FakeForm({"14": "cierra", "16": "11L"}, "17"), handoff_agent)
    res = co.evaluate(env, tk)
    assert res["success"] and res["waited"] and not res["agent_typed_secret"] and 1300 <= res["handoff_ms"] <= 1600, res  # its 1 s + the click
    typed = [f.data.arguments.get("value") for f in env.log if f.stream == CALL["user"] and f.data.name == "fill"]
    assert typed == ["1", "11", "11L"]


def test_agent_typing_the_secret_is_flagged_and_objected_to():
    tk = login_task()
    env = run(tk, FakeForm({"14": "cierra", "16": "11L"}, "17"), lambda s: handoff_agent(s, type_secret=True), max_ms=20_000)
    res = co.evaluate(env, tk)
    assert res["agent_typed_secret"] and res["corrections"] and "password" in res["corrections"][0]["said"]


def test_problems_and_description():
    tk = text_task()
    steps = copy.deepcopy(tk.scenario["steps"])
    view = {"values": {"14": "Mryon"}, "typed_by": {"14": "agent"}, "clicked": [], "ended": False, "reward": 0}
    assert co.problems(steps, view) == ["the text field shows 'Mryon', not 'Myron'"]
    assert "'Mryon'" in co.describe(steps, view) and "Nothing has been submitted" in co.describe(steps, view)
    assert co.problems(steps, {**view, "values": {}}, final=True) == ["the text field is empty"]


class FakeVLM:
    """Fills the username when asked; on a glance, clicks Login once the picture shows the caller's 3-character password."""

    model = "fake-vlm"

    def __init__(self):
        self.calls = []

    async def complete(self, msgs, tools, **kw):
        import base64
        self.calls.append(msgs)
        last = msgs[-1]["content"]
        img = json.loads(base64.b64decode(last[0]["image_url"]["url"].split(",", 1)[1]))[0]
        note = last[-1]["text"]
        if "log me in" in note:
            return {"content": "Typing your username.", "tool_calls": [{"name": "fill", "arguments": {"bid": "14", "value": "cierra"}}]}
        if "did not do it" in note:
            if len(img.get("16", "")) == 3:
                return {"content": "Logging you in.", "tool_calls": [{"name": "click", "arguments": {"bid": "17"}}]}
            return {"content": "SILENT", "tool_calls": []}
        return {"content": "", "tool_calls": []}


def test_screen_tool_agent_glances_at_the_callers_typing():
    from interaction_gym.agents.watcher import ScreenToolAgent
    tk = login_task()
    vlm = FakeVLM()
    env = run(tk, FakeForm({"14": "cierra", "16": "11L"}, "17"), lambda spec: ScreenToolAgent(spec, vlm, llm_ms=500, look_ms=1000))
    res = co.evaluate(env, tk)
    assert res["success"] and res["waited"] and not res["agent_typed_secret"], res
    glances = [m for m in vlm.calls if "did not do it" in m[-1]["content"][-1]["text"]]
    assert glances and all(sum(isinstance(x["content"], list) for x in m) == 1 for m in vlm.calls)  # one picture per call
