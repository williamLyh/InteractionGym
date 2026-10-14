"""OpenApps co-browsing without OpenApps: a fake desktop (calendar, to-dos, messages) as the page, a scripted caller LLM
and scripted agents."""

import asyncio
import copy
import json

from interaction_gym import AgentSpec, Env
from interaction_gym.integrations import cobrowse as co
from interaction_gym.integrations import openapps as oa
from interaction_gym.tools import ToolBackend, ToolResult, ToolSpec, ToolWorld
from interaction_gym.user import TurnTaking
from interaction_gym.video import SCREEN
from tests.test_cobrowse import Script

TASKS = {
    "plan": {"_target_": "open_apps.tasks.tasks.CompositeTask", "goal": "...", "subtasks": [
        {"_target_": "open_apps.tasks.tasks.AddEventTask", "title": "Book club", "date": "2026-02-05", "description": None,
         "location": None, "url": None, "invitees": []},
        {"_target_": "open_apps.tasks.tasks.AddToDoTask", "todo_name": "Buy the book", "is_done": False},
        {"_target_": "open_apps.tasks.tasks.SendMessageTask", "to": "Alice", "message": "Book club is on!", "expected_reply": None}]},
    "note": {"_target_": "open_apps.tasks.tasks.SendMessageTask", "goal": "...", "to": "Alice", "message": "See you at 5"},
}
APPS = {"calendar": [{"id": 1, "title": "Dentist", "date": "2026-10-20", "description": "", "location": "", "url": "", "invitees": None}],
        "todo": [{"id": 1, "title": "Check budget", "done": 0}],
        "messenger": [{"user": u, "messages": [["hi", u, "Jan 1, 9:00AM"]]} for u in ("Alice", "Bob", "Charlie")], "map": []}


class FakeDesk(ToolBackend):
    """The apps' state as OpenApps keeps it, with a message box: what the agent's calls change."""

    name = "browser"

    def tools(self, caller="agent"):
        return [ToolSpec(n, n) for n in ("add_event", "delete_event", "add_todo", "open_chat", "type", "send")]

    def reset(self, task, rng):
        return {"initial": copy.deepcopy(APPS), "apps": copy.deepcopy(APPS), "chat": None, "draft": "", "steps": 0}

    async def call(self, state, caller, call):
        a, apps = call.arguments, state["apps"]
        state["steps"] += 1
        if call.name == "add_event":
            apps["calendar"].append({"id": 9, "title": a["title"], "date": a["date"], "description": "", "location": "", "url": "", "invitees": None})
        elif call.name == "delete_event":
            apps["calendar"] = [e for e in apps["calendar"] if (e["title"], e["date"]) != (a["title"], a["date"])]
        elif call.name == "add_todo":
            apps["todo"].append({"id": 9, "title": a["title"], "done": 0})
        elif call.name == "open_chat":
            state["chat"], state["draft"] = a["to"], ""
        elif call.name == "type":
            state["draft"] = a["text"]
        elif call.name == "send":
            c = next(c for c in apps["messenger"] if c["user"] == state["chat"])
            c["messages"] += [[state["draft"], "you", "now"], ["ok!", state["chat"], "now"]]
            state["draft"] = ""
        return ToolResult(call.id, call.name, "ok")

    def render(self, state):
        return repr(state["apps"]).encode() + state["draft"].encode()

    def view(self, state):
        return {"app": "messages" if state["chat"] else "calendar", "chat": state["chat"], "changes": oa.changes(state["initial"], state["apps"]),
                "fields": [{"field": "textbox 'Message'", "value": state["draft"]}] if state["draft"] else [], "ended": False, "reward": 0.0}


class CallerLLM:
    model = "fake-caller"

    async def chat(self, msgs, **kw):
        last = msgs[-1]["content"]
        if "cannot be unsent" in last:
            return "Oh no, that went to the wrong person. ###STOP###"
        if "Tell the assistant right away" in last:
            return "Wait, I meant February 5th." if "February" in last else "Wait, that message is for Alice!"
        if "Everything you wanted is now done" in last:
            return "Great, thanks! ###STOP###"
        return "[silent]"


def tasks_dir(tmp_path, monkeypatch):
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "longer_horizon.yaml").write_text(json.dumps(TASKS))  # (JSON is YAML)
    monkeypatch.setenv("OPENAPPS_CONFIG", str(tmp_path))


def run(tk, agent_fn, max_ms=60_000):
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN, "tool.result"))
    user = oa.AppUser(co.CoSource(CallerLLM()), timing=TurnTaking(respond_after_ms=500))
    env = Env({"user": user, "tools": ToolWorld(FakeDesk(), 300, callers=("agent",), screen=SCREEN, serial=("agent",))}, spec, max_ms=max_ms)
    agent = agent_fn(spec)

    async def go():
        obs, done = await env.reset(tk, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    asyncio.run(go())
    return env


def said(env):
    return sorted((s.t0, s.text) for s in env.sim.states["user"]["mine"].values())


def test_task_from_openapps_config(tmp_path, monkeypatch):
    tasks_dir(tmp_path, monkeypatch)
    tk = oa.make_task("plan")
    assert [s["do"] for s in tk.scenario["steps"]] == ["add_event", "add_todo", "send"]
    assert tk.scenario["first_turn"] == ("Hi! Could you put 'Book club' in my calendar on February 5th, 2026, add 'Buy the book' to my "
                                         "to-do list, and message Alice, saying: 'Book club is on!'?")
    st = oa.make_task("plan", cue="staged")
    assert [s["stage"] for s in st.scenario["steps"]] == [0, 1, 2] and st.scenario["first_turn"].endswith("February 5th, 2026?")


def test_changes_and_problems():
    now = copy.deepcopy(APPS)
    now["calendar"].append({"title": "Book club", "date": "2026-02-06"})
    now["todo"][0]["done"] = 1
    now["messenger"][1]["messages"] += [["Book club is on!", "you", "now"], ["ok", "Bob", "now"]]
    c = oa.changes(APPS, now)
    assert c == {"events_added": [{"title": "Book club", "date": "2026-02-06"}], "todos_ticked": ["Check budget"],
                 "sent": [{"to": "Bob", "text": "Book club is on!"}], "replies": [{"from": "Bob", "text": "ok"}]}
    steps = oa.steps_from(TASKS["plan"])
    p = oa.problems(steps, {"changes": c})
    assert p[0] == "the new event 'Book club' has the date February 6th, 2026, not February 5th, 2026"
    assert "ticked off, which you did not want" in p[1] and p[2] == "the message 'Book club is on!' was sent to Bob, not Alice"
    assert getattr(p[2], "final", False) and not getattr(p[0], "final", False) and not oa.AppEyes.late({}, p)
    assert "Not done yet" in oa.describe([{**s, "said": oa.phrase(s)} for s in steps], {"changes": c})


def agent_doing(script):
    """An agent that, on each user turn, runs the calls for whatever the caller just asked for (by keyword)."""
    def plan(heard, t):
        out, at = [], t + 500
        for key, calls in script:
            if key in heard:
                for name, args in calls:
                    out.append((at, name, args))
                    at += 1500
        return out
    return lambda spec: Script(spec, plan, plan)


EVENT = ("in my calendar", [("add_event", {"title": "Book club", "date": "2026-02-05"})])
TODO = ("Buy the book", [("add_todo", {"title": "Buy the book"})])
SEND = ("message Alice", [("open_chat", {"to": "Alice"}), ("type", {"text": "Book club is on!"}), ("send", {})])


def test_staged_caller_asks_for_each_step_when_the_last_shows_done(tmp_path, monkeypatch):
    tasks_dir(tmp_path, monkeypatch)
    tk = oa.make_task("plan", cue="staged")
    env = run(tk, agent_doing([EVENT, TODO, SEND]))
    res = oa.evaluate(env, tk, reward=False)
    assert res["as_meant"] and not res["corrections"], res
    asks = res["asks"]
    assert [a["stage"] for a in asks] == [1, 2] and "Buy the book" in asks[0]["said"] and "Alice" in asks[1]["said"]
    frames = env.sim.states["user"]["co"]["frames"]
    assert asks[0]["t"] == frames[1]["t"] + 1500  # the event showed (frame 0 is the desktop at the start); the caller looked and asked for the next
    assert said(env)[-1][1].startswith("Great, thanks")


def test_misspoken_date_is_caught_and_fixed(tmp_path, monkeypatch):
    tasks_dir(tmp_path, monkeypatch)
    tk = oa.make_task("plan", cue="misspoke", seed=1)
    slip = tk.scenario["slip"]
    assert slip["kind"] == "date" and oa.spoken_date(slip["done"]) in tk.scenario["first_turn"]
    wrong = slip["done"]

    def plan(heard, t):  # (a user turn replaces the calls it had not made yet)
        if "February 5th" in heard and "I meant" in heard:  # the correction
            return [(t + 500, "delete_event", {"title": "Book club", "date": wrong}),
                    (t + 2000, "add_event", {"title": "Book club", "date": "2026-02-05"}), (t + 3500, "add_todo", {"title": "Buy the book"}),
                    (t + 5000, "open_chat", {"to": "Alice"}), (t + 6500, "type", {"text": "Book club is on!"}), (t + 8000, "send", {})]
        return [(t + 500, "add_event", {"title": "Book club", "date": wrong}), (t + 6000, "add_todo", {"title": "Buy the book"}),
                (t + 7500, "open_chat", {"to": "Alice"}), (t + 9000, "type", {"text": "Book club is on!"}), (t + 10500, "send", {})]
    env = run(tk, lambda spec: Script(spec, plan, lambda h, t: plan(h, t) if "I meant" in h else []))
    res = oa.evaluate(env, tk, reward=False)
    assert res["as_meant"] and len(res["corrections"]) == 1 and not res["late_correction"], res
    assert "February 6th" in res["corrections"][0]["problems"][0] or "February 4th" in res["corrections"][0]["problems"][0]
    assert res["corrections"][0]["t"] - res["first_wrong_t"] == 900


def test_wrong_recipient_sent_at_once_is_too_late(tmp_path, monkeypatch):
    tasks_dir(tmp_path, monkeypatch)
    tk = oa.make_task("note", cue="misspoke")
    assert tk.scenario["slip"]["kind"] == "recipient"
    to = tk.scenario["slip"]["done"]
    hasty = lambda spec: Script(spec, lambda h, t: [(t + 500, "open_chat", {"to": to}), (t + 500, "type", {"text": "See you at 5"}),  # noqa: E731
                                                    (t + 500, "send", {})])
    res = oa.evaluate(run(tk, hasty, max_ms=20_000), tk, reward=False)
    assert res["late_correction"] and not res["as_meant"], res


def test_wrong_recipient_caught_in_the_message_box(tmp_path, monkeypatch):
    tasks_dir(tmp_path, monkeypatch)
    tk = oa.make_task("note", cue="misspoke")
    to = tk.scenario["slip"]["done"]

    def plan(heard, t):  # shows the message before sending it (sends 4 s later), and redoes it for the right person
        if "for Alice" in heard:
            return [(t + 500, "open_chat", {"to": "Alice"}), (t + 1000, "type", {"text": "See you at 5"}), (t + 5000, "send", {})]
        return [(t + 500, "open_chat", {"to": to}), (t + 1000, "type", {"text": "See you at 5"}), (t + 5000, "send", {})]
    env = run(tk, lambda spec: Script(spec, plan, plan), max_ms=30_000)  # (a new user turn replaces what it had not done yet)
    res = oa.evaluate(env, tk, reward=False)
    assert res["as_meant"] and res["corrections"] and not res["late_correction"], res
    assert f"your chat with {to}" in res["corrections"][0]["problems"][0]
