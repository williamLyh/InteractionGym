"""AutoWebWorld co-browsing without a browser: a fake shop store (the site's Pinia ``signature``) as the page, a scripted
caller LLM and scripted agents."""

import asyncio
import copy

from interaction_gym import AgentSpec, Env
from interaction_gym.integrations import autowebworld as aw
from interaction_gym.integrations import cobrowse as co
from interaction_gym.tools import ToolBackend, ToolResult, ToolSpec, ToolWorld
from interaction_gym.user import TurnTaking
from interaction_gym.video import SCREEN
from tests.test_cobrowse import Script

DATA = {"data": {"products": [{"id": "p1", "title": "Desk Lamp", "variants": [{"sku": "s1", "title": "White"}, {"sku": "s2", "title": "Black"}]},
                              {"id": "p2", "title": "Yoga Mat", "variants": [{"sku": "s3", "title": "Default"}]}]}}
SIG = {"currentPageId": "HOME", "cart_items": [], "email": None, "shipping_city": None, "card_number": None}
TERMINAL = ["CHECKOUT_SUCCESS_MAIN"]


class FakeShop(ToolBackend):
    name = "browser"

    def tools(self, caller="agent"):
        return [ToolSpec(n, n) for n in ("add", "set", "remove", "pay")]

    def reset(self, task, rng):
        return {"initial": {"signature": copy.deepcopy(SIG)}, "store": {"signature": copy.deepcopy(SIG)}, "terminal": TERMINAL, "steps": 0}

    async def call(self, state, caller, call):
        sig, a = state["store"]["signature"], call.arguments
        state["steps"] += 1
        if sig["currentPageId"] in TERMINAL:
            return ToolResult(call.id, call.name, "the order is placed", error=True)
        if call.name == "add":
            sig["cart_items"] = sig["cart_items"] + [{"title": a["title"], "variant_title": a["variant"], "quantity": a["quantity"]}]
        elif call.name == "remove":
            sig["cart_items"] = [it for it in sig["cart_items"] if it["title"] != a["title"]]
        elif call.name == "set":
            sig[a["key"]] = a["value"]
        elif call.name == "pay":
            sig["currentPageId"] = TERMINAL[0]
        return ToolResult(call.id, call.name, "ok")

    def render(self, state):
        return repr(state["store"]).encode()

    def view(self, state):
        sig = state["store"]["signature"]
        return {"page": sig["currentPageId"], "path": "/", "changes": aw._changed(state["initial"], state["store"]), "fields": [],
                "ended": sig["currentPageId"] in TERMINAL, "reward": 0.0}


class CallerLLM:
    model = "fake-caller"

    async def chat(self, msgs, **kw):
        last = msgs[-1]["content"]
        if "cannot be changed" in last:
            return "Oh no, that went through wrong. ###STOP###"
        if "Tell the assistant right away" in last:
            return "Wait, that's not right, please fix it."
        if "Everything you wanted is now done" in last:
            return "Great, thanks! ###STOP###"
        return "[silent]"


def run(tk, agent_fn, max_ms=60_000):
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN, "tool.result"))
    user = aw.SiteUser(co.CoSource(CallerLLM()), timing=TurnTaking(respond_after_ms=500))
    env = Env({"user": user, "tools": ToolWorld(FakeShop(), 300, callers=("agent",), screen=SCREEN, serial=("agent",))}, spec, max_ms=max_ms)
    agent = agent_fn(spec)

    async def go():
        obs, done = await env.reset(tk, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    asyncio.run(go())
    return env


def shop_steps(tk):
    return {s.get("key", s["do"]): s for s in tk.scenario["steps"]}


def doer(tk, said_values=True):
    """Does everything in the caller's words (``said_values``: as said, slips included), paying 2 s after the last field."""
    steps, slip = shop_steps(tk), tk.scenario.get("slip")

    def value(key):
        s = steps[key]
        v = copy.deepcopy(s.get("value") if s["do"] == "set" else s["match"])
        if said_values and slip and tk.scenario["steps"][slip["step"]] is s:
            if s["do"] == "set":
                v = slip["done"]
            else:
                v[slip["key"]] = slip["done"]
        return v

    def plan(heard, t, fix=False):
        it = value("cart_items")
        out = [(t + 500, "add", {"title": it["title"], "variant": it["variant_title"], "quantity": it["quantity"]})]
        if fix:
            out = [(t + 300, "remove", {"title": it["title"]}), (t + 500, "add", {"title": it["title"], "variant": it["variant_title"], "quantity": it["quantity"]})]
        at = t + 1500
        for k, s in steps.items():
            if s["do"] == "set":
                out.append((at, "set", {"key": k, "value": value(k)}))
                at += 700
        return out + [(at + 2000, "pay", {})]
    return plan, value


def test_task_and_spoken_request():
    tk = aw.make_task("shopify", data=DATA, seed=3)
    st = shop_steps(tk)
    assert st["cart_items"]["match"]["title"] in tk.scenario["first_turn"] and st["reach"]["pages"] == ["CHECKOUT_SUCCESS_MAIN"]
    assert tk.scenario["first_turn"].endswith("and place the order?") and "and and" not in tk.scenario["first_turn"]
    staged = aw.make_task("shopify", data=DATA, seed=3, cue="staged")
    assert {s["stage"] for s in staged.scenario["steps"]} == {0, 1, 2} and "postcode" not in staged.scenario["first_turn"]


def test_problems_compare_as_read():
    tk = aw.make_task("shopify", data=DATA, seed=3)
    steps = tk.scenario["steps"]
    st = shop_steps(tk)
    m = st["cart_items"]["match"]
    view = {"page": "CART", "changes": {"cart_items": [{**m, "quantity": m["quantity"] + 1}], "card_number": "4242424242424242",
                                        "shipping_city": "Nowhere"}, "ended": False}
    p = aw.problems(steps, view)
    assert any("quantity" in x for x in p) and any("city on the screen is 'Nowhere'" in x for x in p) and not any("card" in x for x in p)
    assert not aw.SiteEyes.late(view, p) and aw.SiteEyes.late({**view, "ended": True}, aw.problems(steps, {**view, "ended": True}))


def test_everything_as_asked_is_success():
    tk = aw.make_task("shopify", data=DATA, seed=1)
    plan, _ = doer(tk)
    env = run(tk, lambda spec: Script(spec, plan))
    res = aw.evaluate(env, tk)
    assert res["success"] and res["final_page"] == "CHECKOUT_SUCCESS_MAIN" and not res["corrections"], res


def test_misspoken_value_caught_before_paying_and_fixed():
    for seed in range(20):
        tk = aw.make_task("shopify", data=DATA, seed=seed, cue="misspoke")
        if tk.scenario["slip"]["key"] == "value":
            break
    plan, _ = doer(tk)
    fixed, _ = doer(tk, said_values=False)
    env = run(tk, lambda spec: Script(spec, plan, lambda h, t: fixed(h, t, fix=True) if "fix" in h else []))
    res = aw.evaluate(env, tk)
    assert res["success"] and len(res["corrections"]) == 1 and not res["late_correction"], res
    assert res["corrections"][0]["t"] - res["first_wrong_t"] == 900


def test_paying_at_once_makes_the_slip_too_late():
    for seed in range(20):
        tk = aw.make_task("shopify", data=DATA, seed=seed, cue="misspoke")
        if tk.scenario["slip"]["key"] == "value":
            break
    plan, _ = doer(tk)
    hasty = lambda h, t: [(t + 300, n, a) for _, n, a in plan(h, t)]  # noqa: E731  (everything at once, then pay)
    res = aw.evaluate(run(tk, lambda spec: Script(spec, hasty), max_ms=30_000), tk)
    assert not res["success"] and res["late_correction"], res


def test_patch_main():
    head = "import router from './router'\nconst app = createApp(App)\n"
    out = aw.patch_main(head + "\napp.use(createPinia())\napp.use(router)\n")
    assert "window.__pinia = __pinia" in out and "window.__router = router" in out and "import.meta.glob('./stores/*.js'" in out
    assert aw.patch_main(out) == out
    out = aw.patch_main(head + "const pinia = createPinia()\npinia.use(persist)\n\napp.use(pinia)\napp.use(router)\n")
    assert out.index("window.__pinia = pinia") > out.index("app.use(pinia)") and "{ try { f(pinia) }" in out


CHAT = {"data": {"channels": [{"id": "ch_01", "name": "general"}, {"id": "ch_02", "name": "design", "private": False}],
                 "dms": [{"id": "dm_u1", "user_name": "Sarah Connor"}, {"id": "dm_u2", "user_name": "John Smith"}]}}
TASKS = {"data": {"projects": [{"id": "p1", "name": "Project 1: Mobile App"}, {"id": "p2", "name": "Project 2: Audit"}],
                  "tasks": [{"id": "t1", "name": "Old", "project_id": "p1"}]}}


def test_chat_and_task_sites():
    for seed in range(6):
        tk = aw.make_task("slack", data=CHAT, seed=seed, cue="misspoke")
        slip, first = tk.scenario["slip"], tk.scenario["first_turn"]
        assert slip["done"].lstrip("#") in first and slip["intended"].lstrip("#") not in first, first
        s = tk.scenario["steps"][0]
        view = {"page": "CHANNEL_DETAIL", "changes": {s["key"]: slip["values"][1]}}
        assert aw.SiteEyes.misspoke(slip, tk.scenario["steps"], view) and f"is {slip['done']!r}, but you want" in aw.problems(tk.scenario["steps"], view)[0]
    tk = aw.make_task("asana", data=TASKS, seed=0)
    st = shop_steps(tk)
    who = st["new_task_assignee"]["value"]
    view = {"page": "TASK_CREATE_SUCCESS", "ended": True, "changes": {"selected_project_id": st["selected_project_id"]["value"],
            "new_task_name": st["new_task_name"]["value"], "new_task_description": "Something else", "new_task_assignee": who}}
    p = aw.problems(tk.scenario["steps"], view)
    assert p and all(getattr(x, "final", False) for x in p) and "task description on the screen is 'Something else'" in p[0], p
    draft = {"page": "TASK_CREATE_FORM", "changes": {}, "fields": [{"field": "textbox 'e.g., Draft Q4 Report'", "value": "Wrong name"}]}
    assert aw.problems(tk.scenario["steps"], draft) == [f"the task name box says 'Wrong name', but you want {st['new_task_name']['value']!r}"]


def test_long_first_turn_is_clipped_as_the_voice_reference():
    from array import array
    from interaction_gym.audio import Audio
    from interaction_gym.user import clip_reference
    sr = 1000  # 40 s of "speech" with one pause at 27-27.2 s
    s = array("h", [1000] * (40 * sr))
    for i in range(27 * sr, 27 * sr + 200):
        s[i] = 0
    text = " ".join(f"w{i}" for i in range(100))
    a, t = clip_reference(Audio(s, sr), text, 29_000)
    assert 27_000 <= a.dur_ms <= 27_200 and t.split() == text.split()[:68]
    assert clip_reference(Audio(s[:10 * sr], sr), "short one", 29_000)[1] == "short one"
