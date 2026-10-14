"""AutoWebWorld sites as the shared screen of a co-browsing call (``integrations.cobrowse``).

AutoWebWorld (arXiv 2602.14296; MIT) synthesizes web apps from a finite-state machine: each site (29 ship with it:
shops, messaging, project tools, travel, ...) is a Vue single-page app whose whole state lives in a Pinia store,
``signature`` (the FSM's state: what is in the cart, what the form fields hold, which page is shown, ...). It ships no
tasks; here a task is a target for that state, written per site (``SITES``), and the caller asks for it by voice.

**State, read by everyone.** The sites are built with their Pinia instance on ``window.__pinia`` (``patch_main``).
After each action the backend reads the store; the frames' ``view`` carries the page (the route shown: the FSM's page id), the
store's fields that differ from the start (``changes``), and what is typed in the page's text boxes. The caller reads
the screen from it (``SiteEyes``), and whether something is wrong is decided from it and the caller's real intent,
never by the LLM. A terminal page of the FSM (an order placed, a message sent) ends the task: the state then is what
is scored (the site takes no more actions; a synthesized site may reset its data afterwards), and what is wrong then
is too late to fix.

A task's steps: ``{"do": "set", "key", "value", "label"}`` (a store field must hold this value),
``{"do": "item", "key", "match", "label"}`` (a list in the store must hold an item with these fields, e.g. a cart line)
and ``{"do": "reach", "pages", "label"}`` (finish on one of these pages).

**Sites.** Built outside this package, from an AutoWebWorld checkout: in ``<site>/web``, patch ``src/main.js``
(``patch_main``), ``npm install --legacy-peer-deps --ignore-scripts && npx vite build``; the backend serves
``<site>/web/dist`` itself (``AWW_SITES``: a directory holding ``<site>/dist`` and ``<site>/fsm.json``).

**Licence.** AutoWebWorld (github.com/Evanwu1125/AutoWebWorld) is MIT. This module copies none of its code; the sites are built from
your own checkout. The task generators (``SITES``) are ours.
"""

from __future__ import annotations

import copy
import functools
import http.server
import json
import os
import random
import re
import threading
from pathlib import Path
from typing import Any

from ..core import Task
from ..tools import ToolResult, ToolWorld
from ..video import SCREEN
from . import browsergym as bg
from . import cobrowse as co
from .openapps import ALL_DONE, USER, AppUser, Final, _join, fields, norm  # noqa: F401  (AppUser: the caller)

CUES = ("none", "staged", "misspoke")

AGENT_INSTRUCTIONS = (
    "You operate a website for the caller on a voice call. The screen is shared: the caller sees everything you do, "
    "and you see the screen too, as an image with each message. Each tool acts on the page and returns the page "
    "afterwards as an accessibility tree, one element per line with its id in brackets, e.g. [12] button 'Pay now'; "
    "pass that id as bid. Call read_page first to get the ids. Do what the caller asks; tell them briefly what you are "
    "doing, in plain words (never mention ids, tools or the accessibility tree). The caller may add or change things "
    "while you work: keep listening. Placing an order cannot be undone: before the final button, say what you are "
    "about to submit. When everything they asked for is done, tell them in one sentence. Keep everything you say to "
    "one short sentence."
)

_EXPOSE = """window.__pinia = {p}  // (read by InteractionGym: the whole state, and the page shown)
window.__router = router
for (const m of Object.values(import.meta.glob('./stores/*.js', {{ eager: true }})))  // every store from the start
  for (const [k, f] of Object.entries(m)) if (k.startsWith('use') && typeof f === 'function') {{ try {{ f({p}) }} catch (e) {{}} }}
"""


def patch_main(src: str) -> str:
    """A site's ``src/main.js`` with its Pinia instance on ``window.__pinia`` and its router on ``window.__router``,
    every store created at the start (Pinia creates a store when a page first uses it), so the whole state and the page
    shown can be read from the first page on."""
    if "window.__pinia" in src:
        return src
    if "app.use(createPinia())" in src:
        src = src.replace("app.use(createPinia())", "const __pinia = createPinia()\napp.use(__pinia)")
    m = re.search(r"^const (\w+) = createPinia\(\)", src, re.M)
    use = m and re.search(rf"^app\.use\({m.group(1)}\)\n", src, re.M)
    if not use or "import router" not in src:
        raise ValueError("main.js does not create a Pinia instance and a router the usual way")
    return src[:use.end()] + _EXPOSE.format(p=m.group(1)) + src[use.end():]


def sites_dir() -> Path:
    return Path(os.environ.get("AWW_SITES", "aww/sites"))


# ---------------------------------------------------------------- serving a site

class _SPA(http.server.SimpleHTTPRequestHandler):
    """Static files, and the app's index for any other path (the sites route in the browser)."""

    def send_head(self):
        if not os.path.exists(self.translate_path(self.path)):
            self.path = "/index.html"
        return super().send_head()

    def log_message(self, *a):
        pass


_SERVERS: dict[str, str] = {}


def serve(site: str) -> str:
    """The URL of ``site`` (served from ``sites_dir()/<site>/dist``, one server per site per process)."""
    if site not in _SERVERS:
        dist = sites_dir() / site / "dist"
        if not (dist / "index.html").exists():
            raise FileNotFoundError(f"{dist} has no built site (see integrations/autowebworld.py)")
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_SPA, directory=str(dist)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        _SERVERS[site] = f"http://127.0.0.1:{srv.server_address[1]}/"
    return _SERVERS[site]


def fsm(site: str) -> dict:
    return json.loads((sites_dir() / site / "fsm.json").read_text())


def _store(page) -> dict:
    """The site's whole state (every Pinia store), and the route shown (``_route``: the FSM's page id)."""
    return page.evaluate("() => window.__pinia ? Object.assign(JSON.parse(JSON.stringify(window.__pinia.state.value)), "
                         "{_route: window.__router ? window.__router.currentRoute.value.name : null}) : {}")


# ---------------------------------------------------------------- the backend

def _changed(initial: dict, now: dict) -> dict:
    """What differs from the start: the ``signature`` store's fields (``currentPageId`` aside), and the ``data`` store's
    (as ``data.<field>``: e.g. ``data.tasks``, a list a new task was added to)."""
    a, b = initial.get("signature") or {}, now.get("signature") or {}
    out = {k: v for k, v in b.items() if k != "currentPageId" and a.get(k) != v}
    a, b = initial.get("data") or {}, now.get("data") or {}
    out.update({f"data.{k}": v for k, v in b.items() if a.get(k) != v})
    return out


def _page_id(store: dict) -> str | None:
    return store.get("_route") or (store.get("signature") or {}).get("currentPageId")


class AWWBackend(bg.BrowserGymBackend):
    """An AutoWebWorld site as tools (BrowserGym's, on a real browser) and a screen. Each episode opens the site in a
    fresh browser (the sites keep their state in memory: a fresh page is a fresh state)."""

    name = "browser"
    description = "A website in a browser; click, type and read it."
    visible_only = False
    skip_generic = False  # the sites' custom controls are plain <div>s with a click handler: they show as `generic`

    def __init__(self, instructions: str = AGENT_INSTRUCTIONS, viewport: tuple[int, int] = (1024, 640), **kw):
        super().__init__(instructions=instructions, **kw)
        self.viewport = viewport

    def reset(self, task, rng):
        site = task.scenario["aww"]["site"]
        url = serve(site)
        b = bg._Browser(_page_env(url, VIEWPORT.get(site, self.viewport)), 0, headless=self.headless)
        initial = b.pool.submit(lambda: _store(b.env.unwrapped.page)).result()
        return {"browser": b, "site": site, "terminal": fsm(site)["meta"].get("terminal_pages", []), "initial": initial, "store": initial, "done": None,
                "reward": 0.0, "terminated": False, "truncated": False, "steps": 0, "pointer": None, "values": {}, "clicked": [], "by": {}}

    async def call(self, state, caller, call):
        if state.get("done") is not None:  # the FSM's task is over once a terminal page shows (what the site does after is not)
            return ToolResult(call.id, call.name, "Error: the task is complete (the site has finished it)", error=True)
        res = await super().call(state, caller, call)
        if call.name != "read_page":
            b = state["browser"]
            state["store"] = await b.run(lambda: _store(b.env.unwrapped.page))
            if _page_id(state["store"]) in state["terminal"]:
                state["done"] = state["store"]
        return res

    def render(self, state):
        b: bg._Browser = state["browser"]
        return b.screenshot(b.obs, state["pointer"])

    def view(self, state):
        b: bg._Browser = state["browser"]
        page = _page_id(state["store"])
        return {"page": page, "path": "/" + b.obs.get("url", "").split("://", 1)[-1].split("/", 1)[-1] if "://" in b.obs.get("url", "") else "/",
                "changes": _changed(state["initial"], state["store"]), "fields": fields(b.obs), "ended": page in state["terminal"], "reward": 0.0}


def _page_env(url: str, viewport):
    from browsergym.core.env import BrowserEnv
    from browsergym.core.task import AbstractBrowserTask

    class Page(AbstractBrowserTask):
        @classmethod
        def get_task_id(cls):
            return "autowebworld"

        def __init__(self, seed, url):
            super().__init__(seed)
            self.url, self.viewport, self.slow_mo, self.timeout = url, {"width": viewport[0], "height": viewport[1]}, 0, 5000

        def setup(self, page):
            page.on("dialog", lambda d: d.dismiss())  # (the sites' alert() boxes for a missing field)
            page.goto(self.url)
            page.wait_for_function("() => !!window.__pinia", timeout=10000)
            return "", {}

        def validate(self, page, chat_messages):
            return 0.0, False, "", {}

        def teardown(self):
            pass

    return lambda headless=True: BrowserEnv(task_entrypoint=Page, task_kwargs={"url": url}, headless=headless)


def tool_world(**kw) -> ToolWorld:
    latency = kw.pop("latency_ms", None)
    return ToolWorld(AWWBackend(screenshots=False, **kw), bg.action_latency() if latency is None else latency, callers=("agent",),
                     screen=SCREEN, serial=("agent",))


# ---------------------------------------------------------------- what the caller sees

def _same(a: Any, b: Any) -> bool:
    """Equal as the caller reads it: numbers by value; text ignoring case, punctuation and spacing ("12 / 30" is "12/30")."""
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return norm(a).replace(" ", "") == norm(b).replace(" ", "")


def _name(s: dict) -> str:
    """A step as the caller means it (``want``: their words without a slip), or its label."""
    return s.get("want", s["said"]) or s["label"]


def _show(s: dict, v: Any, key: str | None = None) -> str:
    """A value as the screen shows it (an id as its name: ``s["names"]``)."""
    names = s.get("names") or {}
    names = names.get(key, {}) if key is not None else names
    return repr(names.get(v, v)) if v is not None else "nothing"


def _item_text(it: dict, keys, s: dict | None = None) -> str:
    return ", ".join(f"{k.replace('_id', '').replace('_', ' ')} {_show(s or {}, it.get(k), k)}" for k in keys)


def _drafts(s: dict, view: dict) -> dict:
    """What is typed (not yet saved) in the page's boxes for ``s``'s values: ``{key: text}`` (``s["inputs"]``: key ->
    a part of the box's label)."""
    out = {}
    for key, label in (s.get("inputs") or {}).items():
        box = next((f["value"] for f in view.get("fields", []) if label.lower() in f["field"].lower()), None)
        if box:
            out[key] = box
    return out


def _state(view: dict, key: str, initial: Any = None) -> Any:
    return (view.get("changes") or {}).get(key, initial)


def satisfied(s: dict, view: dict) -> bool:
    if s["do"] == "set":
        return _same(_state(view, s["key"]), s["value"])
    if s["do"] == "item":
        items = _state(view, s["key"]) or []
        if s["key"].startswith("data."):
            items = items[s.get("before", 0):]
        return any(all(_same(it.get(k), v) for k, v in s["match"].items()) for it in items)
    return view.get("page") in s["pages"]


def problems(steps: list[dict], view: dict, final: bool = False) -> list[str]:
    """What the screen shows that the caller did not want (empty: nothing wrong so far). A task that has ended (an
    order placed) with something wrong is too late to fix (``Final``)."""
    out, ended = [], bool(view.get("ended"))
    mk = Final if ended else str
    for s in steps:
        if s["do"] == "set":
            v = _state(view, s["key"])
            if v not in (None, "") and not _same(v, s["value"]):
                out.append(mk(f"the {s['label']} on the screen is {_show(s, v)}, but you want {_show(s, s['value'])}"))
        if not ended and not satisfied(s, view):  # typed but not saved yet
            want = s["match"] if s["do"] == "item" else {"value": s.get("value")}
            for key, text in _drafts(s, view).items():
                if key in want and not _same(text, want[key]):
                    out.append(f"the {key.replace('_', ' ') if key != 'value' else s['label']} box says {text!r}, but you want {want[key]!r}")
    lists: dict[str, list[dict]] = {}
    for s in steps:
        if s["do"] == "item":
            lists.setdefault(s["key"], []).append(s)
    for key, want in lists.items():
        name = want[0].get("list", key.replace("_", " "))
        keys = list(want[0]["match"])
        before = want[0].get("before", 0)  # (a list that held items at the start: only new ones count)
        for it in (_state(view, key) or [])[before:] if key.startswith("data.") else _state(view, key) or []:
            if any(all(_same(it.get(k), v) for k, v in s["match"].items()) for s in want):
                continue
            first = keys[0]
            near = next((s for s in want if _same(it.get(first), s["match"][first])), None)
            if near is not None:
                diff = [f"{k.replace('_id', '').replace('_', ' ')} {_show(near, it.get(k), k)}, not {_show(near, near['match'][k], k)}"
                        for k in keys if not _same(it.get(k), near["match"][k])]
                out.append(mk(f"the {name} has {it.get(first)!r} with " + " and ".join(diff)))
            else:
                out.append(mk(f"the {name} has {_item_text(it, keys, want[0])}, which you did not ask for"))
    if ended:
        out += [Final(f"it has finished ({view.get('page')}) without: {_name(s)}") for s in steps
                if s["do"] != "reach" and s.get("asked", True) and not satisfied(s, view)]
    return out


def describe(steps: list[dict], view: dict | None) -> str:
    if view is None:
        return "The shared screen shows the website's home page; nothing has been done yet."
    lines = [f"The shared screen shows the page {view.get('page')} ({view.get('path')})."]
    ch = {k: v for k, v in (view.get("changes") or {}).items() if v not in (None, "", [], 0)}
    lists = {s["key"]: s for s in steps if s["do"] == "item"}
    sets = {s["key"]: s for s in steps if s["do"] == "set"}
    shown = []
    for k, v in ch.items():
        if k in lists:
            s = lists[k]
            new = v[s.get("before", 0):] if k.startswith("data.") else v
            if new:
                shown.append(f"{s.get('list', k)}: " + "; ".join(_item_text(it, list(s["match"]), s) for it in new))
        elif not isinstance(v, (dict, list)) and not k.startswith("data."):
            shown.append(f"{k.replace('_', ' ')}: {_show(sets.get(k, {}), v)}")
    lines.append("What it holds now: " + ("; ".join(shown) if shown else "nothing yet") + ".")
    typed = [f"{f['field']}: {f['value']!r}" for f in view.get("fields", [])]
    if typed:
        lines.append("Typed in the page's boxes: " + "; ".join(typed) + ".")
    asked = [s for s in steps if s.get("asked", True)]
    done = [_name(s) for s in asked if satisfied(s, view)]
    todo = [_name(s) for s in asked if not satisfied(s, view)]
    if done:
        lines.append("Of what you asked for, done as you want it: " + "; ".join(done) + ".")
    if todo:
        lines.append("Not done yet (as you want it): " + "; ".join(todo) + ".")
    if view.get("ended"):
        lines.append("The site has finished the task (it cannot be changed any more).")
    return "\n".join(lines)


class SiteEyes:
    problems = staticmethod(problems)
    describe = staticmethod(describe)
    satisfied = staticmethod(satisfied)

    @staticmethod
    def want(steps: list[dict]) -> str:
        return "; ".join(s["said"] for s in steps)

    @staticmethod
    def misspoke(slip: dict | None, steps: list[dict], view: dict) -> bool:
        if slip is None:
            return False
        said = copy.deepcopy(steps[slip["step"]])
        if said["do"] == "set":
            said["value"] = slip.get("values", (None, slip["done"]))[1]
        else:
            said["match"][slip["key"]] = slip["done"]
        return satisfied(said, view)

    @staticmethod
    def late(view: dict, problems: list[str]) -> bool:
        return bool(problems) and all(getattr(p, "final", False) for p in problems)

    late_note = ("(You look at the shared screen: {problems}. It has gone through and cannot be changed. Say so briefly, "
                 "and end the call.)")


class SiteUser(AppUser):
    """The caller on an AutoWebWorld site (``openapps.AppUser``'s staging and wrap-up, with ``SiteEyes``)."""

    eyes = SiteEyes


# ---------------------------------------------------------------- tasks, per site

NAMES = [("Maya", "Patel"), ("Tom", "Becker"), ("Lena", "Novak"), ("Omar", "Haddad"), ("Grace", "Kim"), ("Jonas", "Berg")]
STREETS = ["12 Mill Lane", "48 Harbour Road", "7 Orchard Close", "230 King Street", "5 Station Approach", "91 Elm Grove"]
CITIES = [("Leeds", "LS6 2AB"), ("Bristol", "BS1 5TR"), ("York", "YO1 7HH"), ("Cambridge", "CB2 1TN"), ("Bath", "BA1 1LZ")]
TEST_CARD = {"card_number": "4242 4242 4242 4242", "card_expiry": "12/30", "card_cvv": "123"}


def _shop_task(data: dict, rng: random.Random) -> dict:
    """Shopify-like store: buy one variant of one product, check out with an address, a shipping method and a card."""
    products = [p for p in data["products"] if any(v.get("available", True) for v in p["variants"])]
    p = rng.choice(products)
    v = rng.choice([v for v in p["variants"] if v.get("available", True)])
    q = rng.choice([1, 2, 3])
    first, last = rng.choice(NAMES)
    street, (city, post) = rng.choice(STREETS), rng.choice(CITIES)
    email = f"{first.lower()}.{last.lower()}@example.com"
    method = rng.choice(["standard", "express"])
    variant = f" in {v['title']}" if len(p["variants"]) > 1 else ""
    steps = [
        {"do": "item", "key": "cart_items", "list": "cart", "match": {"title": p["title"], "variant_title": v["title"], "quantity": q},
         "label": "cart", "said": f"get me {q} of the {p['title']}{variant}", "stage": 0},
        {"do": "set", "key": "email", "value": email, "label": "email", "said": f"my email is {email}", "stage": 1},
        {"do": "set", "key": "shipping_first_name", "value": first, "label": "first name", "said": f"the name is {first} {last}", "stage": 1},
        {"do": "set", "key": "shipping_last_name", "value": last, "label": "last name", "said": "", "stage": 1},
        {"do": "set", "key": "shipping_address1", "value": street, "label": "address", "said": f"ship it to {street}", "stage": 1},
        {"do": "set", "key": "shipping_city", "value": city, "label": "city", "said": f"in {city}", "stage": 1},
        {"do": "set", "key": "shipping_postcode", "value": post, "label": "postal code", "said": f"postcode {post}", "stage": 1},
        {"do": "set", "key": "shipping_method", "value": method, "label": "shipping method", "said": f"{method} shipping", "stage": 2},
        {"do": "set", "key": "card_name", "value": f"{first} {last}", "label": "name on card", "said": "", "stage": 2},
        {"do": "set", "key": "card_number", "value": TEST_CARD["card_number"], "label": "card number",
         "said": f"pay with my test card {TEST_CARD['card_number']}, expiry {TEST_CARD['card_expiry']}, security code {TEST_CARD['card_cvv']}, name on the card as before", "stage": 2},
        {"do": "set", "key": "card_expiry", "value": TEST_CARD["card_expiry"], "label": "card expiry", "said": "", "stage": 2},
        {"do": "set", "key": "card_cvv", "value": TEST_CARD["card_cvv"], "label": "security code", "said": "", "stage": 2},
        {"do": "reach", "pages": ["CHECKOUT_SUCCESS_MAIN"], "label": "order placed", "said": "place the order", "stage": 2},
    ]
    slips = [(0, "quantity", lambda: rng.choice([x for x in (1, 2, 3) if x != q])),
             (5, "value", lambda: rng.choice([c for c, _ in CITIES if c != city])),
             (6, "value", lambda: post[:-1] + ("X" if post[-1] != "X" else "Y"))]
    if len(p["variants"]) > 1:
        slips.append((0, "variant_title", lambda: rng.choice([x["title"] for x in p["variants"] if x["title"] != v["title"]])))
    return {"steps": steps, "slips": slips, "what": f"buy {q} × {p['title']} ({v['title']})"}


MESSAGES = ["The release is moving to Thursday, please update your plans", "Lunch and learn at 1pm today in the big room",
            "Can everyone review the new onboarding doc by Friday?", "Reminder: the office is closed on Monday",
            "Great work on the launch, thank you all", "Who has the slides from yesterday's demo?"]


def _chat_task(data: dict, rng: random.Random) -> dict:
    """Slack-like workspace: post a message in a channel, or send someone a direct message."""
    msg = rng.choice(MESSAGES)
    if rng.random() < 0.5:
        chans = [c for c in data["channels"] if not c.get("private")]
        c = rng.choice(chans)
        names = {x["id"]: "#" + x["name"] for x in data["channels"]}
        steps = [
            {"do": "set", "key": "selected_channel_id", "value": c["id"], "names": names, "label": "channel",
             "said": f"go to the #{c['name']} channel", "stage": 0},
            {"do": "set", "key": "compose_text", "value": msg, "label": "message", "inputs": {"value": "message"},
             "said": f"post this: '{msg}'", "stage": 1},
            {"do": "reach", "pages": ["SEND_MESSAGE_SUCCESS"], "label": "message sent", "said": "send it", "stage": 1}]
        slips = [(0, "value", lambda: rng.choice([x["id"] for x in chans if x["id"] != c["id"]]))]
        return {"steps": steps, "slips": slips, "what": f"post in #{c['name']}"}
    u = rng.choice(data["dms"])
    names = {x["id"]: x["user_name"] for x in data["dms"]}
    steps = [
        {"do": "set", "key": "selected_dm_id", "value": u["id"], "names": names, "label": "conversation",
         "said": f"open my direct messages with {u['user_name']}", "stage": 0},
        {"do": "set", "key": "dm_compose_text", "value": msg, "label": "message", "inputs": {"value": "message"},
         "said": f"send them this: '{msg}'", "stage": 1},
        {"do": "reach", "pages": ["START_DM_SUCCESS"], "label": "message sent", "said": "", "stage": 1}]
    slips = [(0, "value", lambda: rng.choice([x["id"] for x in data["dms"] if x["id"] != u["id"]]))]
    return {"steps": steps, "slips": slips, "what": f"message {u['user_name']}"}


TASK_NAMES = ["Book the venue for the offsite", "Draft the Q4 budget", "Update the hiring scorecard", "Fix the login timeout bug",
              "Prepare the board deck", "Order new laptops for the interns"]
DETAILS = ["Get three quotes before Friday", "Use last year's numbers as a starting point", "Check with Priya first",
           "Needs sign-off from legal", "Keep it under two pages", "Ask IT for the specs"]


def _task_task(data: dict, rng: random.Random) -> dict:
    """Asana-like project tool: add a task to a project, with a description and an assignee."""
    p = rng.choice(data["projects"])
    name, detail, mine = rng.choice(TASK_NAMES), rng.choice(DETAILS), rng.random() < 0.5
    projects = {x["id"]: x["name"] for x in data["projects"]}
    steps = [  # (the site keeps the new task's fields in the FSM's signature when it is created)
        {"do": "set", "key": "selected_project_id", "value": p["id"], "names": projects, "label": "project",
         "said": f"open {p['name']}", "stage": 0},
        {"do": "set", "key": "new_task_name", "value": name, "label": "task name", "inputs": {"value": "Draft Q4"},
         "said": f"add a task called '{name}'", "stage": 1},
        {"do": "set", "key": "new_task_description", "value": detail, "label": "task description", "inputs": {"value": "Details"},
         "said": f"with the description '{detail}'", "stage": 1},
        {"do": "set", "key": "new_task_assignee", "value": "me" if mine else None, "names": {"me": "me", None: "nobody"},
         "label": "assignee", "said": "assign it to me" if mine else "leave it unassigned", "stage": 1},
        {"do": "reach", "pages": ["TASK_CREATE_SUCCESS"], "label": "task created", "said": "", "stage": 1}]
    slips = [(0, "value", lambda: rng.choice([x["id"] for x in data["projects"] if x["id"] != p["id"]]))]
    return {"steps": steps, "slips": slips, "what": f"add a task to {p['name']}"}


SITES = {"shopify": _shop_task, "slack": _chat_task, "asana": _task_task}
# Sites whose desktop layout hides a way back (Slack's conversation page shows its back button only on a narrow
# screen, and no sidebar instead): shown at a phone-like width.
VIEWPORT = {"slack": (480, 860)}


def site_data(site: str, headless: bool = True) -> dict:
    """The site's starting store (opens it once)."""
    b = bg._Browser(_page_env(serve(site), (1024, 640)), 0, headless=headless)
    try:
        return b.pool.submit(lambda: _store(b.env.unwrapped.page)).result()
    finally:
        b.close()


def _say(steps: list[dict]) -> str:
    parts = [s["said"] for s in steps if s["said"]]
    return _join(parts)


def make_task(site: str, cue: str = "none", seed: int = 0, data: dict | None = None) -> Task:
    """A co-browsing task on AutoWebWorld site ``site`` (one of ``SITES``). ``cue``: one of ``CUES``."""
    assert cue in CUES and site in SITES, (site, cue)
    rng = random.Random(f"{site}:{seed}")
    data = data if data is not None else site_data(site)
    spec = SITES[site](data.get("data") or data, rng)
    steps = spec["steps"]
    if cue != "staged":
        for s in steps:
            s["stage"] = 0
    for s in steps:
        s["want"] = s["said"]  # (as meant; ``said`` below is as said, with the slip)
    said, slip = copy.deepcopy(steps), None
    if cue == "misspoke":
        i, key, wrong = rng.choice(spec["slips"])
        done = wrong()
        s = said[i]
        if key == "value":
            names = s.get("names") or {}
            old_name, new_name = str(names.get(s["value"], s["value"])).lstrip("#"), str(names.get(done, done)).lstrip("#")
            s["said"] = s["said"].replace(old_name, new_name)
            slip = {"step": i, "key": "value", "intended": names.get(s["value"], s["value"]), "done": names.get(done, done), "kind": steps[i]["label"]}
            slip["values"] = (s["value"], done)
            s["value"] = done
        else:
            old = steps[i]["match"][key]
            s["said"] = re.sub(rf"\b{re.escape(str(old))}\b", str(done), s["said"], count=1)
            slip = {"step": i, "key": key, "intended": old, "done": done, "kind": key.replace("_", " ")}
            s["match"][key] = done
    first = "Hi! Could you " + _say([w for w in said if w["stage"] == 0]) + "?"
    what = _say(steps) + (" (as you meant it: when you asked, you said one thing wrong without noticing)" if slip else "")
    for s, w in zip(steps, said):  # what the caller says for each step (with the slip, if planted; "" for a detail that
        s["said"] = w["said"]      # goes with the one before, e.g. the last name with the name)
    return Task(
        id=f"aww-{site}" + (f"-{cue}" if cue != "none" else "") + f"-s{seed}",
        scenario={"aww": {"site": site, "what": spec["what"]}, "first_turn": first, "mode": "drive", "cue": cue, "steps": steps, "slip": slip,
                  "persona": "Someone who asked an assistant to do something on a website for them, watching on a shared screen.",
                  "instructions": USER.format(what=what, how="You ask for these one at a time: you have asked for the first; you ask for each next one when you are told to."
                                              if cue == "staged" else "You have already asked for all of it.").replace(
                      "your computer for you: a desktop with your calendar, messages and to-do list", "a website for you"),
                  "categories": ["screen_share", "co_browse", "autowebworld"]},
        initial_state={"seed": seed}, criteria={"reward": "state", "slip": slip})


# ---------------------------------------------------------------- evaluation

VERIFIER = {
    "name": "AutoWebWorld co-browsing: the site's state, the caller's intent and the timeline of the shared screen",
    "description": ("The site's whole state is its store (the FSM's signature). The task is a target for it: fields holding given "
                    "values, list items (cart lines), and a terminal page reached. Every frame of the shared screen is checked "
                    "against the caller's real intent (which may differ from what they first said)."),
    "checks": {
        "state_reward": "1 if the final store holds every target value and the terminal page was reached, else 0",
        "late_correction": "the caller found a mistake only once the site had finished (an order placed)",
    },
}


def evaluate(env, task: Task, node: str = "user", tools: str = "tools") -> dict:
    c = env.sim.states[node]["co"]
    st = bg.browser_state(env, tools)
    final = st.get("done") or st["store"]  # (the state when the site finished the task, if it did)
    page = _page_id(final)
    view = {"page": page, "changes": _changed(st["initial"], final), "ended": page in st["terminal"]}
    wrong = problems(c["steps"], {**view, "ended": False})
    missing = [_name(s) for s in c["steps"] if not satisfied(s, view)]
    out = {"reward": float(not wrong and not missing), "as_meant": not wrong and not missing, "left": wrong + [f"not done: {m}" for m in missing],
           "slip": task.scenario.get("slip"), "cue": task.scenario.get("cue"), "corrections": list(c["corrections"]),
           "late_correction": any(x["late"] for x in c["corrections"]), "asks": list(c["asks"]), "final_page": view["page"],
           "changes": {k: v for k, v in view["changes"].items() if k not in ("products", "orders", "addresses")}, "steps": st["steps"],
           "user_llm": list(c.get("raw", []))}
    out["success"] = out["as_meant"]
    out["first_wrong_t"] = next((f["t"] for f in c["frames"] if f["problems"]), None)
    out["all_done_t"] = c["all_done_t"]
    return out
