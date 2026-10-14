"""OpenApps: a desktop of everyday apps as the shared screen of a co-browsing call (``integrations.cobrowse``).

OpenApps (Meta FAIR; arXiv 2511.20766, ICLR 2026) serves small apps (calendar, messages, to-dos, maps, banking, a
shop, ...) from one Python process; their state is plain data behind ``/<app>_all`` endpoints, and its tasks are
checked against that state (the target state of every sub-task applied to the initial one, compared field by field;
anything else changed fails it). Its longer-horizon tasks combine 2-4 sub-tasks across apps, e.g. "add 'NeurIPS trip'
to my calendar on 2026-12-08 at San Diego, add 'Print the poster' to my to-dos, message Bob: '...', and mark 'Check
budget' as done".

Here the caller asks for such a task by voice, the agent operates the desktop with BrowserGym's tools while both see
the screen (``video.SCREEN``), and the caller speaks up when the screen shows something they did not want. Cues
(``CUES``) make the caller harder to serve:

- **staged**: they ask for one sub-task at a time, the next once the screen shows the last one done (so the agent
  hears new requests while it works, and must not stop listening once it has a plan);
- **misspoke**: they say one value wrong (an event's date, or who a message is for) and only realise when they see it
  on the screen. A sent message cannot be unsent: a slip caught only after the agent sent it is too late (``late``),
  so an agent that reads the message out, or shows it, before sending gives the caller their chance.

**One state, read by everyone.** After each action the backend reads the apps' state; the frames' ``view`` carries
what changed since the start (``changes``) and what is typed in the page's text boxes (``fields``, read from the
page). The caller reads the screen from that view (``AppEyes``), and whether something is wrong is decided from it
and the caller's real intent, never by the LLM. The episode is scored by OpenApps' own check of the final state
(``evaluate``), plus the corrections and their timing.

**Licence.** OpenApps is CC BY-NC 4.0 (non-commercial use only). This module copies none of it: it imports OpenApps
from your own installation and reads its task file (``longer_horizon.yaml``, ``OPENAPPS_CONFIG``) at run time. Episodes
recorded on it (screenshots, task text) fall under OpenApps' licence.

**Licence.** OpenApps is CC BY-NC 4.0 (non-commercial): it is not a dependency of this package. Install it from a
checkout into its own environment (it pins Starlette < 1): ``pip install --no-deps -e OpenApps`` plus its runtime
pins (python-fasthtml 0.12.18, starlette 0.46.2, uvicorn 0.34.2, hydra-core, deepdiff 8.6.1, fastlite 0.1.3, fastcore
1.8.2, apsw, apswutils 0.0.2), with ``browsergym-core`` and Playwright's Chromium. Its task files are read from the
checkout (``OPENAPPS_CONFIG``, default: ``open_apps.config_dir()``).
"""

from __future__ import annotations

import copy
import datetime as dt
import difflib
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..core import Task
from ..tools import ToolWorld
from ..video import SCREEN
from . import browsergym as bg
from . import cobrowse as co
from ..user import UserTurn

CUES = ("none", "staged", "misspoke")

AGENT_INSTRUCTIONS = (
    "You operate the caller's computer for them on a voice call: a desktop with apps (calendar, messages, to-dos, maps "
    "and more) in a browser. The screen is shared: the caller sees everything you do, and you see the screen too, as an "
    "image with each message. Each tool acts on the page and returns the page afterwards as an accessibility tree, one "
    "element per line with its id in brackets, e.g. [12] button 'Save'; pass that id as bid. Call read_page first to get "
    "the ids. The dock at the bottom opens the apps. Do what the caller asks; tell them briefly what you are doing, in "
    "plain words (never mention ids, tools or the accessibility tree). The caller may ask for more while you work: keep "
    "listening. A sent message cannot be unsent: before sending one, say who it is for and what it says. When everything "
    "they asked for is done, tell them in one sentence. Keep everything you say to one short sentence."
)

USER = (
    "You are on a call with an assistant that is using your computer for you: a desktop with your calendar, messages "
    "and to-do list. The screen is shared: you see everything it does. What you want done: {what}. {how} You never touch "
    "the computer yourself. You watch quietly and only speak when you have something to say; if the assistant's message "
    "needs no answer (it says what it is doing, or acknowledges), reply with exactly [silent]. If it asks you to confirm "
    "something that matches what you want, say yes in a few words. Keep replies to one or two short sentences. Once "
    "everything you want is done on the screen, thank the assistant and end the call."
)
HOW = {"all": "You have already asked for all of it.",
       "staged": "You ask for these one at a time: you have asked for the first; you ask for each next one when you are told to."}
ALL_DONE = "(Everything you wanted is now done, as you wanted it, on the shared screen.)"


# ---------------------------------------------------------------- the server (one per process)

_SERVER = None


def server():
    """The process's OpenApps server (started on first use; upstream allows one per process)."""
    global _SERVER
    if _SERVER is None:
        from open_apps.mcp.appserver import AppServer
        _SERVER = AppServer("calendar")
    return _SERVER


def app_state(url: str) -> dict:
    from open_apps.state import get_current_state
    return get_current_state(url)


# Links that open a new tab ("Add Event") open in the same one: the shared screen is one tab.
_SAME_TAB = ("document.addEventListener('click', e => { const a = e.target.closest && e.target.closest('a[target]'); "
             "if (a) a.removeAttribute('target'); }, true);")


def _page_env(url: str, viewport: tuple[int, int] = (1024, 640)):
    """A function making a BrowserGym env on ``url`` (no benchmark task: the page is just opened)."""
    from browsergym.core.env import BrowserEnv
    from browsergym.core.task import AbstractBrowserTask

    class Page(AbstractBrowserTask):
        @classmethod
        def get_task_id(cls):
            return "openapps"

        def __init__(self, seed, url):
            super().__init__(seed)
            self.url, self.viewport, self.slow_mo, self.timeout = url, {"width": viewport[0], "height": viewport[1]}, 0, 5000

        def setup(self, page):
            page.context.add_init_script(_SAME_TAB)
            page.goto(self.url)
            return "", {}

        def validate(self, page, chat_messages):
            return 0.0, False, "", {}

        def teardown(self):
            pass

    return lambda headless=True: BrowserEnv(task_entrypoint=Page, task_kwargs={"url": url}, headless=headless)


# ---------------------------------------------------------------- what changed

def norm(s: Any) -> str:
    """OpenApps' string comparison: case, punctuation and spacing ignored."""
    s = re.sub(r"[^\w\s]", "", str(s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _event(e: dict) -> dict:
    out = {"title": e.get("title") or "", "date": e.get("date") or ""}
    for k in ("location", "description", "url"):
        if e.get(k):
            out[k] = e[k]
    inv = e.get("invitees")
    if isinstance(inv, str):
        inv = [x.strip() for x in inv.split(",") if x.strip()]
    if inv:
        out["invitees"] = sorted(inv)
    return out


def _ekey(e: dict) -> tuple:
    return tuple((k, norm(v) if k != "invitees" else tuple(map(norm, v))) for k, v in sorted(e.items()))


def _diff(a: list, b: list, key) -> tuple[list, list]:
    """(added, removed) between two lists, as multisets under ``key``."""
    ca, cb = Counter(map(key, a)), Counter(map(key, b))
    added, removed = [], []
    for x in b:
        if cb[key(x)] > ca[key(x)]:
            added.append(x)
            cb[key(x)] -= 1
    cb = Counter(map(key, b))
    for x in a:
        if ca[key(x)] > cb[key(x)]:
            removed.append(x)
            ca[key(x)] -= 1
    return added, removed


def changes(initial: dict, now: dict) -> dict:
    """What the apps show now that they did not at the start: events, to-dos and places added or removed, to-dos
    ticked or unticked, messages sent (and the replies that came back)."""
    out: dict[str, list] = {}
    ea, er = _diff([_event(e) for e in initial.get("calendar") or []], [_event(e) for e in now.get("calendar") or []], _ekey)
    out["events_added"], out["events_removed"] = ea, er
    ta, tb = initial.get("todo") or [], now.get("todo") or []
    titles_a, titles_b = Counter(norm(x["title"]) for x in ta), Counter(norm(x["title"]) for x in tb)
    out["todos_added"] = [{"title": x["title"], "done": bool(int(x.get("done") or 0))} for x in _diff(ta, tb, lambda x: norm(x["title"]))[0]]
    out["todos_removed"] = [x["title"] for x in _diff(ta, tb, lambda x: norm(x["title"]))[1]]
    was = {norm(x["title"]): bool(int(x.get("done") or 0)) for x in ta}
    out["todos_ticked"] = [x["title"] for x in tb if norm(x["title"]) in was and titles_a[norm(x["title"])] == titles_b[norm(x["title"])]
                           and bool(int(x.get("done") or 0)) and not was[norm(x["title"])]]
    out["todos_unticked"] = [x["title"] for x in tb if norm(x["title"]) in was and titles_a[norm(x["title"])] == titles_b[norm(x["title"])]
                             and not bool(int(x.get("done") or 0)) and was[norm(x["title"])]]
    before = {c["user"]: len(c["messages"]) for c in initial.get("messenger") or []}
    out["sent"], out["replies"] = [], []
    for c in now.get("messenger") or []:
        for m in c["messages"][before.get(c["user"], 0):]:
            (out["sent"] if m[1] == "you" else out["replies"]).append({"to" if m[1] == "you" else "from": c["user"], "text": m[0]})
    pa, pb = initial.get("map") or [], now.get("map") or []
    name = lambda p: norm(str(p["name"]).split(",", 1)[0])  # noqa: E731
    out["places_added"] = [str(p["name"]).split(",", 1)[0] for p in _diff(pa, pb, name)[0]]
    out["places_removed"] = [str(p["name"]).split(",", 1)[0] for p in _diff(pa, pb, name)[1]]
    return {k: v for k, v in out.items() if v}


def fields(obs: dict) -> list[dict]:
    """The page's text boxes that have something typed in them: ``[{"field": "textbox 'Title'", "value": ...}]``."""
    out = []
    for n in (obs.get("axtree_object") or {}).get("nodes", []):
        role = (n.get("role") or {}).get("value")
        if role not in ("textbox", "searchbox") or n.get("ignored"):
            continue
        val = (n.get("value") or {}).get("value")
        if val:
            out.append({"field": f"{role} '{(n.get('name') or {}).get('value') or ''}'", "value": str(val)})
    return out


def _where(url: str) -> dict:
    path = "/" + url.split("://", 1)[-1].split("/", 1)[-1] if "://" in url else url
    parts = [p for p in path.split("?")[0].split("/") if p]
    app = parts[0] if parts else "desktop"
    out = {"app": app, "path": path}
    if app == "messages" and len(parts) > 1:
        out["chat"] = parts[1]
    return out


class OpenAppsBackend(bg.BrowserGymBackend):
    """The OpenApps desktop as tools (BrowserGym's, on a real browser) and a screen. ``reset`` resets every app's state
    and opens ``start``; after each action the apps' state is read back, for the frames' ``view`` and the evaluation."""

    name = "browser"
    description = "The caller's computer: a desktop of apps (calendar, messages, to-dos, maps, ...) in a browser."
    visible_only = False  # the apps' pages are taller than the window (the calendar's "Add Event" is below the month)
    skip_generic = True

    def __init__(self, instructions: str = AGENT_INSTRUCTIONS, start: str = "/", viewport: tuple[int, int] = (1024, 640), **kw):
        super().__init__(instructions=instructions, **kw)
        self.start, self.viewport = start, viewport

    def reset(self, task, rng):
        srv = server()
        theme = (task.scenario.get("openapps") or {}).get("theme")
        def fresh():
            if theme or getattr(srv, "_ig_theme", None):
                srv.reconfigure(theme=theme or "default")
                srv._ig_theme = theme
            srv.reset()
            return app_state(srv.base_url)
        initial = bg._POOL.submit(fresh).result()
        b = bg._Browser(_page_env(srv.base_url + self.start, self.viewport), 0, headless=self.headless)
        return {"browser": b, "url": srv.base_url, "initial": initial, "apps": initial, "reward": 0.0, "terminated": False,
                "truncated": False, "steps": 0, "pointer": None, "values": {}, "clicked": [], "by": {}}

    async def call(self, state, caller, call):
        res = await super().call(state, caller, call)
        if call.name != "read_page":
            state["apps"] = await state["browser"].run(app_state, state["url"])
        return res

    def render(self, state):  # (the page draws its own pointer where the last action was)
        b: bg._Browser = state["browser"]
        return b.screenshot(b.obs)

    def view(self, state):
        b: bg._Browser = state["browser"]
        return {**_where(b.obs.get("url", "")), "changes": changes(state["initial"], state["apps"]), "fields": fields(b.obs),
                "ended": False, "reward": 0.0}


def tool_world(**kw) -> ToolWorld:
    """The desktop as a ToolWorld whose picture goes to ``video.SCREEN``; only the agent acts on it."""
    latency = kw.pop("latency_ms", None)
    return ToolWorld(OpenAppsBackend(screenshots=False, **kw), bg.action_latency() if latency is None else latency, callers=("agent",),
                     screen=SCREEN, serial=("agent",))


# ---------------------------------------------------------------- the task: what the caller wants

_KIND = {"AddEventTask": "add_event", "RemoveEventTask": "remove_event", "AddToDoTask": "add_todo", "MarkToDoDoneTask": "tick_todo",
         "DeleteToDoTask": "delete_todo", "SendMessageTask": "send", "RemoveLandmarkTask": "remove_place"}


def steps_from(cfg: dict) -> list[dict]:
    """OpenApps task config (a CompositeTask's ``subtasks``, or one task) -> the caller's steps, one per sub-task."""
    subs = cfg.get("subtasks") or [cfg]
    out = []
    for s in subs:
        kind = _KIND.get(str(s.get("_target_", "")).rsplit(".", 1)[-1])
        if kind is None:
            raise ValueError(f"no co-browsing wording for {s.get('_target_')}")
        if kind == "add_event":
            st = {"title": s["title"], "date": str(s["date"])}
            for k in ("location", "description", "url"):
                if s.get(k):
                    st[k] = s[k]
            if s.get("invitees"):
                st["invitees"] = sorted(s["invitees"])
        elif kind == "remove_event":
            st = {"title": s["title"], "date": str(s["date"])}
        elif kind in ("add_todo", "tick_todo", "delete_todo"):
            st = {"title": s["todo_name"]}
        elif kind == "send":
            st = {"to": s["to"], "text": s["message"]}
        else:
            st = {"name": s["name"]}
        out.append({"do": kind, **st})
    return out


def _day(d: int) -> str:
    return f"{d}{'th' if 10 <= d % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def spoken_date(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.strftime('%B')} {_day(d.day)}, {d.year}"


def phrase(s: dict) -> str:
    """One step as the caller says it."""
    k = s["do"]
    if k == "add_event":
        out = f"put '{s['title']}' in my calendar on {spoken_date(s['date'])}"
        if s.get("location"):
            out += f", at {s['location']}"
        if s.get("description"):
            out += f", with the description '{s['description']}'"
        if s.get("invitees"):
            out += ", inviting " + " and ".join(s["invitees"])
        return out
    if k == "remove_event":
        return f"delete the '{s['title']}' event on {spoken_date(s['date'])} from my calendar"
    if k == "add_todo":
        return f"add '{s['title']}' to my to-do list"
    if k == "tick_todo":
        return f"tick off '{s['title']}' on my to-do list"
    if k == "delete_todo":
        return f"delete '{s['title']}' from my to-do list"
    if k == "send":
        return f"message {s['to']}, saying: '{s['text']}'"
    return f"remove {s['name']} from my saved places on the map"


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + ", and " + parts[-1]


def ask(steps: list[dict], first: bool = True) -> str:
    text = _join([phrase(s) for s in steps])
    return ("Hi! Could you " if first else "Could you also ") + text + "?"


def config_dir() -> Path:
    if os.environ.get("OPENAPPS_CONFIG"):
        return Path(os.environ["OPENAPPS_CONFIG"])
    from open_apps import config_dir as cd
    return Path(cd())


def load(key: str, file: str = "longer_horizon") -> dict:
    text = (config_dir() / "tasks" / f"{file}.yaml").read_text()
    try:
        import yaml
        cfg = yaml.safe_load(text)
    except ImportError:  # (a JSON file is YAML too)
        import json
        cfg = json.loads(text)
    if key not in cfg:
        raise KeyError(f"{key} is not in {file}.yaml")
    return cfg[key]


CONTACTS = ("Alice", "Bob", "Charlie")


def plant_slip(steps: list[dict], rng: random.Random) -> dict | None:
    """One value the caller says wrong: an event's date (a day later or earlier), or who a message is for."""
    options = [i for i, s in enumerate(steps) if s["do"] in ("add_event", "send")]
    if not options:
        return None
    i = rng.choice(options)
    s = steps[i]
    if s["do"] == "add_event":
        d = dt.date.fromisoformat(s["date"]) + dt.timedelta(days=rng.choice([1, -1]))
        return {"step": i, "key": "date", "intended": s["date"], "done": d.isoformat(), "kind": "date"}
    other = rng.choice([c for c in CONTACTS if c != s["to"]])
    return {"step": i, "key": "to", "intended": s["to"], "done": other, "kind": "recipient"}


def make_task(key: str, cue: str = "none", seed: int = 0, file: str = "longer_horizon", theme: str | None = None) -> Task:
    """A co-browsing task from OpenApps task ``key`` (in ``config/tasks/<file>.yaml``). ``cue``: one of ``CUES``."""
    assert cue in CUES, cue
    cfg = load(key, file)
    steps = steps_from(cfg)
    stages = list(range(len(steps))) if cue == "staged" else [0] * len(steps)
    for s, k in zip(steps, stages):
        s["stage"] = k
    said, slip = copy.deepcopy(steps), None
    if cue == "misspoke":
        slip = plant_slip(steps, random.Random(f"{key}:{seed}"))
        if slip is None:
            raise ValueError(f"{key}: nothing to misspeak")
        said[slip["step"]][slip["key"]] = slip["done"]
    for s, w in zip(steps, said):
        s["said"] = phrase(w)
    first = ask([w for w in said if w["stage"] == 0])
    what = _join([phrase(s) for s in steps]) + (" (as you meant it: when you asked, you said one thing wrong without noticing)" if slip else "")
    return Task(
        id=f"oa-{key}" + (f"-{cue}" if cue != "none" else "") + (f"-s{seed}" if seed else ""),
        scenario={"openapps": {"task": key, "file": file, "goal": cfg.get("goal"), "theme": theme}, "first_turn": first, "mode": "drive",
                  "cue": cue, "steps": steps, "slip": slip, "persona": "Someone who asked an assistant to use their computer for them, watching on a shared screen.",
                  "instructions": USER.format(what=what, how=HOW["staged" if cue == "staged" else "all"]),
                  "categories": ["screen_share", "co_browse", "openapps"]},
        initial_state={"seed": seed}, criteria={"reward": "openapps", "slip": slip})


# ---------------------------------------------------------------- what the caller sees

class Final(str):
    """A problem that can no longer be fixed (a sent message)."""

    final = True


def _close(a: str, b: str) -> bool:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() >= 0.75


def _event_matches(s: dict, e: dict) -> bool:
    want = {k: s[k] for k in ("title", "date", "location", "description", "url", "invitees") if s.get(k)}
    return _ekey(want) == _ekey(e)


def satisfied(s: dict, view: dict) -> bool:
    c = view.get("changes") or {}
    k = s["do"]
    if k == "add_event":
        return any(_event_matches(s, e) for e in c.get("events_added", []))
    if k == "remove_event":
        return any(norm(e["title"]) == norm(s["title"]) and e["date"] == s["date"] for e in c.get("events_removed", []))
    if k == "add_todo":
        return any(norm(x["title"]) == norm(s["title"]) and not x["done"] for x in c.get("todos_added", []))
    if k == "tick_todo":
        return any(norm(x) == norm(s["title"]) for x in c.get("todos_ticked", []))
    if k == "delete_todo":
        return any(norm(x) == norm(s["title"]) for x in c.get("todos_removed", []))
    if k == "send":
        return any(m["to"] == s["to"] and norm(m["text"]) == norm(s["text"]) for m in c.get("sent", []))
    return any(norm(x) == norm(s["name"]) for x in c.get("places_removed", []))


def _when(iso: str) -> str:
    try:
        return spoken_date(iso)
    except ValueError:
        return repr(iso)


def problems(steps: list[dict], view: dict, final: bool = False) -> list[str]:
    """What the screen shows that the caller did not want, in words (empty: nothing wrong so far). Things not done yet
    are not problems (see ``describe``)."""
    c, out = view.get("changes") or {}, []
    want = [s for s in steps if s["do"] == "add_event"]
    used = Counter()
    for e in c.get("events_added", []):
        ok = next((s for s in want if _event_matches(s, e) and used[id(s)] == 0), None)
        if ok is not None:
            used[id(ok)] += 1
            continue
        same = next((s for s in want if norm(s["title"]) == norm(e["title"])), None)
        if same is not None:
            if any(_event_matches(same, x) for x in c["events_added"]) and _event_matches(same, e):
                out.append(f"'{e['title']}' is in the calendar twice")
                continue
            for k in ("date", "location", "description", "url"):
                if norm(e.get(k)) != norm(same.get(k)):
                    got = _when(e[k]) if k == "date" else repr(e.get(k)) if e.get(k) else "nothing"
                    wanted = _when(same[k]) if k == "date" else repr(same.get(k)) if same.get(k) else "nothing"
                    out.append(f"the new event '{e['title']}' has {'its ' + k if k != 'date' else 'the date'} {got}, not {wanted}")
            continue
        near = next((s for s in want if _close(s["title"], e["title"])), None)
        out.append(f"the new event is called '{e['title']}', not '{near['title']}'" if near else
                   f"an event '{e['title']}' on {_when(e['date'])} was added, which you did not ask for")
    for e in c.get("events_removed", []):
        if not any(s["do"] == "remove_event" and norm(s["title"]) == norm(e["title"]) and s["date"] == e["date"] for s in steps):
            out.append(f"the event '{e['title']}' on {_when(e['date'])} has been deleted, which you did not want")
    todos = [s for s in steps if s["do"] == "add_todo"]
    for x in c.get("todos_added", []):
        if any(norm(s["title"]) == norm(x["title"]) for s in todos):
            if x["done"]:
                out.append(f"the new to-do '{x['title']}' is ticked off already")
            continue
        near = next((s for s in todos if _close(s["title"], x["title"])), None)
        out.append(f"the new to-do says '{x['title']}', not '{near['title']}'" if near else
                   f"a to-do '{x['title']}' was added, which you did not ask for")
    for x in c.get("todos_removed", []):
        if not any(s["do"] == "delete_todo" and norm(s["title"]) == norm(x) for s in steps):
            out.append(f"the to-do '{x}' has been deleted, which you did not want")
    for x in c.get("todos_ticked", []):
        if not any(s["do"] == "tick_todo" and norm(s["title"]) == norm(x) for s in steps):
            out.append(f"the to-do '{x}' has been ticked off, which you did not want")
    for x in c.get("todos_unticked", []):
        out.append(f"the to-do '{x}' has been unticked, which you did not want")
    sends = [s for s in steps if s["do"] == "send"]
    used = Counter()
    for m in c.get("sent", []):
        ok = next((s for s in sends if s["to"] == m["to"] and norm(s["text"]) == norm(m["text"]) and used[id(s)] == 0), None)
        if ok is not None:
            used[id(ok)] += 1
            continue
        same_to = next((s for s in sends if s["to"] == m["to"]), None)
        same_text = next((s for s in sends if norm(s["text"]) == norm(m["text"])), None)
        if same_text is not None and same_text["to"] != m["to"]:
            out.append(Final(f"the message '{m['text']}' was sent to {m['to']}, not {same_text['to']}"))
        elif same_to is not None and any(x["to"] == m["to"] and norm(x["text"]) == norm(same_to["text"]) for x in c["sent"]):
            out.append(Final(f"a second message '{m['text']}' was sent to {m['to']}"))
        elif same_to is not None:
            out.append(Final(f"the message sent to {m['to']} says '{m['text']}', not '{same_to['text']}'"))
        else:
            out.append(Final(f"a message '{m['text']}' was sent to {m['to']}, which you did not ask for"))
    for p in c.get("places_removed", []):
        if not any(s["do"] == "remove_place" and norm(s["name"]) == norm(p) for s in steps):
            out.append(f"{p} has been removed from your saved places, which you did not want")
    for p in c.get("places_added", []):
        out.append(f"{p} has been added to your saved places, which you did not ask for")
    chat = view.get("chat")
    if chat:  # a message being written, before it is sent
        draft = next((f["value"] for f in view.get("fields", []) if f["field"].startswith(("textbox", "searchbox"))), "")
        if draft.strip():
            to_here = [s for s in sends if s["to"] == chat and not satisfied(s, view)]
            elsewhere = [s for s in sends if s["to"] != chat and norm(s["text"]) == norm(draft)]
            if elsewhere:
                out.append(f"it is writing '{draft}' in your chat with {chat}, but that message is for {elsewhere[0]['to']}")
            elif not to_here:
                out.append(f"it is writing '{draft}' in your chat with {chat}, which you did not ask for")
            elif not any(norm(draft) == norm(s["text"]) for s in to_here) and len(norm(draft)) >= len(norm(to_here[0]["text"])):
                out.append(f"the message box to {chat} says '{draft}', not '{to_here[0]['text']}'")
    return out


def describe(steps: list[dict], view: dict | None) -> str:
    """The shared screen in words, as the caller reads it."""
    if view is None:
        return "The shared screen shows the desktop; nothing has been done yet."
    c = view.get("changes") or {}
    lines = [f"The shared screen shows the {view.get('app', 'desktop')} app" + (f" (your chat with {view['chat']})" if view.get("chat") else "") + "."]
    say = []
    for e in c.get("events_added", []):
        extra = "".join(f", {k} {e[k]!r}" for k in ("location", "description", "url") if e.get(k))
        say.append(f"new event '{e['title']}' on {_when(e['date'])}{extra}")
    say += [f"event '{e['title']}' on {_when(e['date'])} deleted" for e in c.get("events_removed", [])]
    say += [f"new to-do '{x['title']}'" + (" (ticked)" if x["done"] else "") for x in c.get("todos_added", [])]
    say += [f"to-do '{x}' deleted" for x in c.get("todos_removed", [])]
    say += [f"to-do '{x}' ticked off" for x in c.get("todos_ticked", [])]
    say += [f"to-do '{x}' unticked" for x in c.get("todos_unticked", [])]
    say += [f"message sent to {m['to']}: '{m['text']}'" for m in c.get("sent", [])]
    say += [f"{p} removed from saved places" for p in c.get("places_removed", [])]
    say += [f"{p} added to saved places" for p in c.get("places_added", [])]
    lines.append("Changes so far: " + ("; ".join(say) if say else "none") + ".")
    typed = [f"{f['field']}: {f['value']!r}" for f in view.get("fields", [])]
    if typed:
        lines.append("Typed on the page but not saved or sent yet: " + "; ".join(typed) + ".")
    asked = [s for s in steps if s.get("asked", True)]
    done = [s["said"] for s in asked if satisfied(s, view)]
    todo = [s["said"] for s in asked if not satisfied(s, view)]
    if done:
        lines.append("Of what you asked for, done as you want it: " + "; ".join(done) + ".")
    if todo:
        lines.append("Not done yet (as you want it): " + "; ".join(phrase(s) for s in asked if not satisfied(s, view)) + ".")
    return "\n".join(lines)


class AppEyes:
    """How the caller reads the OpenApps desktop (for ``cobrowse.CoUser``)."""

    problems = staticmethod(problems)
    describe = staticmethod(describe)
    satisfied = staticmethod(satisfied)

    @staticmethod
    def want(steps: list[dict]) -> str:
        return "; ".join(phrase(s) for s in steps)

    @staticmethod
    def misspoke(slip: dict | None, steps: list[dict], view: dict) -> bool:
        """The screen shows what the caller said by mistake (an event on the said date, a message to the said person)."""
        if slip is None:
            return False
        said = {**steps[slip["step"]], slip["key"]: slip["done"]}
        if said["do"] == "send":
            return said["to"] == view.get("chat") or any(m["to"] == said["to"] for m in (view.get("changes") or {}).get("sent", []))
        return any(e["date"] == said["date"] and norm(e["title"]) == norm(said["title"]) for e in (view.get("changes") or {}).get("events_added", []))

    @staticmethod
    def late(view: dict, problems: list[str]) -> bool:
        return bool(problems) and all(getattr(p, "final", False) for p in problems)

    late_note = ("(You look at the shared screen: {problems}. A sent message cannot be unsent. Say so briefly, and say "
                 "what you want now.)")


class AppUser(co.CoUser):
    """The caller watching the desktop (``cobrowse.CoUser`` with ``AppEyes``; other sites bring their own eyes, with a
    ``satisfied(step, view)``). Staged tasks: ``think_ms`` after the
    screen shows everything asked so far done (and nothing wrong), they ask for the next step. When it is all done,
    they say so (thank and hang up) on their next turn, or ``idle_ms`` after it shows if nobody speaks."""

    eyes = AppEyes

    def __init__(self, source: co.CoSource, *args, think_ms: int = 1500, idle_ms: int = 6000, **kw):
        super().__init__(source, *args, **kw)
        self.think_ms, self.idle_ms = think_ms, idle_ms

    def init_state(self, task, rng):
        st = super().init_state(task, rng)
        c = st["co"]
        c["steps"] = copy.deepcopy(task.scenario["steps"])
        for s in c["steps"]:
            s["asked"] = s["stage"] == 0
        c["stage"], c["next_at"], c["all_done_t"], c["asks"] = 0, None, None, []
        return st

    def _see(self, c, t, view):
        super()._see(c, t, view)
        ok = not c["frames"][-1]["problems"] and all(self.eyes.satisfied(s, view) for s in c["steps"] if s["asked"])
        more = any(not s["asked"] for s in c["steps"])
        if not ok:
            c["next_at"], c["all_done_t"], c["extra"] = None, None, ""
        elif more:
            if c["next_at"] is None:
                c["next_at"] = t + self.think_ms
        elif c["all_done_t"] is None:
            c["all_done_t"], c["next_at"], c["extra"] = t, t + self.idle_ms, "\n\n" + ALL_DONE

    async def _notice(self, st, t):
        frames, wake = await super()._notice(st, t)
        c = st["co"]
        if frames or c["notice_at"] is not None or c["next_at"] is None:
            return frames, min((w for w in (wake, c["next_at"]) if w is not None), default=None)
        if t < c["next_at"]:
            return [], c["next_at"]
        mine = self._mine(st)
        if (mine is not None and mine.end > t) or st["planned"] is not None or st["pending"]:
            c["next_at"] = mine.end if mine is not None and mine.end > t else t + 200
            return [], c["next_at"]
        agent = self._latest(st["agent"], t)
        talking = agent is not None and agent.active(t)
        c["next_at"] = None
        if c["all_done_t"] is not None:  # all done and nobody has said so for a while: the caller does
            if talking:
                c["next_at"] = agent.end + self.idle_ms
                return [], c["next_at"]
            return await self._speak(st, t, t, intent="done")
        nxt = min(s["stage"] for s in c["steps"] if not s["asked"])
        new = [s for s in c["steps"] if s["stage"] == nxt]
        # the next request in the task's own words (an LLM rewording it drops parts: a lost detail is a different task)
        text = "Great. Next, could you " + _join([s["said"] for s in new if s["said"]]) + "?"
        frames, wake = await self._speak(st, t, t, UserTurn(text), barge_in=talking, intent="next")
        if frames:
            for s in new:
                s["asked"] = True
            c["asks"].append({"t": t, "stage": nxt, "barge_in": talking, "said": frames[0].data.text})
        return frames, wake


# ---------------------------------------------------------------- evaluation

VERIFIER = {
    "name": "OpenApps co-browsing: the apps' final state, the caller's intent and the timeline of the shared screen",
    "description": ("OpenApps checks the final state of the apps against the task's target state (every sub-task applied to the "
                    "initial state; anything else changed fails it). What the caller wanted is known (it may differ from what "
                    "they first said); every frame of the shared screen is checked against it, giving when something wrong "
                    "first showed, when the caller said so, and whether it could still be fixed."),
    "checks": {
        "openapps_reward": "OpenApps' own check of the final state (1 or 0)",
        "as_meant": "every sub-task done as the caller meant it, and nothing else changed (from the view)",
        "late_correction": "the caller found a mistake only once it could not be undone (a sent message)",
    },
}


def openapps_reward(task: Task, initial: dict, final: dict) -> float:
    """OpenApps' own check (needs ``open_apps`` importable)."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    o = task.scenario["openapps"]
    t = instantiate(OmegaConf.create(load(o["task"], o["file"])))
    return 1.0 if t.check_if_task_is_complete(initial, final) else 0.0


def evaluate(env, task: Task, node: str = "user", tools: str = "tools", reward: bool = True) -> dict:
    c = env.sim.states[node]["co"]
    st = bg.browser_state(env, tools)
    view = {"changes": changes(st["initial"], st["apps"])}
    left = problems(c["steps"], view) + [f"not done: {s['said']}" for s in c["steps"] if not satisfied(s, view)]
    out = {"reward": openapps_reward(task, st["initial"], st["apps"]) if reward else None, "as_meant": not left, "left": left,
           "slip": task.scenario.get("slip"), "cue": task.scenario.get("cue"), "corrections": list(c["corrections"]),
           "late_correction": any(x["late"] for x in c["corrections"]), "asks": list(c["asks"]), "changes": view["changes"],
           "steps": st["steps"], "user_llm": list(c.get("raw", []))}
    out["success"] = bool(out["reward"]) if reward else out["as_meant"]
    frames = c["frames"]
    out["first_wrong_t"] = next((f["t"] for f in frames if f["problems"]), None)
    out["all_done_t"] = c["all_done_t"]
    return out
