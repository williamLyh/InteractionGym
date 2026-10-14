"""Screen sharing: the caller fills in a web page themselves while an assistant watches their screen and listens.

A rendered world seen as video. The page is a real browser (BrowserGym MiniWoB++); the simulated user acts on it with
its own hands (the browser's tools, called on ``user.tool_call``) and the page's picture is streamed to the agent on
``video.SCREEN`` (``ToolWorld(screen=...)``). The agent cannot touch the page. Its job is the one a person looking over
your shoulder has: say so, in time, when what you do does not match what you said you wanted.

**One world state, every stream derived from it.** The page is the state. The video is its picture (rendered, never
generated). The user's hands follow a plan made from the page and the task (``make_task``), with one planted slip (a
typo, the wrong field, the wrong case) or none. The user's speech comes from an LLM that is told, at every turn, what
it has done so far (``ScreenShareSource``): what it *meant* to do, since people do not see their own slips. When the
agent says something is wrong, the user looks: then it is told what the page really shows, and whether there is a
mistake is read from the page, not decided by the LLM; if there is, its hands correct it. The LLM writes only the
words. So what the user says, what it does and what the agent sees cannot drift apart.

**Timing is the test.** After typing, the user pauses for ``review_ms`` and then submits; the slip can only be fixed if
the agent speaks up between the moment it becomes visible and the submit. The hands wait while the agent is talking and
while the user is answering it (a person stops to listen), but not for an agent that stays silent.

``evaluate`` reports the page's own reward and whether the page, as submitted, showed what the user meant (the pages'
checks are not always complete: form-sequence-2 takes any ticked radio button), whether and how fast the slip was
caught, and false alarms (the agent claimed a mistake the user, looking, could not see).

Supported pages (``TASKS``): enter-text, enter-text-2, login-user, form-sequence-2. Password fields are skipped as
slip targets (their characters are dots, so nobody could see the slip).
"""

from __future__ import annotations

import contextvars
import random
import re

from ..core import Frame, Segment, Task
from ..tools import CALL, RESULT, ToolCall, ToolResult, tool_log
from ..user import LLMSource, UserSim, UserTurn
from ..video import SCREEN
from . import browsergym as bg

TASKS = ("enter-text", "enter-text-2", "login-user", "form-sequence-2")
SLIPS = ("typo", "drop", "case", "wrong_field", "none")

AGENT_INSTRUCTIONS = (
    "You are an assistant on a voice call. The caller shares their screen with you: you see it as images, updated as it "
    "changes. You cannot click or type; the caller does everything on the page themselves. They told you what they want "
    "to enter. Watch their screen while they work. If what they do does not match what they told you (a typo, a wrong "
    "box, wrong capitalisation), tell them at once, briefly and specifically, before they submit. Password fields show "
    "dots: you cannot read them, so never comment on them. Otherwise stay quiet while they work, and answer when they "
    "speak to you; once something you pointed out is fixed, say nothing more about it. Keep every reply to one short sentence."
)

USER_INSTRUCTIONS = (
    "You are on a call with an assistant who can see your screen; you share it so that someone checks your work. You are "
    "filling in a form on a web page yourself: {what}. You do all the clicking and typing yourself; the assistant only "
    "watches and cannot touch your screen, so never ask it to type or click. You have already told it what you are "
    "entering. You work quietly and only speak when you have something to say. If the assistant's message needs no answer (it acknowledges, or says all is fine), reply with exactly "
    "[silent] and you will say nothing. Say that you have done something (typed, clicked, submitted) only when what "
    "you have done so far, as you are told it, shows it. Keep replies to one short sentence. Once the page has taken your submission, thank the assistant "
    "and end the call."
)
CHECK_NOTE = (
    "(You look at your screen to check what the assistant just said. {screen}\n{verdict} Answer in a few words.)"
)
LOOKED = {True: "There is indeed a mistake on it: agree, and say you will fix it (you then fix it).",
          False: "Everything on it is as you meant it: say briefly that it looks right to you.",
          "late": "There was indeed a mistake, but you have already submitted the form and it cannot be changed any more: say so."}

CLAIMS = ("A caller is filling in a form while an assistant watches their screen. Does the assistant's message below point "
          "out that something on the caller's screen is wrong or not what the caller meant (a wrong value, spelling, "
          "capitalisation or place)? Saying that things look right, acknowledging, encouraging, or describing the screen "
          "without saying anything is wrong: no.\nExamples: \"Wait, 16 is in the second box, not the third.\" yes. "
          "\"You typed Jhon, it should be John.\" yes. \"That looks correct, go ahead.\" no. \"I can't see the password.\" "
          "no.\nAnswer yes or no.\n\nAssistant: {said}")


# ---------------------------------------------------------------- the plan: what the user's hands do


def _fields(page: str) -> list[dict]:
    """Text boxes, radio buttons and buttons of a flattened MiniWoB page, in order, with the label before each."""
    out, label = [], ""
    for line in page.splitlines():
        m = re.search(r"StaticText '([^']*)'", line)
        if m and not re.search(r"\[\d+\]", line):
            label = m.group(1)
        m = re.search(r"\[(\d+)\] (textbox|radio|button) '([^']*)'", line)
        if m:
            out.append({"bid": m.group(1), "role": m.group(2), "name": m.group(3) or label})
            label = ""
    return out


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"number {n}")


def plan(name: str, goal: str, page: str) -> dict:
    """The steps that do the task, and how the user says what they want: ``{"steps", "say", "what"}``.
    A step is ``{"do": "type" | "click", "bid", "value"?, "field"}``; the last one submits (``"submit": True``)."""
    f = _fields(page)
    boxes = [x for x in f if x["role"] == "textbox"]
    button = [x for x in f if x["role"] == "button"][-1]
    quoted = re.findall(r'"([^"]*)"', goal)
    steps, say = [], []
    if name in ("enter-text", "enter-text-2"):
        value = quoted[0].upper() if "upper case" in goal else quoted[0]
        steps.append({"do": "type", "bid": boxes[0]["bid"], "value": value, "field": "the text field"})
        say.append(f"I need to enter {value}, {_spell(value)}" + (", all in capitals" if "upper case" in goal else ""))
        what = f"entering {value} into the text field and pressing {button['name']}"
    elif name == "login-user":
        user, pw = quoted[:2]
        steps += [{"do": "type", "bid": boxes[0]["bid"], "value": user, "field": "the username"},
                  {"do": "type", "bid": boxes[1]["bid"], "value": pw, "field": "the password", "secret": True}]
        say.append(f"I'm logging in, my username is {user}, {_spell(user)}, and I have the password")
        what = f"logging in with the username {user} and the password {pw}"
    elif name == "form-sequence-2":
        radio = int(re.search(r"(\d+)(?:st|nd|rd|th) radio", goal).group(1))
        box = int(re.search(r"(\d+)(?:st|nd|rd|th) textbox", goal).group(1))
        radios = [x for x in f if x["role"] == "radio"]
        steps += [{"do": "click", "bid": radios[radio - 1]["bid"], "field": f"the {_ordinal(radio)} radio button", "index": radio - 1,
                   "options": [x["bid"] for x in radios], "names": {x["bid"]: f"the {_ordinal(j + 1)} radio button" for j, x in enumerate(radios)}},
                  {"do": "type", "bid": boxes[box - 1]["bid"], "value": quoted[0], "field": f"the {_ordinal(box)} box", "index": box - 1,
                   "options": [x["bid"] for x in boxes], "names": {x["bid"]: f"the {_ordinal(j + 1)} box" for j, x in enumerate(boxes)}}]
        say.append(f"I have to tick the {_ordinal(radio)} radio button and put {quoted[0]} in the {_ordinal(box)} box")
        what = f"ticking the {_ordinal(radio)} radio button, entering {quoted[0]} into the {_ordinal(box)} text box and pressing {button['name']}"
    else:
        raise ValueError(f"no plan for {name}; supported: {TASKS}")
    steps.append({"do": "click", "bid": button["bid"], "field": f"the {button['name']} button", "submit": True})
    return {"steps": steps, "say": say, "what": what}


def _spell(word: str) -> str:
    return " ".join(word) if word.isalpha() else word


def slip(steps: list[dict], kind: str, rng: random.Random) -> dict | None:
    """Plant one slip of ``kind`` in one visible step; returns ``{"step", "kind", "intended", "done"}`` (``done``: the
    value typed, or the element clicked/typed into instead), or None if this page has no place for it."""
    if kind == "none":
        return None
    if kind == "wrong_field":
        # boxes only: MiniWoB's form-sequence-2 accepts any ticked radio button, so a wrong radio would go unnoticed by the page
        cands = [i for i, s in enumerate(steps) if s["do"] == "type" and len(s.get("options", ())) > 1]
        if not cands:
            return None
        i = rng.choice(cands)
        s = steps[i]
        others = [b for b in s["options"] if b != s["bid"]]
        near = [b for b in others if abs(s["options"].index(b) - s["index"]) == 1]  # a neighbour: the likely slip
        done = rng.choice(near or others)
        return {"step": i, "kind": kind, "intended": s["bid"], "done": done, "intended_name": s["names"][s["bid"]], "done_name": s["names"][done]}
    cands = [i for i, s in enumerate(steps) if s["do"] == "type" and not s.get("secret") and len(s["value"]) >= (1 if kind == "case" else 3)]
    if not cands:
        return None
    i = rng.choice(cands)
    v = steps[i]["value"]
    if kind == "typo":  # two neighbouring letters swapped (never the first letter: people rarely get that wrong)
        ks = [k for k in range(1, len(v) - 1) if v[k] != v[k + 1]]
        if not ks:
            return None
        k = rng.choice(ks)
        done = v[:k] + v[k + 1] + v[k] + v[k + 2:]
    elif kind == "drop":
        k = rng.randrange(1, len(v))
        done = v[:k] + v[k + 1:]
    elif kind == "case":
        done = v.lower() if v != v.lower() else v.capitalize()
        if done == v:
            return None
    else:
        raise ValueError(kind)
    return {"step": i, "kind": kind, "intended": v, "done": done}


def make_task(name: str, seed: int = 0, slip_kind: str = "typo", review_ms: int = 3000, char_ms: int = 220,
              headless: bool = True) -> Task:
    """A screen-sharing task on MiniWoB page ``name`` (one of ``TASKS``) with one planted slip (``SLIPS``; "none" for a
    control episode in which any mistake the agent claims is a false alarm). Opens the page once to read it."""
    env_id = f"browsergym/miniwob.{name}"
    b = bg._Browser(env_id, seed, headless=headless)
    try:
        goal, page = bg._goal(b.obs), bg._page(b.obs)
    finally:
        b.close()
    p = plan(name, goal, page)
    s = slip(p["steps"], slip_kind, random.Random(f"{name}:{seed}:{slip_kind}"))
    if slip_kind != "none" and s is None:
        raise ValueError(f"{name} seed {seed}: no place for a {slip_kind} slip")
    first = "Hi! I'm sharing my screen, can you keep an eye on it while I fill this in? " + ". ".join(p["say"]) + "."
    return Task(
        id=f"screen-{name}-s{seed}-{slip_kind}",
        scenario={"browsergym": env_id, "goal": goal, "first_turn": first,
                  "persona": "Someone filling in a web form who has asked an assistant to watch their screen and catch mistakes.",
                  "instructions": USER_INSTRUCTIONS.format(what=p["what"]),
                  "hands": {"steps": p["steps"], "slip": s, "review_ms": review_ms, "char_ms": char_ms},
                  "categories": ["screen_share"]},
        initial_state={"seed": seed}, criteria={"reward": "benchmark", "slip": s})


# ---------------------------------------------------------------- the user: one mind, speech and hands

_HANDS: contextvars.ContextVar[dict] = contextvars.ContextVar("screenshare_hands")


def screen_note(hands: dict, actual: bool = False) -> str:
    """The user's screen in words. By default what the user *believes* it shows (what they meant to do, as far as they
    have got): people do not see their own slips. With ``actual``: what it really shows, for when they look to check."""
    lines = []
    for k, s in enumerate(hands["steps"]):
        done = hands["i"] > k  # begun (the user thinks of a step as done once they start it)
        if s["do"] == "type":
            if actual:
                v = hands["typed"].get(s["bid"], "")
            else:
                v = s["value"] if done else ""
            lines.append(f"- {s['field']}: " + ("(dots: a password)" if s.get("secret") and v else repr(v) if v else "empty"))
        elif not s.get("submit"):
            if actual:
                got = hands["clicked"].get(tuple(s.get("options", (s["bid"],))))
                names = s.get("names") or {}
                lines.append("- radio buttons: " + ("none ticked" if got is None else f"{names.get(got, 'one')} is ticked"))
            else:
                lines.append(f"- {s['field']}: " + ("ticked" if done else "not ticked yet"))
    if actual:
        for bid, v in hands["typed"].items():  # typed into a box that is not one of the planned ones
            if v and not any(s["do"] == "type" and s["bid"] == bid for s in hands["steps"]):
                names = next((x.get("names") for x in hands["steps"] if x.get("names") and bid in x["names"]), {})
                lines.append(f"- {names.get(bid, 'another box')}: {v!r}")
    state = ("You have pressed the submit button and the page has taken it." if hands["submitted"] else
             "You have filled everything in and are about to press the submit button." if hands["typed_all"] else
             "You are still filling it in.")
    return ("Your screen actually shows:\n" if actual else "What you have done so far:\n") + "\n".join(lines) + "\n" + state


class _Tap:
    """The user LLM, with every raw reply kept on the episode's hands (the spoken text loses the bracketed tags)."""

    def __init__(self, inner):
        self.inner = inner

    async def chat(self, msgs, **kw):
        raw = await self.inner.chat(msgs, **kw)
        hands = _HANDS.get(None)
        if hands is not None:
            hands["raw"].append(raw or "")
        return raw


class ScreenShareSource(LLMSource):
    """``LLMSource`` that tells the user LLM what it has done so far before every turn. When the agent's last words
    say something on the screen is wrong (``CLAIMS``, one yes/no call), the user looks: the turn is written with the
    page as it really is and the page's verdict (``wrong``), and a real mistake sets the hands to fix it."""

    def __init__(self, llm, *args, **kw):
        super().__init__(_Tap(llm), *args, **kw)

    def messages(self, task, convo, barge_in=False):
        msgs = super().messages(task, convo, barge_in)
        hands = _HANDS.get(None)
        if hands is not None:
            look = hands.get("checking")  # None, or what the user saw when they looked: "mistake" | "none"
            if look:
                verdict = LOOKED["late" if hands["submitted"] and look == "mistake" else look == "mistake"]
                note = CHECK_NOTE.format(screen=screen_note(hands, actual=True), verdict=verdict)
            else:
                note = "(" + screen_note(hands) + ")"
            msgs[-1] = {"role": msgs[-1]["role"], "content": msgs[-1]["content"] + "\n\n" + note}
        return msgs

    async def next(self, i, task, convo, barge_in=False):
        hands = _HANDS.get(None)
        if hands is None:
            return await super().next(i, task, convo, barge_in)
        said = " ".join(u.text for u in convo if u.role == "agent" and u.t0 > max((v.t0 for v in convo if v.role == "user"), default=-1))
        verdict = (await self.llm.inner.chat([{"role": "user", "content": CLAIMS.format(said=said)}])).strip() if said else ""
        if said:
            hands["raw"].append(f"<claim? {verdict!r} for {said!r}>")
        if not verdict.lower().startswith("yes"):
            return await super().next(i, task, convo, barge_in)
        # told that something is wrong: the user looks. What they see is the page itself, so whether there is a mistake
        # comes from the page; the LLM only says it
        mistake = wrong(hands)
        hands["claims"].append({"t": convo[-1].t0 if convo else None, "agent": said, "found": mistake})
        if mistake and not hands["submitted"]:
            hands["fix"] = True
        hands["checking"] = "mistake" if mistake else "none"
        try:
            return await super().next(i, task, convo, barge_in)
        finally:
            hands["checking"] = None


def wrong(hands: dict) -> bool:
    """The page, as it is now, differs from what the user meant (only steps already begun count)."""
    for k, s in enumerate(hands["steps"][: hands["i"]]):
        if s["do"] == "type" and not s.get("secret"):
            if hands["typed"].get(s["bid"], "") != s["value"][: len(hands["typed"].get(s["bid"], ""))] or any(
                    v for bid, v in hands["typed"].items() if bid in s.get("options", ()) and bid != s["bid"]):
                return True
        elif s["do"] == "click" and s.get("options") and hands["clicked"].get(tuple(s["options"])) not in (None, s["bid"]):
            return True
    return False


class ScreenUser(UserSim):
    """A simulated user with hands: says what it wants (``task.scenario["first_turn"]``), then fills in the page
    following ``task.scenario["hands"]`` on ``user.tool_call``, one keystroke at a time, while talking with the agent
    through ``ScreenShareSource``. Pass a ``ScreenShareSource`` as ``source``.

    The hands start ``start_ms`` after the first turn ends, take ``move_ms`` to get to an element, ``char_ms`` per
    character, pause ``review_ms`` before submitting, and only submit once nobody has spoken for ``quiet_ms`` (but
    never wait for an agent that says nothing)."""

    def __init__(self, source: ScreenShareSource, *args, start_ms: int = 800, move_ms: int = 700, quiet_ms: int = 1500,
                 patience_ms: int = 6000, **kw):
        super().__init__(source, *args, **kw)
        self.start_ms, self.move_ms, self.quiet_ms, self.patience_ms = start_ms, move_ms, quiet_ms, patience_ms
        self.reads = self.reads + (RESULT["user"],)

    def init_state(self, task, rng):
        st = super().init_state(task, rng)
        h = task.scenario["hands"]
        steps = [dict(s) for s in h["steps"]]
        s = h.get("slip")
        if s is not None:  # the slip replaces what one step does
            steps[s["step"]]["slipped"] = s["done"]
        st["hands"] = {"steps": steps, "slip": s, "review_ms": h["review_ms"], "char_ms": h["char_ms"], "i": 0, "queue": [],
                       "pending": None, "next_t": None, "n": 0, "typed": {}, "clicked": {}, "typed_all": False, "submitted": False,
                       "fix": False, "fixed": False, "claims": [], "log": [], "raw": [], "due": 0}
        return st

    async def step(self, st, t, inbox):
        h = st["hands"]
        for f in inbox:
            if f.stream == RESULT["user"] and h["pending"] is not None and f.data.id == h["pending"]["id"]:
                self._done(h, f.t, f.data)
        speech = [f for f in inbox if f.stream != RESULT["user"]]
        token = _HANDS.set(h)
        try:
            frames, wake = await super().step(st, t, speech)
        finally:
            _HANDS.reset(token)
        mine, hw = self._hands(st, t)
        if hw is not None:
            wake = hw if wake is None else min(wake, hw)
        return frames + mine, wake

    # ------------------------------------------------------------ hands

    @staticmethod
    def _done(h, t: int, res: ToolResult) -> None:
        a = h["pending"]
        h["pending"] = None
        h["log"].append({"t": t, **{k: v for k, v in a.items() if k != "id"}, "error": res.error})
        if a["name"] == "fill":
            h["typed"][a["bid"]] = a["value"]
        if a.get("submit"):
            h["submitted"] = True
            h["submit_t"] = t

    def _talking(self, st, t: int) -> bool:
        """No submitting while the agent speaks or within ``quiet_ms`` after it (a person stops to listen to the one
        watching), nor while the conversation goes on, but that only for ``patience_ms`` past the moment the user meant
        to submit: then they click while talking."""
        def recent(segs):
            return any(s.end > t - self.quiet_ms for s in segs if s.t0 <= t and s.kind is None)
        if recent(st["agent"].values()):
            return True
        own = recent(st["mine"].values()) or st["planned"] is not None or bool(st["pending"])
        return own and t < st["hands"]["due"] + self.patience_ms

    def _expand(self, h, step: dict) -> list[dict]:
        """The atomic actions of one step: move to the element and click it, then (to type) one key at a time. A
        slipped step clicks or types into the wrong element, or types the wrong value."""
        bid, value = step["bid"], step.get("value")
        if "slipped" in step:
            if step.get("options"):
                bid = step["slipped"]
            else:
                value = step["slipped"]
        out = [{"name": "click", "bid": bid, "gap": self.move_ms, "submit": step.get("submit", False), "options": step.get("options")}]
        if step["do"] == "type":
            out += [{"name": "fill", "bid": bid, "value": value[:k], "gap": h["char_ms"]} for k in range(1, len(value) + 1)]
        return out

    def _fix(self, h) -> None:
        """The user agreed there is a slip: do the step as meant (clearing a wrong box or value first), then go on;
        the submit comes after a fresh, shorter look."""
        h["fixed"] = True
        s = h["slip"]
        k = s["step"]
        step = h["steps"][k]
        started = h["i"] > k
        step.pop("slipped", None)
        if not started:  # pointed out before it happened: it will simply be done right
            return
        undo = []
        if step["do"] == "type" and step.get("options"):  # typed into the wrong box: empty it
            undo += [{"name": "click", "bid": s["done"], "gap": self.move_ms}, {"name": "fill", "bid": s["done"], "value": "", "gap": 300}]
        redo = self._expand(h, step)
        if step["do"] == "type" and not step.get("options"):  # the wrong value: select it all away, then type again
            redo.insert(1, {"name": "fill", "bid": step["bid"], "value": "", "gap": 300})
        rest = [a for a in h["queue"] if not a.get("submit")] if h["i"] > k + 1 else []  # a later step underway goes on after
        if any(a.get("submit") for a in h["queue"]) or (h["i"] == len(h["steps"]) and not h["queue"]):
            h["i"] = len(h["steps"]) - 1  # the submit is not done yet: it comes again, after a look
        h["queue"] = undo + redo + rest
        h["typed_all"] = False

    def _hands(self, st, t: int) -> tuple[list[Frame], int | None]:
        h = st["hands"]
        if h["submitted"] or h["pending"] is not None:
            return [], None
        if h["next_t"] is None:  # not started: after the first turn
            first = min(st["mine"].values(), key=lambda s: s.t0, default=None)
            if first is None or first.end > t:
                return [], first.end if first is not None else None
            h["next_t"] = first.end + self.start_ms
        if h["fix"] and not h["fixed"]:
            self._fix(h)
            h["next_t"] = max(h["next_t"], t + 400)
        if not h["queue"]:
            if h["i"] >= len(h["steps"]):
                return [], None
            step = h["steps"][h["i"]]
            if step.get("submit") and not h["typed_all"]:  # everything is in: a look over it before submitting
                h["typed_all"] = True
                h["next_t"] = max(h["next_t"], t) + (h["review_ms"] if not h["fixed"] else 1200)
                h["due"] = h["next_t"]
            h["queue"] = self._expand(h, step)
            h["i"] += 1
        a = h["queue"][0]
        if t < h["next_t"]:
            return [], h["next_t"]
        if a.get("submit") and self._talking(st, t):  # nobody submits mid-conversation: look again once it is quiet
            return [], t + 200
        h["queue"].pop(0)
        h["n"] += 1
        args = {"bid": a["bid"]} if a["name"] == "click" else {"bid": a["bid"], "value": a["value"]}
        call = ToolCall(f"h{h['n']}", a["name"], args)
        h["pending"] = {"id": call.id, **{k: v for k, v in a.items() if k != "options"}}
        if a["name"] == "click" and a.get("options"):
            h["clicked"][tuple(a["options"])] = a["bid"]
        h["next_t"] = t + (h["queue"][0]["gap"] if h["queue"] else self.move_ms)
        return [Frame(CALL["user"], t, call)], None


def tool_world(**kw):
    """The page: the user's hands act on it, its picture goes to ``video.SCREEN``; the agent has no tools."""
    return bg.tool_world(screen=SCREEN, users=True, agent_tools=False, instructions=AGENT_INSTRUCTIONS, crop=bg.MINIWOB_CROP, **kw)


# ---------------------------------------------------------------- evaluation


VERIFIER = {
    "name": "Screen sharing: the page's reward and the slip's timeline",
    "description": ("The page checks what was submitted (MiniWoB reward). The planted slip's timeline comes from the user's "
                    "own actions: when the slip became visible on the shared screen, when the agent said something was wrong (a "
                    "yes/no LLM judgement of its words), and whether the user, looking, fixed it before submitting."),
    "checks": {
        "benchmark_reward": "the page's reward for what was finally submitted (1 = the intended values)",
        "caught": "the agent said something was wrong after the slip was visible and before the submit, so the user looked and fixed it",
        "false_alarm": "the agent said something was wrong (judged by the user LLM) while the page showed what the user meant",
    },
}


def evaluate(env, task: Task, node: str = "user") -> dict:
    """``{"reward", "success", "slip", "visible_t", "submit_t", "caught", "catch_ms", "false_alarms"}``.
    ``visible_t``: when the slip first showed on the screen; ``catch_ms``: from then to the start of the agent's
    utterance that the user agreed with."""
    st = env.sim.states[node]
    h = st["hands"]
    page = bg.evaluate(env)
    s = h["slip"]
    meant = all(h["typed"].get(x["bid"], "") == x["value"] for x in h["steps"] if x["do"] == "type") and all(
        h["clicked"].get(tuple(x["options"])) == x["bid"] for x in h["steps"] if x["do"] == "click" and x.get("options"))
    stray = any(v for bid, v in h["typed"].items() if not any(x["do"] == "type" and x["bid"] == bid for x in h["steps"]))
    out = {"reward": page["reward"], "success": page["success"] and meant and not stray, "as_meant": meant and not stray,
           "slip": s, "submitted": h["submitted"], "claims": list(h["claims"]), "user_llm": list(h["raw"]),
           "submit_t": h.get("submit_t"), "false_alarms": sum(not c["found"] for c in h["claims"])}
    if s is None:
        out.update(visible_t=None, caught=None, catch_ms=None)
        return out
    step = h["steps"][s["step"]]
    visible = None
    for a in h["log"]:  # the first action whose result differs from what was meant
        if s["kind"] == "wrong_field" and a["bid"] == s["done"] and (a["name"] == "fill" and a["value"] or step["do"] == "click"):
            visible = a["t"]
        elif a["name"] == "fill" and a["bid"] == step["bid"] and not s["intended"].startswith(a["value"]):
            visible = a["t"]
        if visible is not None:
            break
    out["visible_t"] = visible
    agent = sorted((f.data for f in env.log if f.stream == "policy.speech" and isinstance(f.data, Segment)), key=lambda x: x.t0)
    first_submit = min((a["t"] for a in h["log"] if a.get("submit")), default=None)
    said = [a for a in agent if visible is not None and a.t0 >= visible and (first_submit is None or a.t0 < first_submit)]
    out["caught"] = bool(h["fixed"] and said)  # the user only fixes before submitting
    out["catch_ms"] = said[0].t0 - visible if out["caught"] else None
    return out
