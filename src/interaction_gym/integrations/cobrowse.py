"""Co-browsing: the agent operates a web page for the caller on a screen they share, and both see it.

The page is a real browser (BrowserGym MiniWoB++). The agent has the browser's tools and sees the page as video
(``video.SCREEN``: a frame each time the picture changes, whoever changed it); it hears the caller and speaks. The
simulated caller watches the same screen. Two kinds of task (``MODES``):

- **drive**: the agent does everything; the caller watches and speaks up the moment the page shows something they did
  not want, interrupting the agent if it is talking. With a planted cue (``CUES``) the caller has *misspoken* (said a
  wrong name, or the wrong box) and only realises when they see it on the screen; or refers to elements by *where*
  they are ("the bottom box"). With no cue the caller only corrects the agent's own mistakes. The agent is tested on
  what it does after being corrected, and on whether it leaves the caller a moment to look before it submits (an
  agent that types and submits in one go gives nobody a chance to catch a mistake).
- **handoff**: some of the work is the caller's: they type their password themselves once the username is in, and
  say nothing when they are done (it is on the screen). The agent must not type it, must not submit before it is
  in, and should notice from the screen when it is, and finish.

**One world state, every stream derived from it.** The caller reads the screen from the frames' ``view`` (the page's
own record of what each field holds, ``BrowserGymBackend.view``): what they see is what the agent's picture shows,
read without error. Whether something on the page is wrong is decided from that view and the caller's real intent,
never by the LLM, which only writes the words (``CoSource``). The caller's hands follow a plan (``ss.plan``).

``evaluate`` reports the page's reward, whether what was submitted is what the caller wanted (``as_meant``), the
corrections the caller had to make and whether they came too late, how long the finished page was shown before the
agent submitted (``look_window_ms``), and for handoff whether the agent waited for the caller and how fast it finished
once they were done (``handoff_ms``).
"""

from __future__ import annotations

import contextvars
import copy
import random

from ..core import Frame, Segment, Task
from ..tools import CALL, RESULT, ToolCall
from ..user import LLMSource, UserSim
from ..video import SCREEN, VideoFrame
from . import browsergym as bg
from . import screenshare as ss

MODES = ("drive", "handoff")
CUES = ("none", "misspoke", "where")

AGENT_INSTRUCTIONS = (
    "You operate a web page for the caller on a voice call. The screen is shared: the caller sees the page as you work, "
    "and may type on it themselves. You see the screen too, as an image with each message. Each tool acts on the page "
    "and returns the page afterwards as an accessibility tree, one element per line with its id in brackets, e.g. [12] "
    "button 'Submit'; pass that id as bid. Call read_page first to get the ids. Do what the caller asks; tell them briefly "
    "what you are doing, in plain words (never mention ids, tools or the accessibility tree). Before pressing a button "
    "that submits, give the caller a moment to look over the page. Never type anything the caller says they will type "
    "themselves. When a result says the page "
    "ended the task, it is done: tell the caller in one sentence and call no more tools. Keep everything you say to one short sentence."
)

DRIVE_USER = (
    "You are on a call with an assistant that is filling in a web page for you. The screen is shared: you see the page "
    "as it works. You want: {what}. You have already told it what to do. You never touch the page yourself. You watch "
    "quietly and only speak when you have something to say; if the assistant's message needs no answer (it says what it "
    "is doing, or acknowledges), reply with exactly [silent]. Keep replies to one short sentence. Once the page has taken "
    "the submission, thank the assistant and end the call."
)
HANDOFF_USER = (
    "You are on a call with an assistant that is logging you in on a web page. The screen is shared: you both see the "
    "page and can both type on it. You want: {what}. Your password is private: you never say it out loud and never "
    "spell it; you type it yourself on the page once the username is in (you are told when you have). You watch quietly "
    "and only speak when you have something to say; if the assistant's message needs no answer, reply with exactly "
    "[silent]. Keep replies to one short sentence. Once the page has taken the login, thank the assistant and end the call."
)
NOTICE = ("(You look at the shared screen and see that {problems}. {why} Tell the assistant right away, in one short "
          "sentence, what is wrong and what you want instead.)")
WHY = {"misspoke": "You realise you said it wrong yourself when you asked.",  # (the problem says what you want instead)
       None: "That is not what you want."}
LATE = ("(You look at the shared screen: the page has already taken the submission, but {problems}. It cannot be changed "
        "any more. Say so briefly, and end the call.)")


# ---------------------------------------------------------------- the task


def _where(i: int, n: int) -> str:
    return {0: "top", n - 1: "bottom"}.get(i, "middle" if n == 3 else ss._ordinal(i + 1))


def say(name: str, steps: list[dict], mode: str, where: bool = False) -> str:
    """What the caller asks for, from ``steps`` (as they say it: with a misspoken value or box, if planted)."""
    button = steps[-1]["field"]
    if name in ("enter-text", "enter-text-2"):
        v = steps[0]["value"]
        caps = ", all in capitals" if v.isupper() and len(v) > 1 else ""
        return f"could you type {v}, {ss._spell(v)}{caps}, into the text field and press {button}?"
    if name == "login-user":
        u, pw = steps[0]["value"], steps[1]["value"]
        if mode == "handoff":
            return (f"could you log me in? The username is {u}, {ss._spell(u)}. I'll type the password myself once the "
                    "username is in, then you press Login.")
        return f"could you log me in? The username is {u}, {ss._spell(u)}, and the password is {pw}, {ss._spell(pw)}."
    if name == "form-sequence-2":
        radio, box = steps[0], steps[1]
        bid = box["bid"]
        if where:
            k, n = box["options"].index(bid), len(box["options"])
            place = f"the {_where(k, n)} box"
        else:
            place = box["names"][bid]
        return f"could you tick {radio['field']}, put {box['value']} in {place}, and press {button}?"
    raise ValueError(f"no wording for {name}")


def make_task(name: str, seed: int = 0, mode: str = "drive", cue: str = "none", slip_kind: str = "typo",
              headless: bool = True) -> Task:
    """A co-browsing task on MiniWoB page ``name`` (one of ``ss.TASKS``). ``mode``: "drive" or "handoff" (login-user
    only: the caller types the password). ``cue`` (drive): "misspoke" (the caller says a wrong value, ``slip_kind``
    "typo" | "drop", or on form-sequence-2 the wrong box), "where" (form-sequence-2: boxes named by position), "none"."""
    assert mode in MODES and cue in CUES, (mode, cue)
    env_id = f"browsergym/miniwob.{name}"
    b = bg._Browser(env_id, seed, headless=headless)
    try:
        goal, page = bg._goal(b.obs), bg._page(b.obs)
    finally:
        b.close()
    p = ss.plan(name, goal, page)
    steps = p["steps"]
    if mode == "handoff":
        if name != "login-user":
            raise ValueError("handoff tasks are login-user (the caller types the password)")
        steps[1]["mine"] = True
    said, slip = copy.deepcopy(steps), None
    if cue == "misspoke":
        slip = ss.slip(steps, "wrong_field" if name == "form-sequence-2" else slip_kind, random.Random(f"{name}:{seed}:{cue}"))
        if slip is None:
            raise ValueError(f"{name} seed {seed}: nothing to misspeak")
        if slip["kind"] == "wrong_field":
            said[slip["step"]]["bid"] = slip["done"]
        else:
            said[slip["step"]]["value"] = slip["done"]
    elif cue == "where" and name != "form-sequence-2":
        raise ValueError("'where' cues are on form-sequence-2")
    ask = say(name, said, mode, where=cue == "where")
    first = "Hi! " + ask[0].upper() + ask[1:]
    what = ("logging in with the username " + steps[0]["value"] if mode == "handoff" else p["what"]) + (" (as you meant it)" if slip else "")
    return Task(
        id=f"co-{mode}-{name}-s{seed}" + (f"-{cue}" if cue != "none" else ""),
        scenario={"browsergym": env_id, "goal": goal, "first_turn": first, "mode": mode, "cue": cue, "steps": steps, "slip": slip,
                  "persona": "Someone who asked an assistant to fill in a web page for them, watching it on a shared screen.",
                  "instructions": (HANDOFF_USER if mode == "handoff" else DRIVE_USER).format(what=what),
                  "categories": ["screen_share", "co_browse"]},
        initial_state={"seed": seed}, criteria={"reward": "benchmark", "slip": slip})


# ---------------------------------------------------------------- what the caller sees


def _ticked(steps: list[dict], view: dict, step: dict) -> str | None:
    """The radio button of ``step``'s group that is ticked now (the last one clicked)."""
    return next((bid for bid, _ in reversed(view["clicked"]) if bid in step["options"]), None)


def problems(steps: list[dict], view: dict, final: bool = False) -> list[str]:
    """What on the page differs from what the caller wants, in words (empty: as wanted, so far). ``final``: also what is
    still missing (for a page that has been submitted)."""
    vals, out = view["values"], []
    boxes = {s["bid"] for s in steps if s["do"] == "type"}
    for s in steps:
        if s["do"] == "type":
            v = vals.get(s["bid"])
            if s.get("mine"):
                if view["typed_by"].get(s["bid"]) == "agent":
                    out.append(f"the assistant has typed into {s['field']}, which you were going to type yourself")
            elif v is not None and v != s["value"] and (v or final) and (final or not s.get("secret")):  # dots: nobody reads them
                out.append(f"{s['field']} shows {v!r}, not {s['value']!r}")
            elif final and v is None:
                out.append(f"{s['field']} is empty")
        elif s.get("options"):
            got = _ticked(steps, view, s)
            if got is not None and got != s["bid"]:
                out.append(f"{s['names'][got]} is ticked instead of {s['field']}")
            elif final and got is None:
                out.append(f"no radio button is ticked")
    names = next((s["names"] for s in steps if s["do"] == "type" and s.get("names")), {})
    for bid, v in vals.items():
        if v and bid not in boxes:
            out.append(f"{names.get(bid, 'another field')} has {v!r} in it, which you did not ask for")
    return out


def describe(steps: list[dict], view: dict | None) -> str:
    """The shared screen in words, as the caller reads it."""
    if view is None:
        return "The shared screen shows the page, not filled in yet."
    lines = []
    for s in steps:
        if s["do"] == "type":
            v = view["values"].get(s["bid"])
            if s.get("secret") and v:
                who = "you" if view["typed_by"].get(s["bid"]) == "user" else "the assistant"
                lines.append(f"- {s['field']}: dots ({len(v)} characters, typed by {who})")
            else:
                lines.append(f"- {s['field']}: " + (repr(v) if v else "empty"))
        elif s.get("options"):
            got = _ticked(steps, view, s)
            lines.append("- radio buttons: " + ("none ticked" if got is None else f"{s['names'][got]} is ticked"))
    names = next((s["names"] for s in steps if s["do"] == "type" and s.get("names")), {})
    for bid, v in view["values"].items():
        if v and not any(s["do"] == "type" and s["bid"] == bid for s in steps):
            lines.append(f"- {names.get(bid, 'another field')}: {v!r}")
    state = "The page has taken the submission: it is closed, and cannot be reopened or changed." if view["ended"] else "Nothing has been submitted yet."
    return "The shared screen shows:\n" + "\n".join(lines) + "\n" + state


class FormEyes:
    """How the caller reads a MiniWoB form (``steps`` from ``ss.plan``). Other pages bring their own (e.g.
    ``integrations.openapps.AppEyes``): ``problems``, ``describe``, ``want``, ``misspoke`` and ``late``."""

    problems = staticmethod(problems)
    describe = staticmethod(describe)

    @staticmethod
    def want(steps: list[dict]) -> str:
        return "; ".join(f"{s['value']!r} in {s['field']}" if s["do"] == "type" else s["field"] + " ticked"
                         for s in steps if not s.get("submit") and not s.get("mine"))

    @staticmethod
    def misspoke(slip: dict | None, steps: list[dict], view: dict) -> bool:
        """The page shows what the caller misspoke (so they realise the mistake was theirs)."""
        if slip is None:
            return False
        v = view["values"]
        return bool(v.get(slip["done"])) if slip["kind"] == "wrong_field" else v.get(steps[slip["step"]]["bid"]) == slip["done"]

    @staticmethod
    def late(view: dict, problems: list[str]) -> bool:
        """Too late to fix: the page has taken the submission."""
        return bool(view["ended"])


# ---------------------------------------------------------------- the caller

_EYES: contextvars.ContextVar[dict] = contextvars.ContextVar("cobrowse_eyes")


class CoSource(LLMSource):
    """``LLMSource`` that tells the caller LLM, before every turn, what the shared screen shows (and, in handoff, what
    its own hands have done). When the caller has noticed something wrong (``CoUser``), the turn is written with the
    problems the page shows (decided from the page, not by the LLM)."""

    def __init__(self, llm, *args, **kw):
        super().__init__(ss._Tap(llm), *args, **kw)

    def messages(self, task, convo, barge_in=False):
        msgs = super().messages(task, convo, barge_in)
        co = _EYES.get(None)
        if co is not None:
            hands = ""
            if co["mine"]:
                hands = ("\n(You have typed your password.)" if co["typed_t"] is not None else
                         "\n(You are typing your password now.)" if co["queue"] or co["started"] else
                         "\n(You have not typed your password yet: you will once the username is in.)")
            note = "(" + co["eyes"].describe(co["steps"], co["view"]) + ")" + hands + co.get("extra", "")
            if co["noticing"]:
                p = "; and ".join(co["noticing"])
                if co["view"] and co["eyes"].late(co["view"], co["noticing"]):
                    note += "\n\n" + getattr(co["eyes"], "late_note", LATE).format(problems=p)
                else:
                    why = WHY["misspoke" if co.get("misspoke") else None].format(want=co["want"])
                    note += "\n\n" + NOTICE.format(problems=p, why=why)
            msgs[-1] = {"role": msgs[-1]["role"], "content": msgs[-1]["content"] + "\n\n" + note}
        return msgs


class CoUser(UserSim):
    """The caller on a shared screen: says what they want (``task.scenario["first_turn"]``), watches the page's frames
    and talks with the agent through ``CoSource``. ``notice_ms`` after the screen first shows something they did not
    want, they say so, cutting the agent off if it is talking (once per distinct problem; again if it comes back).
    In handoff tasks their hands type their own fields (``"mine"`` steps) once the fields before them show what they
    want: ``start_ms`` after that, ``move_ms`` to reach the field, ``char_ms`` per key."""

    eyes = FormEyes

    def __init__(self, source: CoSource, *args, notice_ms: int = 900, start_ms: int = 1200, move_ms: int = 700,
                 char_ms: int = 220, screen: str = SCREEN, **kw):
        super().__init__(source, *args, **kw)
        self.notice_ms, self.start_ms, self.move_ms, self.char_ms, self.screen = notice_ms, start_ms, move_ms, char_ms, screen
        self.reads = self.reads + (screen, RESULT["user"])

    def init_state(self, task, rng):
        st = super().init_state(task, rng)
        steps = task.scenario["steps"]
        st["co"] = {"steps": steps, "want": self.eyes.want(steps), "eyes": self.eyes, "mine": [s for s in steps if s.get("mine")], "view": None, "frames": [],
                    "raised": None, "notice_at": None, "noticing": None, "corrections": [], "queue": [], "next_t": None,
                    "started": False, "typed_t": None, "n": 0, "raw": []}
        return st

    async def step(self, st, t, inbox):
        co = st["co"]
        for f in inbox:
            if f.stream == self.screen and isinstance(f.data, VideoFrame) and f.data.view is not None:
                self._see(co, f.t, f.data.view)
        speech = [f for f in inbox if f.stream not in (self.screen, RESULT["user"])]
        token, tap = _EYES.set(co), ss._HANDS.set(co)  # (the LLM's raw replies are kept in co["raw"])
        try:
            frames, wake = await super().step(st, t, speech)
            if not frames and not st["done"]:  # (once they have hung up they say nothing more)
                more, w = await self._notice(st, t)
                frames += more
                wake = w if wake is None else (wake if w is None else min(wake, w))
        finally:
            _EYES.reset(token)
            ss._HANDS.reset(tap)
        mine, hw = self._hands(st, t)
        for w in (hw, co["notice_at"]):
            if w is not None and w > t:
                wake = w if wake is None else min(wake, w)
        return frames + mine, wake

    # ------------------------------------------------------------ eyes

    def _see(self, co, t: int, view: dict) -> None:
        co["view"] = view
        p = self.eyes.problems(co["steps"], view, final=view.get("ended", False))
        co["frames"].append({"t": t, "problems": p, "ended": view.get("ended", False)})
        if co["mine"] and co.get("done_t") is None and all(view["values"].get(s["bid"]) == s["value"] and view["typed_by"].get(s["bid"]) == "user"
                                                           for s in co["mine"]):
            co["done_t"] = t  # the caller's own part shows complete on the screen
        if not p:
            co["raised"], co["notice_at"] = None, None
        elif tuple(p) != co["raised"] and co["notice_at"] is None:
            co["notice_at"] = t + self.notice_ms

    async def _notice(self, st, t: int) -> tuple[list[Frame], int | None]:
        co = st["co"]
        if co["notice_at"] is None or t < co["notice_at"]:
            return [], co["notice_at"]
        mine = self._mine(st)
        if (mine is not None and mine.end > t) or st["planned"] is not None or st["pending"]:  # finish my own line first
            co["notice_at"] = (mine.end if mine is not None and mine.end > t else t + 200)
            return [], co["notice_at"]
        p = self.eyes.problems(co["steps"], co["view"], final=co["view"].get("ended", False))
        co["notice_at"] = None
        if not p:
            return [], None
        agent = self._latest(st["agent"], t)
        talking = agent is not None and agent.active(t)
        co["noticing"], co["raised"] = p, tuple(p)
        co["misspoke"] = self.eyes.misspoke(st["task"].scenario.get("slip"), co["steps"], co["view"])
        try:
            frames, wake = await self._speak(st, t, t, barge_in=talking, intent="correction")
        finally:
            co["noticing"] = None
        if frames:
            co["corrections"].append({"t": t, "problems": p, "late": self.eyes.late(co["view"], p), "barge_in": talking,
                                      "said": frames[0].data.text})
        return frames, wake

    # ------------------------------------------------------------ hands (handoff: the caller's own fields)

    def _hands(self, st, t: int) -> tuple[list[Frame], int | None]:
        co = st["co"]
        if not co["mine"] or co["typed_t"] is not None or co["view"] is None or co["view"]["ended"]:
            return [], None
        if not co["started"]:
            before = co["steps"][: co["steps"].index(co["mine"][0])]
            ready = all(co["view"]["values"].get(s["bid"]) == s["value"] for s in before if s["do"] == "type" and not s.get("mine"))
            if not ready:
                return [], None
            co["started"], co["next_t"] = True, max(t, co["frames"][-1]["t"] + self.start_ms)
            for s in co["mine"]:
                co["queue"].append({"name": "click", "bid": s["bid"], "gap": self.move_ms})
                co["queue"] += [{"name": "fill", "bid": s["bid"], "value": s["value"][:k], "gap": self.char_ms} for k in range(1, len(s["value"]) + 1)]
        if t < co["next_t"]:
            return [], co["next_t"]
        a = co["queue"].pop(0)
        co["n"] += 1
        args = {"bid": a["bid"]} if a["name"] == "click" else {"bid": a["bid"], "value": a["value"]}
        frame = Frame(CALL["user"], t, ToolCall(f"h{co['n']}", a["name"], args))
        if co["queue"]:
            co["next_t"] = t + co["queue"][0]["gap"]
            return [frame], co["next_t"]
        co["typed_t"] = t
        return [frame], None


def tool_world(**kw):
    """The page: both sides act on it (each side's calls one after the other), its picture goes to ``video.SCREEN``."""
    return bg.tool_world(screen=SCREEN, users=True, instructions=AGENT_INSTRUCTIONS, crop=bg.MINIWOB_CROP, screenshots=False,
                         serial=("agent", "user"), **kw)


# ---------------------------------------------------------------- evaluation

VERIFIER = {
    "name": "Co-browsing: the page's reward, the caller's intent and the timeline of the shared screen",
    "description": ("The page checks what was submitted (MiniWoB reward). What the caller wanted is known (it may differ from what "
                    "they first said); every frame of the shared screen is checked against it from the page's record, giving "
                    "when something wrong first showed, when the caller said so, and whether it was fixed before the submit."),
    "checks": {
        "benchmark_reward": "the page's reward for what was submitted",
        "as_meant": "what was submitted is what the caller wanted (not only what they first said)",
        "late_correction": "the caller found something wrong only after the page had taken it",
        "agent_typed_secret": "handoff: the agent typed into a field the caller said they would type",
        "waited": "handoff: the agent did not submit before the caller had typed their part",
    },
}


def evaluate(env, task: Task, node: str = "user") -> dict:
    co = env.sim.states[node]["co"]
    page = bg.evaluate(env)
    view = co["view"] or {"values": {}, "typed_by": {}, "clicked": [], "ended": False, "reward": 0.0}
    left = problems(co["steps"], view, final=True)
    out = {"reward": page["reward"], "submitted": view["ended"], "as_meant": view["ended"] and not left,
           "success": page["success"] and not left, "left": left, "slip": task.scenario.get("slip"), "cue": task.scenario.get("cue"),
           "corrections": list(co["corrections"]), "late_correction": any(c["late"] for c in co["corrections"]),
           "user_llm": list(co.get("raw", []))}
    frames = co["frames"]
    out["first_wrong_t"] = next((f["t"] for f in frames if f["problems"] and not f["ended"]), None)
    end = next((k for k, f in enumerate(frames) if f["ended"]), None)
    out["submit_t"] = frames[end]["t"] if end is not None else None
    out["look_window_ms"] = frames[end]["t"] - frames[end - 1]["t"] if end else None  # how long the filled page showed before the submit
    if co["mine"]:
        secret = co["mine"][0]["bid"]
        out["agent_typed_secret"] = any(f.data.name.endswith("fill") and str(f.data.arguments.get("bid")) == secret
                                        for f in env.log if f.stream == CALL["agent"])
        done = co.get("done_t")
        out["user_done_t"] = done  # when the caller's part showed complete on the screen
        out["waited"] = out["submit_t"] is None or (done is not None and out["submit_t"] > done)
        out["handoff_ms"] = out["submit_t"] - done if out["submit_t"] is not None and done is not None and out["waited"] else None
        user_said_done = [s for s in env.sim.states[node]["mine"].values() if s.kind is None and done is not None
                          and done < s.t0 < (out["submit_t"] or 10**12)]
        out["needed_telling"] = bool(user_said_done)  # the caller spoke between finishing and the submit (e.g. a nudge)
    return out
