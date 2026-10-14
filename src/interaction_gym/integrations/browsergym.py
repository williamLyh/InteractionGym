"""BrowserGym adapter: a real browser as a tool environment (computer use).

BrowserGym (ServiceNow, Apache-2.0; arXiv 2412.05467) wraps a Playwright-driven Chromium as a gym
env over many web benchmarks (MiniWoB++, WebArena, WorkArena, ...). Here it is one ``ToolBackend``:

- **Tools.** BrowserGym's element-level actions, one tool each: ``click(bid)``, ``fill(bid, value)``,
  ``select_option(bid, option)``, ``press(bid, key)``, ``scroll(dx, dy)``, plus ``read_page()``.
  ``bid`` is the element id shown in the page text. Every result is the page *after* the action: the
  accessibility tree flattened to text (and the action's error, if any), so a text-only LLM can drive it.
  Each result also carries a screenshot of the page (``ToolResult.images``): saved with the trajectory and
  shown by the viewer; a text agent ignores it, a vision agent can use it.
- **Task.** The page's own instruction (e.g. MiniWoB's "Click on the 'Submit' button") is what the
  simulated user *says*: ``make_task`` opens the page once with the episode's seed to read it and puts it
  in ``scenario["turns"]``; the agent never gets it as text. ``reset`` checks that the page shows the
  same instruction (the benchmarks seed their pages, so it does).
- **Reward.** The benchmark's own: ``evaluate(env)`` returns the reward of the last action
  (MiniWoB: 1 on success, 0 or negative otherwise) and whether the page ended the episode.

**Seen by others.** With ``users=True`` the simulated user gets the same tools (its hands on the page, e.g. a caller
filling in a form while sharing their screen), and ``render`` gives the page's picture with a mouse pointer where the
last action happened, so ``tool_world(screen=...)`` streams the page as video (``integrations.screenshare``).

The browser runs in real time while the env runs in simulated time: ``latency_ms`` sets how long each
action takes in the episode (default ``action_latency``). Browsers cannot be copied, so ``fork`` raises.

Not a uv dependency: ``pip install browsergym-miniwob``, ``playwright install chromium``, and a
MiniWoB++ checkout served as files: ``export MINIWOB_URL=file:///.../miniwob-plusplus/miniwob/html/miniwob/``.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..core import Task
from ..tools import ToolBackend, ToolCall, ToolResult, ToolSpec, ToolWorld

INSTRUCTIONS = (
    "You can operate a web browser for the caller. Each tool acts on the current page and returns the page "
    "afterwards as an accessibility tree, one element per line with its id in brackets, e.g. [12] button 'Submit'; elements marked clickable can be clicked. "
    "Pass that id as bid. Call read_page first to see the page. Do what the caller asks on the page; tell them "
    "briefly what you are doing. When a result says the page ended the task, it is done: tell the caller in one "
    "sentence and call no more tools."
)

_BID = {"bid": {"type": "string", "description": "the element id, as shown in brackets in the page text"}}
TOOLS = [
    ToolSpec("read_page", "Return the current page as an accessibility tree, without acting."),
    ToolSpec("click", "Click an element.", {"type": "object", "properties": dict(_BID), "required": ["bid"]}),
    ToolSpec("fill", "Clear a text field and type a value into it.",
             {"type": "object", "properties": {**_BID, "value": {"type": "string"}}, "required": ["bid", "value"]}),
    ToolSpec("select_option", "Choose an option of a select / list element by its visible text.",
             {"type": "object", "properties": {**_BID, "option": {"type": "string"}}, "required": ["bid", "option"]}),
    ToolSpec("press", "Focus an element and press a key or key combination, e.g. Enter, Tab, Control+a.",
             {"type": "object", "properties": {**_BID, "key": {"type": "string"}}, "required": ["bid", "key"]}),
    ToolSpec("scroll", "Scroll the page by dx, dy pixels (positive dy scrolls down).",
             {"type": "object", "properties": {"dx": {"type": "number"}, "dy": {"type": "number"}}, "required": ["dx", "dy"]}),
]


def _action(call: ToolCall) -> str | None:
    """The BrowserGym action string for a call (``None`` for ``read_page``)."""
    a, q = call.arguments, json.dumps
    if call.name == "read_page":
        return None
    if call.name == "click":
        return f"click({q(str(a['bid']))})"
    if call.name == "fill":
        return f"fill({q(str(a['bid']))}, {q(str(a['value']))})"
    if call.name == "select_option":
        return f"select_option({q(str(a['bid']))}, {q(str(a['option']))})"
    if call.name == "press":
        return f"press({q(str(a['bid']))}, {q(str(a['key']))})"
    if call.name == "scroll":
        return f"scroll({float(a['dx'])}, {float(a['dy'])})"
    raise KeyError(call.name)


def _goal(obs: dict) -> str:
    return " ".join(m.get("text", "") for m in obs.get("goal_object") or [] if m.get("type") == "text") or obs.get("goal", "")


def _page(obs: dict, visible_only: bool = True, skip_generic: bool = False) -> str:
    from browsergym.utils.obs import flatten_axtree_to_str
    return flatten_axtree_to_str(obs["axtree_object"], extra_properties=obs["extra_element_properties"],
                                filter_visible_only=visible_only, with_clickable=True, skip_generic=skip_generic)


# Playwright's sync API refuses to run inside an event loop, and BrowserGym keeps one Playwright instance per
# process, bound to the thread that started it: every browser of the process is driven from this one thread.
_POOL = ThreadPoolExecutor(1, thread_name_prefix="browsergym")


class _Browser:
    """One BrowserGym env (one page in its own browser context). ``env_id``: a registered gym id, or a function
    ``(**kwargs) -> env`` for a page that is not a registered benchmark (``integrations.openapps``)."""

    def __init__(self, env_id: str | Callable[..., Any], seed: int, **kwargs):
        self.pool = _POOL
        self.env = None
        self.obs, self.info = self.pool.submit(self._open, env_id, seed, kwargs).result()

    def _open(self, env_id, seed, kwargs):
        if callable(env_id):
            self.env = env_id(**kwargs)
        else:
            import gymnasium as gym
            import browsergym.miniwob  # noqa: F401  (registers browsergym/miniwob.*)
            self.env = gym.make(env_id, **kwargs)
        return self.env.reset(seed=seed)

    async def run(self, fn, *args):
        """``fn(*args)`` on the browser's thread (anything that touches the page, or must be ordered with its actions)."""
        return await asyncio.get_running_loop().run_in_executor(self.pool, fn, *args)

    async def step(self, action: str):
        return await asyncio.get_running_loop().run_in_executor(self.pool, self.env.step, action)

    def screenshot(self, obs: dict, pointer: tuple[float, float] | None = None, crop: tuple | None = None, ended: bool = False) -> bytes:
        import io
        from PIL import Image, ImageDraw
        im = Image.fromarray(obs["screenshot"])
        if ended:  # the page has taken it: greyed out under a banner, as a site's confirmation would show (MiniWoB's own
            # page just stays as it was, which reads as "still open")
            im = Image.blend(im.convert("RGB"), Image.new("RGB", im.size, (235, 235, 235)), 0.6)
            d = ImageDraw.Draw(im)
            w = (crop[2] - crop[0]) if crop else im.size[0]
            d.rectangle([0, 60, w, 100], fill=(60, 60, 60))
            d.text((10, 74), "Submitted. This form is closed.", fill="white")
        if pointer is not None:  # an arrow pointer, tip at the point
            x, y = pointer
            arrow = [(0, 0), (0, 17), (4, 13), (7, 20), (10, 19), (7, 12), (12, 12)]
            ImageDraw.Draw(im).polygon([(x + dx, y + dy) for dx, dy in arrow], fill="black", outline="white")
        if crop is not None:
            im = im.crop(crop)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()

    def close(self) -> None:
        if self.env is not None:
            self.pool.submit(self.env.close).result()
            self.env = None


class BrowserGymBackend(ToolBackend):
    """A BrowserGym task as tools. ``task.scenario["browsergym"]`` names the gym id (e.g.
    ``"browsergym/miniwob.click-button"``) and ``task.initial_state["seed"]`` the page seed (see ``make_task``)."""

    name = "browser"
    description = "A web browser showing one page; click, type and read it."
    visible_only = True  # the page text lists only what is in view (a tall page: also what is scrolled away)
    skip_generic = False  # leave out the page's unnamed containers (``generic`` elements)

    def __init__(self, instructions: str = INSTRUCTIONS, headless: bool = True, screenshots: bool = True, users: bool = False,
                 crop: tuple | None = None, agent_tools: bool = True):
        """``users``: the simulated user can act on the page too; ``agent_tools=False``: the agent cannot (it only watches). ``crop``: (left, top, right, bottom) of the
        screenshots and rendered picture, e.g. MiniWoB's page without the empty margin: ``MINIWOB_CROP``."""
        self._instructions, self.headless, self.screenshots, self.users, self.crop = instructions, headless, screenshots, users, crop
        self.agent_tools = agent_tools

    def tools(self, caller="agent"):
        return list(TOOLS) if (caller == "agent" and self.agent_tools) or (caller == "user" and self.users) else []

    def instructions(self, caller="agent"):
        return self._instructions if caller == "agent" else ""

    def reset(self, task, rng):
        b = _Browser(task.scenario["browsergym"], task.initial_state["seed"], headless=self.headless)
        goal = _goal(b.obs)
        want = task.scenario.get("goal")
        if want is not None and goal != want:
            b.close()
            raise RuntimeError(f"the page's instruction changed for this seed: {goal!r} != {want!r}")
        return {"browser": b, "goal": goal, "reward": 0.0, "terminated": False, "truncated": False, "steps": 0, "pointer": None,
                "values": {}, "clicked": [], "by": {}}

    async def call(self, state, caller, call):
        b: _Browser = state["browser"]
        if state["terminated"] or state["truncated"]:
            return ToolResult(call.id, call.name, "Error: the page has closed (the task is over)", error=True)
        try:
            action = _action(call)
        except (KeyError, ValueError, TypeError) as e:
            return ToolResult(call.id, call.name, f"Error: bad arguments: {e}", error=True)
        if action is not None:
            box = (b.obs.get("extra_element_properties") or {}).get(str(call.arguments.get("bid")), {}).get("bbox")
            if box:  # the pointer moves to the element acted on
                state["pointer"] = (box[0] + box[2] / 2, box[1] + box[3] / 2)
            obs, reward, term, trunc, info = await b.step(action)
            b.obs, state["steps"] = obs, state["steps"] + 1
            if not obs.get("last_action_error"):  # what each field holds, as typed, and who typed it, and what was clicked
                # (for evaluators and for people looking at the screen: the page's own record is gone once it ends)
                bid = str(call.arguments.get("bid"))
                if call.name == "fill":
                    state["values"][bid] = str(call.arguments["value"])
                    state["by"][bid] = caller
                elif call.name == "click":
                    state["clicked"].append([bid, caller])
            state["reward"], state["terminated"], state["truncated"] = float(reward), bool(term), bool(trunc)
        obs = b.obs
        err = obs.get("last_action_error") or ""
        if state["terminated"]:
            body = "The page ended the task."
        else:
            body = _page(obs, self.visible_only, self.skip_generic)
        shot = (b.screenshot(obs, crop=self.crop),) if self.screenshots and caller == "agent" else ()  # a user sees the screen itself
        return ToolResult(call.id, call.name, (f"Error: {err}\n\n" if err and action else "") + body, error=bool(err and action), images=shot)

    def render(self, state):
        b: _Browser = state["browser"]
        return b.screenshot(b.obs, state["pointer"], self.crop, ended=state["terminated"])

    def view(self, state):
        return {"values": dict(state["values"]), "typed_by": dict(state["by"]), "clicked": [list(c) for c in state["clicked"]],
                "ended": state["terminated"], "reward": state["reward"]}

    def fork(self, state, n):
        raise NotImplementedError("a live browser cannot be copied; BrowserGym episodes do not fork")

    @staticmethod
    def close(state) -> None:
        state["browser"].close()


def make_task(env_id: str, seed: int = 0, headless: bool = True) -> Task:
    """A task whose user says the page's instruction. Opens the page once (with ``seed``) to read it."""
    b = _Browser(env_id, seed, headless=headless)
    try:
        goal = _goal(b.obs)
    finally:
        b.close()
    say = goal if goal.endswith((".", "?", "!")) else goal + "."
    return Task(id=f"{env_id.split('/')[-1]}-s{seed}", scenario={"browsergym": env_id, "goal": goal, "turns": [f"Hi, could you do this for me? {say}"]},
                initial_state={"seed": seed}, criteria={"reward": "benchmark"})


def action_latency(click_ms: int = 400, type_ms_per_char: int = 60, read_ms: int = 150) -> Callable[[ToolCall], int]:
    """Simulated time per action: reading is quick, typing scales with the text."""
    def latency(call: ToolCall) -> int:
        if call.name.endswith("read_page"):
            return read_ms
        if call.name.endswith("fill"):
            return click_ms + type_ms_per_char * len(str(call.arguments.get("value", "")))
        return click_ms
    return latency


MINIWOB_CROP = (0, 0, 240, 315)  # a MiniWoB page's area in BrowserGym's screenshot (the rest is an empty margin)


def tool_world(latency_ms: int | Callable[[ToolCall], int] | None = None, screen: str | None = None, serial=(), **kwargs) -> ToolWorld:
    """The browser as a ToolWorld. ``screen``: stream the page's picture there (``video.SCREEN``); ``users=True``: the
    simulated user acts on the page too; ``serial``: callers whose calls run one after the other (``ToolWorld``)."""
    callers = ("agent", "user") if kwargs.get("users") else ("agent",)
    return ToolWorld(BrowserGymBackend(**kwargs), action_latency() if latency_ms is None else latency_ms, callers=callers, screen=screen,
                     serial=serial)


def browser_state(env, node: str = "tools") -> dict[str, Any]:
    sim = getattr(env, "sim", env)
    return sim.states[node]["services"][BrowserGymBackend.name]


VERIFIER = {
    "name": "MiniWoB task reward (BrowserGym)",
    "description": ("The web page itself checks the task: a MiniWoB page knows the goal and, when the agent's actions end the task, "
                    "gives a reward (positive when it was done right). Read from the browser after the episode."),
    "checks": {
        "benchmark_reward": "the reward the MiniWoB page gave when the task ended (0 if it never ended)",
        "page_ended_task": "the page reported the task as ended",
    },
}


def evaluate(env, node: str = "tools") -> dict:
    """The benchmark's reward for the episode: ``{"reward", "success", "terminated", "steps"}``."""
    st = browser_state(env, node)
    return {"reward": st["reward"], "success": st["terminated"] and st["reward"] > 0, "terminated": st["terminated"], "steps": st["steps"]}
