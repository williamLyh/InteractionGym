"""Agents that see a video stream (e.g. a shared screen), hear the user's transcript and speak through a TTS:
``ScreenWatchAgent`` only watches; ``ScreenToolAgent`` also operates the page with tools."""

from __future__ import annotations

import base64
import json
import re

from ..core import AgentSpec, Chunk, Frame, Segment
from ..tools import CALL, ToolCall, ToolResult
from ..video import SCREEN, VideoFrame
from .cascaded import speakable

SILENT = "SILENT"
LOOK = ("(Your view of the caller's screen has changed: the image above is what it shows now. If something on it needs the "
        "caller's attention right now, say it in one short sentence. Otherwise reply with exactly " + SILENT + ".)")


def _image(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


class ScreenWatchAgent:
    """Watches ``screen`` and listens. Two kinds of LLM round, each taking ``llm_ms`` of simulated time:

    - **reply**: when a user turn ends, answer it, with the screen as it is now;
    - **look**: when the screen has changed and the agent last looked at least ``look_ms`` ago (so at most
      ``1000 / look_ms`` looks per second, like sampling the video at that rate), decide whether to say something
      (anything but ``SILENT``).

    Only the newest picture goes to the model; earlier ones are dropped from the history (the words stay). No round
    starts while the user or the agent is talking; the agent stops when the user starts talking."""

    def __init__(self, spec: AgentSpec, llm, llm_ms: int = 800, look_ms: int = 1000, tts=None, voice: str = "aiden",
                 language: str = "English", screen: str = SCREEN, wps: float = 2.8):
        self.spec, self.llm, self.llm_ms, self.look_ms, self.tts, self.voice, self.language = spec, llm, llm_ms, look_ms, tts, voice, language
        self.screen, self.wps = screen, wps
        self.system, self.history, self.heard, self.inbox = "", [], {}, []
        self.frame: VideoFrame | None = None
        self.frame_t, self.looked_t, self.last_look, self.n, self.looks = -1, -1, -10**9, 0, 0
        self.speaking: Segment | None = None
        self.trace: list[dict] = []  # every round: when, kind, what it said (SILENT included)

    def describe(self) -> dict:
        return {"name": "vision LLM watching the screen" + (" (TTS out)" if self.tts else " (text)"), "kind": "cascaded" if self.tts else "text",
                "model": getattr(self.llm, "model", None), "hears": "the user's transcript (no ASR)", "sees": f"{self.screen} (newest frame, "
                f"at most one look per {self.look_ms} ms)", "llm_ms": self.llm_ms, "tts": getattr(self.tts, "model", None),
                "voice": self.voice if self.tts else None}

    async def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        out = []
        for f in obs:
            if f.stream == "session":
                self.system = f.data.instructions
            elif f.stream == self.screen and isinstance(f.data, VideoFrame):
                self.frame, self.frame_t = f.data, f.t
            elif isinstance(f.data, Chunk):
                if self.speaking is not None and t < self.speaking.end:  # the user is talking: stop, or never start
                    at = max(t, self.speaking.t0)
                    out.append(Frame(self.spec.out, at, self.speaking.cut(at)))
                    self.speaking = None
                self.heard[f.data.id] = self.heard.get(f.data.id, "") + f.data.text
                if f.data.last and self.heard[f.data.id].strip():
                    self.inbox.append(self.heard.pop(f.data.id).strip())
        if self.heard or (self.speaking is not None and t < self.speaking.end):
            return out
        if self.inbox:
            words = " ".join(self.inbox)
            self.inbox = []
            self.history.append({"role": "user", "content": words})
            text = await self._round(t, "reply", None)
        elif self.frame is not None and self.frame_t > self.looked_t and t - self.last_look >= self.look_ms:
            text = await self._round(t, "look", LOOK)
            self.looks += 1
        else:
            return out
        if text:
            out.append(await self._say(t + self.llm_ms, text))
        return out

    async def _round(self, t: int, kind: str, note: str | None) -> str:
        msgs = [{"role": "system", "content": self.system}] + self.history
        if self.frame is not None:  # the newest picture, with the last user message (or a look's own note)
            parts = [_image(self.frame.png)]
            if note is None:
                last = msgs.pop()
                parts.append({"type": "text", "text": last["content"]})
                msgs.append({"role": "user", "content": parts})
            else:
                parts.append({"type": "text", "text": note})
                msgs.append({"role": "user", "content": parts})
        elif note is not None:
            msgs.append({"role": "user", "content": note})
        self.looked_t, self.last_look = self.frame_t, t
        raw = await self.llm.chat(msgs)
        text = speakable(raw or "")
        silent = kind == "look" and (not text or re.fullmatch(rf"\W*{SILENT}\W*", text, re.I) or SILENT in text.upper())
        self.trace.append({"t": t, "kind": kind, "frame_t": self.frame_t, "said": None if silent else text})
        if silent:
            return ""
        if kind == "look":  # a remark of its own: kept in the history as the agent's line
            self.history.append({"role": "user", "content": "(You looked at the caller's screen.)"})
        self.history.append({"role": "assistant", "content": text})
        return text

    async def _say(self, t0: int, text: str) -> Frame:
        self.n += 1
        if self.tts is not None:
            audio = await self.tts.synth(text, self.voice, None, language=self.language)
            seg = Segment(f"a{self.n}", t0, audio.dur_ms, audio, text=text)
        else:
            seg = Segment(f"a{self.n}", t0, max(400, int(len(text.split()) / self.wps * 1000)), text)
        self.speaking = seg
        return Frame(self.spec.out, t0, seg)


GLANCE = ("(Your screen has changed and you did not do it: the image above is what it shows now. If you need to act or "
          "say something now, do; otherwise reply with exactly " + SILENT + ".)")
NOW = "(The image above is your screen now.)"


class ScreenToolAgent:
    """A tool-using vision LLM on a shared screen: it operates the page with the session's tools, sees the page as
    the video stream ``screen`` (the newest frame goes with every LLM call; earlier pictures are dropped from the
    history), hears the user's transcript and speaks through a TTS. The other side may act on the same page.

    Rounds, each taking ``llm_ms`` of simulated time: at the end of a user turn; when all tool results are back; and
    a **glance** when the screen changed without the agent's doing (the other side acted) and it last glanced at least
    ``look_ms`` ago (reply ``SILENT`` to do nothing). A round's tool calls go out when its words start. While the user
    is talking, the agent starts no new round, and stops (or never starts) speaking. ``keep_results``: only the newest
    so many tool results are sent in full (older pages are dropped from the history, like older pictures)."""

    def __init__(self, spec: AgentSpec, llm, llm_ms: int = 800, look_ms: int = 1000, max_rounds: int = 12, tts=None,
                 voice: str = "aiden", language: str = "English", screen: str = SCREEN, wps: float = 2.8, keep_results: int | None = None):
        self.spec, self.llm, self.llm_ms, self.look_ms, self.max_rounds = spec, llm, llm_ms, look_ms, max_rounds
        self.keep_results = keep_results
        self.tts, self.voice, self.language, self.screen, self.wps = tts, voice, language, screen, wps
        self.msgs, self.tools, self.pending, self.n, self.rounds = [], [], set(), 0, 0
        self.heard: dict[str, str] = {}
        self.inbox: list[str] = []
        self.speaking: Segment | None = None
        self.frame: VideoFrame | None = None
        self.frame_t, self.seen_t, self.glanced = -1, -1, -10**9
        self.own_t = -1  # the newest frame that came with one of its own results
        self.trace: list[dict] = []

    def describe(self) -> dict:
        return {"name": "vision LLM + tools on a shared screen" + (" (TTS out)" if self.tts else " (text)"), "kind": "cascaded" if self.tts else "text",
                "model": getattr(self.llm, "model", None), "hears": "the user's transcript (no ASR)",
                "sees": f"{self.screen} (newest frame with each call; a glance when the other side changes it, at most one per {self.look_ms} ms)",
                "acts": "the session's tools", "llm_ms": self.llm_ms, "tts": getattr(self.tts, "model", None), "voice": self.voice if self.tts else None}

    async def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        out, results = [], False
        for f in obs:
            if f.stream == "session":
                self.tools = [s.openai() for s in f.data.tools]
                self.msgs = [{"role": "system", "content": f.data.instructions}]
            elif f.stream == self.screen and isinstance(f.data, VideoFrame):
                self.frame, self.frame_t = f.data, f.t
                if f.data.note.startswith("agent:"):
                    self.own_t = f.t
            elif isinstance(f.data, Chunk):
                if self.speaking is not None and t < self.speaking.end:  # the user is talking: stop, or never start
                    at = max(t, self.speaking.t0)
                    out.append(Frame(self.spec.out, at, self.speaking.cut(at)))
                    self.speaking = None
                self.heard[f.data.id] = self.heard.get(f.data.id, "") + f.data.text
                if f.data.last and self.heard[f.data.id].strip():
                    self.inbox.append(self.heard.pop(f.data.id).strip())
            elif isinstance(f.data, ToolResult) and f.data.id in self.pending:
                self.msgs.append({"role": "tool", "tool_call_id": f.data.id, "content": f.data.content})
                self.pending.discard(f.data.id)
                results = True
        if self.pending or self.heard:  # finish the tool round, and let the user finish talking, first
            return out
        kind = None
        if self.inbox:
            self.msgs += [{"role": "user", "content": u} for u in self.inbox]
            self.inbox, self.rounds, kind = [], 0, "reply"
        elif results and self.rounds < self.max_rounds:
            kind = "results"
        elif (self.frame is not None and self.frame_t > self.seen_t and self.frame_t > self.own_t and t - self.glanced >= self.look_ms
              and not (self.speaking is not None and t < self.speaking.end)):
            kind = "glance"
            self.glanced = t
        if kind is None:
            return out
        self.rounds += 1
        r = await self.llm.complete(self._with_screen(kind), self.tools)
        self.seen_t = self.frame_t
        text = speakable(r["content"] or "")
        if kind == "glance" and (re.search(rf"\b{SILENT}\b", text, re.I) or not text) and not r["tool_calls"]:
            self.trace.append({"t": t, "kind": kind, "frame_t": self.frame_t, "said": None, "calls": []})
            return out
        if kind == "glance":
            text = re.sub(rf"\b{SILENT}\b", "", text, flags=re.I).strip()
            self.msgs.append({"role": "user", "content": "(Your screen changed: the caller did something on the page.)"})
        self.trace.append({"t": t, "kind": kind, "frame_t": self.frame_t, "said": text or None, "calls": [c["name"] for c in r["tool_calls"]]})
        t_out = t + self.llm_ms
        if text:  # say it, then act: a round's calls go out when its words start (after any speech still playing)
            t_out = max(t_out, self.speaking.end if self.speaking is not None else 0)
        msg = {"role": "assistant", "content": r["content"] or ""}
        if r["tool_calls"]:
            msg["tool_calls"] = []
            for c in r["tool_calls"]:
                self.n += 1
                call = ToolCall(f"c{self.n}", c["name"], c["arguments"] if isinstance(c["arguments"], dict) else {})
                msg["tool_calls"].append({"id": call.id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}})
                self.pending.add(call.id)
                out.append(Frame(CALL["agent"], t_out, call))
        self.msgs.append(msg)
        if text:
            self.n += 1
            if self.tts is not None:
                audio = await self.tts.synth(text, self.voice, None, language=self.language)
                seg = Segment(f"a{self.n}", t_out, audio.dur_ms, audio, text=text)
            else:
                seg = Segment(f"a{self.n}", t_out, max(400, int(len(text.split()) / self.wps * 1000)), text)
            self.speaking = seg
            out.append(Frame(self.spec.out, t_out, seg))
        return out

    def _with_screen(self, kind: str) -> list[dict]:
        """The history for this call, with the newest picture of the screen (and, for a glance, its question) last."""
        msgs = list(self.msgs)
        if self.keep_results is not None:
            old = [k for k, m in enumerate(msgs) if m["role"] == "tool"][: -self.keep_results or None]
            for k in old:
                msgs[k] = {**msgs[k], "content": "(an earlier page; dropped: see the newest result)"}
        if self.frame is None:
            return msgs
        note = GLANCE if kind == "glance" else NOW
        parts = [_image(self.frame.png), {"type": "text", "text": note}]
        if msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], str):  # the user's words, with the picture
            parts = [_image(self.frame.png), {"type": "text", "text": msgs[-1]["content"] + "\n\n" + note}]
            msgs[-1] = {"role": "user", "content": parts}
        else:
            msgs.append({"role": "user", "content": parts})
        return msgs
