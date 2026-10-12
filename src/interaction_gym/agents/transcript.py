"""``TranscriptToolAgent``: an LLM with tools that hears the user's transcript and speaks through a TTS."""

from __future__ import annotations

import json

from ..core import AgentSpec, Chunk, Frame, Segment
from ..tools import CALL, ToolCall, ToolResult
from .cascaded import ToolChat, speakable


class TranscriptToolAgent:
    """A tool-using LLM agent that hears the user's transcript (no ASR: the text of the user's speech chunks) and
    speaks through a TTS, for tool environments where speech recognition is not what is being studied.

    At each end of a user turn, and whenever all tool results are back: one LLM round with the session's
    instructions and tools. Tool calls go out after ``llm_ms``; speech is synthesized by ``tts`` (or timed at
    ``wps`` words per second). A user turn that ends while results are awaited is read with them; while the user
    is talking, the agent starts no new round, and stops (or never starts) speaking. A round's tool calls go out when
    its words start, so the agent says what it is about to do as it does it."""

    def __init__(self, spec: AgentSpec, llm: ToolChat, llm_ms: int = 700, wps: float = 2.8, max_rounds: int = 12, tts=None,
                 voice: str = "aiden", language: str = "English"):
        self.spec, self.llm, self.llm_ms, self.wps, self.max_rounds = spec, llm, llm_ms, wps, max_rounds
        self.tts, self.voice, self.language = tts, voice, language
        self.msgs, self.tools, self.pending, self.n, self.rounds = [], [], set(), 0, 0
        self.heard: dict[str, str] = {}  # user segment id -> text so far (each chunk carries only the new words)
        self.inbox: list[str] = []  # finished user turns not yet given to the LLM
        self.speaking: Segment | None = None

    def describe(self) -> dict:
        return {"name": "LLM + tools (text in, TTS out)" if self.tts else "LLM + tools (text)", "kind": "cascaded" if self.tts else "text",
                "model": self.llm.model, "hears": "the user's transcript (no ASR)", "llm_ms": self.llm_ms,
                "tts": getattr(self.tts, "model", None), "voice": self.voice if self.tts else None}

    async def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        out, results = [], False
        for f in obs:
            if f.stream == "session":
                self.tools = [s.openai() for s in f.data.tools]
                ctx = f"\n\nContext:\n{json.dumps(f.data.context, ensure_ascii=False, indent=1)}" if f.data.context else ""
                self.msgs = [{"role": "system", "content": f.data.instructions + ctx}]
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
        if self.pending or self.heard:  # finish the tool round, and let the user finish talking, before the next LLM round
            return out
        if self.inbox:
            self.msgs += [{"role": "user", "content": u} for u in self.inbox]
            self.inbox, self.rounds, results = [], 0, True
        if not results or self.rounds >= self.max_rounds:
            return out
        self.rounds += 1
        r = await self.llm.complete(self.msgs, self.tools)
        text = speakable(r["content"] or "")
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
            t0 = t_out
            if self.tts is not None:
                audio = await self.tts.synth(text, self.voice, None, language=self.language)
                seg = Segment(f"a{self.n}", t0, audio.dur_ms, audio, text=text)
            else:
                seg = Segment(f"a{self.n}", t0, max(400, int(len(text.split()) / self.wps * 1000)), text)
            self.speaking = seg
            out.append(Frame(self.spec.out, t0, seg))
        return out
