"""Scripted stand-ins for the model under training.

``CannedAgent`` observes the user exactly like a real session does (streamed
``Chunk``s, never future content) and either submits each reply as a whole
``Segment`` (event-driven "realtime" models) or streams it one chunk per step
(frame-synchronous full-duplex models).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import AgentSpec, Chunk, Frame, Segment


@dataclass
class _Heard:
    t0: int
    end: int
    done: bool = False
    text: str = ""  # what has been heard of it so far


class CannedAgent:
    """Replies with fixed texts in order; waits ``reply_after`` ms of user silence,
    yields after ``yield_after`` ms of the user talking over it.

    ``min_words`` makes it judge by what it actually heard: user sounds with fewer words
    (a cough, "mm-hmm") neither make it yield nor get a reply. It never sees ``kind``."""

    def __init__(
        self,
        replies: list[str],
        spec: AgentSpec = AgentSpec(),
        *,
        streaming: bool = False,
        yield_after: int | None = 160,
        reply_after: int = 300,
        words_per_sec: float = 3.4,
        min_words: int = 0,
    ):
        self.replies, self.spec, self.streaming = replies, spec, streaming
        self.yield_after, self.reply_after, self.wps = yield_after, reply_after, words_per_sec
        self.min_words = min_words
        self.user: dict[str, _Heard] = {}
        self.answered: set[str] = set()
        self.plan: Segment | None = None  # the reply currently being spoken, as planned
        self.stop_at: int | None = None  # when it actually ends (plan.end unless it yields)

    def describe(self) -> dict:
        """``meta.agent``."""
        return {"name": "CannedAgent", "kind": "text", "replies": len(self.replies), "streaming": self.streaming}

    def _speaking(self, t: int) -> bool:
        return self.plan is not None and t < self.stop_at

    def _next_chunk(self, t: int) -> list[Frame]:
        b = min(t + self.spec.chunk_ms, self.stop_at)
        seg = self.plan
        c = Chunk(seg.id, t, b - t, seg.play(t, b), seg.play(t, b), first=t == seg.t0, last=b == seg.end)
        return [Frame(self.spec.out, t, c)]

    def act(self, t: int, obs: list[Frame]) -> list[Frame]:
        for f in obs:
            c = f.data
            if isinstance(c, Chunk):
                h = self.user.setdefault(c.id, _Heard(c.t0, c.t0))
                h.end, h.done, h.text = c.t0 + c.dur, h.done or c.last, h.text + c.text
        # with min_words, only what has sounded like words so far counts (a cough or "mm-hmm" never does)
        heard = {k: h for k, h in self.user.items() if len(h.text.split()) >= self.min_words}
        last_id = max(heard, key=lambda k: heard[k].t0, default=None)
        last = self.user.get(last_id)

        if self._speaking(t):
            barged = last is not None and last.t0 > self.plan.t0 and not last.done
            if self.yield_after is not None and barged and t - last.t0 >= self.yield_after:
                self.stop_at = t
                return [] if self.streaming else [Frame(self.spec.out, t, self.plan.cut(t))]
            return self._next_chunk(t) if self.streaming else []

        n = len(self.answered)
        if last is not None and last.done and last_id not in self.answered and t - last.end >= self.reply_after and n < len(self.replies):
            self.answered.add(last_id)
            text = self.replies[n]
            self.plan = Segment(f"a{n}", t, round(len(text.split()) / self.wps * 1000), text)
            self.stop_at = self.plan.end
            return self._next_chunk(t) if self.streaming else [Frame(self.spec.out, t, self.plan)]
        return []
