"""Post-hoc evaluation from the episode log.

Timing metrics follow τ-Voice's definitions: response latency is the gap from
the end of a user turn to the start of the agent's reply; yield latency is the
gap from the start of a user barge-in to the agent's speech being cut.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

from .core import Frame, Segment


@dataclass(frozen=True)
class Turn:
    planned: Segment  # as first emitted (for streamed segments: the final version, since no plan exists)
    actual: Segment  # latest version (shorter if it was cut)
    grown: bool = False  # built from streamed chunks: its end is simply where the speaker stopped

    @property
    def was_cut(self) -> bool:
        return self.actual.end < self.planned.end


@dataclass(frozen=True)
class Span:
    """A measured interval, e.g. from the end of a user turn to the agent's reply."""

    kind: str  # "response" | "yield"
    t0: int
    t1: int | None  # None: the expected reaction never happened

    @property
    def ms(self) -> int | None:
        return None if self.t1 is None else self.t1 - self.t0


def turns(log: list[Frame], stream: str) -> list[Turn]:
    first: dict[str, Segment] = {}
    last: dict[str, Segment] = {}
    for f in log:
        if f.stream == stream and isinstance(f.data, Segment):
            first.setdefault(f.data.id, f.data)
            last[f.data.id] = f.data
    out = []
    for i in first:
        grown = last[i].end > first[i].end
        out.append(Turn(last[i] if grown else first[i], last[i], grown))
    return sorted(out, key=lambda x: x.planned.t0)


def response_spans(log: list[Frame], user: str = "user.speech", agent: str = "policy.speech") -> list[Span]:
    """One span per user turn: its end → the agent's next reply (t1=None if the user spoke again first)."""
    us, ag = turns(log, user), turns(log, agent)
    out = []
    for i, u in enumerate(us):
        next_user = us[i + 1].planned.t0 if i + 1 < len(us) else None
        reply = next((a for a in ag if a.planned.t0 >= u.actual.end), None)
        ok = reply is not None and (next_user is None or reply.planned.t0 < next_user)
        out.append(Span("response", u.actual.end, reply.planned.t0 if ok else None))
    return out


def yield_spans(log: list[Frame], user: str = "user.speech", agent: str = "policy.speech") -> list[Span]:
    """One span per user turn that starts while the agent speaks: barge-in → agent stops (t1=None if it never yields).

    A whole segment yields when it is cut; a streamed one when it stops before the user's turn is over.
    """
    out = []
    for u in turns(log, user):
        for a in turns(log, agent):
            if a.planned.t0 < u.planned.t0 < a.planned.end:
                stopped = a.was_cut or (a.grown and a.actual.end < u.actual.end)
                out.append(Span("yield", u.planned.t0, a.actual.end if stopped else None))
    return out


def response_latencies(log: list[Frame], user: str = "user.speech", agent: str = "policy.speech") -> list[int | None]:
    return [s.ms for s in response_spans(log, user, agent)]


def yield_latencies(log: list[Frame], user: str = "user.speech", agent: str = "policy.speech") -> list[int | None]:
    return [s.ms for s in yield_spans(log, user, agent)]


# ---------------------------------------------------------------- duplex behaviours on schema-v1 turns

NONDIRECTED = ("backchannel", "aside", "noise")
AWAY = "away"  # the user is gone for a while (called away): the agent should wait, at most briefly check in
SIDE = NONDIRECTED + (AWAY,)  # user turns that are not lines addressed to the agent


def _speech(turns: list[dict]) -> list[dict]:
    return [t for t in turns if t["end_time"] > t["start_time"]]


def reply_after(u: dict, a: dict, agents: list[dict], others: list[dict], window_ms: int = 2000) -> dict | None:
    """An agent turn that answers a non-directed sound ``u`` heard while agent turn ``a`` was playing: ``a`` ran
    to its end (not cut) and a new agent turn starts within ``window_ms`` of the later of the two ends, with no
    directed user turn in between ("Oh no, are you okay?" after "Dad, the microwave quit"). ``a`` already
    answered the user's question, so a fresh turn this soon is taken as a reaction to the sound."""
    if "unsaid" in a:
        return None
    ref = max(u["end_time"], a["end_time"])
    nxt = next((b for b in agents if b is not a and ref <= b["start_time"] <= ref + window_ms), None)
    # a directed user turn after the sound, or still going when it started (a noise may overlap the user's own speech),
    # is what the new agent turn may answer
    if nxt is None or any(o.get("kind") is None and u["start_time"] < o["end_time"] and o["start_time"] < nxt["start_time"] for o in others):
        return None
    return nxt


def away_reaction(u: dict, agents: list[dict], check_in_ms: int = 3000, check_in_after_ms: int = 2000) -> tuple[str, dict | None]:
    """What the agent did while the user was away (a ``kind: "away"`` turn): ``waited`` (no agent turn started in it),
    ``checked_in`` (exactly one short turn, at most ``check_in_ms`` long, at least ``check_in_after_ms`` into the
    absence: "Are you still there?"), or ``talked`` (anything more); with the first agent turn involved."""
    during = [b for b in agents if u["start_time"] <= b["start_time"] < u["end_time"]]
    if not during:
        return "waited", None
    b = during[0]
    if len(during) == 1 and b["end_time"] - b["start_time"] <= check_in_ms and b["start_time"] >= u["start_time"] + check_in_after_ms:
        return "checked_in", b
    return "talked", b


def duplex(turns: list[dict], agent: str = "agent", respond_window_ms: int = 2000, stop_grace_ms: int = 1000,
           min_reaction_ms: int = 200, answer_window_ms: int = 5000, end_ms: int | None = None) -> dict:
    """Special duplex behaviours of an episode (docs/FORMAT.md §6.2), keyed by the turn that triggers them.

    - ``barge_in``: a normal non-agent turn starting while an agent turn plays; ``yielded`` if that
      agent turn was cut (``unsaid``) or stopped before the barge-in ended.
    - ``backchannel`` / ``aside`` / ``noise`` (from ``kind``): the agent should not react —
      ``continued`` / ``stopped`` while it was speaking (``responded`` if it then answered an aside or
      noise, ``reply_after``; going on after a backchannel is what the user wants), ``stayed_silent`` /
      ``responded`` otherwise.
    - ``away`` (from ``kind``): the user was called away; ``waited`` / ``checked_in`` / ``talked`` (``away_reaction``).
    - ``agent_interrupt``: an agent turn starting while a normal non-agent turn plays.

    A turn with ``unsaid`` was cut: that is a reaction. One without it may simply have finished, so
    ending during the other's turn counts as a reaction only ``min_reaction_ms`` or more after that
    turn started (sooner is too fast to be one). An agent turn after a non-directed sound counts as
    responding to it unless it may be the answer to the user's previous question: one not yet answered
    that ended at most ``answer_window_ms`` before this agent turn started. With ``end_ms`` (the episode's
    end), an agent turn cut only because the episode ended is not a stop.
    """
    speech = _speech(turns)
    agents = [t for t in speech if t["role"] == agent]
    others = [t for t in speech if t["role"] != agent]
    playing = lambda pool, t0: next((a for a in pool if a["start_time"] <= t0 < a["end_time"]), None)  # noqa: E731
    cut = lambda a: "unsaid" in a and not (end_ms is not None and a["end_time"] >= end_ms)  # noqa: E731  (by the agent, not the end)

    def ended_during(a: dict, u: dict) -> bool:  # finished (not cut) while u was going, late enough to be a reaction
        return "unsaid" not in a and u["start_time"] + min_reaction_ms <= a["end_time"] < u["end_time"]

    def answered(q: dict, before: int) -> bool:  # an agent turn started after question q, before time `before`
        return any(q["end_time"] <= b["start_time"] <= before for b in agents)

    out = {}
    for u in others:
        a = playing(agents, u["start_time"])
        kind = u.get("kind")
        if kind is None:
            if a is not None:
                stopped = cut(a) or ended_during(a, u)
                rec = {"behavior": "barge_in", "target": a["id"], "reaction": "yielded" if stopped else "kept_talking"}
                if stopped:
                    rec["latency_ms"] = a["end_time"] - u["start_time"]
                out[u["id"]] = rec
        elif kind == AWAY:
            r, b = away_reaction(u, agents)
            out[u["id"]] = {"behavior": AWAY, "reaction": r, **({"target": b["id"]} if b is not None else {})}
        elif kind in NONDIRECTED:
            if a is not None:
                stopped = ended_during(a, u) or (cut(a) and a["end_time"] <= u["end_time"] + stop_grace_ms)
                replied = kind != "backchannel" and reply_after(u, a, agents, others, respond_window_ms)
                reaction = "stopped" if stopped else "responded" if replied else "continued"
                out[u["id"]] = {"behavior": kind, "target": a["id"], "reaction": reaction}
            else:
                nxt = next((b for b in agents if u["end_time"] <= b["start_time"] <= u["end_time"] + respond_window_ms), None)
                between = nxt is not None and any(
                    o.get("kind") is None and u["end_time"] <= o["start_time"] < nxt["start_time"] for o in others
                )
                # the user's directed turns before the sound, or going on during it (a cough over the user's own question)
                prev = [o for o in others if o.get("kind") is None and o["start_time"] < u["end_time"] and o is not u]
                pending = (bool(prev) and nxt is not None and not answered(prev[-1], u["start_time"])
                           and nxt["start_time"] - prev[-1]["end_time"] <= answer_window_ms)  # the reply may be to that question
                out[u["id"]] = {"behavior": kind, "reaction": "responded" if nxt is not None and not between and not pending else "stayed_silent"}
    for a in agents:
        u = playing([o for o in others if o.get("kind") is None], a["start_time"])
        if u is not None and u["start_time"] < a["start_time"]:  # starting together is a barge-in (above), not this
            out[a["id"]] = {"behavior": "agent_interrupt", "target": u["id"]}
    start = {t["id"]: t["start_time"] for t in turns}
    return dict(sorted(out.items(), key=lambda kv: start[kv[0]]))


# ---------------------------------------------------------------- scores: did the agent do what the user expected?

EXPECTATIONS = ("respond", "yield", "ignore", "wait", "interrupt")

# A piece of a turn that stops mid-thought trails off (user.pieces); the user will go on after a pause.
TRAILING = ("...", "…")


# The latency curve (``latency_score``): a logistic in milliseconds, half score at LATENCY_MID_MS.
LATENCY_MID_MS = 950.0
LATENCY_WIDTH_MS = 100.0


def latency_score(ms: float, mid_ms: float = LATENCY_MID_MS, width_ms: float = LATENCY_WIDTH_MS) -> float:
    """How satisfying a reaction delay is, for taking the floor after the user ends (``respond``) and for stopping
    when talked over (``yield``, from the user's start to the agent's stop): a logistic in milliseconds,

        s(l) = 1 / (1 + exp((l - mid_ms) / width_ms)),   mid_ms = 950, width_ms = 100,

    so 200-500 ms is full score, 500-700 ms still high, and from ~800 ms it falls off quickly:

    ====== ====== ====== ====== ====== ====== ====== ====== ====== ======
    l (ms)    0    200    500    700    800    900   1000   1200   1500
    s(l)   1.000  0.999  0.989  0.924  0.818  0.622  0.378  0.076  0.004
    ====== ====== ====== ====== ====== ====== ====== ====== ====== ======

    Monotone: a very early reaction (under 200 ms, but after the user ended / after the barge-in started) also scores
    ~1. Negative delays are clamped to 0 (they do not occur in ``scores``: a reply that starts during the user's turn
    is ``talked_over``)."""
    x = (max(ms, 0.0) - mid_ms) / width_ms
    if x > 50:
        return 0.0
    return 1.0 / (1.0 + math.exp(x))


def expectation(u: dict, agent_playing: bool) -> str:
    """What the user expects of the agent for one of its turns, one of ``EXPECTATIONS``. The producer may say it
    (``expects``); otherwise it follows from ``kind``: a backchannel / aside / noise is to be ignored, an absence
    (``away``) waited through, and a normal turn answered (``respond``) — or, if it starts while the agent talks,
    yielded to and then answered (``yield``). ``expects: "wait"`` (a piece before a mid-thought pause, "hold on")
    means: do not take the floor (and stop, if the agent is talking); ``expects: "interrupt"``: cut in."""
    kind, explicit = u.get("kind"), u.get("expects")
    if kind in NONDIRECTED and explicit in (None, "ignore", "continue"):
        return "ignore"
    if kind == AWAY or explicit == "wait":
        return "wait"
    if explicit == "interrupt":
        return "interrupt"
    if agent_playing and explicit in (None, "yield"):
        return "yield"
    return "respond"


def expectations(u: dict, agent_playing: bool) -> list[str]:
    """``[expectation(u, agent_playing)]``: every user turn has exactly one expectation (kept for callers
    of the list form)."""
    return [expectation(u, agent_playing)]


def scores(turns: list[dict], agent: str = "agent", respond_within_ms: int = 5000, mid_ms: float = LATENCY_MID_MS,
           width_ms: float = LATENCY_WIDTH_MS, end_ms: int | None = None) -> dict:
    """One rule-based score in [0, 1] per user turn: did the agent do what the user expected (``expectation``),
    and — where doing it means taking over the conversation — how fast. Outcomes (docs/FORMAT.md §6.3):

    - ``respond`` (a normal turn): ``talked_over`` (0) if an agent turn started during the user's turn; else
      ``responded`` — ``latency_score`` of the gap from the user's end to the reply — if the agent started speaking
      after the turn ended, before the next directed user turn and within ``respond_within_ms``; else
      ``no_response`` (0). With ``end_ms`` (the episode's end), a turn left unanswered because the episode ended
      before its window closed is ``censored``: kept without a score and left out of the means (replayed benchmark
      clips often stop right after the user's last word).
    - ``yield`` (a normal turn that starts while the agent talks): ``kept_talking`` (0) if the agent did not stop
      (cut, or ended before the user did; a turn cut only by the episode end is not a stop); ``talked_over`` (0) if
      it started again while the user was still talking; ``no_response`` (0) if it never took the floor again in the
      ``respond`` window; else ``yielded``: ``latency_score`` of the time from the user's start to the stop. A barge-in that trails off ("Actually, wait..."; the user goes on after a pause) is to be yielded to
      and then waited through: ``took_floor`` (0) if the agent spoke before the user went on. A censored window
      leaves the stop alone to score (``yielded``).
    - ``ignore`` (a backchannel / aside / noise): ``ignored`` (1) if the agent neither stopped (cut within 1 s of
      the sound's end) nor replied to it; ``stopped`` / ``replied`` (0). A reply right after an aside or noise
      heard while the agent spoke counts (``reply_after``), one that may answer the user's own open question does
      not (``duplex``); going on after a backchannel is what the user wants.
    - ``wait`` (``expects: "wait"``: a piece before a mid-thought pause, "hold on"): ``kept_talking`` (0) if the
      agent was talking and did not stop, ``took_floor`` (0) if it started speaking during the turn or after it
      before the user's next directed turn, else ``waited`` (1). For an absence (``kind: "away"``): ``waited`` /
      ``checked_in`` (1: at most one short check-in, ``away_reaction``) or ``took_floor`` (0).
    - ``interrupt`` (``expects: "interrupt"``): ``cut_in`` (1) if an agent turn started during the user's turn,
      else ``listened`` (0).

    Returns ``{"events": [one per user turn], "by_expectation": {name: mean}, "total": mean over the scored
    events}``; ``total`` is ``None`` when nothing was scored. The same scores serve as evaluation metrics and as a
    (stage-1) RL reward.

    Every event carries position fields (they never change a score), so a trainer can assign it to the part of the
    episode that caused it: ``user_start_ms`` / ``user_end_ms`` (the user turn), ``resolved_ms`` (when the outcome
    was settled: the reply's / stop's / cut-in's start, or the end of the window in which nothing happened; absent
    for ``censored``), ``latency_ms`` (the latency that was graded: the reply's for ``respond``, the stop's for
    ``yield`` / a ``wait`` that stopped the agent) and ``target``: the agent turn that decided the outcome (the reply,
    the turn that yielded or kept talking, the one that talked over the user or took the floor, the reaction to a
    sound; absent when the agent did nothing). A ``yield`` that was answered also names the ``reply`` and its
    ``response_ms``."""
    speech = _speech(turns)
    agents = [t for t in speech if t["role"] == agent]
    users = [t for t in speech if t["role"] != agent]
    playing = lambda pool, t0: next((a for a in pool if a["start_time"] <= t0 < a["end_time"]), None)  # noqa: E731
    cut_by_end = lambda a: end_ms is not None and a["end_time"] >= end_ms  # noqa: E731  (the episode ended, not the agent)
    events = []
    for i, u in enumerate(users):
        u0, u1 = u["start_time"], u["end_time"]
        a = playing(agents, u0)
        nxt_user = next((o for o in users[i + 1 :] if o.get("kind") not in NONDIRECTED), None)
        limit = nxt_user["start_time"] if nxt_user is not None else math.inf
        window_end = min(limit, u1 + respond_within_ms)
        cut_in = next((b for b in agents if u0 < b["start_time"] < u1), None)  # an agent turn starting during the user's
        took = next((b for b in agents if u1 <= b["start_time"] < limit), None)  # the first one after it
        reply = took if took is not None and took["start_time"] < window_end else None
        censored = reply is None and end_ms is not None and window_end > end_ms
        stopped = a is not None and (("unsaid" in a and not cut_by_end(a)) or a["end_time"] < u1)
        exp = expectation(u, a is not None)
        ev: dict = {"turn": u["id"], "expects": exp}
        if exp == "respond":
            if cut_in is not None:
                ev.update(outcome="talked_over", score=0.0, target=cut_in["id"], resolved_ms=cut_in["start_time"])
            elif reply is not None:
                gap = reply["start_time"] - u1
                ev.update(outcome="responded", latency_ms=gap, score=latency_score(gap, mid_ms, width_ms), target=reply["id"],
                          resolved_ms=reply["start_time"])
            elif censored:
                ev.update(outcome="censored")
            else:
                ev.update(outcome="no_response", score=0.0, resolved_ms=window_end)
        elif exp == "yield":
            if not stopped:
                ev.update(outcome="kept_talking", score=0.0, target=a["id"], resolved_ms=u1)
            else:
                lat = a["end_time"] - u0
                ok = latency_score(lat, mid_ms, width_ms)
                if cut_in is not None:
                    ev.update(outcome="talked_over", score=0.0, latency_ms=lat, target=cut_in["id"], resolved_ms=cut_in["start_time"])
                elif (u.get("text") or "").rstrip().endswith(TRAILING):  # a piece: the user goes on after a pause
                    if took is not None:
                        ev.update(outcome="took_floor", score=0.0, latency_ms=lat, target=took["id"], resolved_ms=took["start_time"])
                    else:
                        ev.update(outcome="yielded", score=ok, latency_ms=lat, target=a["id"], resolved_ms=a["end_time"])
                elif reply is not None:
                    ev.update(outcome="yielded", score=ok, latency_ms=lat, target=a["id"], resolved_ms=reply["start_time"],
                              reply=reply["id"], response_ms=reply["start_time"] - u1)
                elif censored:
                    ev.update(outcome="yielded", score=ok, latency_ms=lat, target=a["id"], resolved_ms=a["end_time"])
                else:
                    ev.update(outcome="no_response", score=0.0, latency_ms=lat, resolved_ms=window_end)
        elif exp == "ignore":
            if a is not None:
                stop = "unsaid" in a and not cut_by_end(a) and a["end_time"] <= u1 + 1000
                replied = not stop and u.get("kind") != "backchannel" and reply_after(u, a, agents, users)
                outcome = "stopped" if stop else "replied" if replied else "ignored"
                ev.update(outcome=outcome, score=1.0 if outcome == "ignored" else 0.0, target=(replied or a)["id"],
                          resolved_ms=a["end_time"] if stop else replied["start_time"] if replied else u1)
            else:
                # replying to it: an agent turn soon after that is not the answer to an open question
                responded = duplex(turns, agent, end_ms=end_ms).get(u["id"], {}).get("reaction") == "responded"
                nxt = next((b for b in agents if u1 <= b["start_time"] <= u1 + 2000), None)  # = duplex()'s
                if responded and nxt is not None:
                    ev.update(outcome="replied", score=0.0, target=nxt["id"], resolved_ms=nxt["start_time"])
                else:
                    ev.update(outcome="ignored", score=1.0, resolved_ms=u1 + 2000)  # duplex()'s respond window
        elif exp == "wait" and u.get("kind") == AWAY:
            reaction, b = away_reaction(u, agents)  # "waited" | "checked_in" | "talked"
            if reaction == "talked":
                ev.update(outcome="took_floor", score=0.0, target=b["id"], resolved_ms=b["start_time"])
            else:
                ev.update(outcome=reaction, score=1.0, resolved_ms=u1)  # outcome "waited" / "checked_in"
                if b is not None:
                    ev["target"] = b["id"]
        elif exp == "wait":
            if a is not None and not stopped:
                ev.update(outcome="kept_talking", score=0.0, target=a["id"], resolved_ms=u1)
            elif (b := cut_in or took) is not None:
                ev.update(outcome="took_floor", score=0.0, target=b["id"], resolved_ms=b["start_time"])
            else:
                ev.update(outcome="waited", score=1.0, resolved_ms=limit if limit != math.inf else u1)
                if a is not None:  # it stopped talking, as asked
                    ev.update(latency_ms=a["end_time"] - u0, target=a["id"])
        else:  # interrupt
            ev.update(outcome="cut_in" if cut_in else "listened", score=1.0 if cut_in else 0.0,
                      resolved_ms=cut_in["start_time"] if cut_in else u1)
            if cut_in is not None:
                ev["target"] = cut_in["id"]
        ev.update(user_start_ms=u0, user_end_ms=u1)
        events.append(ev)
    by: dict[str, list[float]] = {}
    for ev in events:
        if "score" in ev:
            by.setdefault(ev["expects"], []).append(ev["score"])
    vals = [ev["score"] for ev in events if "score" in ev]
    return {"events": events, "by_expectation": {k: round(sum(v) / len(v), 4) for k, v in by.items()},
            "total": round(sum(vals) / len(vals), 4) if vals else None}


def _checks(turns: list[dict], agent: str = "agent", respond_within_ms: int = 5000, end_ms: int | None = None) -> list[dict]:
    """The elementary observations behind the timing metrics (``timing_counts``), per directed user turn: did the
    agent take the turn after it (``respond``: ``responded`` with ``latency_ms`` / ``no_response``, censored left
    out), stop when it started while the agent talked (``yield``: ``yielded`` with ``latency_ms`` /
    ``kept_talking``) and start talking during it (``cut``: ``cut_in`` / ``listened``); per non-directed sound, the
    ``ignore`` score. The rules ``scores`` used before it gave one event per user turn (pre-release, until 2026-10-08)."""
    speech = _speech(turns)
    agents = [t for t in speech if t["role"] == agent]
    users = [t for t in speech if t["role"] != agent]
    sc = {e["turn"]: e for e in scores(turns, agent, respond_within_ms, end_ms=end_ms)["events"]}
    playing = lambda t0: next((a for a in agents if a["start_time"] <= t0 < a["end_time"]), None)  # noqa: E731
    out = []
    for i, u in enumerate(users):
        e = sc[u["id"]]
        if e["expects"] == "ignore":
            out.append({"turn": u["id"], "check": "ignore", "score": e["score"]})
            continue
        if u.get("kind") == AWAY:
            continue
        u0, u1 = u["start_time"], u["end_time"]
        a = playing(u0)
        explicit = u.get("expects")
        if explicit == "yield" or (explicit in (None, "wait") and a is not None):
            if a is not None:
                stopped = ("unsaid" in a and not (end_ms is not None and a["end_time"] >= end_ms)) or a["end_time"] < u1
                out.append({"turn": u["id"], "check": "yield", "outcome": "yielded" if stopped else "kept_talking",
                            **({"latency_ms": a["end_time"] - u0} if stopped else {})})
        cut_in = next((b for b in agents if u0 < b["start_time"] < u1), None)
        out.append({"turn": u["id"], "check": "cut", "outcome": "cut_in" if cut_in else "listened"})
        if explicit != "wait":
            nxt = next((o for o in users[i + 1 :] if o.get("kind") not in NONDIRECTED), None)
            window_end = min(nxt["start_time"] if nxt is not None else math.inf, u1 + respond_within_ms)
            reply = next((b for b in agents if u1 <= b["start_time"] < window_end), None)
            if reply is not None:
                out.append({"turn": u["id"], "check": "respond", "outcome": "responded", "latency_ms": reply["start_time"] - u1})
            elif not (end_ms is not None and window_end > end_ms):
                out.append({"turn": u["id"], "check": "respond", "outcome": "no_response"})
    return out


# ---------------------------------------------------------------- duplex timing metrics, collisions kept apart

INTENDED = ("yield", "interrupt")  # a user turn that means to cut in (UserSim marks its barge-ins expects="yield")


def collisions(turns: list[dict], agent: str = "agent", intended=None) -> dict[str, dict]:
    """Directed user turns that start while an agent turn plays, split by intent: ``{turn id: {"agent": id,
    "intended": bool}}``. ``intended`` (a set of turn ids, or a predicate on the turn dict) says which user turns
    meant to cut in; by default those marked ``expects`` "yield" / "interrupt" (a live ``UserSim`` barge-in, or one
    frozen from it). An overlap that is not intended is a *collision*: a replayed (open-loop) line whose fixed start
    time happened to fall inside the agent's speech — the script never meant to barge in."""
    speech = _speech(turns)
    agents = [t for t in speech if t["role"] == agent]
    is_int = (intended if callable(intended) else (lambda u: u["id"] in intended) if intended is not None
              else (lambda u: u.get("expects") in INTENDED))
    out = {}
    for u in speech:
        if u["role"] == agent or u.get("kind") in SIDE:
            continue
        a = next((a for a in agents if a["start_time"] <= u["start_time"] < a["end_time"]), None)
        if a is not None:
            out[u["id"]] = {"agent": a["id"], "intended": bool(is_int(u))}
    return out


def timing_counts(turns: list[dict], events: list[dict] | None = None, agent: str = "agent", intended=None,
                  end_ms: int | None = None) -> dict:
    """Duplex timing counts of one episode (summed over episodes by ``timing_summary``), from the turns (the
    elementary observations of ``_checks``; ``events`` is accepted for compatibility and ignored: everything is
    recomputed from ``turns``):

    - ``respond`` (turn-taking): the agent took the turn after a directed user turn (``responded`` / ``no_response``;
      censored turns left out), with its latency; ``respond_clean`` the same without collision turns;
    - ``barge_in``: intended user barge-ins: the agent ``yielded`` (with latency) or ``kept_talking``;
    - ``collision``: unintended overlaps (``collisions``): the same outcomes, reported apart and never counted
      in the agent's yield / cut-in rates;
    - ``cut_in``: the agent started talking during a directed user turn, over non-collision turns;
      ``cut_in_collision`` over collision turns;
    - ``nondirected``: backchannels / asides / noises the agent correctly ignored (``ignore`` scores)."""
    checks = _checks(turns, agent, end_ms=end_ms)
    col = collisions(turns, agent, intended)
    c = {k: 0 for k in ("user_turns", "responded", "no_response", "responded_clean", "no_response_clean", "barge_in",
                        "barge_in_yielded", "collision", "collision_yielded", "cut_in_n", "cut_in", "cut_in_collision_n",
                        "cut_in_collision", "nondirected", "nondirected_ok")}
    lat = {"response": [], "response_clean": [], "yield": [], "collision_yield": []}
    c["user_turns"] = len({t["id"] for t in _speech(turns) if t["role"] != agent and t.get("kind") not in SIDE})
    for e in checks:
        x, o = col.get(e["turn"]), e.get("outcome")
        if e["check"] == "respond":
            ok = o == "responded"
            c["responded" if ok else "no_response"] += 1
            if ok:
                lat["response"].append(e["latency_ms"])
            if x is None or x["intended"]:
                c["responded_clean" if ok else "no_response_clean"] += 1
                if ok:
                    lat["response_clean"].append(e["latency_ms"])
        elif e["check"] == "yield":
            k = "barge_in" if x is None or x["intended"] else "collision"
            c[k] += 1
            if o == "yielded":
                c[k + "_yielded"] += 1
                lat["yield" if k == "barge_in" else "collision_yield"].append(e["latency_ms"])
        elif e["check"] == "cut":
            k = "cut_in_collision" if x is not None and not x["intended"] else "cut_in"
            c[k + "_n"] += 1
            c[k] += o == "cut_in"
        elif e["check"] == "ignore":
            c["nondirected"] += 1
            c["nondirected_ok"] += e["score"] == 1.0
    return {**c, "latencies": lat}


def _q(xs: list, p: float):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def timing_summary(counts: list[dict]) -> dict:
    """Pool ``timing_counts`` over episodes: turn-take rate (FD-Bench-style take-over rate after a user turn),
    response latency median / p90, yield rate and latency on intended barge-ins, the agent's cut-in rate, the share
    of user turns that were collisions and how the agent behaved on them, and non-directed sound handling.
    Rates are ``None`` when there was nothing to count."""
    s = {k: sum(c[k] for c in counts) for k in counts[0] if k != "latencies"} if counts else {}
    lat = {k: [x for c in counts for x in c["latencies"][k]] for k in ("response", "response_clean", "yield", "collision_yield")}
    r = lambda a, b: s[a] / (s[a] + s[b]) if s and s[a] + s[b] else None  # noqa: E731
    d = lambda a, n: s[a] / s[n] if s and s[n] else None  # noqa: E731
    return {"episodes": len(counts), "user_turns": s.get("user_turns", 0),
            "turn_take_rate": r("responded", "no_response"), "turn_take_rate_clean": r("responded_clean", "no_response_clean"),
            "latency_median_ms": _q(lat["response"], 0.5), "latency_p90_ms": _q(lat["response"], 0.9),
            "latency_median_clean_ms": _q(lat["response_clean"], 0.5),
            "barge_ins": s.get("barge_in", 0), "yield_rate": d("barge_in_yielded", "barge_in"),
            "yield_latency_median_ms": _q(lat["yield"], 0.5),
            "cut_in_rate": d("cut_in", "cut_in_n"),
            "collisions": s.get("collision", 0), "collision_share": d("collision", "user_turns"),
            "collisions_per_episode": s["collision"] / len(counts) if counts else None,
            "collision_yield_rate": d("collision_yielded", "collision"), "collision_yield_latency_median_ms": _q(lat["collision_yield"], 0.5),
            "collision_cut_in_rate": d("cut_in_collision", "cut_in_collision_n"),
            "nondirected": s.get("nondirected", 0), "nondirected_ok_rate": d("nondirected_ok", "nondirected")}
