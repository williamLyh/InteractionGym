"""Plain-language commentary for the viewer's live playback (docs/FORMAT.md §6).

``headlines(ep)`` turns an episode's recorded evaluation into one-line headlines ("User barged in —
agent yielded in 340 ms ✓"), each placed on the timeline and carrying the raw records it was
derived from. Nothing is invented: every word of a headline comes from ``eval.scores`` /
``eval.duplex`` and the turns they point to. ``units(trace)`` condenses an agent trace
(docs/AGENT_TRACE.md) to what the env viewer's unit strip needs.
"""

from __future__ import annotations

import math

NONDIRECTED = ("backchannel", "aside", "noise")
GOOD = {  # (expects, outcome) pairs that are what the user wanted (eval.scores: one event per user turn)
    ("respond", "responded"), ("yield", "yielded"), ("ignore", "ignored"), ("wait", "waited"), ("wait", "checked_in"),
    ("interrupt", "cut_in"),
}
NEUTRAL = {("respond", "censored")}  # recorded but unscored: neither what the user wanted nor not


def _snip(text: str, n: int = 40) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _secs(ms: float) -> str:
    return f"{ms} ms" if abs(ms) < 1000 else f"{ms / 1000:.1f} s"


def _what(u: dict) -> str:
    """How the user's turn reads in a headline: 'User said “mm-hmm”', 'User made a noise', ..."""
    kind, text = u.get("kind"), _snip(u.get("text", ""))
    if kind == "noise":
        label = (u.get("label") or "").replace("_", " ")
        return f"Noise “{text}”" if text else f"A noise ({label})" if label else "A noise"
    if kind == "away":
        return f"User away for {_secs(u['end_time'] - u['start_time'])}"
    if kind == "backchannel":
        return f"User said “{text}” (backchannel)" if text else "User backchannelled"
    if kind == "aside":
        return f"User aside “{text}”" if text else "User aside"
    return f"User: “{text}”" if text else "User turn"


def _mark(ok: bool | None) -> str:
    return "" if ok is None else (" ✓" if ok else " ✗")


def _speech(turns: list[dict]) -> list[dict]:
    return [t for t in turns if t["end_time"] > t["start_time"]]


def _next_normal_user(turns: list[dict], u: dict, agent: str) -> dict | None:
    return next((o for o in _speech(turns) if o["role"] != agent and o.get("kind") not in NONDIRECTED
                 and o["start_time"] > u["start_time"]), None)


def _over(u: dict, ep: dict, agent: str) -> bool:
    """Whether an agent turn was playing when the user turn started."""
    return any(b["role"] == agent and b["start_time"] <= u["start_time"] < b["end_time"] for b in _speech(ep["turns"]))


def _score_headline(ev: dict, u: dict, by_id: dict, ep: dict, agent: str) -> tuple[str, int, int]:
    """(text, t, t_resolved) for one ``eval.scores`` event. ``t`` is where the situation starts (the
    playhead jumps there); ``t_resolved`` is when its outcome is visible on the timeline."""
    exp, out, lat = ev["expects"], ev["outcome"], ev.get("latency_ms")
    tg = by_id.get(ev.get("target")) or {}
    u0, u1, dur = u["start_time"], u["end_time"], ep["meta"]["duration_ms"]
    t1 = ev.get("resolved_ms", u1)
    said = f"“{_snip(u.get('text', ''))}”"
    if exp == "respond":
        if out == "censored":  # the episode ended before the answer window closed: not scored
            return f"{said} cut off by the episode end — not scored", u1, dur
        if out == "responded":
            return f"Agent answered {said} after {_secs(lat)}", u1, tg.get("start_time", u1 + lat)
        if out == "talked_over":
            return f"Agent talked over {said}", u0, tg.get("start_time", u0)
        nxt = _next_normal_user(ep["turns"], u, agent)
        until = f"user spoke again {_secs(nxt['start_time'] - u1)} later" if nxt else f"episode ended {_secs(dur - u1)} later"
        return f"No reply to {said} ({until})", u1, nxt["start_time"] if nxt else dur
    if exp == "yield":
        head = "User barged in"
        if out == "kept_talking":
            if "unsaid" in tg and tg.get("end_time", 0) >= dur:  # cut by the episode end, not by the agent
                return f"{head} — agent kept talking until the episode ended", u0, u1
            return f"{head} — agent kept talking", u0, u1
        stop = f"agent yielded in {_secs(lat)}" if lat is not None else "agent yielded"
        if out == "yielded":
            then = f", answered {_secs(ev['response_ms'])} after the user finished" if ev.get("response_ms") is not None else ""
            return f"{head} — {stop}{then}", u0, t1
        if out == "talked_over":
            return f"{head} — {stop}, then talked over the rest", u0, tg.get("start_time", t1)
        if out == "took_floor":
            return f"{head} and paused mid-thought — {stop}, then took the floor before the user went on", u0, tg.get("start_time", t1)
        return f"{head} — {stop} but never took the floor again", u0, t1  # no_response
    if exp == "ignore":
        head = _what(u)
        if out == "ignored":
            text = "agent kept talking" if tg and tg.get("start_time", u0) <= u0 else "agent stayed silent"
            return f"{head} — {text}", u0, u1
        if out == "stopped":
            return f"{head} — agent stopped talking", u0, tg.get("end_time", u1)
        if tg and _over(u, ep, agent) and tg.get("start_time", 0) >= u1:  # eval.reply_after: answered once its own turn was done
            return f"{head} — agent finished its turn, then replied to it after {_secs(tg['start_time'] - u1)}", u0, tg["start_time"]
        return f"{head} — agent replied to it", u0, tg.get("start_time", u1)
    if exp == "wait" and u.get("kind") == "away":
        text = {"waited": "agent waited", "checked_in": "agent briefly checked in", "took_floor": "agent talked to an absent user"}.get(out, out)
        return f"{_what(u)} — {text}", u0, tg.get("start_time", u1) if out == "took_floor" else u1
    if exp == "wait":
        over = _over(u, ep, agent)
        head = f"User cut in with {said} (wait)" if over else "User paused mid-thought"
        if out == "kept_talking":
            return f"{head} — agent kept talking", u0, u1
        if out == "took_floor":
            return f"{head} — agent took the floor", u0 if over else u1, tg.get("start_time", t1)
        nxt = _next_normal_user(ep["turns"], u, agent)
        stopped = f"agent stopped in {_secs(lat)} and waited" if lat is not None else "agent waited"
        return f"{head} — {stopped}", u0 if over else u1, nxt["start_time"] if nxt else u1
    if exp == "interrupt":
        if out == "cut_in":
            return f"Agent cut in on {said} (user wanted to be interrupted)", u0, tg.get("start_time", u0)
        return f"Agent let the user finish {said} (user wanted to be interrupted)", u0, u1
    return f"{exp}: {out}", u0, u1


def _duplex_headline(d: dict, t: dict, by_id: dict, ep: dict) -> tuple[str, bool | None, int, int]:
    b, r, tg = d["behavior"], d.get("reaction"), by_id.get(d.get("target")) or {}
    if b == "barge_in":
        if r == "yielded":
            return f"User barged in — agent yielded in {_secs(d['latency_ms'])}", True, t["start_time"], tg.get("end_time", t["start_time"])
        return "User barged in — agent kept talking", False, t["start_time"], t["end_time"]
    if b in NONDIRECTED:
        text = {"continued": "agent kept talking", "stopped": "agent stopped talking",
                "stayed_silent": "agent stayed silent", "responded": "agent replied to it"}.get(r, r)
        t1 = tg.get("end_time", t["end_time"]) if r == "stopped" else t["end_time"]
        if r == "responded" and tg:  # it was talking: finished that turn, then replied (eval.reply_after)
            text, t1 = "agent finished its turn, then replied to it", max(t["end_time"], tg.get("end_time", 0))
        return f"{_what(t)} — {text}", r in ("continued", "stayed_silent"), t["start_time"], t1
    if b == "away":
        text = {"waited": "agent waited", "checked_in": "agent briefly checked in", "talked": "agent kept talking to an absent user"}.get(r, r)
        return f"{_what(t)} — {text}", r in ("waited", "checked_in"), t["start_time"], t["end_time"]
    if b == "agent_interrupt":
        wanted = tg.get("expects") == "interrupt"
        return f"Agent cut in on “{_snip(tg.get('text', ''))}”", wanted, tg.get("start_time", t["start_time"]), t["start_time"]
    return f"{b}: {r}", None, t["start_time"], t["end_time"]


def _pending(h: dict, u: dict) -> str:
    """Neutral wording while the situation is still unfolding on the playhead (before ``t_resolved``):
    it says what happened so far and nothing about the outcome."""
    exp = h.get("expects") or {"barge_in": "yield", "agent_interrupt": "interrupt"}.get(h.get("behavior"), "ignore")
    if exp == "yield":
        return "User barges in while the agent talks…"
    if exp == "wait" and h.get("over"):
        return f"User cuts in with “{_snip(u.get('text', ''))}” while the agent talks…"
    if exp == "respond":
        return f"User finished “{_snip(u.get('text', ''))}” — awaiting a reply…"
    if exp == "wait":
        return "User is away…" if u.get("kind") == "away" else "User pauses mid-thought…"
    if exp == "ignore":
        return _what(u) + "…"
    return f"User is speaking: “{_snip(u.get('text', ''))}”…"


def _tally(scored: list[tuple[str, float]]) -> float | None:
    """``eval.scores.total`` over the events so far: the mean score over the user turns scored so far."""
    return round(sum(s for _, s in scored) / len(scored), 4) if scored else None


def headlines(ep: dict, agent: str = "agent") -> list[dict]:
    """One headline per ``eval.scores`` event (one per user turn), or — for episodes evaluated without scores — per
    ``eval.duplex`` record. Each item: ``text``, ``ok`` (True / False, or None when nothing says what
    was expected), ``t`` / ``t_resolved`` (ms), ``turn``, ``source``, ``raw`` (the event, its turn,
    the target turn and the matching duplex record, exactly as recorded); scored items add
    ``expects``, ``outcome``, ``score`` and ``tally`` — the episode's total over the events
    resolved up to this one, which ends at ``eval.scores.total``. A ``censored`` event (the episode ended
    before the user's turn could be answered) has ``score`` None, ``scored`` False, ``ok`` None and stays out of
    the tally, as it stays out of ``eval.scores``. Sorted by ``t_resolved``."""
    ev_all = ep.get("eval") or {}
    by_id = {t["id"]: t for t in ep.get("turns", [])}
    dup = ev_all.get("duplex") or {}
    out: list[dict] = []
    sc = ev_all.get("scores")
    if sc and sc.get("events"):
        for i, ev in enumerate(sc["events"]):
            u = by_id.get(ev["turn"])
            if u is None:
                continue
            text, t, t1 = _score_headline(ev, u, by_id, ep, agent)
            key = (ev["expects"], ev["outcome"])
            ok = None if key in NEUTRAL else key in GOOD
            raw = {"event": ev, "turn": u}
            if ev.get("target") in by_id:
                raw["target"] = by_id[ev["target"]]
            if ev["turn"] in dup:
                raw["duplex"] = dup[ev["turn"]]
            out.append({"i": i, "text": text + _mark(ok), "ok": ok, "t": t, "t_resolved": max(t, t1), "turn": ev["turn"],
                        "source": "eval.scores", "expects": ev["expects"], "outcome": ev["outcome"],
                        "score": ev.get("score"), "scored": "score" in ev, "latency_ms": ev.get("latency_ms"),
                        "over": ev["expects"] == "wait" and _over(u, ep, agent), "raw": raw})
    else:
        for i, (tid, d) in enumerate(dup.items()):
            t = by_id.get(tid)
            if t is None:
                continue
            text, ok, t0, t1 = _duplex_headline(d, t, by_id, ep)
            raw = {"duplex": {tid: d}, "turn": t}
            if d.get("target") in by_id:
                raw["target"] = by_id[d["target"]]
            out.append({"i": i, "text": text + _mark(ok), "ok": ok, "t": t0, "t_resolved": max(t0, t1), "turn": tid,
                        "source": "eval.duplex", "behavior": d["behavior"], "outcome": d.get("reaction"),
                        "latency_ms": d.get("latency_ms"), "raw": raw})
    for h in out:
        h["pending"] = _pending(h, by_id[h["turn"]] if h["source"] == "eval.scores" else h["raw"]["turn"])
    out.sort(key=lambda h: (h["t_resolved"], h["i"]))
    scored: list[tuple[str, float]] = []
    for h in out:
        if h.get("score") is not None:
            scored.append((h["expects"], h["score"]))
            h["tally"] = _tally(scored)
        del h["i"]
    return out


def _detok(piece: str) -> str:
    """Byte-level BPE pieces as text (Ġ = space, Ċ = newline)."""
    return piece.replace("Ġ", " ").replace("Ċ", "\n")


def units(trace: dict | None) -> dict | None:
    """The agent trace condensed for the env viewer's unit strip: per unit its input window, decision,
    the text it generated, and the special tokens that matter for turn-taking (``<|speak|>``,
    ``<|listen|>``, ``<|turn_eos|>``, ``<|chunk_eos|>`` …) in output order. Token ids are dropped."""
    if not trace:
        return None
    unit_ms = trace.get("unit_ms") or 1000
    rows = []
    for u in trace.get("units", []):
        if u.get("unit_index", -1) < 0 or u.get("end_ms") is None:
            continue
        special = set(u.get("special_ids") or [])
        text, marks = [], []
        for st in u.get("stages") or []:
            if st.get("stage") not in (None, "thinker"):
                continue
            for tok_id, piece in st.get("output") or []:
                if tok_id in special:
                    marks.append(piece)
                else:
                    text.append(_detok(piece))
        rows.append({"i": u["unit_index"], "t0": u["end_ms"] - unit_ms, "t1": u["end_ms"], "decision": u.get("decision"),
                     "text": "".join(text), "marks": marks})
    return {"unit_ms": unit_ms, "server": trace.get("server"), "model": trace.get("model"), "clock": trace.get("clock"), "units": rows}


def view_data(episodes: list[dict], traces: list[dict] | None = None, agent_pages: dict[str, str] | None = None) -> dict:
    """What the env viewer adds to the episodes for playback, keyed by ``episode_id``."""
    by_id = {t["episode_id"]: t for t in traces or [] if t}
    out = {}
    for ep in episodes:
        eid = ep.get("meta", {}).get("episode_id")
        if eid is None:
            continue
        sc = (ep.get("eval") or {}).get("scores") or {}
        total = sc.get("total")
        out[eid] = {"headlines": headlines(ep), "total": None if total is None or (isinstance(total, float) and math.isnan(total)) else total,
                    "units": units(by_id.get(eid)), "agent_page": (agent_pages or {}).get(eid)}
    return out
