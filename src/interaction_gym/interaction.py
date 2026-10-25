"""Closed-loop interaction metrics: what an agent's behaviour cost the user over a whole call.

An open-loop evaluation scores each agent reply against a fixed next user turn, so it cannot see consequences: the
user repeating a question the agent missed, prompting a silent agent, being talked over and starting again, a call
that never reaches its end. These need a user who reacts. ``metrics(ep)`` reads them off one closed-loop episode
(schema v1), by rules only:

- ``nudges``: the user spoke again after their own turn because the agent stayed silent (no agent turn between the
  two user turns and a gap of at least ``nudge_gap_ms``);
- ``agent_cut_ins``: user turns the agent talked over (``eval.scores`` ``respond`` → ``talked_over``) and
  ``user_cut_by_agent``: user turns cut short while the agent was talking (the user gave up the floor);
- ``barge_ins_not_yielded``: the user barged in and the agent kept talking (``yield`` → ``kept_talking``);
- ``repeats``: user turns that say again most of an earlier user turn (content-word overlap ≥ ``repeat_overlap``);
- ``completed``: the call ended because the user hung up (``meta.end_reason == "idle"``), not at the time limit;
- ``user_turns``, ``agent_turns``, ``duration_s``, ``user_turns_per_min``.

``label_user_turns(ep, llm)`` adds an LLM label per user turn after the first (``answer`` / ``confirm`` /
``correct`` / ``repeat`` / ``clarify`` / ``close`` / ``other``); ``effort`` = correct + repeat + clarify, the turns the
user spent fixing the conversation instead of moving it forward. ``summarize`` averages over episodes.
"""

from __future__ import annotations

import json
import re

STOP = set("a an the and or but so to of in on at for with is are was were be it its this that i you we they he she me my "
           "your our do did does can could would will just yeah yes no okay ok um uh oh like what how".split())

LABELS = ("answer", "confirm", "correct", "repeat", "clarify", "close", "other")
EFFORT = ("correct", "repeat", "clarify")

LABEL_PROMPT = """Below is a spoken conversation between a USER and a voice ASSISTANT.

{dialogue}

For each USER turn after the first (numbered), label what it does, choosing one:
- "answer": answers a question the assistant asked (gives a detail, chooses an option)
- "confirm": confirms or accepts what the assistant said, proposed or did
- "correct": corrects a mistake of the assistant (a wrong detail, a wrong action) or points out something it skipped
- "repeat": says again or re-asks something the user already said, because the assistant did not act on it, did not respond or got it wrong
- "clarify": asks the assistant to repeat or explain itself ("sorry?", "what do you mean?", "can you say that again?")
- "close": thanks / goodbye
- "other": anything else (a new question, a story, a reaction)
Respond with ONLY a JSON object: {{"labels": ["...", ...]}} (one label per numbered user turn, in order)."""


def controlled(ep: dict) -> str:
    """The role the external policy played (``meta.controlled``; ``agent`` in files written before roles were recorded)."""
    return ep.get("meta", {}).get("controlled") or "agent"


def is_user(t: dict, ep: dict) -> bool:
    """A turn of the other side of the call: any role but the controlled one."""
    return t["role"] != controlled(ep)


def _speech(ep: dict, role: str) -> list[dict]:
    """Directed speech of ``role``: ``"agent"`` is the controlled role, ``"user"`` every other role."""
    me = controlled(ep)
    mine = (lambda t: t["role"] != me) if role == "user" else (lambda t: t["role"] == (me if role == "agent" else role))
    return sorted((t for t in ep.get("turns", []) if mine(t) and t.get("kind") is None and t["end_time"] > t["start_time"]),
                  key=lambda t: t["start_time"])


def _words(text: str) -> set[str]:
    if re.search(r"[一-鿿]", text):  # Mandarin: character bigrams
        cs = [c for c in text if "一" <= c <= "鿿"]
        return {a + b for a, b in zip(cs, cs[1:])}
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP and len(w) > 1}


def metrics(ep: dict, nudge_gap_ms: int = 4000, repeat_overlap: float = 0.6) -> dict:
    users, agents = _speech(ep, "user"), _speech(ep, "agent")
    nudges = 0
    for u0, u1 in zip(users, users[1:]):
        between = any(u0["end_time"] <= a["start_time"] < u1["start_time"] for a in agents)
        if not between and u1["start_time"] - u0["end_time"] >= nudge_gap_ms:
            nudges += 1
    repeats = 0
    for i, u in enumerate(users[1:], 1):
        w = _words(u.get("text", ""))
        if len(w) >= 3 and any(len(w & (p := _words(v.get("text", "")))) / max(1, min(len(w), len(p))) >= repeat_overlap
                               for v in users[:i] if len(_words(v.get("text", ""))) >= 3):
            repeats += 1
    events = ep.get("eval", {}).get("scores", {}).get("events", [])
    cut_ins = sum(e.get("expects") == "respond" and e.get("outcome") == "talked_over" for e in events)
    not_yielded = sum(e.get("expects") == "yield" and e.get("outcome") == "kept_talking" for e in events)
    user_cut = sum("unsaid" in u and any(a["start_time"] <= u["end_time"] <= a["end_time"] for a in agents) for u in users)
    dur = (ep.get("meta", {}).get("duration_ms") or max([t["end_time"] for t in users + agents] or [0])) / 1000
    return {"user_turns": len(users), "agent_turns": len(agents), "duration_s": round(dur, 1),
            "user_turns_per_min": round(len(users) / dur * 60, 2) if dur else None,
            "nudges": nudges, "repeats": repeats, "agent_cut_ins": cut_ins, "user_cut_by_agent": user_cut,
            "barge_ins_not_yielded": not_yielded, "completed": ep.get("meta", {}).get("end_reason") == "idle"}


def dialogue(ep: dict) -> str:
    """USER / ASSISTANT lines in time order, user turns numbered from the second on (for ``LABEL_PROMPT``)."""
    lines, k = [], 0
    for t in sorted((t for t in ep.get("turns", []) if t.get("kind") is None and t.get("text", "").strip()), key=lambda t: t["start_time"]):
        if is_user(t, ep):
            tag = "USER" if k == 0 else f"USER ({k})"
            k += 1
        else:
            tag = "ASSISTANT"
        lines.append(f"{tag}: {t['text'].strip()}")
    return "\n".join(lines)


async def label_user_turns(ep: dict, llm) -> list[str]:
    """One label per user turn after the first (``LABELS``); unknown labels become ``other``."""
    n = len([t for t in ep.get("turns", []) if is_user(t, ep) and t.get("kind") is None and t.get("text", "").strip()]) - 1
    if n <= 0:
        return []
    raw = await llm.chat([{"role": "user", "content": LABEL_PROMPT.format(dialogue=dialogue(ep))}])
    m = re.search(r"\{.*\}", raw, re.S)
    labels = json.loads(m.group(0)).get("labels", []) if m else []
    labels = [x if x in LABELS else "other" for x in labels][:n]
    return labels + ["other"] * (n - len(labels))


def effort(labels: list[str]) -> int:
    return sum(x in EFFORT for x in labels)


def summarize(rows: list[dict]) -> dict:
    """Means of ``metrics`` rows (``completed`` as a rate); with ``labels`` in the rows also label rates and effort."""
    if not rows:
        return {}
    out = {}
    for k in ("user_turns", "agent_turns", "duration_s", "user_turns_per_min", "nudges", "repeats", "agent_cut_ins",
              "user_cut_by_agent", "barge_ins_not_yielded"):
        v = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(sum(v) / len(v), 3) if v else None
    out["completed_rate"] = round(sum(bool(r["completed"]) for r in rows) / len(rows), 3)
    out["calls_with_nudge"] = round(sum(r["nudges"] > 0 for r in rows) / len(rows), 3)
    labelled = [r for r in rows if "labels" in r]
    if labelled:
        n = sum(len(r["labels"]) for r in labelled)
        out["label_share"] = {x: round(sum(r["labels"].count(x) for r in labelled) / max(1, n), 3) for x in LABELS}
        out["effort_per_call"] = round(sum(effort(r["labels"]) for r in labelled) / len(labelled), 3)
    return out
