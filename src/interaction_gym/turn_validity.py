"""Open-loop user-turn validity: is a replayed (fixed) user turn still a valid turn against what the agent did?

An open-loop evaluation replays user turns that were written or recorded against some *other* assistant (a reference
dialogue, a recorded call, the base policy's closed-loop call). Once the evaluated agent says something different, a
replayed turn can stop making sense. Each user turn after the first is flagged invalid for any of:

- ``timing``: it starts while the agent is speaking although it was not meant to cut in (``eval.collisions``: in the
  reference it was said after the assistant had finished) — a rule;
- ``content``: it responds to content the agent never said (answers a question that was not asked, accepts an offer
  that was not made, acts as if a step happened that did not) — LLM;
- ``ignored_question``: the agent asked the user something since the user's previous turn and this turn does not
  address it — rule (the agent's speech since then contains a question) AND LLM;
- ``stale``: it gives or asks for information that is already settled (confirmed back, answered, or superseded) and
  adds nothing — LLM.

The LLM part is a checker with its own false-positive rate: run the same checker on closed-loop turns (a live user
who reacts to the agent) and report the open-loop rate *above* that floor. ``prompt`` / ``parse`` build and read the
LLM call (T = 0, cached by the caller); ``rule_flags`` and ``combine`` give the per-turn verdict.
"""

from __future__ import annotations

import json
import re

from .eval import SIDE as NONDIRECTED, collisions

TYPES = ("timing", "content", "ignored_question", "stale")
LLM_TYPES = ("content", "ignored_question", "stale")

PROMPT = """Below is a spoken phone conversation between a CALLER and an ASSISTANT, transcribed up to the moment the caller starts a new line.{context}

{dialogue}

The caller's next line is:
CALLER: "{line}"

Judge this line ONLY against what the ASSISTANT ACTUALLY SAID above (not what a typical assistant would have said). Three questions:
A. unsaid: Does the line respond to, answer, confirm, accept, thank for or correct something the assistant never said or offered? (E.g. it answers a question the assistant did not ask, accepts an offer that was not made, reacts to a price / time / option the assistant never mentioned, or acts as if a step happened that did not.)
B. ignored: If the assistant, since the caller's previous line, asked the caller a question or for some information, does this line fail to address it (neither answering it, nor declining, nor saying they will come to it)? If the assistant asked nothing, answer false. Volunteering other details is fine only if the question is ALSO addressed.
C. stale: Does the line give or ask for information that is already settled in the conversation, adding nothing new? (E.g. repeats a detail the assistant has already confirmed back, re-asks something the assistant already answered, restates a request that was already fulfilled, or states a value that was later changed.) A short "yes" / "that's right" in answer to a confirmation question is NOT stale.

Respond with ONLY a JSON object: {{"assistant_asked": "<the open question, or empty>", "unsaid": true/false, "ignored": true/false, "stale": true/false, "why": "<at most 15 words>"}}"""

_Q = re.compile(r"\?|\b(could you (tell|give|let)|can you (tell|give|let)|may i (have|get|ask)|please (tell|give|let|provide|confirm)|"
                r"let me know|what (is|are|was|would)|which one)\b", re.I)


def _speech(turns: list[dict]) -> list[dict]:
    return sorted((t for t in turns if t.get("text") and t["end_time"] > t["start_time"]), key=lambda t: t["start_time"])


def _heard(t: dict, at: int) -> tuple[str, bool]:
    """The part of a turn heard by time ``at`` (words in proportion to the time spoken) and whether it was cut there."""
    if t["end_time"] <= at:
        return t["text"], False
    words = t["text"].split()
    k = max(1, round(len(words) * (at - t["start_time"]) / (t["end_time"] - t["start_time"])))
    return " ".join(words[:k]), True


def user_turns(turns: list[dict], agent: str = "agent", end_ms: int | None = None) -> list[dict]:
    """The user turns to check: directed (not backchannels / asides / noises), with words, after the first one, and
    starting before ``end_ms``."""
    us = [t for t in _speech(turns) if t["role"] != agent and t.get("kind") not in NONDIRECTED]
    return [u for u in us[1:] if end_ms is None or u["start_time"] < end_ms]


def dialogue(turns: list[dict], at: int, agent: str = "agent", user_label: str = "CALLER", agent_label: str = "ASSISTANT") -> str:
    """The conversation as heard up to time ``at`` (non-directed sounds left out; a turn still running is cut at ``at``)."""
    lines = []
    for t in _speech(turns):
        if t["start_time"] >= at or t.get("kind") in NONDIRECTED:
            continue
        text, cut = _heard(t, at)
        lines.append(f"{agent_label if t['role'] == agent else user_label}: {text}" + (" [still talking]" if cut else ""))
    return "\n".join(lines) or "(nothing yet)"


def agent_since_previous(turns: list[dict], u: dict, agent: str = "agent") -> str:
    """What the agent said (heard by ``u``'s start) since the previous directed user turn started."""
    us = [t for t in _speech(turns) if t["role"] != agent and t.get("kind") not in NONDIRECTED and t["start_time"] < u["start_time"]]
    t0 = us[-1]["start_time"] if us else -1
    return " ".join(_heard(a, u["start_time"])[0] for a in _speech(turns)
                    if a["role"] == agent and t0 <= a["start_time"] < u["start_time"])


def rule_flags(turns: list[dict], agent: str = "agent", intended=None, end_ms: int | None = None) -> dict[str, dict]:
    """Per checked user turn: ``timing`` (an unintended collision) and ``question_open`` (the agent's speech since the
    user's previous turn contains a question: gates ``ignored_question``)."""
    col = collisions(turns, agent, intended)
    out = {}
    for u in user_turns(turns, agent, end_ms):
        x = col.get(u["id"])
        out[u["id"]] = {"timing": x is not None and not x["intended"], "question_open": bool(_Q.search(agent_since_previous(turns, u, agent)))}
    return out


def prompt(turns: list[dict], u: dict, agent: str = "agent", context: str = "") -> str:
    return PROMPT.format(context=(" " + context) if context else "", dialogue=dialogue(turns, u["start_time"], agent), line=u["text"])


def parse(raw: str | None) -> dict | None:
    """The checker's reply → {"content", "ignored_question", "stale"} (bools), or None if unreadable."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0) if m else raw)
    except Exception:  # noqa: BLE001
        return None
    b = lambda k: j.get(k) is True or str(j.get(k)).lower() == "true"  # noqa: E731
    return {"content": b("unsaid"), "ignored_question": b("ignored"), "stale": b("stale")}


def combine(rule: dict, llm: dict | None) -> dict:
    """One turn's verdict: rule flags + LLM flags (``ignored_question`` needs both the rule and the LLM)."""
    f = {"timing": rule["timing"], "content": bool(llm and llm["content"]),
         "ignored_question": bool(llm and llm["ignored_question"] and rule["question_open"]), "stale": bool(llm and llm["stale"])}
    f["invalid"] = any(f[k] for k in TYPES)
    f["invalid_llm"] = any(f[k] for k in LLM_TYPES)
    f["llm_ok"] = llm is not None
    return f


def summarize(per_episode: list[list[dict]]) -> dict:
    """Pool turn verdicts (``combine``) of several episodes: the invalid-turn rate overall and by type, and the share of
    episodes with at least one invalid turn (episodes without a checked turn are left out)."""
    flat = [f for ep in per_episode for f in ep]
    eps = [ep for ep in per_episode if ep]
    r = lambda k, xs: sum(f[k] for f in xs) / len(xs) if xs else None  # noqa: E731
    out = {"turns": len(flat), "episodes": len(eps), "invalid": r("invalid", flat), "invalid_llm": r("invalid_llm", flat),
           **{k: r(k, flat) for k in TYPES},
           "episodes_with_invalid": sum(any(f["invalid"] for f in ep) for ep in eps) / len(eps) if eps else None,
           "episodes_with_invalid_llm": sum(any(f["invalid_llm"] for f in ep) for ep in eps) / len(eps) if eps else None}
    return out
