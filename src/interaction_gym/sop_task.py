"""Spoken SOP tasks: a caller wants something done; the agent must follow an SOP (``interaction_gym.sop``) before it
acts, or decline. For agents that only talk (e.g. a full-duplex speech model without tool calls): the information the
agent would look up is in its instructions (what a front desk sees on screen), the rest it must ask the caller.

A task carries, in ``task.scenario``:

- the user: ``persona``, ``profile``, ``first_turn``, ``instructions`` (the caller's goal and the details they know);
- ``sop``: the SOP source (``sop.validate``); ``sop_case``: ``{"action", "facts", "truth", "expect"}``: the action
  the caller asks for, the facts the agent can see (text), the ground truth of every check of that action
  (``{check id: holds?}``, which decides whether the action should be done) and, when it should, the details a
  correct booking has (``{field: value}``);
- ``agent_role``: who the agent is; ``categories``: ``["sop"]``.

``agent_instructions(task, fmt)`` is the agent's prompt: role, the SOP rendered in format ``fmt`` (``sop.render``),
the facts, and a phone-call note. ``policy(task, fmt)`` is the same SOP for ``meta.policy``.

**Scoring** (``judge``): an LLM reads the transcript and reports, without judging, what happened: the decision
(done / declined / unfinished) and its turn, which conditions the agent established and when (asked the caller and
got the answer, or stated the information it has), the reason it gave for declining, the details it committed to
and whether it offered the SOP's alternative. Rules on the SOP tree then score it (``VERIFIER``): the decision
against the ground truth, the procedure (``sop.witnessed`` / ``sop.refuted`` on what was established before the
decision; ``look_up`` conditions are on the agent's screen, so they count as known without being said) and the details or alternative. Reward 1 only if all three hold.
"""

from __future__ import annotations

import json
import re

from . import sop as S
from .core import Task

SPOKEN_NOTE = ("This is a phone call: what you write is spoken to the caller. Speak briefly and naturally, one or two short "
               "sentences at a time; no lists or markdown. Ask for what you need one or two items at a time.")

# the agent's instructions around the SOP, by the SOP's "lang"
FRAME = {
    "en": {"follow": "Follow this standard operating procedure (SOP). Do an action only when its conditions hold; otherwise decline as it says.",
           "screen": "Information you can see (your screen):", "note": SPOKEN_NOTE},
    "zh": {"follow": "请按照下面的标准作业流程（SOP）办事：只有条件都满足时才办理；不满足时按流程说明婉拒。",
           "screen": "你能看到的信息（你的屏幕）：",
           "note": "这是一通电话：你写的内容会被读给来电者听。请全程只用普通话，说话简短自然，每次一两句；不要列表或格式符号。需要的信息一次只问一两项。"},
}

VERIFIER = {
    "name": "SOP transcript check (LLM extraction + rules)",
    "description": ("After the call, an LLM reads the transcript and reports what happened, without scoring it: the agent's "
                    "decision and when, which SOP conditions it established before deciding (asked the caller and got the "
                    "answer, or stated the information it has), the reason it gave for declining, the details it committed to. "
                    "Rules on the SOP's condition tree then score it against the case's ground truth. Reward 1 only if all "
                    "three checks pass."),
    "checks": {
        "decision_correct": "the agent did the action exactly when the SOP allows it for this caller, and declined otherwise",
        "procedure_followed": ("done: every required condition the agent must ask about (a holding alternative of each ANY; IN ORDER "
                               "steps in order) was established before it committed; what it looks up on its screen counts as known. Declined: a failing condition that rules the action out "
                               "was established, and given as the reason"),
        "details_correct": ("done: what the agent committed to matches the caller's request and the facts. Declined: it offered "
                            "the SOP's alternative, where the failing condition has one. Never for a wrong decision"),
    },
}


def action_of(task: Task) -> tuple[str, dict]:
    case = task.scenario["sop_case"]
    return case["action"], task.scenario["sop"]["actions"][case["action"]]


def should_do(task: Task) -> bool:
    _, act = action_of(task)
    return S.holds(act.get("requires"), task.scenario["sop_case"]["truth"])


def agent_instructions(task: Task, fmt: str = "structured") -> str:
    sc = task.scenario
    facts = sc["sop_case"].get("facts", "")
    f = FRAME.get(sc["sop"].get("lang", "en"), FRAME["en"])
    return "\n\n".join(x for x in (
        sc.get("agent_role", ""),
        f["follow"] + "\n\n" + S.render(sc["sop"], fmt),
        f"{f['screen']}\n{facts}" if facts else "",
        f["note"]) if x)


def policy(task: Task, fmt: str = "structured") -> dict:
    """``meta.policy``: the SOP as the agent got it (``text``), the requested action's own part (``target``), and the
    source tree itself (``source``), so the trajectory can be re-rendered or re-checked."""
    sc = task.scenario
    name, act = action_of(task)
    one = {**sc["sop"], "rules": [], "actions": {name: act}}
    return {"name": sc["sop"].get("name", "SOP"), "format": fmt, "target_action": name, "target": S.render(one, fmt),
            "text": S.render(sc["sop"], fmt), "source": sc["sop"], "case": sc["sop_case"],
            "note": "In the agent's instructions (meta.env.instructions); nothing enforces it, the verifier checks it after the call."}


# ---------------------------------------------------------------- judging

JUDGE = """You read the transcript of a phone call between a caller and an agent who must follow an SOP. Report what happened; do not judge whether it was right.

The caller asked for: {action} ({description})
The SOP conditions for it (id: condition [how the agent establishes it]):
{conditions}
Alternatives the SOP offers when a condition fails:
{alternatives}
Detail fields of a booking / outcome: {fields}

Transcript (turn id, speaker, text):
{transcript}

Answer with one JSON object and nothing else:
{{"decision": "done" | "declined" | "unfinished",   // done: the agent told the caller the action is done or confirmed; declined: it said it cannot be done; unfinished: neither
 "decision_turn": "<agent turn id where it confirmed or declined, or null>",
 "established": [{{"check": "<condition id>", "turn": "<for [ask]: the CALLER turn that gave the answer; for [look_up]: the agent turn that stated it>", "evidence": "<the words that establish it, quoted>", "holds": true | false}}],
   // conditions the agent established, in order: [ask] = the caller said it (asked or volunteered) and the agent acknowledged or used it; [look_up] = the agent stated or clearly used the information it has. A condition that names several items (e.g. a name AND a phone number) is established only when every item was actually said; a cut-off or partial answer does not count. Leave out conditions that were never established.
 "decline_reason": "<condition id the agent gave as the reason for declining, or null>",
 "offered_alternative": true | false,   // did it offer the SOP's alternative for the failing condition
 "committed": {{"<field>": "<value the agent confirmed, or null>"}}}}"""

DETAILS = """Does what the agent confirmed match what a correct outcome has? For each field, answer true if they mean the same thing for this call: equivalent forms are equal ("7pm" = "19:00", "two" = 2, "tonight" = today's date), the confirmed value need not repeat what is clear from the call (e.g. a weekday without its date), and where the correct value lists alternatives ("X, or Y"), any one of them matches. Answer false if it is different, or missing (null).
Correct: {expect}
Confirmed by the agent: {committed}
Answer with one JSON object: {{"match": {{"<field>": true | false}}}}"""


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON in judge reply: {text[:200]!r}")
    return json.loads(re.sub(r"//[^\n]*", "", m.group(0)))


def transcript(ep: dict) -> tuple[str, dict[str, str]]:
    """The spoken turns as "[a3] AGENT: ..." lines (short aliases: agent turn ids can be long), and alias -> turn id."""
    lines, alias, n = [], {}, {"agent": 0, "user": 0}
    me = ep.get("meta", {}).get("controlled") or "agent"  # the role under evaluation; every other role is the caller side
    for t in ep["turns"]:
        if not t.get("text", "").strip() or t.get("kind"):
            continue
        r = "agent" if t["role"] == me else "user"
        a = f"{r[0]}{n[r]}"
        n[r] += 1
        alias[a] = t["id"]
        lines.append(f"[{a}] {'AGENT' if r == 'agent' else 'CALLER'}: {t['text'].strip()}")
    return "\n".join(lines), alias


async def judge(ep: dict, task: Task, llm) -> dict:
    """``{"reward": {"total", "parts"}, "report": the LLM's report, "established", "should_do", ...}`` for an episode
    (schema v1). ``llm``: an ``OpenAIChat``-like client (``await llm.chat(messages)``)."""
    case = task.scenario["sop_case"]
    name, act = action_of(task)
    truth, req = case["truth"], act.get("requires")
    conds = S.checks(req)
    alts = [f"- {c['check']}: {c['on_fail']}" for c in conds if c.get("on_fail")] + ([f"- otherwise: {act['otherwise']}"] if act.get("otherwise") else [])
    prompt = JUDGE.format(action=name, description=act.get("description", ""),
                          conditions="\n".join(f"- {c['check']}: {c['text']} [{c.get('by', 'look_up')}]" for c in conds),
                          alternatives="\n".join(alts) or "(none)", fields=", ".join(case.get("expect", {})) or "(none)",
                          transcript=(tx := transcript(ep))[0])
    report = _json(await llm.chat([{"role": "user", "content": prompt}]))
    alias = tx[1]
    for k in ("decision_turn",):  # back to the episode's turn ids
        report[k] = alias.get(report.get(k), report.get(k))
    for e in report.get("established", []):
        e["turn"] = alias.get(e.get("turn"), e.get("turn"))
    order = {t["id"]: i for i, t in enumerate(ep["turns"])}
    cut = order.get(report.get("decision_turn"), len(order))
    known = {c["check"] for c in conds}
    # an [ask] condition is established by the caller's answer: an agent turn (its own question left unanswered, or the
    # agent asserting the fact) does not count; nor does an entry with no turn of this call (DuplexInteractionRL
    # docs/review/reward_audit_r3r4.md: both were paid in training)
    # The agent turn right after a caller turn still counts: judges cite the agent acknowledging the answer (r4 review:
    # 40-90 % of their ask citations), and only a turn chained onto the agent's own turn can be its unanswered question.
    role = {t["id"]: t["role"] for t in ep["turns"]}
    by = {c["check"]: c.get("by", "look_up") for c in conds}
    said = sorted((t for t in ep["turns"] if t.get("kind") is None), key=lambda t: (t.get("start_time", 0), order[t["id"]]))
    after_caller = {b["id"] for a, b in zip(said, said[1:]) if a["role"] == "user" and b["role"] == "agent"}
    est = [e["check"] for e in report.get("established", []) if e.get("check") in known and e.get("turn") in order
           and order[e["turn"]] <= cut and (by[e["check"]] != "ask" or role.get(e["turn"]) != "agent" or e["turn"] in after_caller)]
    want = S.holds(req, truth)
    dec = report.get("decision")
    decision_ok = dec == ("done" if want else "declined")
    seen = {c["check"] for c in conds if c.get("by", "look_up") == "look_up"}  # on the agent's screen: known, need not be spoken
    if dec == "done":
        procedure_ok = S.witnessed(req, est, truth, seen)
    elif dec == "declined":
        procedure_ok = S.refuted(req, est, truth, seen) and report.get("decline_reason") in S.failing(req, truth)
    else:
        procedure_ok = False
    match = {}
    if dec == "done" and case.get("expect"):
        match = _json(await llm.chat([{"role": "user", "content": DETAILS.format(expect=json.dumps(case["expect"], ensure_ascii=False),
                                                                                committed=json.dumps(report.get("committed", {}), ensure_ascii=False))}])).get("match", {})
        details_ok = bool(match) and all(match.get(k) is True for k in case["expect"])
    elif dec == "declined":
        has_alt = any(c.get("on_fail") for c in conds if c["check"] in S.failing(req, truth))
        details_ok = bool(report.get("offered_alternative")) or not has_alt
    else:
        details_ok = False
    details_ok = details_ok and decision_ok  # the details of a wrong decision are not right (a wrong decline offered no alternative)
    parts = {"decision_correct": float(decision_ok), "procedure_followed": float(procedure_ok), "details_correct": float(details_ok)}
    return {"reward": {"total": float(all(parts.values())), "parts": parts}, "should_do": want, "decision": dec,
            "established": est, "failing": S.failing(req, truth), "details_match": match, "report": report}
