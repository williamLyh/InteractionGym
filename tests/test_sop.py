"""SOP source format: condition logic, formats (built-in and plugins), spoken SOP tasks and their judge."""

import asyncio
import json

import pytest

import examples.sop_calls as calls
from interaction_gym import sop as S
from interaction_gym import sop_task as st

TREE = {"chain": [
    {"check": "who", "text": "Caller identified.", "by": "ask"},
    {"all": [{"check": "open", "text": "Open then.", "by": "look_up", "on_fail": "Suggest another time."},
             {"any": [{"check": "small", "text": "At most 8.", "by": "look_up"},
                      {"check": "deposit", "text": "Deposit agreed.", "by": "ask"}]}]}]}
SOP = {"name": "t", "rules": ["Be brief."], "actions": {"book": {"description": "Book.", "requires": TREE, "then": ["Confirm."], "otherwise": "Decline."}}}


def test_holds_witnessed_refuted():
    ok = {"who": True, "open": True, "small": False, "deposit": True}
    assert S.holds(TREE, ok) and not S.holds(TREE, {**ok, "deposit": False})
    assert S.witnessed(TREE, ["who", "open", "deposit"], ok)
    assert S.witnessed(TREE, ["who", "deposit", "open"], ok)            # inside an ALL the order is free
    assert not S.witnessed(TREE, ["open", "deposit", "who"], ok)        # the chain's first step came last
    assert not S.witnessed(TREE, ["who", "open", "small"], ok)          # the alternative that was checked does not hold
    assert not S.witnessed(TREE, ["who", "open"], ok)
    bad = {**ok, "deposit": False}
    assert S.refuted(TREE, ["who", "open", "small", "deposit"], bad)
    assert not S.refuted(TREE, ["who", "open", "small"], bad)           # one failing alternative does not rule it out
    assert S.failing(TREE, bad) == ["small", "deposit"]


def test_formats_keep_the_logic():
    s = S.render(SOP, "structured")
    assert "IN THIS ORDER" in s and "ANY ONE" in s and "Be brief." in s
    li = S.render(SOP, "list")
    assert "1. Caller identified." in li and "At least one of: (a) At most 8." in li and " OR (b) Deposit agreed." in li
    code = S.render(SOP, "code")
    assert "(small() or deposit())" in code and "def book():" in code
    mm = S.render(SOP, "mermaid")
    assert mm.count("-->|yes|") == 4 and "flowchart TD" in mm
    with pytest.raises(KeyError):
        S.render(SOP, "no-such-format")


def test_a_format_plugin():
    @S.register_format("one-line")
    def one_line(sop):
        return "; ".join(c["text"] for c in S.checks(sop["actions"]["book"]["requires"]))
    try:
        assert S.render(SOP, "one-line") == "Caller identified.; Open then.; At most 8.; Deposit agreed."
        assert "one-line" in S.formats()
    finally:
        S._FORMATS.pop("one-line")


def test_validate():
    with pytest.raises(ValueError):
        S.validate({"actions": {"a": {"requires": {"all": [], "any": []}}}})
    with pytest.raises(ValueError):
        S.validate({"actions": {"a": {"requires": {"check": "x", "text": "t", "by": "guess"}}}})


def test_scenarios_are_consistent():
    for name, sc in calls.SCENARIOS.items():
        t = calls.make_task(name, sc)
        act = sc["sop"]["actions"][sc["sop_case"]["action"]]
        assert {c["check"] for c in S.checks(act["requires"])} == set(sc["sop_case"]["truth"]), name
        assert st.should_do(t) == bool(sc["sop_case"].get("expect")), name   # cases to do say what a correct outcome has
        for fmt in ("structured", "list", "code", "mermaid"):
            assert sc["sop_case"]["facts"] in st.agent_instructions(t, fmt)
        p = st.policy(t)
        assert p["target_action"] == sc["sop_case"]["action"] and p["source"] == sc["sop"]


class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)

    async def chat(self, messages, **kw):
        return self.replies.pop(0)


def _ep(turns):
    return {"turns": [{"id": i, "role": r, "text": x} for i, r, x in turns]}


def test_judge_scores_a_correct_decline_and_a_skipped_step():
    t = calls.make_task("booking-mina", calls.SCENARIOS["booking-mina"])
    ep = _ep([("u0", "user", "Table for ten Saturday 8pm."), ("a0", "agent", "For ten we need a deposit of 20 euros each."),
              ("u1", "user", "We can't pay a deposit."), ("a1", "agent", "Then I can't book it; call our events line 020 7946 0321.")])
    report = {"decision": "declined", "decision_turn": "a1", "decline_reason": "deposit_agreed", "offered_alternative": True, "committed": {},
              "established": [{"check": "request_details", "turn": "u0"}, {"check": "party_of_8_or_fewer", "turn": "a0"},
                              {"check": "deposit_agreed", "turn": "u1"}]}
    res = asyncio.run(st.judge(ep, t, FakeLLM(json.dumps(report))))
    assert res["reward"]["total"] == 1.0 and not res["should_do"]
    report["established"] = [{"check": "request_details", "turn": "u0"}]       # declined without establishing why
    res = asyncio.run(st.judge(ep, t, FakeLLM(json.dumps(report))))
    assert res["reward"]["parts"] == {"decision_correct": 1.0, "procedure_followed": 0.0, "details_correct": 1.0}


def test_judge_checks_details_when_done():
    t = calls.make_task("booking-alex", calls.SCENARIOS["booking-alex"])
    ep = _ep([("u0", "user", "Tonight at 7 for two."), ("a0", "agent", "19:00 is free. Name and phone?"), ("u1", "user", "Alex Chen, 07700 900123."),
              ("a1", "agent", "Booked: tonight 7pm, two people, Alex Chen.")])
    est = [{"check": c, "turn": "a0"} for c in ("request_details", "opening_hours", "party_of_8_or_fewer", "table_free")] + [{"check": "name_and_phone", "turn": "u1"}]
    report = {"decision": "done", "decision_turn": "a1", "established": est, "committed": {"day": "tonight", "time": "7pm", "people": "two", "name": "Alex Chen"}}
    match = {"match": {"day": True, "time": True, "people": True, "name": True}}
    res = asyncio.run(st.judge(ep, t, FakeLLM("```json\n" + json.dumps(report) + "\n```", json.dumps(match))))
    assert res["reward"]["total"] == 1.0
    res = asyncio.run(st.judge(ep, t, FakeLLM(json.dumps({**report, "established": est[:1]}), json.dumps(match))))
    assert res["reward"]["parts"]["procedure_followed"] == 0.0                # confirmed without asking for a name and phone
    res = asyncio.run(st.judge(ep, t, FakeLLM(json.dumps({**report, "established": est[:1] + est[-1:]}), json.dumps(match))))
    assert res["reward"]["total"] == 1.0                                      # the free tables are on its screen: need not be said


def test_known_checks_need_not_be_established_and_do_not_constrain_order():
    ok = {"who": True, "open": True, "small": True, "deposit": False}
    assert not S.witnessed(TREE, ["who"], ok)
    assert S.witnessed(TREE, ["who"], ok, known={"open", "small"})
    assert S.witnessed(TREE, ["who"], ok, known={"open", "small", "who"})
    assert not S.witnessed(TREE, [], {**ok, "small": False}, known={"open", "small"})   # deposit was never agreed
    assert S.refuted(TREE, ["deposit"], {**ok, "small": False}, known={"small"})


def test_a_chinese_sop_is_framed_and_rendered_in_chinese():
    zh = {**SOP, "lang": "zh"}
    assert "以下步骤，按顺序：" in S.render(zh, "structured") and "以下步骤" not in S.render(SOP, "structured")
    assert "至少满足其一：" in S.render(zh, "list") and S.render(zh, "mermaid").count("-->|是|") == 4
    t = calls.make_task("zh-hotel-he", calls.SCENARIOS["zh-hotel-he"])
    ins = st.agent_instructions(t)
    assert "标准作业流程" in ins and "你能看到的信息" in ins and "Follow this" not in ins
