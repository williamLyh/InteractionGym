"""turn_validity: rule flags, prompt, parser and pooling of open-loop user-turn verdicts."""

from interaction_gym import turn_validity as tv


def T(tid, role, start, end, text="x", **kw):
    return {"id": tid, "role": role, "start_time": start, "end_time": end, "text": text, **kw}


TURNS = [
    T("u0", "user", 0, 2000, "I'd like a pizza."),
    T("a0", "agent", 2500, 6500, "Sure. What size would you like? We have small and large."),
    T("u1", "user", 4500, 6000, "Medium is fine."),                        # starts inside a0: a collision
    T("a1", "agent", 7000, 8000, "Got it."),
    T("u2", "user", 9000, 10000, "Thanks!", kind="backchannel"),            # not checked
    T("u3", "user", 11000, 12000, "Yes please.", expects="yield"),
]


def test_rules_and_dialogue():
    r = tv.rule_flags(TURNS)
    assert set(r) == {"u1", "u3"}
    assert r["u1"] == {"timing": True, "question_open": True}  # half of a0 heard: the question was already asked
    assert r["u3"] == {"timing": False, "question_open": False}
    d = tv.dialogue(TURNS, 4500)
    assert d.splitlines()[-1].startswith("ASSISTANT: Sure. What size") and d.endswith("[still talking]")
    assert "Medium is fine." in tv.prompt(TURNS, TURNS[2]) and "Thanks!" not in tv.prompt(TURNS, TURNS[5])
    assert [u["id"] for u in tv.user_turns(TURNS, end_ms=10_000)] == ["u1"]


def test_parse_combine_summarize():
    assert tv.parse('```json\n{"assistant_asked": "size", "unsaid": false, "ignored": true, "stale": "false", "why": "x"}\n```') == \
        {"content": False, "ignored_question": True, "stale": False}
    assert tv.parse("not json") is None
    a = tv.combine({"timing": False, "question_open": False}, {"content": False, "ignored_question": True, "stale": False})
    assert not a["ignored_question"] and not a["invalid"]  # the rule found no question: the LLM flag is dropped
    b = tv.combine({"timing": True, "question_open": True}, None)
    assert b["invalid"] and not b["invalid_llm"] and not b["llm_ok"]
    s = tv.summarize([[a, b], [], [a]])
    assert s["turns"] == 3 and s["episodes"] == 2 and s["invalid"] == 1 / 3 and s["timing"] == 1 / 3 and s["episodes_with_invalid"] == 0.5
