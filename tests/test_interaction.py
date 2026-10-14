"""Closed-loop interaction metrics (interaction.py)."""

import asyncio
import json

from interaction_gym import interaction as I
from interaction_gym.clients import FakeChat


def _t(i, role, start, end, text, **kw):
    return {"id": i, "role": role, "start_time": start, "end_time": end, "text": text, **kw}


EP = {"meta": {"duration_ms": 30_000, "end_reason": "idle"},
      "turns": [_t("u0", "user", 0, 2000, "Can you book a table for four tonight?"),
                _t("a0", "agent", 2600, 5000, "Sure, what time?"),
                _t("u1", "user", 5500, 6500, "Eight pm."),
                # the agent stays silent: the user prompts it after 6 s
                _t("u2", "user", 12_500, 13_500, "Hello? Are you there?"),
                _t("a1", "agent", 14_000, 18_000, "Sorry! A table for four at eight is booked."),
                _t("u3", "user", 17_000, 17_800, "Wait, a table for four tonight please", unsaid=" at eight"),
                _t("u4", "user", 19_000, 20_000, "Thanks, bye.")],
      "eval": {"scores": {"events": [{"turn": "u3", "expects": "yield", "outcome": "kept_talking"},
                                     {"turn": "u2", "expects": "respond", "outcome": "responded"}]}}}


def test_metrics_from_rules():
    m = I.metrics(EP)
    assert m["user_turns"] == 5 and m["agent_turns"] == 2 and m["duration_s"] == 30.0
    assert m["nudges"] == 1  # u1 -> u2 with no agent turn between, 6 s gap
    assert m["repeats"] == 1  # u3 says most of u0 again
    assert m["user_cut_by_agent"] == 1 and m["barge_ins_not_yielded"] == 1 and m["agent_cut_ins"] == 0
    assert m["completed"] is True


def test_mandarin_repeats_use_character_bigrams():
    ep = {"meta": {}, "turns": [_t("u0", "user", 0, 1000, "我想改一下快递的收件地址"), _t("a0", "agent", 1500, 2000, "好的"),
                                _t("u1", "user", 2500, 3500, "我是说改快递的收件地址")]}
    assert I.metrics(ep)["repeats"] == 1


def test_labels_and_summary():
    llm = FakeChat([json.dumps({"labels": ["answer", "repeat", "correct", "bogus"]})])
    labels = asyncio.run(I.label_user_turns(EP, llm))
    assert labels == ["answer", "repeat", "correct", "other"]
    assert "USER (1): Eight pm." in llm.calls[0][0]["content"] and "ASSISTANT: Sure, what time?" in llm.calls[0][0]["content"]
    s = I.summarize([{**I.metrics(EP), "labels": labels}, {**I.metrics(EP), "completed": False, "labels": ["close"]}])
    assert s["completed_rate"] == 0.5 and s["effort_per_call"] == 1.0 and s["label_share"]["repeat"] == 0.2
