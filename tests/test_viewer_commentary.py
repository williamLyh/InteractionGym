"""Headlines for the viewer's live playback: derived only from recorded eval fields."""

import json
import re

from interaction_gym.eval import duplex, scores
from interaction_gym.viewer import export_run, headlines, render_html, units


def T(id, role, t0, t1, text="", **kw):
    return {"id": id, "role": role, "start_time": t0, "end_time": t1, "text": text, **kw}


def ep_of(turns, duration=20000, with_scores=True, eid="e"):
    ev = {"reward": {"total": None, "parts": {}}, "duplex": duplex(turns)}
    if with_scores:
        ev["scores"] = scores(turns)
    return {"schema": 1, "meta": {"episode_id": eid, "duration_ms": duration, "end_reason": "idle"}, "turns": turns, "eval": ev}


TURNS = [
    T("u0", "user", 0, 1500, "What's the weather like tomorrow?"),
    T("a0", "agent", 1800, 3400, "Tomorrow will be sunny with ", unsaid="a high of twenty four."),
    T("u1", "user", 3100, 5100, "Sorry, I meant the day after tomorrow."),
    T("a1", "agent", 5600, 9000, "Got it, rainy, bring an umbrella."),
    T("u2", "user", 7000, 7300, "mm-hmm", kind="backchannel"),
    T("u3", "user", 11000, 12500, "And on Sunday?"),
]


def test_one_headline_per_score_event_and_tally_ends_at_total():
    ep = ep_of(TURNS)
    hs = headlines(ep)
    assert len(hs) == len(ep["eval"]["scores"]["events"])
    assert [h["t_resolved"] for h in hs] == sorted(h["t_resolved"] for h in hs)
    assert hs[-1]["tally"] == ep["eval"]["scores"]["total"]
    assert all(h["source"] == "eval.scores" and h["t"] <= h["t_resolved"] for h in hs)


def test_headline_texts_come_from_recorded_fields():
    hs = {(h["turn"], h["expects"]): h for h in headlines(ep_of(TURNS))}
    y = hs["u1", "yield"]
    assert y["text"] == "User barged in — agent yielded in 300 ms, answered 500 ms after the user finished ✓" and y["ok"] and y["latency_ms"] == 300
    assert y["t"] == 3100 and y["t_resolved"] == 5600  # barge-in start -> agent stops, then takes the floor again
    assert y["raw"]["event"]["outcome"] == "yielded" and y["raw"]["target"]["id"] == "a0" and y["raw"]["duplex"]["behavior"] == "barge_in"
    b = hs["u2", "ignore"]
    assert b["text"] == "User said “mm-hmm” (backchannel) — agent kept talking ✓" and b["ok"]
    r = hs["u0", "respond"]
    assert r["text"].startswith("Agent answered “What's the weather") and "after 300 ms" in r["text"] and r["t_resolved"] == 1800
    n = hs["u3", "respond"]  # never answered: the gap is the episode's own end
    assert n["text"] == "No reply to “And on Sunday?” (episode ended 7.5 s later) ✗" and n["ok"] is False and n["score"] == 0.0
    assert y["pending"] == "User barges in while the agent talks…"  # shown before the outcome: no verdict in it


def test_bad_reactions_are_marked():
    turns = [
        T("u0", "user", 0, 1000, "Hi there."),
        T("a0", "agent", 1300, 2600, "Hello! How can", unsaid=" I help you today?"),
        T("u1", "user", 2000, 2300, "mm-hmm", kind="backchannel"),
        T("u2", "user", 7000, 9000, "I want to book something for later."),
        T("a1", "agent", 8000, 10000, "Sure."),
    ]
    hs = {(h["turn"], h["expects"]): h for h in headlines(ep_of(turns))}
    assert hs["u1", "ignore"]["text"] == "User said “mm-hmm” (backchannel) — agent stopped talking ✗"
    assert hs["u2", "respond"]["text"] == "Agent talked over “I want to book something for later.” ✗" and hs["u2", "respond"]["ok"] is False
    assert len(hs) == 3  # one headline per user turn


def test_duplex_only_episodes_get_headlines_without_scores():
    hs = headlines(ep_of(TURNS, with_scores=False))
    assert [h["behavior"] for h in hs] == ["barge_in", "backchannel"]
    assert all("score" not in h and "tally" not in h for h in hs)
    assert hs[0]["text"] == "User barged in — agent yielded in 300 ms ✓"
    assert headlines({"meta": {"episode_id": "x", "duration_ms": 0}, "turns": []}) == []


def test_units_condense_the_trace():
    tr = {"episode_id": "e", "unit_ms": 1000, "units": [
        {"unit_index": -1, "end_ms": 0, "decision": "prompt", "stages": [{"stage": "thinker", "input": [[1, "sys"]], "output": []}]},
        {"unit_index": 0, "end_ms": 1000, "decision": "speak", "special_ids": [9, 8], "stages": [
            {"stage": "thinker", "input": [[5, "<unit>"]], "output": [[9, "<|speak|>"], [11, "ĠSure"], [12, ","], [8, "<|turn_eos|>"]]},
            {"stage": "talker", "input": [], "output": [[4192, "s3:4192"]]}]}]}
    u = units(tr)
    assert u["unit_ms"] == 1000 and u["units"] == [{"i": 0, "t0": 0, "t1": 1000, "decision": "speak", "text": " Sure,", "marks": ["<|speak|>", "<|turn_eos|>"]}]
    assert units(None) is None


def _view(html):
    return json.loads(re.search(r'<script id="view" type="application/json">(.*?)</script>', html, re.S).group(1).replace("<\\/", "</"))


def test_page_carries_view_data_but_not_token_dumps(tmp_path):
    ep = ep_of(TURNS)
    html = render_html([ep])
    assert "__VIEW__" not in html and "__DATA__" not in html
    v = _view(html)["e"]
    assert v["total"] == ep["eval"]["scores"]["total"] and v["units"] is None and len(v["headlines"]) == len(ep["eval"]["scores"]["events"])
    tr = {"episode_id": "e", "unit_ms": 1000, "units": [{"unit_index": 0, "end_ms": 1000, "decision": "listen", "special_ids": [7],
                                                         "stages": [{"stage": "thinker", "input": [[7, "<|audio|>"]] * 10, "output": [[7, "<|listen|>"]]}]}]}
    export_run([ep], tmp_path, traces=[tr])
    env = (tmp_path / "env.html").read_text()
    v = _view(env)["e"]
    assert v["agent_page"] == "e.agent.html" and v["units"]["units"][0]["decision"] == "listen"
    assert "Tokens per unit" not in env and "<|audio|>" not in env  # full token lists stay on the agent page


def test_censored_turns_are_neutral_unscored_and_left_out_of_the_tally():
    turns = TURNS[:-1] + [T("u3", "user", 11000, 12500, "And on Sunday?")]
    ep = ep_of(turns, duration=13000)
    ep["eval"]["scores"] = scores(turns, end_ms=13000)                         # as traj.episode records it
    hs = {(h["turn"], h["expects"]): h for h in headlines(ep)}
    c = hs["u3", "respond"]
    assert c["text"] == "“And on Sunday?” cut off by the episode end — not scored"
    assert c["ok"] is None and c["score"] is None and c["scored"] is False and "tally" not in c and c["t_resolved"] == 13000
    assert c["pending"].startswith("User finished “And on Sunday?”")
    tallied = [h for h in headlines(ep) if "tally" in h]
    assert tallied[-1]["tally"] == ep["eval"]["scores"]["total"] and all(h["scored"] for h in tallied)
    assert ep["eval"]["scores"]["total"] != scores(turns)["total"]            # the censored turn would have scored 0


def _all_outcome_scenarios():
    """(turns, end_ms) that between them make eval.scores emit every outcome it has."""
    S = [(TURNS, None)]                                                      # responded, no_response, yielded, ignored
    S.append(([T("u0", "user", 0, 1000, "Hi there."), T("a0", "agent", 1300, 2600, "Hello! How can", unsaid=" I help?"),
               T("u1", "user", 2000, 2300, "mm-hmm", kind="backchannel"), T("u2", "user", 7000, 9000, "Book it for later."),
               T("a1", "agent", 8000, 10000, "Sure.")], None))                # stopped, talked_over
    S.append(([T("u0", "user", 0, 2000, "Hi."), T("a0", "agent", 2500, 12000, "Long answer", unsaid="..."),
               T("u1", "user", 6000, 8000, "Hang on.")], 12000))               # kept_talking (episode end)
    S.append(([T("u0", "user", 0, 2000, "Tell me a story."), T("a0", "agent", 2300, 9000, "Once upon a time"),
               T("u1", "user", 4000, 5000, "Wait, stop.", expects="wait"), T("u2", "user", 12000, 13000, "Honey, dinner!", kind="aside"),
               T("u3", "user", 20000, 21000, "So, um", expects="wait"), T("a1", "agent", 21500, 23000, "Yes?"),
               T("u4", "user", 26000, 27000, "Go on", kind="aside"), T("a2", "agent", 27500, 28000, "Hm?")], None))   # kept_talking, ignored, took_floor, replied
    S.append(([T("u0", "user", 0, 1500, "Q?"), T("a0", "agent", 3000, 8000, "Answer."), T("u1", "user", 5900, 7500, "Microwave quit", kind="aside"),
               T("a1", "agent", 9000, 12000, "Oh no, are you okay?"), T("u2", "user", 15000, 16000, "Um", expects="wait"),
               T("u3", "user", 17000, 18000, "Interrupt me now.", expects="interrupt"), T("a2", "agent", 17500, 19000, "OK!"),
               T("u4", "user", 22000, 24000, "Interrupt me again.", expects="interrupt")], None))   # replied (reply_after), waited, interrupt cut_in / listened
    S.append(([T("u0", "user", 0, 1000, "Go."), T("a0", "agent", 1200, 6000, "Sure, so", unsaid=" first..."),
               T("u1", "user", 3000, 4000, "Wait.", expects="wait"), T("u2", "user", 8000, 9000, "Carry on.")], None))   # wait over the agent: stopped, waited
    S.append(([T("u0", "user", 0, 1000, "Hi."), T("a0", "agent", 1200, 2000, "Hello."), T("u1", "user", 3000, 9000, "", kind="away", expects="wait"),
               T("u2", "user", 9000, 10000, "Back."), T("a1", "agent", 10300, 11000, "Great."), T("u3", "user", 12000, 18000, "", kind="away", expects="wait"),
               T("a2", "agent", 15000, 16000, "Still there?"), T("u4", "user", 19000, 25000, "", kind="away", expects="wait"),
               T("a3", "agent", 19500, 24000, "So as I was saying")], None))  # away: waited, checked_in, took_floor
    S.append(([T("u0", "user", 0, 1000, "Go."), T("a0", "agent", 1200, 6000, "Sure", unsaid="..."), T("u1", "user", 5800, 7000, "No"),
               T("a1", "agent", 6500, 8000, "Right")], None))                 # yield: talked_over
    S.append(([T("u0", "user", 0, 1000, "Go."), T("a0", "agent", 1200, 6000, "Sure", unsaid="..."), T("u1", "user", 5800, 7000, "No")], None))  # yield: no_response
    S.append(([T("u0", "user", 0, 1000, "Go."), T("a0", "agent", 1200, 6000, "Sure", unsaid="..."), T("u1", "user", 5800, 7000, "Actually..."),
               T("a1", "agent", 7500, 8000, "Yes?"), T("u2", "user", 9000, 10000, "Sunday.")], None))   # yield: took_floor
    S.append(([T("u0", "user", 0, 1000, "Hi."), T("a0", "agent", 1200, 2000, "Hello."), T("u1", "user", 3000, 9800, "And?")], 10000))  # censored
    return S


def test_every_outcome_eval_scores_can_emit_has_a_headline():
    import inspect

    from interaction_gym import eval as ev_mod
    from interaction_gym.viewer.commentary import GOOD, NEUTRAL
    src = inspect.getsource(ev_mod.scores)
    emitted = {w for line in src.splitlines() if "outcome" in line for w in re.findall(r'"(\w+)"(?!\])', line)}  # not ["id"]
    assert {"censored", "no_response", "responded", "talked_over", "yielded", "kept_talking", "stopped", "ignored", "replied",
            "took_floor", "waited", "cut_in", "listened", "checked_in"} <= emitted
    seen = {}
    for turns, end in _all_outcome_scenarios():
        ep = ep_of(turns, duration=end or 30000)
        ep["eval"]["scores"] = scores(turns, end_ms=end)
        hs = headlines(ep)
        assert [h for h in hs if "tally" in h][-1]["tally"] == ep["eval"]["scores"]["total"]
        for h in hs:
            seen.setdefault((h["expects"], h["outcome"]), h)
    assert emitted == {o for _, o in seen}  # every literal is reached
    for (exp, out), h in seen.items():
        assert not h["text"].startswith(f"{exp}: "), h["text"]                  # not the fallback
        assert h["ok"] is (None if (exp, out) in NEUTRAL else (exp, out) in GOOD)
        assert h["t"] <= h["t_resolved"] and h["pending"]
    assert "User cut in with “Wait, stop.” (wait) — agent kept talking ✗" in [h["text"] for h in _hs_of(_all_outcome_scenarios()[3])]
    assert any("until the episode ended" in h["text"] for h in _hs_of(_all_outcome_scenarios()[2]))
    assert "User cut in with “Wait.” (wait) — agent stopped in 3.0 s and waited ✓" in [h["text"] for h in _hs_of(_all_outcome_scenarios()[5])]
    r = next(h for h in _hs_of(_all_outcome_scenarios()[4]) if h["turn"] == "u1")
    assert r["text"] == "User aside “Microwave quit” — agent finished its turn, then replied to it after 1.5 s ✗" and r["t_resolved"] == 9000


def _hs_of(scenario):
    turns, end = scenario
    ep = ep_of(turns, duration=end or 30000)
    ep["eval"]["scores"] = scores(turns, end_ms=end)
    return headlines(ep)
