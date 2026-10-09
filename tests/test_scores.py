"""eval.scores: rule-based scores of how the agent met the user's expectations."""

import pytest

from interaction_gym.eval import latency_score, scores


def T(tid, role, start, end, **kw):
    return {"id": tid, "role": role, "start_time": start, "end_time": end, "text": "x", **kw}


def ev(sc, turn, expects=None):
    e = next(e for e in sc["events"] if e["turn"] == turn)
    assert expects is None or e["expects"] == expects
    return e


def test_latency_curve():
    # logistic in ms, half score at 950 ms, width 100 ms (the table in latency_score's docstring)
    table = {0: 1.000, 200: 0.999, 500: 0.989, 700: 0.924, 800: 0.818, 900: 0.622, 1000: 0.378, 1200: 0.076, 1500: 0.004}
    assert {ms: round(latency_score(ms), 3) for ms in table} == table
    assert latency_score(950) == 0.5 and latency_score(-100) == latency_score(0) and latency_score(10**7) == 0.0
    xs = [0, 100, 300, 600, 900, 1100, 2000, 5000]
    assert all(latency_score(a) > latency_score(b) for a, b in zip(xs, xs[1:]))  # monotone: earlier is never worse


def test_every_kind_of_expectation_is_scored():
    turns = [
        T("u0", "user", 0, 2000),                                   # question -> respond (agent at 2200: 200 ms, ~full score)
        T("a0", "agent", 2200, 8000, unsaid="..."),                 # cut at 8000
        T("u1", "user", 3000, 3300, kind="backchannel"),            # agent keeps talking -> ignored
        T("u2", "user", 7500, 9000),                                # barge-in: agent stops 500 ms later, answers at 10 s
        T("a1", "agent", 10000, 12000),
        T("u3", "user", 13000, 14000, expects="wait"),              # pause mid-thought ...
        T("a2", "agent", 14500, 15000),                             # ... agent jumps in -> took_floor
        T("u4", "user", 16000, 18000),                              # user goes on
        T("a3", "agent", 17000, 19000),                             # agent cuts into u4 -> talked_over
        T("u5", "user", 25000, 26000),                              # never answered -> no_response
    ]
    sc = scores(turns)
    assert [e["turn"] for e in sc["events"]] == ["u0", "u1", "u2", "u3", "u4", "u5"]   # exactly one event per user turn
    assert ev(sc, "u0", "respond") == {"turn": "u0", "expects": "respond", "outcome": "responded", "latency_ms": 200, "score": pytest.approx(1.0, abs=1e-3), "target": "a0",
                                       "resolved_ms": 2200, "user_start_ms": 0, "user_end_ms": 2000}
    assert (ev(sc, "u1", "ignore")["outcome"], ev(sc, "u1")["score"]) == ("ignored", 1.0)
    y = ev(sc, "u2", "yield")   # graded by the stop latency; answering afterwards is required but not graded
    assert y["outcome"] == "yielded" and y["latency_ms"] == 500 and 0.98 < y["score"] < 0.99 and y["target"] == "a0"
    assert (y["reply"], y["response_ms"], y["resolved_ms"]) == ("a1", 1000, 10000)
    assert ev(sc, "u3", "wait")["outcome"] == "took_floor" and ev(sc, "u3")["score"] == 0.0
    assert ev(sc, "u4", "respond")["outcome"] == "talked_over" and ev(sc, "u4")["score"] == 0.0 and ev(sc, "u4")["target"] == "a3"
    assert ev(sc, "u5", "respond")["outcome"] == "no_response"
    assert set(sc["by_expectation"]) == {"respond", "ignore", "yield", "wait"}
    assert sc["total"] == round(sum(e["score"] for e in sc["events"]) / 6, 4)   # the mean over user turns


def test_failures_score_zero():
    turns = [T("a0", "agent", 0, 6000), T("u0", "user", 1000, 3000),           # barge-in, agent never stops
             T("u1", "user", 4000, 4200, kind="noise")]                         # the agent was cut by... nothing: continues
    sc = scores(turns)
    assert ev(sc, "u0", "yield")["outcome"] == "kept_talking" and ev(sc, "u0")["score"] == 0.0
    turns = [T("u0", "user", 0, 1000), T("a0", "agent", 1300, 2000), T("u1", "user", 3000, 3300, kind="aside"),
             T("a1", "agent", 3500, 4500)]                                      # replies to an aside
    assert ev(scores(turns), "u1", "ignore")["outcome"] == "replied"
    assert scores([])["total"] is None


def test_a_yield_must_be_followed_by_taking_the_floor():
    base = [T("u0", "user", 0, 1000), T("a0", "agent", 1200, 6000, unsaid="...")]
    stop_only = base + [T("u1", "user", 5800, 7000)]                            # stops 200 ms in, never speaks again
    e = ev(scores(stop_only), "u1", "yield")
    assert (e["outcome"], e["score"], e["latency_ms"]) == ("no_response", 0.0, 200) and "target" not in e
    e = ev(scores(stop_only, end_ms=9000), "u1", "yield")                       # the episode ended in the window: the stop alone
    assert (e["outcome"], e["score"]) == ("yielded", pytest.approx(1.0, abs=1e-3))
    talks_again = stop_only + [T("a1", "agent", 6500, 8000)]                   # stops, then talks over the rest of the barge-in
    assert ev(scores(talks_again), "u1", "yield")["outcome"] == "talked_over"


def test_a_barge_in_that_trails_off_is_yielded_to_and_waited_through():
    turns = [T("u0", "user", 0, 1000), T("a0", "agent", 1200, 6000, unsaid="..."),
             T("u1", "user", 5800, 7000, expects="yield", text="Actually, wait..."), T("u2", "user", 9000, 10000, text="Make it Sunday.")]
    e = ev(scores(turns + [T("a1", "agent", 10300, 12000)]), "u1", "yield")
    assert (e["outcome"], e["score"]) == ("yielded", pytest.approx(1.0, abs=1e-3))                      # stopped, then let the user go on
    assert ev(scores(turns + [T("a1", "agent", 7500, 8500)]), "u1", "yield")["outcome"] == "took_floor"


def test_hold_on_while_the_agent_talks():
    hold = [T("u0", "user", 0, 1000), T("u1", "user", 2000, 2600, expects="wait", text="Hold on.")]
    e = ev(scores(hold + [T("a0", "agent", 1300, 2400, unsaid="...")]), "u1", "wait")  # stops 400 ms into "hold on"
    assert (e["outcome"], e["score"], e["target"], e["latency_ms"]) == ("waited", 1.0, "a0", 400)
    assert ev(scores(hold + [T("a0", "agent", 1300, 5000)]), "u1", "wait")["outcome"] == "kept_talking"
    assert ev(scores(hold + [T("a0", "agent", 1300, 2400, unsaid="..."), T("a1", "agent", 3000, 4000)]), "u1", "wait")["outcome"] == "took_floor"


def test_a_user_that_barges_in_expects_the_agent_to_yield():
    import asyncio

    from interaction_gym.traj import episode
    from examples import user_modes as um

    ep = episode(asyncio.run(um.run("script")), "e")  # the scripted user cuts in when the agent says "four"
    cut_in = [t for t in ep["turns"] if t["role"] == "user" and t.get("expects") == "yield"]
    assert cut_in and any(e["expects"] == "yield" for e in ep["eval"]["scores"]["events"])


def test_unanswered_turn_at_the_episode_end_is_censored():
    turns = [T("u0", "user", 0, 2000), T("a0", "agent", 2500, 3000), T("u1", "user", 4000, 9800)]
    sc = scores(turns, end_ms=10000)
    assert ev(sc, "u1", "respond") == {"turn": "u1", "expects": "respond", "outcome": "censored", "user_start_ms": 4000, "user_end_ms": 9800}
    assert sc["by_expectation"]["respond"] == round(ev(sc, "u0", "respond")["score"], 4)   # u1 left out of the mean
    assert ev(scores(turns, end_ms=20000), "u1", "respond")["outcome"] == "no_response"
    assert ev(scores(turns), "u1", "respond")["outcome"] == "no_response"       # no end given: as before


def test_replying_to_an_aside_after_finishing_the_answer():
    turns = [T("u0", "user", 0, 1500), T("a0", "agent", 3000, 8000), T("u1", "user", 5900, 7500, kind="aside"),
             T("a1", "agent", 9000, 12000)]                                     # "Oh no, are you okay?" 1 s after a0 ended
    e = ev(scores(turns), "u1", "ignore")
    assert e["outcome"] == "replied" and e["score"] == 0.0 and e["target"] == "a1"
    turns[-1] = T("a1", "agent", 11000, 12000)                                   # 3 s later: not a reaction
    assert ev(scores(turns), "u1", "ignore")["outcome"] == "ignored"
    turns[-1] = T("u2", "user", 8500, 9000)                                      # the user asked something in between
    assert ev(scores(turns + [T("a1", "agent", 9200, 12000)]), "u1", "ignore")["outcome"] == "ignored"


def test_a_turn_cut_by_the_episode_end_is_not_a_yield():
    turns = [T("u0", "user", 0, 2000), T("a0", "agent", 2500, 20000, unsaid="..."), T("u1", "user", 6000, 8000)]
    assert ev(scores(turns, end_ms=20000), "u1", "yield")["outcome"] == "kept_talking"
    e = ev(scores(turns, end_ms=30000), "u1", "yield")  # cut before the end: a real stop (but never followed by a reply)
    assert e["outcome"] == "no_response" and e["latency_ms"] == 14000


def test_backchannel_then_agent_goes_on_is_not_a_reply():
    from interaction_gym.eval import duplex, scores
    turns = [
        {"id": "u0", "role": "user", "start_time": 0, "end_time": 1000},
        {"id": "a0", "role": "agent", "start_time": 1500, "end_time": 4000},
        {"id": "u1", "role": "user", "start_time": 2000, "end_time": 2400, "kind": "backchannel"},
        {"id": "a1", "role": "agent", "start_time": 4350, "end_time": 6000},
    ]
    assert duplex(turns)["u1"]["reaction"] == "continued"
    ev = next(e for e in scores(turns)["events"] if e["turn"] == "u1")
    assert ev["outcome"] == "ignored" and ev["score"] == 1.0
    aside = [dict(t, kind="aside") if t["id"] == "u1" else t for t in turns]
    assert duplex(aside)["u1"]["reaction"] == "responded"


def test_duplex_agent_cut_by_episode_end_did_not_yield():
    from interaction_gym.eval import duplex
    turns = [
        {"id": "u0", "role": "user", "start_time": 0, "end_time": 1000},
        {"id": "a0", "role": "agent", "start_time": 1200, "end_time": 8000, "unsaid": "rest of it"},
        {"id": "u1", "role": "user", "start_time": 3000, "end_time": 4000},
    ]
    assert duplex(turns)["u1"]["reaction"] == "yielded"
    assert duplex(turns, end_ms=8000)["u1"]["reaction"] == "kept_talking"


def test_position_fields_locate_every_event():
    """Position fields (user turn span, resolution time, agent turn involved) never change a score."""
    turns = [
        T("u0", "user", 0, 2000),
        T("a0", "agent", 2200, 8000, unsaid="..."),
        T("u1", "user", 3000, 3300, kind="backchannel"),
        T("u2", "user", 7500, 9000),
        T("a1", "agent", 10000, 12000),
        T("u3", "user", 13000, 14000, expects="wait"),
        T("a2", "agent", 14500, 15000),
        T("u4", "user", 16000, 18000),
        T("a3", "agent", 17000, 19000),
        T("u6", "user", 24000, 24400, kind="aside"),                 # heard in silence, answered by a4
        T("a4", "agent", 25000, 26000),
        T("u5", "user", 28000, 29000),
    ]
    sc = scores(turns)
    for e in sc["events"]:
        u = next(t for t in turns if t["id"] == e["turn"])
        assert (e["user_start_ms"], e["user_end_ms"]) == (u["start_time"], u["end_time"])
        assert "resolved_ms" in e
    assert ev(sc, "u2", "yield")["resolved_ms"] == 10000 and ev(sc, "u2", "yield")["target"] == "a0"
    w = ev(sc, "u3", "wait")
    assert w["target"] == "a2" and w["resolved_ms"] == 14500 and w["score"] == 0.0
    assert ev(sc, "u4", "respond")["resolved_ms"] == 17000 and ev(sc, "u4")["target"] == "a3"
    i = ev(sc, "u6", "ignore")
    assert i["outcome"] == "replied" and i["target"] == "a4" and i["resolved_ms"] == 25000
    nr = ev(sc, "u5", "respond")
    assert nr["outcome"] == "no_response" and nr["resolved_ms"] == 34000 and "target" not in nr
    # scores unchanged by the new fields
    strip = lambda s: [{k: v for k, v in e.items() if k not in ("user_start_ms", "user_end_ms", "resolved_ms")} for e in s["events"]]  # noqa: E731
    assert all("score" not in e or 0.0 <= e["score"] <= 1.0 for e in strip(sc))
