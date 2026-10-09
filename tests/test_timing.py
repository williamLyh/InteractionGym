"""eval.collisions / timing_counts / timing_summary: duplex timing metrics with open-loop collisions kept apart."""

from interaction_gym.eval import collisions, scores, timing_counts, timing_summary


def T(tid, role, start, end, **kw):
    return {"id": tid, "role": role, "start_time": start, "end_time": end, "text": "x", **kw}


TURNS = [
    T("u0", "user", 0, 2000),
    T("a0", "agent", 2500, 9000, unsaid="..."),          # 500 ms reply
    T("u1", "user", 6000, 8000),                          # replayed line inside a0: a collision (a0 cut at 9000: yielded, 3000 ms)
    T("a1", "agent", 9000, 15000),                        # answers u1 1000 ms after it ended
    T("u2", "user", 12000, 14000, expects="yield"),       # an intended barge-in; a1 keeps talking
    T("u3", "user", 16000, 19000),
    T("a2", "agent", 17000, 18000),                       # cuts into u3 (clean turn)
    T("u4", "user", 20000, 21000, kind="backchannel"),
]


def test_collisions_split_by_intent():
    c = collisions(TURNS)
    assert c == {"u1": {"agent": "a0", "intended": False}, "u2": {"agent": "a1", "intended": True}}
    assert collisions(TURNS, intended={"u1"})["u1"]["intended"] and not collisions(TURNS, intended=lambda u: False)["u2"]["intended"]


def test_timing_counts_and_summary():
    c = timing_counts(TURNS, scores(TURNS)["events"])
    assert c["user_turns"] == 4 and c["collision"] == 1 and c["collision_yielded"] == 1 and c["latencies"]["collision_yield"] == [3000]
    assert c["barge_in"] == 1 and c["barge_in_yielded"] == 0
    assert c["cut_in_n"] == 3 and c["cut_in"] == 1 and c["cut_in_collision_n"] == 1 and c["cut_in_collision"] == 0
    assert c["nondirected"] == 1 and c["nondirected_ok"] == 1  # a backchannel while the agent is silent: it stayed silent
    s = timing_summary([c, timing_counts(TURNS)])
    assert s["episodes"] == 2 and s["collisions"] == 2 and s["collision_share"] == 0.25 and s["yield_rate"] == 0.0
    assert s["collision_yield_rate"] == 1.0 and s["cut_in_rate"] == 1 / 3 and s["latency_median_ms"] == 1000 and s["latency_median_clean_ms"] == 500
    assert timing_summary([])["turn_take_rate"] is None
