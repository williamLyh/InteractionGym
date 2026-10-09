"""Synthetic scenarios whose correct outcome is derived by hand (in the comments) and checked exactly.

Every number asserted here is computed from the scenario's definition, not read back from a run:
if the env and the derivation disagree, either the env is wrong or the spec is ambiguous.
"""

import asyncio
import copy
import json
import random
from array import array

import pytest

from interaction_gym import AgentSpec, Background, Env, Frame, Segment, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.audio import Audio
from interaction_gym.core import Chunk, Node
from interaction_gym.media import MediaStore
from interaction_gym.tools import CALL, RESULT, FunctionBackend, ToolCall, ToolWorld
from interaction_gym.traj import agent_view, episode, load, save, to_frames
from interaction_gym.user import ReplayUser, ScriptSource, TurnTaking, UserSim, Voice
from tests.test_traj import check

TASK = Task(id="t", scenario={})


def run(coro):
    return asyncio.run(coro)


async def drive(env, agent, task=TASK, seed=0, record=None):
    """Run an episode to its natural end (or max_ms). ``record`` collects (env time, obs) per step."""
    obs = await env.reset(task, seed=seed)
    if record is not None:
        record.append((env.t, obs))
    done = False
    while not done:
        act = agent.act(env.t, obs)
        if asyncio.iscoroutine(act):
            act = await act
        obs, _, done = await env.step(act)
        if record is not None:
            record.append((env.t, obs))
    return env


def turns_of(ep, role=None):
    return [t for t in ep["turns"] if role is None or t["role"] == role]


class Silent:
    """An agent that never does anything."""

    def act(self, t, obs):
        return []


class Scripted:
    """Emits fixed frames at fixed env times (keys must be step times)."""

    def __init__(self, plan: dict[int, list[Frame]]):
        self.plan = plan

    def act(self, t, obs):
        return self.plan.get(t, [])


# =============================================================== 1. turn timing


def test_text_turn_timing_reply_and_natural_end():
    # user u0: t=0, dur 1000 (given). Agent (chunk 100) sees u0's last chunk in window [900,1000) -> at t=1000.
    # reply_after=300 -> first step t with t - 1000 >= 300 is t=1300. Reply: 10 words at 5 w/s = 2000 ms -> 1300..3300.
    # busy_until = 3300; natural end = first step t >= 3300 + end_idle 3000 -> 6300.
    spec = AgentSpec(chunk_ms=100)
    user = ReplayUser([{"t": 0, "text": "hello there my friend", "dur": 1000}])
    env = run(drive(Env({"user": user}, spec), CannedAgent(["a b c d e f g h i j"], spec, reply_after=300, words_per_sec=5)))
    ep = check(episode(env, "e"))
    assert [(t["role"], t["start_time"], t["end_time"], t["text"]) for t in ep["turns"]] == [
        ("user", 0, 1000, "hello there my friend"),
        ("agent", 1300, 3300, "a b c d e f g h i j"),
    ]
    assert ep["meta"]["duration_ms"] == 6300 and ep["meta"]["end_reason"] == "idle"


def test_audio_turn_timing_from_tts_length():
    # FakeSpeech: silence of round(words / 4 * 1000) ms at 16 kHz -> "one two three four" = 1000 ms exactly.
    # User at t=500 -> 500..1500; agent reply at first step t >= 1500 + 300 = 1800 (chunk 100); 4 words @ 2 w/s = 2000 -> 1800..3800.
    from interaction_gym.clients import FakeSpeech

    spec = AgentSpec(chunk_ms=100)
    user = ReplayUser([{"t": 500, "text": "one two three four"}], voice=Voice(FakeSpeech(words_per_sec=4)))
    env = run(drive(Env({"user": user}, spec), CannedAgent(["w x y z"], spec, words_per_sec=2)))
    ep = check(episode(env, "e"))
    assert [(t["role"], t["start_time"], t["end_time"]) for t in ep["turns"]] == [("user", 500, 1500), ("agent", 1800, 3800)]
    assert ep["meta"]["duration_ms"] == 3800 + 3000


def test_max_duration_truncates_and_records_what_was_left_unsaid():
    # Same as the first scenario but max_ms=2000: the reply 1300..3300 is cut at 2000.
    # Text "a b c d e f g h i j" has 19 chars; at 700/2000 = 0.35 of its duration round(19*0.35)=round(6.65)=7 chars
    # are said: "a b c d" (7 chars); unsaid " e f g h i j".
    spec = AgentSpec(chunk_ms=100)
    user = ReplayUser([{"t": 0, "text": "hello there my friend", "dur": 1000}])
    env = run(drive(Env({"user": user}, spec, max_ms=2000), CannedAgent(["a b c d e f g h i j"], spec, words_per_sec=5)))
    ep = check(episode(env, "e"))
    (a,) = turns_of(ep, "agent")
    assert (a["start_time"], a["end_time"], a["text"], a["unsaid"]) == (1300, 2000, "a b c d", " e f g h i j")
    assert ep["meta"]["duration_ms"] == 2000 and ep["meta"]["end_reason"] == "max_duration" and env.truncated


# =============================================================== 2. barge-in


BARGE_USER = [
    {"t": 0, "text": "please tell me the weather", "dur": 1000},
    {"t": 2000, "text": "stop stop", "dur": 1000},  # a normal turn while the agent speaks 1300..3300
]


def test_barge_in_agent_yields_after_yield_after():
    # Agent reply 1300..3300 ("a b c d e f g h i j", 19 chars). u1 starts at 2000; the agent first sees it at
    # t=2100 (window [2000,2100)); yield_after=160 -> first step with t - 2000 >= 160 is t=2200 -> cut at 2200.
    # Said: round(19 * 900/2000) = round(8.55) = 9 chars = "a b c d e"; unsaid " f g h i j".
    # eval: u1 barge_in -> yielded, latency = 2200 - 2000 = 200.
    spec = AgentSpec(chunk_ms=100)
    env = run(drive(Env({"user": ReplayUser(BARGE_USER)}, spec), CannedAgent(["a b c d e f g h i j"], spec, words_per_sec=5)))
    ep = check(episode(env, "e"))
    (a,) = turns_of(ep, "agent")
    assert (a["start_time"], a["end_time"], a["text"], a["unsaid"]) == (1300, 2200, "a b c d e", " f g h i j")
    assert ep["eval"]["duplex"] == {"u1": {"behavior": "barge_in", "target": a["id"], "reaction": "yielded", "latency_ms": 200}}


def test_barge_in_natural_end_counts_from_the_last_activity_not_the_cut_plan():
    # After the barge-in above, the last thing anyone says is u1 (ends 3000); the agent's cut turn ended at 2200.
    # "No new activity for end_idle_ms" -> the episode should end at 3000 + 3000 = 6000.
    spec = AgentSpec(chunk_ms=100)
    env = run(drive(Env({"user": ReplayUser(BARGE_USER)}, spec), CannedAgent(["a b c d e f g h i j"], spec, words_per_sec=5)))
    assert env.t == 6000


def test_barge_in_agent_that_never_yields_keeps_talking():
    spec = AgentSpec(chunk_ms=100)
    env = run(drive(Env({"user": ReplayUser(BARGE_USER)}, spec),
                    CannedAgent(["a b c d e f g h i j"], spec, words_per_sec=5, yield_after=None)))
    ep = check(episode(env, "e"))
    (a,) = turns_of(ep, "agent")
    assert (a["start_time"], a["end_time"], "unsaid" in a) == (1300, 3300, False)
    assert ep["eval"]["duplex"] == {"u1": {"behavior": "barge_in", "target": a["id"], "reaction": "kept_talking"}}


# =============================================================== 3. non-directed sounds


SOUNDS_USER = [
    {"t": 0, "text": "what is on the menu", "dur": 1000},
    {"t": 2000, "text": "mm-hmm", "kind": "backchannel", "dur": 400},
    {"t": 3000, "text": "uh", "kind": "aside", "dur": 600},
    {"t": 4000, "text": "", "kind": "noise", "dur": 400},
]
LONG_REPLY = " ".join(f"w{i}" for i in range(20))  # 20 words @ 5 w/s = 4000 ms -> 1300..5300


def test_listening_agent_continues_through_backchannel_aside_and_noise():
    # min_words=2: "mm-hmm", "uh" and the noise never reach 2 heard words, so the agent never yields.
    # Every sound starts while the agent speaks (1300..5300) and the agent is not cut -> "continued" x3.
    spec = AgentSpec(chunk_ms=100)
    env = run(drive(Env({"user": ReplayUser(SOUNDS_USER)}, spec), CannedAgent([LONG_REPLY], spec, words_per_sec=5, min_words=2)))
    ep = check(episode(env, "e"))
    (a,) = turns_of(ep, "agent")
    assert (a["start_time"], a["end_time"]) == (1300, 5300)
    assert {k: v["reaction"] for k, v in ep["eval"]["duplex"].items()} == {"u1": "continued", "u2": "continued", "u3": "continued"}


def test_naive_agent_stops_for_a_backchannel():
    # min_words=0: the backchannel (2000..2400) counts as a barge-in -> cut at t=2200 (seen at 2100, 2200-2000 >= 160).
    # u1: agent end 2200 < u1 end 2400 -> "stopped". u2/u3 happen while the agent is silent and it never speaks
    # again -> "stayed_silent" for both.
    spec = AgentSpec(chunk_ms=100)
    env = run(drive(Env({"user": ReplayUser(SOUNDS_USER)}, spec), CannedAgent([LONG_REPLY], spec, words_per_sec=5)))
    ep = check(episode(env, "e"))
    (a,) = turns_of(ep, "agent")
    assert (a["start_time"], a["end_time"]) == (1300, 2200)
    assert {k: v["reaction"] for k, v in ep["eval"]["duplex"].items()} == {"u1": "stopped", "u2": "stayed_silent", "u3": "stayed_silent"}


# =============================================================== 4. the agent talks over the user


def test_agent_interrupting_the_user_makes_the_user_yield():
    # UserSim text-only: "aa bb cc dd ee ff gg hh ii jj" = 10 words @ 3.4 w/s -> round(2941.2) = 2941 ms (0..2941).
    # The agent starts a 3000 ms segment at t=500. The user is talked over; yield_after_ms=1000 -> cut at 1500.
    # Said: round(29 * 1500/2941) = round(14.79) = 15 chars = "aa bb cc dd ee "; unsaid "ff gg hh ii jj".
    # eval: the agent turn starts while u0 plays -> agent_interrupt (target u0).
    spec = AgentSpec(chunk_ms=100)
    user = UserSim(ScriptSource(["aa bb cc dd ee ff gg hh ii jj"]), timing=TurnTaking(yield_after_ms=1000))
    agent = Scripted({500: [Frame("policy.speech", 500, Segment("a0", 500, 3000, "I am talking over you now"))]})
    env = run(drive(Env({"user": user}, spec), agent))
    ep = check(episode(env, "e"))
    (u,) = turns_of(ep, "user")
    assert (u["start_time"], u["end_time"], u["text"], u["unsaid"]) == (0, 1500, "aa bb cc dd ee ", "ff gg hh ii jj")
    assert ep["eval"]["duplex"] == {"a0": {"behavior": "agent_interrupt", "target": "u0"}}


# =============================================================== 5. streaming observations


def observed_chunks(record, stream="user.speech"):
    """(step time, chunk) for every observed chunk on ``stream``."""
    return [(t, f.data) for t, obs in record for f in obs if f.stream == stream and isinstance(f.data, Chunk)]


def check_stream_invariants(record, final_turns, chunk_ms, stream="user.speech"):
    """The agent only ever sees what has been played: every chunk lies inside the step window that just ended,
    windows of a segment are contiguous, the first is flagged first, the last ends at the segment's (final) end and
    is flagged last, and the chunk texts concatenate to exactly what was said."""
    by_id: dict[str, list] = {}
    for t, c in observed_chunks(record, stream):
        assert t - chunk_ms <= c.t0 and c.t0 + c.dur <= t, (t, c)  # never ahead of the clock
        by_id.setdefault(c.id, []).append(c)
    for turn in final_turns:
        cs = by_id[turn["id"]]
        assert cs[0].first and cs[0].t0 == turn["start_time"]
        assert all(not c.first for c in cs[1:])
        assert all(a.t0 + a.dur == b.t0 for a, b in zip(cs, cs[1:]))  # contiguous
        assert cs[-1].last and cs[-1].t0 + cs[-1].dur == turn["end_time"] and all(not c.last for c in cs[:-1])
        assert "".join(c.text for c in cs) == turn["text"], (turn["id"], [c.text for c in cs])


@pytest.mark.parametrize("chunk_ms", [40, 80, 200, 1000])
def test_observations_never_run_ahead_and_add_up(chunk_ms):
    spec = AgentSpec(chunk_ms=chunk_ms)
    rec = []
    env = run(drive(Env({"user": ReplayUser(SOUNDS_USER)}, spec), CannedAgent([LONG_REPLY], spec, words_per_sec=5, min_words=2), record=rec))
    ep = check(episode(env, "e"))
    check_stream_invariants(rec, [t for t in turns_of(ep, "user") if t["end_time"] > t["start_time"]], chunk_ms)


def test_a_cut_shows_up_as_an_early_last_chunk():
    # From scenario 4: the user's u0 (planned 0..2941) is cut at 1500 -> the chunk ending at 1500 is flagged last
    # and nothing of u0 is observed after it.
    spec = AgentSpec(chunk_ms=100)
    user = UserSim(ScriptSource(["aa bb cc dd ee ff gg hh ii jj"]), timing=TurnTaking(yield_after_ms=1000))
    agent = Scripted({500: [Frame("policy.speech", 500, Segment("a0", 500, 3000, "I am talking over you now"))]})
    rec = []
    env = run(drive(Env({"user": user}, spec), agent, record=rec))
    ep = check(episode(env, "e"))
    cs = [c for _, c in observed_chunks(rec) if c.id == "u0"]
    assert cs[-1].last and cs[-1].t0 + cs[-1].dur == 1500
    check_stream_invariants(rec, turns_of(ep, "user"), 100)


def test_cut_chunk_texts_still_add_up_to_what_was_said():
    for cut_at in range(110, 1000, 37):  # includes 258: observed "ab" + "" but said "abc"
        _cut_scenario(cut_at)


def _cut_scenario(cut_at):
    # A 10-char, 1000 ms user turn cut at an arbitrary time: whatever the cut point, the text the agent observed
    # chunk by chunk must be exactly the said prefix (no character lost or repeated at the cut).
    spec = AgentSpec(chunk_ms=10)

    class Cutter(Node):  # speaks "abcdefghij" over 0..1000, cut at cut_at
        def init_state(self, task, rng):
            return {"n": 0}

        async def step(self, st, t, inbox):
            seg = Segment("u0", 0, 1000, "abcdefghij")
            st["n"] += 1
            if st["n"] == 1:
                return [Frame("user.speech", 0, seg)], cut_at
            if t == cut_at:
                return [Frame("user.speech", t, seg.cut(t))], None
            return [], None

    rec = []
    env = run(drive(Env({"user": Cutter()}, spec, end_idle_ms=100), Silent(), record=rec))
    ep = check(episode(env, "e"))
    check_stream_invariants(rec, turns_of(ep, "user"), 10)


@pytest.mark.parametrize("chunk_ms", [40, 80, 200, 1000])
def test_reply_start_is_quantized_to_the_agent_step(chunk_ms):
    # User turns are independent of the agent's step; the reply to u0 (ends 1000) starts at the first step boundary
    # t >= 1000 + reply_after(300): ceil(1300 / chunk) * chunk -> 40: 1320, 80: 1360, 200: 1400, 1000: 2000.
    spec = AgentSpec(chunk_ms=chunk_ms)
    env = run(drive(Env({"user": ReplayUser([{"t": 0, "text": "hello there my friend", "dur": 1000}])}, spec),
                    CannedAgent(["a b c d e"], spec, words_per_sec=5)))
    ep = check(episode(env, "e"))
    want = -(-1300 // chunk_ms) * chunk_ms
    assert [(t["role"], t["start_time"], t["end_time"]) for t in ep["turns"]] == [("user", 0, 1000), ("agent", want, want + 1000)]


# =============================================================== 6. microphone audio, sample by sample


def test_microphone_is_the_exact_sample_mix():
    # sr = 1000 Hz, so one sample per ms and timeline ms t <-> sample t. Step 30 ms (does not divide any boundary).
    #  user u0 : t=250, 500 samples, value (t - 250)                 -> on [250, 750)
    #  user u1 : t=1000, 100 samples, value 32700                     -> on [1000, 1100)
    #  bg A    : ramp 0..299, loop, offset 100, start 100, end 900, gain 0 -> on [100, 900): (t - 100 + 100) % 300 = t % 300
    #  bg B    : constant 1000, loop (50 samples), gain -20 dB (x0.1) -> 100 everywhere
    # mic(t) = clip(u0 + u1 + A + B) to int16; at [1000,1100): 32700 + 100 = 32800 -> 32767.
    sr = 1000
    a16 = lambda vals: Audio(array("h", vals), sr)  # noqa: E731
    u0, u1 = a16(range(500)), a16([32700] * 100)
    bg_a = Background(a16(range(300)), loop=True, offset_ms=100, start_time=100, end_time=900)
    bg_b = Background(a16([1000] * 50), gain_db=-20, loop=True)
    spec = AgentSpec(chunk_ms=30, audio="user.audio", sr=sr)
    user = ReplayUser([{"t": 250, "text": "x", "audio": u0}, {"t": 1000, "text": "y", "audio": u1}])
    rec = []
    env = run(drive(Env({"user": user}, spec, background=[bg_a, bg_b], end_idle_ms=200), Silent(), record=rec))

    def want(t):
        v = (t - 250 if 250 <= t < 750 else 0) + (32700 if 1000 <= t < 1100 else 0) + (t % 300 if 100 <= t < 900 else 0) + 100
        return max(-32768, min(32767, v))

    frames = [f for _, obs in rec for f in obs if f.stream == "user.audio"]
    assert [f.t for f in frames] == list(range(0, env.t, 30)) and all(f.dur == 30 and len(f.data) == 30 for f in frames)
    got = [s for f in frames for s in f.data.samples]
    assert got == [want(t) for t in range(env.t)]


# =============================================================== 7. tools


def add(state: dict, x: int) -> int:
    """Add x to a counter."""
    state["n"] += x
    return state["n"]


def boom() -> str:
    """Always fails."""
    raise RuntimeError("boom!")


class CallingUser(Node):
    """Says "hello" over 0..1000 and calls add(x=1) at t=500."""

    async def step(self, st, t, inbox):
        if t:
            return [], None
        return [Frame("user.speech", 0, Segment("u0", 0, 1000, "hello")), Frame(CALL["user"], 500, ToolCall("u1", "add", {"x": 1}))], None


def tool_scenario():
    # latency: add 500 ms, boom 200 ms, anything else 0.
    # user  : add(x=1) at 500  -> n=1, result "1" at 1000 on tool.user_result (never shown to the agent)
    # agent : speaks a0 over 1000..2000; add(x=2) at 1500 -> n=3, result "3" at 2000 (attached to a0)
    #         boom() at 3000 while silent -> error at 3200 (a silent turn "agent:c2" carries it)
    #         nope() at 3100 -> unknown tool, error at 3100 (another silent turn "agent:c3")
    lat = lambda c: {"add": 500, "boom": 200}.get(c.name, 0)  # noqa: E731
    world = ToolWorld(FunctionBackend({"add": add, "boom": boom}, initial_state={"n": 0}), lat)
    spec = AgentSpec(chunk_ms=100, obs=("user.speech", "tool.result"))
    agent = Scripted({
        1000: [Frame("policy.speech", 1000, Segment("a0", 1000, 1000, "let me check that"))],
        1500: [Frame(CALL["agent"], 1500, ToolCall("c1", "add", {"x": 2}))],
        3000: [Frame(CALL["agent"], 3000, ToolCall("c2", "boom"))],
        3100: [Frame(CALL["agent"], 3100, ToolCall("c3", "nope"))],
    })
    rec = []
    # end_idle 2000: the agent is silent from 2000 and calls at 3000 (< 2000 + 2000), so the episode is still running;
    # the last activity is c2's result at 3200 -> natural end at the first step >= 5200.
    env = run(drive(Env({"user": CallingUser(), "tools": world}, spec, end_idle_ms=2000), agent, record=rec))
    return env, rec


def test_tool_latency_errors_and_where_calls_are_attached():
    env, rec = tool_scenario()
    ep = check(episode(env, "e"))
    calls = {t["id"]: t.get("tool_calls") for t in ep["turns"]}
    assert calls == {
        "u0": [{"id": "u1", "name": "add", "arguments": {"x": 1}, "call_time": 500, "result_time": 1000, "content": "1"}],
        "a0": [{"id": "c1", "name": "add", "arguments": {"x": 2}, "call_time": 1500, "result_time": 2000, "content": "3"}],
        "agent:c2": [{"id": "c2", "name": "boom", "arguments": {}, "call_time": 3000, "result_time": 3200, "content": "Error: boom!", "error": True}],
        "agent:c3": [{"id": "c3", "name": "nope", "arguments": {}, "call_time": 3100, "result_time": 3100,
                      "content": "Error: unknown tool nope", "error": True}],
    }
    assert [(t["start_time"], t["end_time"]) for t in ep["turns"] if t["id"].startswith("agent:")] == [(3000, 3000), (3100, 3100)]
    # the agent hears its own results when they arrive, and nothing of the user's
    # (step at which the agent sees it, frame time, id): a call made by act(t) is processed during the step t -> t+100,
    # so c3's instant result (frame t=3100) is first observed after that step, at 3200 — together with c2's (3200).
    seen = [(t, f.t, f.data.id) for t, obs in rec for f in obs if f.stream == RESULT["agent"]]
    assert seen == [(2000, 2000, "c1"), (3200, 3100, "c3"), (3200, 3200, "c2")]
    assert not any(f.stream == RESULT["user"] for _, obs in rec for f in obs)
    assert env.t == 5200


def test_the_agent_does_not_see_the_users_tool_calls_by_default():
    world = ToolWorld(FunctionBackend({"add": add}, initial_state={"n": 0}))
    rec = []
    run(drive(Env({"user": CallingUser(), "tools": world}, AgentSpec(chunk_ms=100), end_idle_ms=500), Silent(), record=rec))
    assert not any(f.stream == CALL["user"] for _, obs in rec for f in obs)


def test_discover_mode_session_update_arrives_with_the_load_result():
    # load_tools at t=100 with 400 ms latency -> result and session update at 500; the agent sees both at t=500.
    shop = FunctionBackend({"add": add}, initial_state={"n": 0}, name="shop")
    weather = FunctionBackend({"boom": boom}, name="weather")
    world = ToolWorld({"shop": shop, "weather": weather}, 400, mode="discover")
    spec = AgentSpec(chunk_ms=100, obs=("tool.result",))
    rec = []
    run(drive(Env({"tools": world}, spec, end_idle_ms=300),
              Scripted({100: [Frame(CALL["agent"], 100, ToolCall("l1", "load_tools", {"service": "shop"}))]}), record=rec))
    sessions = [(t, f.t, sorted(x.name for x in f.data.tools)) for t, obs in rec for f in obs if f.stream == "session"]
    assert sessions == [(0, 0, ["list_services", "load_tools", "search_tools"]),
                        (500, 500, ["list_services", "load_tools", "search_tools", "shop__add"])]
    assert [(t, f.t) for t, obs in rec for f in obs if f.stream == "tool.result"] == [(500, 500)]


# =============================================================== 8. fork and determinism


def test_forks_diverge_independently_and_the_parent_is_untouched():
    # Fork at t=1500 (the agent has been speaking since 1300; u1 barges in at 2000).
    # Branch A's agent never yields (a0 ends 3300); branch B's yields at 2200 (as in scenario 2).
    # The parent, continued afterwards with the same agent as B, must end exactly like B.
    async def main():
        spec = AgentSpec(chunk_ms=100)
        env = Env({"user": ReplayUser(BARGE_USER)}, spec)
        agent = CannedAgent(["a b c d e f g h i j"], spec, words_per_sec=5)
        obs = await env.reset(TASK)
        while env.t < 1500:
            obs, _, _ = await env.step(agent.act(env.t, obs))
        log_at_fork = list(env.log)
        a_env, b_env = env.fork(2)
        a_agent, b_agent = copy.deepcopy(agent), copy.deepcopy(agent)
        a_agent.yield_after = None
        for e, ag in ((a_env, a_agent), (b_env, b_agent)):
            o, done = obs, False
            while not done:
                o, _, done = await e.step(ag.act(e.t, o))
        assert env.log == log_at_fork and env.t == 1500  # the parent did not move
        done = False
        while not done:
            obs, _, done = await env.step(agent.act(env.t, obs))
        return env, a_env, b_env

    parent, a, b = run(main())
    end = lambda e: [(t["id"], t["end_time"]) for t in turns_of(episode(e, "e"), "agent")]  # noqa: E731
    assert end(a)[0][1] == 3300 and end(b)[0][1] == 2200 and end(parent) == end(b)
    strip = lambda ep: {k: v for k, v in ep.items() if k != "meta"}  # noqa: E731
    assert strip(episode(parent, "e")) == strip(episode(b, "e"))


def test_same_seed_same_episode():
    def once():
        spec = AgentSpec(chunk_ms=80)
        return episode(run(drive(Env({"user": ReplayUser(SOUNDS_USER)}, spec), CannedAgent([LONG_REPLY], spec), seed=7)), "e")

    assert json.dumps(once(), sort_keys=True) == json.dumps(once(), sort_keys=True)


# =============================================================== 9. trajectory round trip


class Replayer(Node):
    """Re-emits a recorded event log (from traj.to_frames)."""

    def __init__(self, frames):
        self.frames = frames

    async def step(self, st, t, inbox):
        return (list(self.frames) if t == 0 else []), None


def audio_scenario():
    # sr 1000 (1 sample/ms). User u0: ramp audio 0..999 over 0..1000; u1: constant 5 over 2000..3000 (barges in).
    # Agent: a0 = ramp of 2000 samples (values 7..2006) over 1300..3300, cut at 2200 -> said = first 900 samples.
    sr = 1000
    u0, u1 = Audio(array("h", range(1000)), sr), Audio(array("h", [5] * 1000), sr)
    a_audio = Audio(array("h", range(7, 2007)), sr)
    a0 = Segment("a0", 1300, 2000, a_audio, text="a b c d e f g h i j")
    user = ReplayUser([{"t": 0, "text": "hello", "audio": u0}, {"t": 2000, "text": "wait wait", "audio": u1}])
    agent = Scripted({1300: [Frame("policy.speech", 1300, a0)], 2200: [Frame("policy.speech", 2200, a0.cut(2200))]})
    rec = []
    env = run(drive(Env({"user": user}, AgentSpec(chunk_ms=100, sr=sr), end_idle_ms=500), agent, record=rec))
    return env, rec, (u0, u1, a_audio)


def test_media_refs_slice_exactly_the_audio_that_played(tmp_path):
    env, _, (u0, u1, a_audio) = audio_scenario()
    media = MediaStore(tmp_path)
    ep = check(episode(env, "e", media=media))
    t = {x["id"]: x for x in ep["turns"]}
    assert (t["a0"]["start_time"], t["a0"]["end_time"], t["a0"]["text"], t["a0"]["unsaid"]) == (1300, 2200, "a b c d e", " f g h i j")
    assert media.load(t["u0"]["media"]) == u0 and media.load(t["u1"]["media"]) == u1
    assert media.load(t["a0"]["media"]) == a_audio[:900]  # what was said, from the stored full plan
    assert ep["eval"]["duplex"]["u1"] == {"behavior": "barge_in", "target": "a0", "reaction": "yielded", "latency_ms": 200}


def test_save_load_and_rebuild_from_frames(tmp_path):
    env, _, _ = audio_scenario()
    media = MediaStore(tmp_path)
    ep = episode(env, "e", media=media)
    path = save([ep], tmp_path / "run.jsonl")
    (back,) = load(path)
    assert back == json.loads(json.dumps(ep))
    # replay the trajectory as a runtime log and rebuild the turns from it
    env2 = run(drive(Env({"user": Replayer(to_frames(back, media))}, AgentSpec(chunk_ms=100, sr=1000), end_idle_ms=500), Silent()))
    ep2 = episode(env2, "e", media=media)
    key = lambda ep: [(x["id"], x["role"], x["start_time"], x["end_time"], x["text"], x.get("unsaid"), x.get("kind")) for x in ep["turns"]]  # noqa: E731
    assert key(ep2) == key(back)
    assert ep2["eval"]["duplex"] == back["eval"]["duplex"]


def test_rebuild_from_frames_keeps_the_audio_of_cut_turns(tmp_path):
    env, _, _ = audio_scenario()
    media = MediaStore(tmp_path)
    back = json.loads(json.dumps(episode(env, "e", media=media)))
    env2 = run(drive(Env({"user": Replayer(to_frames(back, media))}, AgentSpec(chunk_ms=100, sr=1000), end_idle_ms=500), Silent()))
    ep2 = episode(env2, "e", media=media)
    for x, y in zip(back["turns"], ep2["turns"]):
        assert media.load(x["media"]) == media.load(y["media"])


def test_agent_view_equals_what_the_agent_observed():
    # agent_view derives per-step hearing from the turns; with replayed (never cut) user turns it must match the
    # chunks the live agent observed, step by step.
    spec = AgentSpec(chunk_ms=100)
    rec = []
    env = run(drive(Env({"user": ReplayUser(SOUNDS_USER)}, spec), CannedAgent([LONG_REPLY], spec, words_per_sec=5, min_words=2), record=rec))
    view = agent_view(episode(env, "e"), chunk_ms=100)
    live = {t: [{"id": f.data.id, "start_time": f.data.t0, "end_time": f.data.t0 + f.data.dur, "text": f.data.text}
                for f in obs if f.stream == "user.speech" and isinstance(f.data, Chunk) and f.data.dur > 0]
            for t, obs in rec}
    for step in view:
        assert step["heard"] == live.get(step["time"] + 100, []), step


# =============================================================== 10. the vLLM-Omni adapter against a fake server


websockets = pytest.importorskip("websockets")
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent  # noqa: E402

SR24 = 24000


async def fake_duplex(ws, *, reply_at_unit=2, audio_ms=700, text=" Hi there", text_only=False, jitter=False, log=None):
    """Lockstep server: 1000 ms units; on completing unit ``reply_at_unit`` it answers (audio ``audio_ms`` + text, or
    text only), then ``response.done``; every append is acknowledged after its outputs."""
    rng = random.Random()
    first = json.loads(await ws.recv())
    assert first["session"]["extra_body"]["clock"] == "input"
    await ws.send(json.dumps({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": 1000}}}))
    heard = units = 0
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] != "input_audio_buffer.append":
            continue
        heard += len(base64.b64decode(ev["audio"])) // 2 * 1000 // ev["sample_rate_hz"]
        if log is not None:
            log.append(heard)
        while (units + 1) * 1000 <= heard:
            units += 1
            if jitter:
                await asyncio.sleep(rng.uniform(0, 0.03))
            if units == reply_at_unit:
                if text_only:
                    await ws.send(json.dumps({"type": "response.output_text.delta", "response_id": "r", "delta": text}))
                else:
                    pcm = base64.b64encode(bytes(2 * SR24 * audio_ms // 1000)).decode()
                    await ws.send(json.dumps({"type": "response.output_audio.delta", "response_id": "r", "delta": pcm}))
                    await ws.send(json.dumps({"type": "response.output_audio_transcript.delta", "response_id": "r", "delta": text}))
                await ws.send(json.dumps({"type": "response.done", "response": {"id": "r", "status": "completed"}}))
        await ws.send(json.dumps({"type": "input_audio_buffer.processed", "audio_end_ms": heard, "unit_end_ms": units * 1000}))


import base64  # noqa: E402


def adapter_episode(**server_kw):
    text_only = server_kw.get("text_only", False)
    log = []

    async def main():
        async with websockets.serve(lambda ws: fake_duplex(ws, log=log, **server_kw), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR24)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", clock="input",
                                        ref_audio=None if text_only else __file__, audio_out=not text_only, speech_cps=10.0)
            env = await drive(Env({}, spec), agent)
            await agent.close()
            return env, agent

    env, agent = run(main())
    return env, agent, log


def test_lockstep_adapter_reply_timing_and_ack_count():
    # Each act(t) sends the mic window [t-200, t) -> the append that completes unit 2 (input reaches 2000 ms) is sent
    # at t=2000; its reply (700 ms audio) plays from t=2000 (lockstep: no prebuffer) -> agent turn 2000..2700.
    # Natural end: first step t >= 2700 + 3000 = 5700 -> 5800 (steps of 200). Appends: one per act after reset,
    # i.e. at t = 200..5600 -> 28, each acknowledged once; the server heard 200, 400, ..., 5600.
    env, agent, log = adapter_episode()
    ep = check(episode(env, "e"))
    (a,) = turns_of(ep, "agent")
    assert (a["start_time"], a["end_time"], a["text"]) == (2000, 2700, "Hi there")
    assert env.t == 5800 and agent.appends == agent.acks == 28 and log == list(range(200, 5601, 200))


def test_lockstep_adapter_text_mode_is_timed_by_the_rate():
    # " Hi there friend" = 16 chars at 10 chars/s -> 1600 ms: 2000..3600 ("Hi there friend" after the leading space
    # is stripped). Natural end at the first step >= 6600 -> 6600.
    env, agent, _ = adapter_episode(text_only=True, text=" Hi there friend")
    ep = check(episode(env, "e"))
    (a,) = turns_of(ep, "agent")
    assert (a["start_time"], a["end_time"], a["text"], "media" in a) == (2000, 3600, "Hi there friend", False)
    assert env.t == 6600


def test_lockstep_adapter_is_reproducible_under_server_jitter():
    key = lambda env: json.dumps(episode(env, "e")["turns"])  # noqa: E731
    runs = [adapter_episode(jitter=True)[0] for _ in range(2)]
    assert key(runs[0]) == key(runs[1]) == key(adapter_episode()[0])


async def scripted_duplex(ws, script):
    """Lockstep server whose outputs are scripted per completed unit: {unit: [(event type, extra fields)]}."""
    first = json.loads(await ws.recv())
    assert first["session"]["extra_body"]["clock"] == "input"
    await ws.send(json.dumps({"type": "session.created", "session": {"capabilities": {"chunk_period_ms": 1000}}}))
    heard = units = 0
    async for raw in ws:
        ev = json.loads(raw)
        if ev["type"] != "input_audio_buffer.append":
            continue
        heard += len(base64.b64decode(ev["audio"])) // 2 * 1000 // ev["sample_rate_hz"]
        while (units + 1) * 1000 <= heard:
            units += 1
            for typ, extra in script.get(units, []):
                await ws.send(json.dumps({"type": typ, "response_id": "r", **extra}))
        await ws.send(json.dumps({"type": "input_audio_buffer.processed", "audio_end_ms": heard}))


def pcm24(ms):
    return base64.b64encode(bytes(2 * SR24 * ms // 1000)).decode()


def scripted_episode(script):
    async def main():
        async with websockets.serve(lambda ws: scripted_duplex(ws, script), "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR24)
            agent = VllmOmniDuplexAgent(spec, f"ws://127.0.0.1:{port}/v1/realtime?duplex=1", clock="input", ref_audio=__file__)
            env = await drive(Env({}, spec), agent)
            await agent.close()
            return env

    return check(episode(run(main()), "e"))


def test_lockstep_adapter_fills_a_mid_response_gap_with_silence():
    # unit 2 (t=2000): 500 ms "Hello"; unit 3 (t=3000): 500 ms " world" + done.
    # Plays 2000..2500, then the response is still open -> silence 2500..3000, then 3000..3500 -> one turn 2000..3500.
    ep = scripted_episode({
        2: [("response.output_audio.delta", {"delta": pcm24(500)}), ("response.output_audio_transcript.delta", {"delta": "Hello"})],
        3: [("response.output_audio.delta", {"delta": pcm24(500)}), ("response.output_audio_transcript.delta", {"delta": " world"}),
            ("response.done", {"response": {"id": "r", "status": "completed"}})],
    })
    assert [(t["start_time"], t["end_time"], t["text"]) for t in turns_of(ep, "agent")] == [(2000, 3500, "Hello world")]


def test_lockstep_adapter_drops_what_had_not_played_when_a_response_is_cancelled():
    # unit 2 (t=2000): one 2000 ms delta; unit 3 (t=3000): response cancelled. By t=3000, 2000..3000 has played
    # (5 steps of 200 ms); the remaining 1000 ms never plays -> turn 2000..3000.
    ep = scripted_episode({
        2: [("response.output_audio.delta", {"delta": pcm24(2000)}), ("response.output_audio_transcript.delta", {"delta": "one two three four"})],
        3: [("response.done", {"response": {"id": "r", "status": "cancelled"}})],
    })
    assert [(t["start_time"], t["end_time"]) for t in turns_of(ep, "agent")] == [(2000, 3000)]


def test_lockstep_adapter_keeps_text_aligned_with_the_audio_that_played():
    # Same as above: half of the 2000 ms delta played, so half of its 18-char text was said: round(18 * 0.5) = 9 chars
    # = "one two t" (and after its first 200 ms only round(18 * 0.1) = 2 chars had been heard, not all 18).
    ep = scripted_episode({
        2: [("response.output_audio.delta", {"delta": pcm24(2000)}), ("response.output_audio_transcript.delta", {"delta": "one two three four"})],
        3: [("response.done", {"response": {"id": "r", "status": "cancelled"}})],
    })
    (a,) = turns_of(ep, "agent")
    assert a["text"] == "one two t"
