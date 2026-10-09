"""Benchmark users have no background: in the open-vs-closed benchmarks (FD-Bench v2 / v3), the agent's microphone in
the closed-loop condition (B, ``benchmark_user``) carries exactly the replayed open-loop condition's audio (A,
``ReplayUser``) until the user's second turn. No models, no data (synthetic recordings, fake clients)."""

import asyncio
import math
from array import array

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents.canned import CannedAgent
from interaction_gym.audio import Audio
from interaction_gym.benchmarks import benchmark_user, fdb2, fdb3
from interaction_gym.clients import FakeChat, FakeSpeech
from interaction_gym.traj import episode
from interaction_gym.user import ReplayUser, ResponseDelay, TurnTaking, UserSim, Voice

SR = 16000
SPEC = AgentSpec(chunk_ms=100, audio="user.audio", sr=SR)


def tone(ms, f=220.0, amp=6000):
    n = SR * ms // 1000
    return Audio(array("h", (int(amp * math.sin(2 * math.pi * f * i / SR)) for i in range(n))), SR)


def silence(ms):
    return Audio(array("h", [0] * (SR * ms // 1000)), SR)


class MicRecorder:
    """A ``CannedAgent`` that keeps every microphone frame it is given."""

    def __init__(self, replies):
        self.inner = CannedAgent(replies, SPEC, reply_after=300)
        self.mic: list[int] = []

    def act(self, t, obs):
        for f in obs:
            if f.stream == SPEC.audio:
                self.mic.extend(f.data.samples)
        return self.inner.act(t, obs)


async def drive(env, task, seed=0):
    agent = MicRecorder(["Sure, I can check order A B C one two three for you."])
    obs = await env.reset(task, seed)
    while True:
        obs, _, done = await env.step(agent.act(env.t, obs))
        if env.truncated or done:
            break
    return episode(env, "e"), agent.mic


def second_user_turn_ms(ep):
    users = [t for t in ep["turns"] if t["role"] == "user"]
    assert len(users) >= 2
    return users[1]["start_time"]


def assert_same_until(a_mic, b_mic, ms):
    n = SR * ms // 1000
    assert len(a_mic) >= n and len(b_mic) >= n
    assert a_mic[:n] == b_mic[:n]


def closed_timing():
    return TurnTaking(response_delay=ResponseDelay(), yield_after_ms=None, nudge_after_ms=10_000)


# ---------------------------------------------------------------- FD-Bench v3


def fdb3_task():
    audio = tone(2500) + silence(300)
    meta = {"id": "ecommerce_01", "domain": "ecommerce_support", "title": "t", "difficulty": "easy",
            "dialogue": [{"user": "um... track my order ABC123", "ai": "I will check order ABC123."}], "acting_notes": "",
            "disfluency_features": ["PAUSE"], "expected_tool_calls": [{"function": "track_order", "args": {"order_id": "ABC123"}}],
            "state_rollback_test": False, "latency_profile": "normal"}
    return fdb3.task_of(fdb3.Recording("ecommerce_01_spk", meta, audio, fdb3.request_end_ms(audio), audio.dur_ms))


def fdb3_b(user_cls):
    task = fdb3_task()
    src = fdb3.UserSource(FakeChat(["Yes, that's the one, thanks. ###STOP###"]), 4)
    kw = dict(voice=Voice(FakeSpeech(sr=SR)), timing=closed_timing())
    user = benchmark_user(src, **kw) if user_cls is None else user_cls(src, **kw)
    return asyncio.run(drive(Env({"user": user}, SPEC, max_ms=30_000, end_idle_ms=3000), task, seed=3))


def test_fdb3_open_and_closed_loop_mic_identical_until_the_users_second_turn():
    a, a_mic = asyncio.run(drive(Env({"user": ReplayUser(voice=Voice())}, SPEC, max_ms=30_000, end_idle_ms=3000), fdb3_task(), seed=3))
    b, b_mic = fdb3_b(None)
    u1 = second_user_turn_ms(b)
    assert u1 > 2800  # after the recording and the agent's reply
    assert_same_until(a_mic, b_mic, u1)
    assert b["meta"].get("background", []) == []
    # the env's general default (a background from the persona's surroundings) would break it
    _, d_mic = fdb3_b(UserSim)
    assert d_mic[: SR * u1 // 1000] != a_mic[: SR * u1 // 1000]


# ---------------------------------------------------------------- FD-Bench v2


def fdb2_task(tmp_path):
    t = {"id": "Correction.x.001", "split": "Correction", "class_id": "x", "scenario_title": "t",
         "examiner_system_prompt": "Act like a customer.", "examiner_task_prompt": "Order a pizza.",
         "staged_reveal": {"T1": "ask", "T2": "size"}, "skills_tested": ["correction"]}
    import json

    p = tmp_path / "prompts.json"
    p.write_text(json.dumps({"settings": {"turn_limit": 5}, "splits": {"Correction": {"classes": [], "tasks": [t]}}}))
    task = fdb2.load(p, splits=["Correction"])[0]
    wav = tmp_path / "e00.wav"
    tone(2000).write_wav(wav)
    return task, str(wav)


def test_fdb2_open_and_closed_loop_mic_identical_until_the_examiners_second_line(tmp_path):
    task_a, wav = fdb2_task(tmp_path)
    task_a.scenario["turns"] = [{"t": 0, "text": "Hi, I'd like to order a pizza.", "audio": wav, "final": False},
                                {"t": 9000, "text": "A large one, please.", "audio": wav, "final": True}]
    a, a_mic = asyncio.run(drive(Env({"user": ReplayUser(voice=Voice())}, SPEC, max_ms=20_000, end_idle_ms=3000), task_a))
    task_b, _ = fdb2_task(tmp_path)
    task_b.scenario["first_turn"] = {"text": "Hi, I'd like to order a pizza.", "audio": wav}
    ex = fdb2.ExaminerSource(FakeChat(["A large one, please. The conversation is over."]), 4)
    user = benchmark_user(ex, voice=Voice(FakeSpeech(sr=SR), voice="aiden", clone=FakeSpeech(sr=SR)),
                          timing=TurnTaking(response_delay=ResponseDelay(), yield_after_ms=1000, nudge_after_ms=10_000))
    b, b_mic = asyncio.run(drive(Env({"user": user}, SPEC, max_ms=20_000, end_idle_ms=3000), task_b))
    u1 = second_user_turn_ms(b)
    assert u1 > 2000
    assert_same_until(a_mic, b_mic, u1)


def test_benchmark_user_defaults_and_overrides():
    from interaction_gym.benchmarks import benchmark_behaviors, benchmark_soundscape
    from interaction_gym.soundscape import Soundscape

    u = benchmark_user(fdb3.UserSource(FakeChat(["x"])))
    assert u.soundscape == benchmark_soundscape() and not u.soundscape.background and u.soundscape.bank is None
    b = benchmark_behaviors()
    assert b.noise_per_min == 0.0 and b.aside_per_min == 0.0
    task = Task(id="t", scenario={"profile": {"surroundings": "street"}})
    assert u.background(task, 0, SR) == ([], [])
    assert u._resolve(task)[0].rates() == {"aside": 0.0, "noise": 0.0}
    assert benchmark_user(fdb3.UserSource(FakeChat(["x"])), soundscape=Soundscape()).soundscape.background
