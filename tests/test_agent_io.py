"""The env ↔ agent-session boundary: streamed observations, streamed actions, chunk size."""

import asyncio

import examples.minimal as minimal
import examples.user_modes as um
from interaction_gym import SESSION, AgentSpec, Chunk, Env, Frame, Segment, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.eval import turns, yield_spans
from interaction_gym.user import ReplayUser, ScriptSource, TurnTaking, UserSim


def run(coro):
    return asyncio.run(coro)


async def rollout(env, agent, task):
    """Run an episode, recording (step time, observation) pairs."""
    seen = []
    obs = await env.reset(task)
    done = False
    while not done:
        seen.append((env.t, [f for f in obs if f.stream != SESSION]))
        obs, _, done = await env.step(agent.act(env.t, obs))
    seen.append((env.t, [f for f in obs if f.stream != SESSION]))
    return seen


def test_agent_never_sees_unplayed_content():
    env = Env({"user": um.make_user("online")}, AgentSpec(chunk_ms=200), max_ms=60_000)
    seen = run(rollout(env, CannedAgent(um.AGENT_REPLIES), um.TASK))
    truth = {t.actual.id: t.actual for t in turns(env.log, "user.speech")}
    so_far: dict[str, str] = {}
    for t, obs in seen:
        for f in obs:
            assert isinstance(f.data, Chunk)  # never a whole segment
            c = f.data
            assert c.t0 + c.dur <= t  # only what has already been played
            so_far[c.id] = so_far.get(c.id, "") + c.text
            assert so_far[c.id] == truth[c.id].heard_text(c.t0 + c.dur)
    assert so_far == {i: s.transcript for i, s in truth.items()}  # everything arrives eventually


def test_cut_user_segment_ends_early_in_the_stream():
    long = "this is a very long sentence that goes on and on and on for quite a while indeed"
    env = Env({"user": UserSim(ScriptSource([long]))}, AgentSpec(chunk_ms=100))

    async def main():
        chunks = await env.reset(Task())
        while env.t < 3000:
            act = [Frame("policy.speech", 500, Segment("a0", 500, 2000, "sorry to cut in here"))] if env.t == 500 else []
            obs, _, _ = await env.step(act)
            chunks += obs
        return chunks

    chunks = [f.data for f in run(main()) if f.stream != SESSION]
    last = [c for c in chunks if c.last]
    assert len(last) == 1 and last[0].t0 + last[0].dur == 500 + TurnTaking().yield_after_ms  # stream stops at the cut
    assert "".join(c.text for c in chunks) == long[: len("".join(c.text for c in chunks))]


def test_streaming_agent_output_grows_one_segment():
    env = run(minimal.run(chunk_ms=200, streaming=True))
    a0, a1 = turns(env.log, "policy.speech")
    assert a0.grown and a1.grown
    versions = [f.data for f in env.log if f.stream == "policy.speech" and f.data.id == "a0"]
    assert [v.dur for v in versions] == sorted(v.dur for v in versions) and len(versions) > 3  # grows chunk by chunk
    assert a0.actual.transcript == minimal.REPLIES[0][: len(a0.actual.transcript)]  # a prefix: it stopped early
    (y,) = yield_spans(env.log)
    assert y.t1 is not None and 160 <= y.ms < 160 + 200


def test_chunk_size_is_an_agent_property():
    for ms in (160, 240, 1000):
        env = Env({"user": ReplayUser([{"t": 0, "text": "hello there"}])}, AgentSpec(chunk_ms=ms))
        seen = run(rollout(env, CannedAgent(["hi"], AgentSpec(chunk_ms=ms)), Task()))
        times = [t for t, _ in seen]
        assert all(b - a == ms for a, b in zip(times, times[1:]))
        assert env.chunk_ms == ms


def test_peek_passes_raw_segments():
    env = Env({"user": ReplayUser([{"t": 0, "text": "hello there"}])}, AgentSpec(), peek=True)
    obs = [f for f in run(env.reset(Task())) if f.stream != SESSION]
    assert isinstance(obs[0].data, Segment) and obs[0].data.transcript == "hello there"


def test_whole_segment_and_chunk_actions_coexist():
    async def main():
        env = Env({}, AgentSpec(chunk_ms=100))
        await env.reset(Task())
        await env.step([Frame("policy.speech", 0, Segment("w", 0, 300, "whole"))])
        await env.step([Frame("policy.speech", 100, Chunk("s", 100, 100, "ab", "ab"))])
        await env.step([Frame("policy.speech", 200, Chunk("s", 200, 100, "cd", "cd"))])
        return env

    env = run(main())
    segs = [f.data for f in env.log if isinstance(f.data, Segment)]
    s = [x for x in segs if x.id == "s"][-1]
    assert (s.t0, s.end, s.data) == (100, 300, "abcd")
    assert [x.id for x in segs].count("w") == 1
