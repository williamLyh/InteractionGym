import asyncio

from examples.minimal import run
from interaction_gym.eval import response_latencies, turns, yield_latencies


def test_barge_in_episode():
    env = asyncio.run(run())
    agent = turns(env.log, "policy.speech")
    user = turns(env.log, "user.speech")

    assert env.done and env.sim.reward == 1.0
    assert [a.was_cut for a in agent] == [True, False]  # first answer interrupted, second finished
    assert [u.planned.data for u in user][1].startswith("Sorry")

    (y,) = yield_latencies(env.log)
    assert 160 <= y < 160 + env.chunk_ms  # yields after 160 ms of overlap, quantized by the agent's chunk size
    assert all(r is not None and r >= 300 for r in response_latencies(env.log)[1:])


def test_deterministic():
    a, b = asyncio.run(run()), asyncio.run(run())
    assert a.log == b.log
