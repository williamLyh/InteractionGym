"""freeze_user: a closed-loop episode's user side becomes a fixed (open-loop) user."""

import asyncio

from interaction_gym import AgentSpec, Env
from interaction_gym.agents import CannedAgent
from interaction_gym.clients import FakeSpeech
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, freeze_user
from interaction_gym.user import ReplayUser, ScriptSource, UserSim, Voice
from examples import user_modes as um
from tests.test_traj import check


def play(env, agent):
    async def main():
        obs = await env.reset(um.TASK, seed=0)
        done = False
        while not done:
            obs, _, done = await env.step(agent.act(env.t, obs))
    asyncio.run(main())
    return env


def user_turns(ep):
    return [(t["start_time"], t["end_time"], t["text"], t.get("kind")) for t in ep["turns"] if t["role"] == "user"]


def test_frozen_user_replays_the_same_user_against_any_agent(tmp_path):
    media = MediaStore(tmp_path)
    spec = AgentSpec(sr=16000)
    voice = Voice(FakeSpeech(sr=16000))
    closed = check(episode(play(Env({"user": UserSim(ScriptSource(), voice)}, spec, max_ms=60_000), CannedAgent(um.AGENT_REPLIES, spec)),
                           "c", media=media))
    turns = freeze_user(closed, media)
    assert all("audio" in t for t in turns) and [t["t"] for t in turns] == [t[0] for t in user_turns(closed)]
    # replayed against the same agent: the same user side
    same = check(episode(play(Env({"user": ReplayUser(turns, voice)}, spec, max_ms=60_000), CannedAgent(um.AGENT_REPLIES, spec)), "o"))
    assert user_turns(same) == user_turns(closed)
    # against a different (slower) agent: the user does not react — same times, same words
    slow = CannedAgent(um.AGENT_REPLIES, spec, reply_after=2500)
    other = check(episode(play(Env({"user": ReplayUser(turns, voice)}, spec, max_ms=60_000), slow), "o2"))
    assert user_turns(other) == user_turns(closed)


def test_a_cut_turn_is_frozen_as_said():
    ep = {"turns": [{"id": "u0", "role": "user", "start_time": 0, "end_time": 1000, "text": "I would like", "unsaid": " a table"},
                    {"id": "a0", "role": "agent", "start_time": 500, "end_time": 2000, "text": "Sure"}]}
    assert freeze_user(ep) == [{"t": 0, "text": "I would like", "dur": 1000}]
