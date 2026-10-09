"""The user's reply gap: random, content-aware, reproducible."""

import asyncio
import dataclasses
import random
import statistics

from interaction_gym import AgentSpec, Env, Segment
from interaction_gym.clients import FakeChat
from interaction_gym.core import Frame
from interaction_gym.traj import episode
from interaction_gym.user import LLMSource, ResponseDelay, TurnTaking, UserSim, _pause, _spoken
from examples import user_modes as um
from tests.test_traj import check


def test_the_gap_is_lognormal_with_distraction_pauses_and_attentiveness():
    d = ResponseDelay(distracted_p=0.0)
    gaps = [d.sample(random.Random(i)) for i in range(2000)]
    assert 250 < statistics.median(gaps) < 350 and sorted(gaps)[int(0.99 * len(gaps))] < 1500  # a long tail, but rare
    assert statistics.median(d.sample(random.Random(i), "long") for i in range(500)) > 2500
    mixed = ResponseDelay(distracted_p=0.5)
    assert sum(mixed.sample(random.Random(i)) > 1500 for i in range(1000)) > 300
    low = sum(ResponseDelay().sample(random.Random(i), attentiveness="low") > 1500 for i in range(2000))
    high = sum(ResponseDelay().sample(random.Random(i), attentiveness="high") > 1500 for i in range(2000))
    assert low > 3 * high
    assert ResponseDelay().sample(random.Random(7)) == ResponseDelay().sample(random.Random(7))


def test_pause_markers_are_parsed_and_never_spoken():
    assert _pause("(long pause) Let me check my calendar.") == "long" and _pause("(pause) Hmm, seven.") == "short"
    assert _spoken("(long pause) Let me check my calendar.") == "Let me check my calendar."
    assert _pause("Seven please.") is None


def run_episode(seed, replies, agent_resumes=False, scenario=None):
    async def main():
        spec = AgentSpec(chunk_ms=100)
        user = UserSim(LLMSource(FakeChat(replies)), timing=TurnTaking(response_delay=ResponseDelay()))
        env = Env({"user": user}, spec, max_ms=40_000)
        obs = await env.reset(dataclasses.replace(um.TASK, scenario=scenario or {}), seed=seed)
        n = 0
        while not env.done:
            act = []
            users = [f for f in env.log if f.stream == "user.speech"]
            if env.t == 3000 and n == 0:  # the agent answers the first user turn, 2 s long
                act, n = [Frame(spec.out, 3000, Segment("a0", 3000, 2000, "Sure, one moment", text="Sure, one moment"))], 1
            if agent_resumes and env.t == 5200 and n == 1:  # and starts talking again shortly after it stopped
                act, n = [Frame(spec.out, 5200, Segment("a1", 5200, 3000, "Also, do you", text="Also, do you"))], 2
            obs, _, _ = await env.step(act)
        return check(episode(env, "e"))
    return asyncio.run(main())


def user_starts(ep):
    return [t["start_time"] for t in ep["turns"] if t["role"] == "user"]


def test_a_reply_comes_after_a_random_gap_reproducibly():
    replies = ["Hi, I need a table.", "For two, please.", "Thanks! ###STOP###"]
    a, b, c = run_episode(1, replies), run_episode(1, replies), run_episode(2, replies)
    assert user_starts(a) == user_starts(b)  # same seed, same gaps
    assert user_starts(a)[1] != user_starts(c)[1]  # another seed, another gap
    assert user_starts(a)[1] - 5000 != 1000  # not the fixed 1 s
    assert a["meta"]["user"]["turn_taking"]["response_delay"]["median_ms"] == 300


def test_a_marked_pause_lengthens_the_gap_and_is_not_spoken():
    quick = run_episode(3, ["Hi, I need a table.", "For two, please.", "Bye ###STOP###"])
    slow = run_episode(3, ["Hi, I need a table.", "(long pause) Let me check... for two, please.", "Bye ###STOP###"])
    assert user_starts(slow)[1] - user_starts(quick)[1] == 2500  # same draw + pause_long_ms
    assert [t["text"] for t in slow["turns"] if t["role"] == "user"][1] == "Let me check... for two, please."


def test_the_user_waits_if_the_agent_starts_again_during_the_gap():
    ep = run_episode(4, ["Hi, I need a table.", "(long pause) For two, please.", "Bye ###STOP###"], agent_resumes=True,
                     scenario={"turn_taking": {"response_delay": {"distracted_p": 0.0}}})
    second = user_starts(ep)[1]
    assert second >= 8200  # not over the agent's second turn (5200-8200)
    assert "agent_interrupt" not in str(ep["eval"]["duplex"]) and not any(v.get("behavior") == "barge_in" for v in ep["eval"]["duplex"].values())


def test_stage_directions_and_inline_pauses_are_not_spoken():
    assert _spoken("[Interrupting]") == ""
    assert _spoken("[User cuts in]\nYeah, FP-77310.") == "Yeah, FP-77310."
    assert _spoken("[INCOMPLETE - assistant is still...") == ""
    assert _spoken("Oh, sorry, I was just— *honey, put the spoon down*—okay.") == "Oh, sorry, I was just— —okay."
    assert _spoken("number is... (pause) 492-881-05.") == "number is... 492-881-05."
    assert _spoken('"Sure [CURRENTLY SPEAKING, INCOMPLETE] thanks"') == "Sure thanks"
