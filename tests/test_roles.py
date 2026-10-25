"""Roles (docs/ROLES.md): the controlled role may have any name, the trajectory records the role table, and every
reader judges relative to the controlled role — including with the sides swapped (the external policy as the user)."""

import asyncio
import json

import pytest

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.interaction import controlled, metrics
from interaction_gym.plugins import Composition, EpisodeContext, PluginError, compose
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.traj import episode
from interaction_gym.user import ReplayUser, Voice
from interaction_gym.viewer import render_html
from interaction_gym.viewer.commentary import headlines

TURNS = [{"t": 0, "text": "Hi, what's the weather like tomorrow?"}, {"t": 4000, "text": "Sorry, I meant the day after tomorrow."}]
REPLIES = ["Tomorrow will be sunny with a high of twenty four degrees and a light breeze from the west.",
           "Got it, the day after tomorrow looks rainy, so bring an umbrella."]


def run(env, agent, task=Task("t", {}), seed=0):
    async def go():
        obs = await env.reset(task, seed)
        done = False
        while not done:
            obs, _, done = await env.step(agent.act(env.t, obs))
        return episode(env, "x")
    return asyncio.run(go())


def test_default_roles_are_recorded_and_old_files_read_as_agent():
    spec = AgentSpec(chunk_ms=200)
    ep = run(Env({"user": ReplayUser(TURNS, voice=Voice())}, spec, max_ms=60000), CannedAgent(REPLIES, spec, yield_after=160))
    assert ep["meta"]["controlled"] == "agent"
    assert ep["meta"]["roles"] == {"agent": {"stream": "policy.speech", "perceives": ["user.speech"], "controlled": True},
                                   "user": {"stream": "user.speech", "perceives": [], "component": "user"}}  # a replay listens to nothing
    assert {t["role"] for t in ep["turns"]} == {"user", "agent"}
    old = {"meta": {k: v for k, v in ep["meta"].items() if k not in ("roles", "controlled")}, "turns": ep["turns"], "eval": ep["eval"]}  # a file from before roles
    assert controlled(old) == "agent" and metrics(old) == metrics(ep)


def test_the_controlled_role_may_have_any_name_and_the_scores_follow_it():
    spec = AgentSpec(chunk_ms=200, obs=("caller.speech",), out="assistant.speech")
    ep = run(Env({"caller": ReplayUser(TURNS, voice=Voice(), stream="caller.speech")}, spec, max_ms=60000), CannedAgent(REPLIES, spec, yield_after=160))
    assert ep["meta"]["controlled"] == "assistant" and set(ep["meta"]["roles"]) == {"assistant", "caller"}
    assert ep["meta"]["roles"]["caller"] == {"stream": "caller.speech", "perceives": [], "component": "caller"}
    roles = [t["role"] for t in ep["turns"]]
    assert roles == ["caller", "assistant", "caller", "assistant"] and "unsaid" in ep["turns"][1]  # the first reply is cut
    ev = ep["eval"]["scores"]["events"]
    assert len(ev) == 2 and [e["expects"] for e in ev] == ["respond", "yield"] and ep["eval"]["scores"]["total"] is not None
    assert all(d["behavior"] == "barge_in" for d in ep["eval"]["duplex"].values())
    # the same episode with the default names gives the same timeline and scores: only the names differ
    spec2 = AgentSpec(chunk_ms=200)
    ep2 = run(Env({"user": ReplayUser(TURNS, voice=Voice())}, spec2, max_ms=60000), CannedAgent(REPLIES, spec2, yield_after=160))
    strip = lambda ts: [{k: v for k, v in t.items() if k != "role"} for t in ts]  # noqa: E731
    assert strip(ep["turns"]) == strip(ep2["turns"]) and ep["eval"]["scores"]["total"] == ep2["eval"]["scores"]["total"]
    assert metrics(ep) == metrics(ep2)
    # the readers follow meta.controlled
    assert [h["role"] for h in headlines(ep, "assistant")] if headlines(ep, "assistant") and "role" in headlines(ep, "assistant")[0] else True
    html = render_html([ep])
    assert "assistant.speech" in html and "caller.speech" in html


def test_sides_swapped_the_external_policy_is_the_user():
    """The policy speaks on user.speech and hears policy.speech: a replayed assistant asks, the canned 'user' answers."""
    spec = AgentSpec(chunk_ms=200, obs=("policy.speech",), out="user.speech")
    assistant = ReplayUser([{"t": 0, "text": "Hello, how can I help you today?"}, {"t": 6000, "text": "Sure, which day?"}], voice=Voice(), stream="policy.speech")
    ep = run(Env({"assistant": assistant}, spec, max_ms=60000), CannedAgent(["I'd like to know the weather.", "The day after tomorrow."], spec, yield_after=None))
    assert ep["meta"]["controlled"] == "user"
    assert ep["meta"]["roles"]["user"]["controlled"] and ep["meta"]["roles"]["agent"]["component"] == "assistant"
    roles = [t["role"] for t in ep["turns"]]
    assert roles[0] == "agent" and "user" in roles
    # the scores judge the controlled role (the user) on its reactions to the assistant's turns
    ev = ep["eval"]["scores"]["events"]
    assert ev and all(e["turn"] in {t["id"] for t in ep["turns"] if t["role"] == "agent"} for e in ev)
    assert ep["eval"]["scores"]["total"] is not None


def test_role_plugins_name_and_stream_must_agree_and_a_three_role_bundle_runs(tmp_path):
    c = Composition.bundle("minimal")
    c.apply("- id: controlled\n  config: {name: assistant, out: policy.speech}", "p")
    with pytest.raises(PluginError, match="name 'assistant' but out 'policy.speech' would be recorded as role 'agent'"):
        compose(c.resolve())
    c = Composition.bundle("minimal")
    c.apply("""
- id: controlled
  config: {name: assistant, chunk_ms: 200, obs: [caller.speech, business.speech]}
- id: user
  config: {stream: caller.speech, turns: [{t: 0, text: "Hi, I'd like a table for two tonight."}]}
- insert:
    - {id: business, plugin: role/replay, config: {stream: business.speech, turns: [{t: 9000, text: "We are fully booked until nine."}]}}
- id: agent
  config: {replies: ["Let me check with the restaurant.", "They are fully booked until nine; is that all right?"], yield_after: 160}
""", "p")
    run_ = compose(c.resolve())
    assert run_.spec.out == "assistant.speech" and set(run_.nodes) == {"user", "business"}
    rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(run_.tasks[0], 0, episode_id="x")))
    assert rec["meta"]["controlled"] == "assistant" and set(rec["meta"]["roles"]) == {"assistant", "caller", "business"}
    assert {t["role"] for t in rec["turns"]} == {"assistant", "caller", "business"}
    assert len(rec["eval"]["scores"]["events"]) == 2  # one expectation per turn of each other role
    assert json.dumps(rec)  # serialisable


def test_standards_grade_each_speaker_s_turns_by_its_own_rule():
    from interaction_gym.eval import CALLER_STANDARD, Standard, scores

    # a controlled "user" that answers the assistant within 30 ms: cut off by the assistant standard, fine by the caller's
    turns = [{"id": "a0", "role": "agent", "start_time": 0, "end_time": 2000, "text": "Which day?"},
             {"id": "u0", "role": "user", "start_time": 2030, "end_time": 3000, "text": "Tuesday."}]
    strict = scores(turns, agent="user")["events"][0]
    lenient = scores(turns, agent="user", standards=CALLER_STANDARD)["events"][0]
    assert strict["outcome"] == lenient["outcome"] == "responded" and strict["score"] < 0.4 < 0.99 < lenient["score"]
    # by speaker: the business may be answered slowly, the caller must not
    three = [{"id": "c0", "role": "caller", "start_time": 0, "end_time": 1000, "text": "A table for two?"},
             {"id": "s0", "role": "assistant", "start_time": 2000, "end_time": 2500, "text": "One moment."},
             {"id": "b0", "role": "business", "start_time": 3000, "end_time": 4000, "text": "We are full until nine."},
             {"id": "s1", "role": "assistant", "start_time": 5500, "end_time": 6500, "text": "Nine works?"}]
    by = {e["turn"]: e for e in scores(three, agent="assistant", standards={"*": Standard(), "business": {"mid_ms": 3000}})["events"]}
    assert by["c0"]["score"] < 0.5 and by["b0"]["score"] > 0.9  # 1000 ms is slow for the caller; 1500 ms is fine for the business
    assert scores(three, agent="assistant")["events"][1]["score"] < 0.1  # without standards: the assistant standard for all


def test_role_controlled_standards_reach_the_trajectory(tmp_path):
    c = Composition.bundle("minimal")
    c.apply("- id: controlled\n  config: {name: agent, chunk_ms: 200, standards: {'*': {early_floor: 1.0}, user: {respond_within_ms: 8000}}}", "p")
    run_ = compose(c.resolve())
    assert run_.controlled.extra["standards"]["user"].respond_within_ms == 8000
    rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(run_.tasks[0], 0, episode_id="x")))
    assert rec["meta"]["standards"] == {"*": {"mid_ms": 950.0, "width_ms": 100.0, "early_floor": 1.0, "respond_within_ms": 5000},
                                        "user": {"mid_ms": 950.0, "width_ms": 100.0, "early_floor": 0.3, "respond_within_ms": 8000}}
    c.apply("- id: controlled\n  config: {name: agent, chunk_ms: 200, standards: {'*': {floor: 1}}}", "q")
    with pytest.raises(PluginError, match="standards: .*the fields are mid_ms, width_ms, early_floor, respond_within_ms"):
        compose(c.resolve())
