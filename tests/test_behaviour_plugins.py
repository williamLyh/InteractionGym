"""Behaviours as objects (interaction_gym.behaviours): the default library equals the historical user frame for frame,
an empty list is a user that only speaks, a subset runs only that subset, and role/user names them as rows."""

import asyncio

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.behaviours import Asides, Behaviour, NoiseEvents, default_behaviours
from interaction_gym.clients import FakeChat, FakeSpeech
from interaction_gym.core import Frame
from interaction_gym.plugins import Composition, EpisodeContext, compose
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.traj import episode
from interaction_gym.user import Behaviors, LLMSource, ScriptSource, UserSim, UserTurn, Voice

TASK = Task("t", {"persona": "a caller", "instructions": "ask about opening hours", "profile": {"surroundings": "street", "language": "en"},
                  "turns": ["Hi, when do you open tomorrow?", "And on Sunday?", "Thanks, bye."]})
REPLIES = ["We open at nine tomorrow morning and we are open until six in the evening, with a short break at noon.",
           "On Sundays we are open from ten to four.", "You're welcome, goodbye."]
NOISY = Behaviors(noise_per_min=30.0, aside_per_min=20.0, away_p=0.5)


def run(user, seed=0, max_ms=120_000):
    spec = AgentSpec(chunk_ms=200)
    env = Env({"user": user}, spec, max_ms=max_ms)
    agent = CannedAgent(REPLIES, spec, yield_after=160)

    async def go():
        obs = await env.reset(TASK, seed)
        done = False
        while not done:
            obs, _, done = await env.step(agent.act(env.t, obs))
        return episode(env, "x"), env
    return asyncio.run(go())


def kinds(ep):
    return [t.get("kind") or "turn" for t in ep["turns"] if t["role"] == "user"]


def test_default_library_equals_the_historical_user_frame_for_frame():
    a, env_a = run(UserSim(ScriptSource(), voice=Voice(FakeSpeech(), clone=False), behaviors=NOISY))
    b, env_b = run(UserSim(ScriptSource(), voice=Voice(FakeSpeech(), clone=False), behaviors=NOISY, behaviours=default_behaviours()))
    assert env_a.log == env_b.log and a["turns"] == b["turns"]
    assert "noise" in kinds(a) and ("aside" in kinds(a) or "away" in kinds(a))
    assert a["meta"]["user"]["behaviours"] == ["noise", "aside", "listening"] and a["meta"]["user"]["random_events"]["noise"] > 0


def test_an_empty_list_is_a_user_that_only_speaks_and_a_subset_runs_only_that_subset():
    none, _ = run(UserSim(ScriptSource(), voice=Voice(FakeSpeech(), clone=False), behaviors=NOISY, behaviours=[]))
    assert kinds(none) == ["turn", "turn", "turn"] and none["meta"]["user"]["random_events"] == {} and "noise_events" not in none["meta"]["user"]
    only_noise, _ = run(UserSim(ScriptSource(), voice=Voice(FakeSpeech(), clone=False), behaviors=NOISY, behaviours=[NoiseEvents()]))
    assert set(kinds(only_noise)) == {"turn", "noise"}
    only_asides, _ = run(UserSim(ScriptSource(), voice=Voice(FakeSpeech(), clone=False), behaviors=NOISY, behaviours=[Asides()]))
    assert "noise" not in kinds(only_asides) and ({"aside", "away"} & set(kinds(only_asides)))
    # each behaviour draws from its own seeded stream: the noise events are the same with or without the asides
    both, _ = run(UserSim(ScriptSource(), voice=Voice(FakeSpeech(), clone=False), behaviors=NOISY))
    noise_times = lambda ep: [t["start_time"] for t in ep["turns"] if t.get("kind") == "noise"]  # noqa: E731
    shorter = min(both["meta"]["duration_ms"], only_noise["meta"]["duration_ms"])  # the asides make the call longer
    assert [x for x in noise_times(both) if x < shorter] == [x for x in noise_times(only_noise) if x < shorter]


def test_a_behaviour_of_ones_own_plugs_in():
    class Sigh(Behaviour):
        """A sigh 2 s after the agent's first turn ends, once: a sound the agent should ignore."""

        name = "sigh"

        def init(self, user, st):
            st["sighed"] = False
            st["events"]["sighs"] = 0

        def due(self, st, t):
            first = min((s.end for s in st["agent"].values()), default=None)
            return None if first is None or st["sighed"] else max(first + 2000, t + 1)

        async def sounds(self, user, st, t):
            first = min((s.end for s in st["agent"].values()), default=None)
            if first is None or st["sighed"] or t < first + 2000:
                return []
            st["sighed"] = True
            st["events"]["sighs"] += 1
            return [await user._emit(st, t, UserTurn("", kind="noise", expects="ignore", label="sigh", dur=600))]

    ep, _ = run(UserSim(ScriptSource(), voice=Voice(FakeSpeech(), clone=False), behaviours=[Sigh()]))
    sighs = [t for t in ep["turns"] if t.get("label") == "sigh"]
    assert len(sighs) == 1 and ep["meta"]["user"]["random_events"] == {"sighs": 1} and ep["meta"]["user"]["behaviours"] == ["sigh"]
    ev = [e for e in ep["eval"]["scores"]["events"] if e["turn"] == sighs[0]["id"]]
    assert ev and ev[0]["expects"] == "ignore"  # the scorer judges the agent against the behaviour's own label


def test_role_user_names_its_behaviour_rows():
    c = Composition.bundle("minimal")
    c.apply("""
- insert:
    - {id: lines, plugin: source/script}
    - {id: tts, plugin: backend/tts_fake}
    - {id: noise, plugin: behaviour/noise}
    - {id: asides, plugin: behaviour/asides}
- id: user
  plugin: role/user
  config: {source: lines, voice: {tts: tts, clone: false}, behaviors: {noise_per_min: 30.0, aside_per_min: 20.0}, behaviours: [asides, noise]}
- id: tasks
  config: {tasks: [{id: t, scenario: {persona: a caller, instructions: ask, profile: {surroundings: street, language: en}, turns: ["Hi, when do you open tomorrow?", "And on Sunday?", "Thanks, bye."]}}]}
- id: agent
  config: {replies: ["We open at nine tomorrow morning and we are open until six in the evening, with a short break at noon.", "On Sundays we are open from ten to four.", "You're welcome, goodbye."], yield_after: 160}
- id: env
  config: {max_ms: 120000}
""", "p")
    run_ = compose(c.resolve())
    rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(run_.tasks[0], 0, episode_id="x")))
    assert rec["meta"]["user"]["behaviours"] == ["aside", "noise"] and "noise" in kinds(rec)  # no listener: no listening behaviour
    c.apply("- id: user\n  config: {source: lines, voice: {tts: tts, clone: false}, behaviors: {noise_per_min: 30.0, aside_per_min: 20.0}, behaviours: []}", "q")
    rec2, _, _ = asyncio.run(run_episode(compose(c.resolve()), EpisodeContext(run_.tasks[0], 0, episode_id="x")))
    assert kinds(rec2) == ["turn", "turn", "turn"]
