"""The MiniCPM-o suite bundles against examples/minicpmo_suite.py: the default set (an evaluation user, a background
only where the scenario has one) and the chat set (the realistic user) give the script's episodes frame for frame;
the agent's prompt comes from the scenario; the pages sink writes the run's notes and index."""

import asyncio
import json

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.benchmarks import benchmark_user
from interaction_gym.clients import Cached, FakeChat, FakeSpeech
from interaction_gym.plugins import Composition, EpisodeContext, compose
from interaction_gym.plugins.builtin.agents import instructions_for
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.plugins.cli import main
from interaction_gym.soundscape import Soundscape
from interaction_gym.traj import episode
from interaction_gym.user import PERSONA, QWEN3_TTS_VOICES, STOP, LLMInterrupt, LLMSource, ResponseDelay, TurnTaking, UserSim, Voice

SR = 24000
CALLER = ["I'd like a table for two tonight at seven.", "Yes, seven is fine, under Chen.", f"Perfect, thank you, bye. {STOP}"]
AGENT = ["Sure, for how many people and at what time?", "Seven for two, under what name?", "Done, see you tonight."]
FAKES = """
- id: user_llm
  plugin: backend/fake_chat
  config: {replies: %s, per_episode: true}
- insert:
    - {id: listener_llm, plugin: backend/fake_chat, config: {replies: ["LISTEN"], per_episode: true}}
- id: tts
  plugin: backend/tts_fake
  config: {sr: 24000}
- id: clone_tts
  plugin: backend/tts_fake
  config: {sr: 24000}
- id: agent
  plugin: agent/canned
  config: {replies: %s, yield_after: 160}
- id: traces
  disabled: true
- id: driver
  config: {concurrency: 2, media: false, run_id: minicpmo-suite}
""" % (json.dumps(CALLER), json.dumps(AGENT))
LISTENER = "- id: user\n  config: {source: lines, listener: listener_llm, benchmark: true, voice: {tts: tts, clone: clone_tts, voices: qwen3}, timing: {response_delay: {}}, soundscape: {background: scenario}}\n"
LISTENER_CHAT = "- id: user\n  config: {source: lines, listener: listener_llm, voice: {tts: tts, clone: clone_tts, voices: qwen3}, timing: {response_delay: {}}, behaviors: !py interaction_gym.user:PERSONA}\n"


def scenarios(which):
    from importlib import resources

    return json.loads((resources.files("interaction_gym") / "data" / "minicpmo_suite.json").read_text())[which]


async def run_as_script(name, sc, seed=0, chunk_ms=200):
    """examples/minicpmo_suite.py ``run`` with the fakes (two FakeChats in place of its one LLM)."""
    voice = Voice(Cached(FakeSpeech(sr=SR)), voices=QWEN3_TTS_VOICES, clone=FakeSpeech(sr=SR))
    if sc.get("realistic"):
        user = UserSim(LLMSource(FakeChat(CALLER)), voice, interrupt=LLMInterrupt(FakeChat(["LISTEN"])), timing=TurnTaking(response_delay=ResponseDelay()),
                       behaviors=PERSONA, soundscape=Soundscape())
    else:
        user = benchmark_user(LLMSource(FakeChat(CALLER)), voice=voice, interrupt=LLMInterrupt(FakeChat(["LISTEN"])),
                              timing=TurnTaking(response_delay=ResponseDelay()), soundscape=Soundscape(background="background" in sc))
    task = Task(id=name, scenario={k: sc[k] for k in ("persona", "profile", "first_turn", "turn_taking", "background", "voice") if k in sc} | {"instructions": sc["goal"]})
    spec = AgentSpec(chunk_ms=chunk_ms, audio="user.audio", sr=SR)
    env = Env({"user": user}, spec, max_ms=sc.get("max_ms", 120_000))
    agent = CannedAgent(AGENT, spec, yield_after=160)
    obs = await env.reset(task, seed=seed)
    done = False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs))
    return episode(env, name)


def test_default_set_equals_the_script(tmp_path):
    c = Composition.bundle("minicpmo_suite")
    c.apply(FAKES + LISTENER, "fakes")
    run_ = compose(c.resolve())
    sc = scenarios("default")
    assert [t.id for t in run_.tasks] == list(sc)
    for task in run_.tasks:
        rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(task, 0, episode_id=task.id)))
        ep = asyncio.run(run_as_script(task.id, sc[task.id]))
        assert rec["turns"] == ep["turns"], task.id
        same = lambda u: json.dumps(u, sort_keys=True).replace("EpisodeFakeChat", "FakeChat")  # noqa: E731  (the fakes differ by class name only)
        assert same(rec["meta"]["user"]) == same(ep["meta"]["user"]), task.id
        assert env.nodes["user"].soundscape.background == ("background" in sc[task.id])
        cfg = Composition.bundle("minicpmo_suite").resolve().configs["agent"]
        assert instructions_for(cfg["instructions"], cfg["instructions_vars"], task) == sc[task.id]["agent"]  # the agent's prompt, per scenario


def test_chat_set_runs_the_realistic_user_as_the_script():
    c = Composition.bundle("minicpmo_suite_chat")
    c.set("tasks.only=[chat-hiking-sam]")
    c.apply(FAKES + LISTENER_CHAT, "fakes")
    run_ = compose(c.resolve())
    task = run_.tasks[0]
    rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(task, 0, episode_id=task.id)))
    sc = scenarios("chat")[task.id]
    ep = asyncio.run(run_as_script(task.id, sc))
    assert rec["turns"] == ep["turns"]
    user = env.nodes["user"]
    assert user.behaviors == PERSONA and user.soundscape.background is True and user.soundscape.bank == "default"
    assert "!py" in json.dumps(run_.resolved.meta()["rows"])  # the preset recorded as plain data


def test_ig_run_writes_the_pages_and_notes(tmp_path):
    out = tmp_path / "run"
    patch = tmp_path / "fakes.yml"
    patch.write_text(FAKES + LISTENER)
    assert main(["run", "minicpmo_suite", "--set", "tasks.only=[booking-alex, return-jake]", "--patch", str(patch), "--out", str(out)]) == 0
    notes = json.loads((out / "notes.json").read_text())
    assert set(notes) == {"booking-alex/s0", "return-jake/s0"} and notes["booking-alex/s0"].startswith("Alex Chen (female, 34) · voice ")
    assert (out / "index.html").exists() and (out / "episodes.jsonl").exists()
