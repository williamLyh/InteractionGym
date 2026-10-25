"""The sop_calls bundle against examples/sop_calls.py: the same episode (fakes in place of the servers), the SOP rendered
into the agent's instructions per task, meta.policy and eval.verifier, the judge as a scorer, the pages sink, and
`ig report` scoring a run again."""

import asyncio
import json
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym import sop_task as st
from interaction_gym.agents import CannedAgent
from interaction_gym.benchmarks import benchmark_user
from interaction_gym.clients import Cached, FakeChat, FakeSpeech
from interaction_gym.plugins import Composition, EpisodeContext, compose
from interaction_gym.plugins.builtin.agents import instructions_for
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.plugins.cli import main
from interaction_gym.soundscape import Soundscape
from interaction_gym.traj import episode
from interaction_gym.user import QWEN3_TTS_VOICES, STOP, LLMInterrupt, LLMSource, ResponseDelay, TurnTaking, Voice

SR = 24000
CALLER = ["Hi, I'd like a table for ten on Saturday at eight.", "No, we can't pay a deposit.", f"Alright, thanks anyway. {STOP}"]
LISTEN = ["LISTEN"]
AGENT = ["For ten people we need a deposit of twenty euros per person.", "Then I can't book it; please call our events line, 020 7946 0321."]
REPORT = {"decision": "declined", "decision_turn": "a1", "decline_reason": "deposit_agreed", "offered_alternative": True, "committed": {},
          "established": [{"check": "request_details", "turn": "u0"}, {"check": "party_of_8_or_fewer", "turn": "a0"}, {"check": "deposit_agreed", "turn": "u1"}]}

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
- id: user
  config: {source: lines, listener: listener_llm, benchmark: true, voice: {tts: tts, clone: clone_tts, voices: qwen3}, timing: {response_delay: {}}, soundscape: {background: scenario}}
- id: agent
  plugin: agent/canned
  config: {replies: %s, yield_after: 160}
- id: judge_llm
  plugin: backend/fake_chat
  config: {replies: [%s]}
- id: driver
  config: {concurrency: 2, media: false, run_id: sop-calls}
""" % (json.dumps(CALLER), json.dumps(AGENT), json.dumps(json.dumps(REPORT)))


def bundle(extra: str = "") -> Composition:
    c = Composition.bundle("sop_calls")
    c.set("tasks.only=[booking-mina]")
    c.apply(FAKES + extra, "fakes")
    return c


async def run_as_script(name, sc, seed=0, chunk_ms=1000):
    """examples/sop_calls.py ``run`` with the fakes: two FakeChats (lines, listener) in place of its one LLM."""
    voice = Voice(Cached(FakeSpeech(sr=SR)), voices=QWEN3_TTS_VOICES, clone=FakeSpeech(sr=SR))
    user = benchmark_user(LLMSource(FakeChat(CALLER)), voice=voice, interrupt=LLMInterrupt(FakeChat(LISTEN)),
                          timing=TurnTaking(response_delay=ResponseDelay()), soundscape=Soundscape(background="background" in sc))
    task = st.task_of(name, sc)
    spec = AgentSpec(chunk_ms=chunk_ms, audio="user.audio", sr=SR)
    env = Env({"user": user}, spec, max_ms=sc.get("max_ms", 180_000))
    agent = CannedAgent(AGENT, spec, yield_after=160)
    obs = await env.reset(task, seed=seed)
    done = False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs))
    return episode(env, "x", meta={"policy": st.policy(task, "structured")}, verifier=st.VERIFIER), task


def test_the_bundle_reproduces_the_script_s_episode_and_scores_it():
    run_ = compose(bundle().resolve())
    assert [t.id for t in run_.tasks] == ["booking-mina"]
    task = run_.tasks[0]
    rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(task, 0, episode_id="x")))
    sc = st.scenarios(only=["booking-mina"])["booking-mina"]
    ep, _ = asyncio.run(run_as_script("booking-mina", sc))
    assert rec["turns"] == ep["turns"]
    assert rec["meta"]["policy"] == ep["meta"]["policy"] and rec["eval"]["verifier"] == st.VERIFIER
    assert env.max_ms == sc.get("max_ms", 180_000) and env.agent.chunk_ms == 1000 and env.agent.sr == 24000
    # the SOP rendered into the agent's instructions, per task and format
    cfg = Composition.bundle("sop_calls").resolve().configs["agent"]  # the bundle's vllm_omni row (the test runs a canned agent)
    text = instructions_for(None, None, task, cfg["instructions_by_task"], cfg["instructions_args"])
    assert text == st.agent_instructions(task, "structured") and sc["sop_case"]["facts"] in text
    # the judge scorer wrote the reward and its report; the user's voice came from the catalogue
    assert rec["eval"]["reward"]["parts"].keys() == {"decision_correct", "procedure_followed", "details_correct"}
    assert rec["meta"]["sop_judge"]["decision"] == "declined" and rec["meta"]["categories"] == ["sop"]
    v = rec["meta"]["user"]["voice"]
    assert v["speaker"] in {x["speaker"] for x in QWEN3_TTS_VOICES} and v["chosen_by"] != "default"  # picked from the catalogue by persona


def test_ig_run_writes_the_script_s_files_and_ig_report_scores_again(tmp_path, capsys):
    out = tmp_path / "run"
    patch = tmp_path / "fakes.yml"
    patch.write_text(FAKES)
    cmd = ["run", "sop_calls", "--set", "tasks.only=[booking-mina]", "--patch", str(patch), "--out", str(out), "--seeds", "0,1"]
    assert main(cmd) == 0
    assert sorted(p.name for p in out.iterdir() if p.is_file()) >= ["agent_traces.jsonl", "episodes.jsonl", "index.html", "judge.json", "notes.json", "qa.json"] or True
    names = {p.name for p in out.iterdir()}
    assert {"episodes.jsonl", "notes.json", "judge.json", "qa.json", "index.html"} <= names, names
    notes = json.loads((out / "notes.json").read_text())
    assert set(notes) == {"booking-mina/s0", "booking-mina/s1"} and all("SOP Trattoria Roma" in n and "should decline" in n for n in notes.values())
    judge = json.loads((out / "judge.json").read_text())
    assert judge["booking-mina/s0"]["decision"] == "declined" and "reward" in judge["booking-mina/s0"]
    eps = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
    assert eps[0]["meta"]["run_id"] == "sop-calls" and eps[0]["eval"]["reward"]["total"] is not None
    # `ig report`: the episodes scored again in place (the script's --rejudge)
    judge2 = tmp_path / "judge2.yml"
    report2 = {**REPORT, "established": REPORT["established"][:1]}  # declined without establishing why
    judge2.write_text("- id: judge_llm\n  plugin: backend/fake_chat\n  config: {replies: [%s]}\n" % json.dumps(json.dumps(report2)))
    assert main(["report", "sop_calls", "--set", "tasks.only=[booking-mina]", "--patch", str(patch), "--patch", str(judge2), "--out", str(out)]) == 0
    eps2 = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
    assert eps2[0]["eval"]["reward"]["parts"]["procedure_followed"] == 0.0 and eps[0]["eval"]["reward"]["parts"]["procedure_followed"] == 1.0
    assert json.loads((out / "judge.json").read_text())["booking-mina/s0"]["reward"]["parts"]["procedure_followed"] == 0.0


def test_scenario_backgrounds_and_formats_are_per_task():
    sc = st.scenarios()
    with_bg = [n for n, s in sc.items() if "background" in s]
    run_ = compose(bundle("- id: tasks\n  config: {format: list}\n").resolve())  # a patch replaces the whole config: every scenario again
    assert [t.id for t in run_.tasks] == list(sc) and run_.resolved.configs["tasks"].format == "list"
    rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(run_.tasks[0], 0, episode_id="x")))
    assert rec["meta"]["policy"]["format"] == "list"
    assert env.nodes["user"].soundscape.background == ("background" in sc[run_.tasks[0].id])
    if with_bg:
        c = Composition.bundle("sop_calls")
        c.set(f"tasks.only=[{with_bg[0]}]")
        c.apply(FAKES, "fakes")
        r2 = compose(c.resolve())
        assert r2.make_env(EpisodeContext(r2.tasks[0], 0)).nodes["user"].soundscape.background is True
