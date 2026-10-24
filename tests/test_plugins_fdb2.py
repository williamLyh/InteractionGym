"""The fdb2 bundle against examples/fdb2_ab.py on main: `ig prepare` writes the same reference scripts, the A and B
episodes equal the ones the script's make_env builds, the output layout and meta fields are what its `report` reads."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from interaction_gym import AgentSpec, Env
from interaction_gym.agents import CannedAgent
from interaction_gym.benchmarks import benchmark_user, fdb2
from interaction_gym.clients import FakeChat, FakeSpeech
from interaction_gym.plugins import Composition, EpisodeContext, PluginError, compose
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.plugins.cli import main
from interaction_gym.traj import episode
from interaction_gym.user import ReplayUser, ResponseDelay, TurnTaking, Voice
from tests.test_fdb2 import _data

EXAMINER = ["Hi, I'd like a pizza.", "A large one please. The conversation is over."]
ASSISTANT = ["Sure, what size?"]
REPLIES = ["Sure, what size would you like?", "A large pizza, coming up.", "Anything else?"]

FAKES = """
- id: examiner_llm
  plugin: backend/fake_chat
  config: {replies: ["Hi, I'd like a pizza.", "A large one please. The conversation is over."], seed: episode, per_episode: true}
- id: reference_llm
  plugin: backend/fake_chat
  config: {replies: ["Sure, what size?"], seed: episode, seed_offset: 10000, per_episode: true}
- id: closing_llm
  plugin: backend/fake_chat
  config: {replies: ['{"T1": true, "T2": false, "T3": false, "T4": false}']}
- id: tts
  plugin: backend/tts_fake
- id: clone_tts
  plugin: backend/tts_fake
- id: agent
  plugin: agent/canned
  config: {replies: ["Sure, what size would you like?", "A large pizza, coming up.", "Anything else?"], yield_after: 160}
- id: controlled
  config: {name: agent, chunk_ms: 200, audio: user.audio, sr: 16000}
- id: traces
  disabled: true
- id: driver
  config: {concurrency: 2, media: false, retries: 1, warmup: false, run_id: "fdb2-ab-{tag}", limit_stride: true, arms: {A: [], B: [{id: user, plugin: role/user, config: {source: examiner, benchmark: true, voice: {tts: tts, voice: serena, clone: clone_tts}, timing: {response_delay: {}, yield_after_ms: 1000, nudge_after_ms: 10000}}}, {id: env, config: {max_ms: 120000, end_idle_ms: 3000}}]}}
"""


def bundle(data: Path, extra: str = "") -> Composition:
    c = Composition.bundle("fdb2")
    c.set(f"tasks.data={data}")
    c.apply(FAKES + extra, "fakes")
    return c


def spec():
    return AgentSpec(chunk_ms=200, audio="user.audio", sr=16000)


def prepared(tmp_path, seeds="0") -> tuple[Path, Path]:
    data = _data(tmp_path)
    out = tmp_path / "run"
    patch = tmp_path / "fakes.yml"
    patch.write_text(FAKES)
    assert main(["prepare", "fdb2", "--set", f"tasks.data={data}", "--patch", str(patch), "--out", str(out), "--seeds", seeds]) == 0
    return data, out


def test_prepare_writes_the_scripts_the_script_wrote(tmp_path, capsys):
    data, out = prepared(tmp_path)
    tasks = fdb2.load(data)
    for t in tasks:
        d = out / "scripts" / t.id / "s0"
        rec = json.loads((d / "script.json").read_text())
        ex = [x for x in rec["lines"] if x["role"] == "examiner"]
        assert [x["role"] for x in rec["lines"]] == ["examiner", "assistant", "examiner"]
        assert [x["text"] for x in ex] == EXAMINER and rec["lines"][1]["text"] == ASSISTANT[0]
        assert rec["task"] == t.id and rec["seed"] == 0 and rec["reached_end"] is True
        assert [x["audio"] for x in ex] == ["e00.wav", "e01.wav"] and all((d / x["audio"]).exists() for x in ex)
        durs = [x["dur"] for x in ex]
        assert [x["t"] for x in ex] == fdb2.schedule(rec["lines"], durs) and rec["script_end_ms"] == ex[-1]["t"] + ex[-1]["dur"]
    # a second prepare finds them and writes nothing new
    stamp = {p: p.stat().st_mtime for p in out.rglob("script.json")}
    patch = tmp_path / "fakes.yml"
    assert main(["prepare", "fdb2", "--set", f"tasks.data={data}", "--patch", str(patch), "--out", str(out), "--seeds", "0"]) == 0
    assert {p: p.stat().st_mtime for p in out.rglob("script.json")} == stamp
    assert "4 prepared" in capsys.readouterr().out


def make_env_as_main(task, rec: dict, d: Path, cond: str) -> Env:
    """examples/fdb2_ab.py ``make_env`` with the fakes in place of the servers (A_MAX_MS = MAX_MS = 120 s)."""
    if cond == "A":
        ex = [x for x in rec["lines"] if x["role"] == "examiner" and x["t"] < fdb2.MAX_MS]
        task.scenario["turns"] = [{"t": x["t"], "text": x["text"], "audio": str(d / x["audio"]), "final": i == len(ex) - 1} for i, x in enumerate(ex)]
        return Env({"user": ReplayUser(voice=Voice())}, spec(), max_ms=min(rec["script_end_ms"] + 15_000, fdb2.MAX_MS), end_idle_ms=8000)
    first = [x for x in rec["lines"] if x["role"] == "examiner"][0]
    task.scenario["first_turn"] = {"text": first["text"], "audio": str(d / first["audio"])}
    user = benchmark_user(fdb2.ExaminerSource(FakeChat(EXAMINER), 12), voice=Voice(FakeSpeech(), voice="serena", clone=FakeSpeech()),
                          timing=TurnTaking(response_delay=ResponseDelay(), yield_after_ms=1000, nudge_after_ms=10_000))
    return Env({"user": user}, spec(), max_ms=fdb2.MAX_MS, end_idle_ms=3000)


async def run_as_main(task, rec, d, cond, seed):
    env, agent = make_env_as_main(task, rec, d, cond), CannedAgent(REPLIES, spec(), yield_after=160)
    obs = await env.reset(task, seed)
    while True:
        obs, _, done = await env.step(agent.act(env.t, obs))
        if env.truncated or (done and not getattr(agent, "busy", False)):
            break
    return episode(env, f"x/{task.id}/s{seed}/{cond}")


def test_arms_reproduce_the_script_s_conditions_frame_for_frame(tmp_path):
    data, out = prepared(tmp_path)
    run = compose(bundle(data).resolve())
    arms = run.driver.arms(run)
    for r in arms.values():
        r.task_sources[0].out = out
    task = run.tasks[1]
    d = out / "scripts" / task.id / "s0"
    rec = json.loads((d / "script.json").read_text())
    for arm in ("A", "B"):
        ep = EpisodeContext(task, 0, arm, {}, f"x/{task.id}/s0/{arm}")
        ep_bundle, env, _ = asyncio.run(run_episode(arms[arm], ep))
        ep_main = asyncio.run(run_as_main(fdb2.load(data, only=[task.id])[0], rec, d, arm, 0))
        assert ep_bundle["turns"] == ep_main["turns"], arm
        ub, um = dict(ep_bundle["meta"]["user"]), dict(ep_main["meta"]["user"])
        assert ub.pop("llm", None) is not None or arm == "A"  # the fakes differ only by class name (EpisodeFakeChat vs FakeChat)
        um.pop("llm", None)
        assert ub == um, arm
        assert ep_bundle["meta"]["max_ms"] == 120000 and ep_bundle["meta"]["closing"] is False
    assert task.scenario.get("turns") is None and task.scenario.get("first_turn") is None  # the run's task is not mutated
    users = [t for t in ep_bundle["turns"] if t["role"] == "user" and t["text"]]
    assert users[0]["text"] == EXAMINER[0] and fdb2.reached_end(ep_bundle)
    assert ep_bundle["meta"]["examiner_log"] and ep_bundle["meta"]["examiner_log"][0]["i"] == 1  # B: the examiner's own log, fresh per episode


def test_a_fresh_examiner_per_episode_and_closing(tmp_path):
    data, out = prepared(tmp_path)
    run = compose(bundle(data, "- id: examiner\n  config: {llm: examiner_llm, max_turns: 12, closing: true, closing_llm: closing_llm}\n").resolve())
    arms = run.driver.arms(run)
    arms["B"].task_sources[0].out = out
    logs = []
    for seed in (0, 0):
        ep = EpisodeContext(run.tasks[0], seed, "B", {}, "x")
        rec, env, _ = asyncio.run(run_episode(arms["B"], ep))
        logs.append(rec["meta"]["examiner_log"])
        assert rec["meta"]["closing"] is True and env.nodes["user"].source.closing is not None
    assert logs[0] == logs[1] and len(logs[0]) >= 1  # two episodes, two examiners: the second log did not grow on the first
    assert "covered" in logs[0][0]  # the closing tracker ran


def test_run_layout_and_meta_match_the_script_and_the_run_needs_scripts(tmp_path, capsys):
    data, out = prepared(tmp_path, seeds="0,1")
    patch = tmp_path / "fakes.yml"
    args = ["run", "fdb2", "--set", f"tasks.data={data}", "--patch", str(patch), "--out", str(out), "--seeds", "0,1", "--limit", "2", "--tag", "minicpmo"]
    assert main(args) == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("episodes.jsonl"))
    assert files == ["minicpmo/A/episodes.jsonl", "minicpmo/B/episodes.jsonl"]
    eps = {c: [json.loads(x) for x in (out / "minicpmo" / c / "episodes.jsonl").read_text().splitlines()] for c in "AB"}
    assert len(eps["A"]) == len(eps["B"]) == 4
    # --limit 2 with limit_stride: every other task (the script's tasks[::n//limit][:limit])
    assert sorted({e["meta"]["task_id"] for e in eps["A"]}) == [t.id for t in fdb2.load(data)][::2]
    m = eps["B"][0]["meta"]
    assert m["run_id"] == "fdb2-ab-minicpmo" and m["condition"] == "B" and m["agent_name"] == "minicpmo" and m["seed"] in (0, 1)
    assert m["max_ms"] == 120000 and m["closing"] is False and m["examiner_log"] and m["agent"]["name"] == "agent"
    assert eps["A"][0]["meta"]["max_ms"] == 120000 and "examiner_log" not in eps["A"][0]["meta"]
    assert all(e["meta"]["duration_ms"] <= 120000 for e in eps["A"] + eps["B"])
    # the script's report reads these rows (judge replaced by a fake that returns one parseable judgement)
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import fdb2_ab

    judgement = '{\n "Turn-taking event and score": [\n  [2.5, 3.5]: 4, 5\n ],\n "Task-specific score": 4\n}'
    fdb2_ab.llm = lambda seed, temperature=0.0, max_tokens=800: FakeChat([judgement])
    asyncio.run(fdb2_ab.report(SimpleNamespace(data=str(data), out=str(out), concurrency=2, window_ms=0, name="")))
    rows = json.loads((out / "rows.json").read_text())
    assert len(rows) == 8 and {r["cond"] for r in rows} == {"A", "B"} and all(r["agent"] == "minicpmo" for r in rows)
    assert (out / "summary.md").exists() and "| minicpmo | all |" in (out / "summary.md").read_text()
    # `ig report` (report/fdb2 with a fake judge row) writes the same files as the script's report
    script_summary, script_rows = (out / "summary.md").read_text(), (out / "rows.json").read_text()
    judge = tmp_path / "judge.yml"
    judge.write_text("- id: judge_llm\n  plugin: backend/fake_chat\n  config: {replies: [%s]}\n" % json.dumps(judgement))
    assert main(["report", "fdb2", "--set", f"tasks.data={data}", "--patch", str(patch), "--patch", str(judge), "--out", str(out), "--tag", "minicpmo"]) == 0
    assert (out / "summary.md").read_text() == script_summary and (out / "rows.json").read_text() == script_rows
    # a run without scripts says what to do
    run = compose(bundle(data).resolve())
    with pytest.raises(PluginError, match="no reference script for .*run `ig prepare` first"):
        asyncio.run(run.driver.run(run, out=tmp_path / "elsewhere", seeds=[7], limit=1, log=lambda *a, **k: None))


def test_py_references_resolve_to_module_attributes():
    res = Composition.bundle("fdb2").resolve()
    assert res.configs["agent"]["instructions"] == fdb2.EXAMINEE_PROMPT  # !py interaction_gym.benchmarks.fdb2:EXAMINEE_PROMPT
    assert "instructions: You are a helpful AI assistant." in res.dump()  # recorded expanded
    c = Composition.bundle("fdb2")
    c.set("agent.instructions=!py interaction_gym.benchmarks.fdb2:NO_SUCH_NAME")
    with pytest.raises(PluginError, match="!py interaction_gym.benchmarks.fdb2:NO_SUCH_NAME"):
        c.resolve()
