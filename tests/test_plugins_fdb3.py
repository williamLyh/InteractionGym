"""The fdb3 bundles against examples/fdb3_ab.py on main: the cascaded A and B episodes equal the ones the script's
make_env / make_agent build (fakes in place of the servers), the per-recording caps and tool latency, the output
layout and the meta / eval fields its report reads; the duplex bundle's per-domain prompt."""

import asyncio
import json
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.cascaded import CascadedAgent, Endpointing, LatencyModel
from interaction_gym.benchmarks import benchmark_user, fdb3
from interaction_gym.clients import CloneWithFallback, FakeChat, FakeSpeech
from interaction_gym.plugins import REGISTRY, Composition, EpisodeContext, compose, plugin
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.plugins.cli import main
from interaction_gym.tools import ToolWorld
from interaction_gym.traj import episode
from interaction_gym.user import STOP, ReplayUser, ResponseDelay, TurnTaking, Voice
from tests.test_fdb3 import FakeASR, FakeToolLLM, silence, tone

SR = 16000
SPEC = AgentSpec(chunk_ms=100, obs=("user.speech", "tool.result"), audio="user.audio", sr=SR)
CALLER = [f"Thanks, that's all. {STOP}"]

if "backend/test_asr" not in {p.key for p in REGISTRY.all()}:
    plugin("backend", "test_asr")(lambda ctx, kw: FakeASR())
    plugin("backend", "test_tool_llm")(lambda ctx, kw: FakeToolLLM())

FAKES = """
- id: user_llm
  plugin: backend/fake_chat
  config: {replies: ["Thanks, that's all. ###STOP###"], seed: episode, per_episode: true}
- id: tts
  plugin: backend/tts_fake
- id: clone_tts
  plugin: backend/tts_fake
- id: asr
  plugin: backend/test_asr
- id: agent_llm
  plugin: backend/test_tool_llm
"""


def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "fdb3"
    for rid, profile in (("ecommerce_01_spk", "normal"), ("ecommerce_02_spk", "instant")):
        meta = {"id": rid.rsplit("_", 1)[0], "domain": "ecommerce_support", "title": "t", "difficulty": "easy",
                "dialogue": [{"user": "um... track my order ABC123", "ai": "I will check order ABC123."}], "acting_notes": "",
                "disfluency_features": ["PAUSE"], "expected_tool_calls": [{"function": "track_order", "args": {"order_id": "ABC123"}}],
                "state_rollback_test": False, "latency_profile": profile}
        (root / rid).mkdir(parents=True)
        (root / rid / "metadata.json").write_text(json.dumps(meta))
        (tone(2500) + silence(300) + silence(2500)).write_wav(root / rid / "input.wav")
    return root


def bundle(data: Path, name: str = "fdb3", extra: str = "") -> Composition:
    c = Composition.bundle(name)
    c.set(f"tasks.data={data}")
    c.apply(FAKES + extra, "fakes")
    return c


def make_env_as_main(task, cond: str) -> Env:
    """examples/fdb3_ab.py ``make_env`` for the cascaded agent, fakes in place of the servers."""
    tools = ToolWorld(fdb3.MockAPIBackend(), fdb3.latency_ms(task))
    if cond == "A":
        return Env({"user": ReplayUser(voice=Voice()), "tools": tools}, SPEC, max_ms=task.scenario["benchmark"]["user_end_ms"] + 60_000, end_idle_ms=3000)
    clone = CloneWithFallback(FakeSpeech(), FakeSpeech())
    user = benchmark_user(fdb3.UserSource(FakeChat(CALLER), 8), voice=Voice(clone, clone=clone),
                          timing=TurnTaking(response_delay=ResponseDelay(), yield_after_ms=None, nudge_after_ms=10_000))
    return Env({"user": user, "tools": tools}, SPEC, max_ms=180_000, end_idle_ms=3000)


async def run_as_main(task, cond, seed, cache):
    env = make_env_as_main(task, cond)
    agent = CascadedAgent(SPEC, FakeASR(), FakeToolLLM(), FakeSpeech(sr=SR), instructions=fdb3.AGENT_INSTRUCTIONS, voice="aiden", seed=seed,
                          cache=cache, endpointing=Endpointing(silence_ms=800), latency=LatencyModel())
    obs = await env.reset(task, seed)
    while True:
        obs, _, done = await env.step(await agent.act(env.t, obs))
        if env.truncated or (done and not agent.busy):
            break
    return episode(env, f"x/{task.id}/s{seed}/{cond}"), agent


def test_arms_reproduce_the_script_s_conditions_frame_for_frame(tmp_path):
    data = dataset(tmp_path)
    run = compose(bundle(data).resolve())
    arms = run.driver.arms(run)
    assert [t.id for t in run.tasks] == ["ecommerce_01_spk", "ecommerce_02_spk"] and run.tasks[0].scenario["profile"] == {"language": "en"}
    for task in run.tasks:
        cache_bundle, cache_main = {}, {}
        for arm in ("A", "B"):
            ep = EpisodeContext(task, 0, arm, cache_bundle, f"x/{task.id}/s0/{arm}")
            rec, env, agent = asyncio.run(run_episode(arms[arm], ep))
            ep_main, agent_main = asyncio.run(run_as_main(fdb3.load(data, only=[task.id])[0][1], arm, 0, cache_main))
            assert rec["turns"] == ep_main["turns"], (task.id, arm)
            assert [c["name"] for t in rec["turns"] for c in t.get("tool_calls", [])] == ["track_order"], (task.id, arm)
            assert rec["meta"]["agent_events"] == agent_main.events and rec["meta"]["recording"] == task.id
            assert env.max_ms == (task.scenario["benchmark"]["user_end_ms"] + 60_000 if arm == "A" else 180_000)
            assert rec["eval"]["benchmark"].keys() == {"calls", "u1_ms", "timing", "user_turns"}  # the runner's quick_metrics
            assert rec["eval"]["benchmark"]["calls"] == fdb3.agent_calls(ep_main)
    # the tool latency follows each recording's profile: a world per episode
    worlds = {t.id: arms["A"].make_env(EpisodeContext(t, 0, "A")).nodes["tools"] for t in run.tasks}
    assert worlds["ecommerce_01_spk"].latency_ms(None) == 500 and worlds["ecommerce_02_spk"].latency_ms(None) == 0


def test_run_layout_and_meta_match_the_script(tmp_path, capsys):
    data = dataset(tmp_path)
    out = tmp_path / "run"
    patch = tmp_path / "fakes.yml"
    patch.write_text(FAKES)
    cmd = ["run", "fdb3", "--set", f"tasks.data={data}", "--patch", str(patch), "--out", str(out), "--seeds", "0", "--tag", "cascaded-official",
           "--set", "driver.media=false", "--set", "driver.concurrency=2"]
    assert main(cmd) == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("episodes.jsonl"))
    assert files == ["cascaded-official/A/episodes.jsonl", "cascaded-official/B/episodes.jsonl"]
    eps = [json.loads(x) for x in (out / "cascaded-official" / "B" / "episodes.jsonl").read_text().splitlines()]
    m = eps[0]["meta"]
    assert m["run_id"] == "fdb3-ab-cascaded-official" and m["condition"] == "B" and m["recording"] == m["task_id"]
    assert m["episode_id"].split("/", 1)[1].rsplit("/", 1)[0] == f"{m['task_id']}/s0"  # what the script's report keys on
    assert m["agent"]["kind"] == "cascaded" and m["agent"]["endpointing"]["silence_ms"] == 800 and "agent_events" in m
    assert m["task"]["scenario"]["benchmark"]["user_end_ms"] and "benchmark" in eps[0]["eval"]
    assert main(cmd) == 0
    assert "0 pairs to run (4 episodes done)" in [x for x in capsys.readouterr().out.splitlines() if x.startswith("ig:")][-1]
    # the script's report and `ig report` (report/fdb3 with a fake judge row) write the same files
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import fdb3_ab

    verdict = '{"correct": true, "asks": false, "claims": false, "labels": ["close"]}'
    fdb3_ab.OpenAIChat = lambda *a, **k: FakeChat([verdict])
    asyncio.run(fdb3_ab.report(SimpleNamespace(out=str(out), tags="cascaded-official", pages=False, agent="cascaded-official", endpoint_ms=800)))
    script_summary = (out / "summary.md").read_text()
    script_rows = (out / "cascaded-official" / "rows.json").read_text()
    rows = json.loads(script_rows)
    assert len(rows) == 2 and all(r["A"]["pass"] and r["B"]["pass_final"] and r["consistency"]["ok"] for r in rows)
    assert "### Pass@1" in script_summary and "| all | 2 | 1.000 (2/2) |" in script_summary
    judge = tmp_path / "judge.yml"
    judge.write_text("- id: judge_llm\n  plugin: backend/fake_chat\n  config: {replies: [%s]}\n" % json.dumps(verdict))
    assert main(["report", "fdb3", "--set", f"tasks.data={data}", "--patch", str(patch), "--patch", str(judge), "--out", str(out), "--tag", "cascaded-official"]) == 0
    assert (out / "summary.md").read_text() == script_summary and (out / "cascaded-official" / "rows.json").read_text() == script_rows


def test_duplex_bundle_names_the_caller_s_domain_in_the_prompt(tmp_path):
    data = dataset(tmp_path)
    extra = """
- id: agent
  plugin: agent/canned
  config: {replies: ["Tracking order ABC123 for you."]}
- id: controlled
  config: {name: agent, chunk_ms: 200, audio: user.audio, sr: 16000}
"""
    run = compose(bundle(data, "fdb3_duplex", extra).resolve())
    assert "tools" not in run.nodes and run.env_config.end_idle_ms == 8000  # no tools; more quiet in the open loop
    from interaction_gym.plugins.builtin.agents import instructions_for

    res = Composition.bundle("fdb3_duplex").resolve()
    cfg = res.configs["agent"]
    text = instructions_for(cfg["instructions"], cfg["instructions_vars"], run.tasks[0])
    assert text.startswith("You are the voice assistant of an online shop (product search, shopping cart, order tracking), answering")
    other = fdb3.load(data, only=["ecommerce_01_spk"])[0][1]
    other.scenario["benchmark"]["domain"] = "unknown"
    assert "voice assistant of a customer service line," in instructions_for(cfg["instructions"], cfg["instructions_vars"], other)
    arms = run.driver.arms(run)
    rec, env, _ = asyncio.run(run_episode(arms["A"], EpisodeContext(run.tasks[0], 0, "A", {}, "x")))
    assert env.end_idle_ms == 8000 and rec["meta"]["recording"] == "ecommerce_01_spk"
    rec, env, _ = asyncio.run(run_episode(arms["B"], EpisodeContext(run.tasks[0], 0, "B", {}, "x")))
    assert env.end_idle_ms == 3000 and env.max_ms == 180_000
