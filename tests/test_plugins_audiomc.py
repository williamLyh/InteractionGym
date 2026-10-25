"""The audiomc bundle against examples/audiomc_ab.py on main: the same A and B episodes, the same output layout and
meta fields (so the script's ``report`` command reads a bundle run unchanged), and the A/B driver's mechanics
(shared cache, retries, resume, warm-up, output-mode check)."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

from interaction_gym import Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.benchmarks import audiomc, benchmark_user
from interaction_gym.clients import FakeChat, FakeSpeech
from interaction_gym.plugins import Composition, EpisodeContext, PluginError, compose
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.plugins.cli import main
from interaction_gym.traj import episode
from interaction_gym.user import ResponseDelay, TurnTaking, Voice
from tests.test_audiomc import _root

REPLIES = ["Sure, I can help with that.", "Done, K4 is added.", "Anything else?"]
REPLAN = ["Thanks! Please add K4 to the list."]

# the bundle without servers: a canned agent in place of MiniCPM-o, fakes for the replanning LLM and the cloning TTS
FAKES = """
- id: user_llm
  plugin: backend/fake_chat
  config: {replies: ["Thanks! Please add K4 to the list."], seed: episode}
- id: tts
  plugin: backend/tts_fake
- id: clone_tts
  plugin: backend/tts_fake
- id: agent
  plugin: agent/canned
  config: {replies: ["Sure, I can help with that.", "Done, K4 is added.", "Anything else?"], yield_after: 160}
- id: controlled
  config: {name: agent, chunk_ms: 200, audio: user.audio, sr: 16000}
- id: traces
  disabled: true
- id: driver
  config: {concurrency: 2, media: false, retries: 1, warmup: false, run_id: audiomc-ab, arms: {A: [], B: [{id: user, config: {source: replan, benchmark: true, voice: {tts: clone, clone: clone}, timing: {response_delay: {}, yield_after_ms: null, nudge_after_ms: 10000}}}]}}
"""


def _data(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return _root(d)


def bundle(data: Path, extra: str = "") -> Composition:
    c = Composition.bundle("audiomc")
    c.set(f"tasks.data={data}")
    c.set("tasks.per_axis=null")
    c.apply(FAKES + extra, "fakes")
    return c


def spec():
    from interaction_gym import AgentSpec

    return AgentSpec(chunk_ms=200, audio="user.audio", sr=16000)


def make_env_as_main(task, cond: str) -> Env:
    """examples/audiomc_ab.py ``make_env`` with the fakes in place of the servers."""
    timing = TurnTaking(response_delay=ResponseDelay(), yield_after_ms=None, nudge_after_ms=10_000)
    if cond == "A":
        user = benchmark_user(audiomc.ScriptSource(), voice=Voice(), timing=timing)
    else:
        clone = FakeSpeech()
        user = benchmark_user(audiomc.ReplanSource(FakeChat(REPLAN)), voice=Voice(clone, clone=clone), timing=timing)
    return Env({"user": user}, spec(), max_ms=900_000, end_idle_ms=8000)


async def run_as_main(task, cond, seed):
    env, agent = make_env_as_main(task, cond), CannedAgent(REPLIES, spec(), yield_after=160)
    obs = await env.reset(task, seed)
    while True:
        obs, _, done = await env.step(agent.act(env.t, obs))
        if env.truncated or (done and not getattr(agent, "busy", False)):
            break
    return episode(env, f"x/{task.id}/s{seed}/{cond}")


def test_arms_reproduce_the_script_s_conditions_frame_for_frame(tmp_path):
    data = _data(tmp_path)
    run = compose(bundle(data).resolve())
    arms = run.driver.arms(run)
    assert list(arms) == ["A", "B"]
    task = run.tasks[0]
    for arm in arms:
        ep_bundle, env, _ = asyncio.run(run_episode(arms[arm], EpisodeContext(task, 0, arm, {}, f"x/{task.id}/s0/{arm}")))
        ep_main = asyncio.run(run_as_main(task, arm, 0))
        assert ep_bundle["turns"] == ep_main["turns"], arm
        assert ep_bundle["meta"]["user"] == ep_main["meta"]["user"], arm  # same user profile (mode, content, voice, soundscape)
    assert arms["B"].make_env().nodes["user"].source.__class__ is audiomc.ReplanSource
    assert arms["A"].make_env().nodes["user"].source.__class__ is audiomc.ScriptSource


def test_run_layout_and_meta_match_the_script_and_its_report_reads_them(tmp_path, monkeypatch, capsys):
    data = _data(tmp_path)
    out = tmp_path / "run"
    comp = bundle(data)
    patch = tmp_path / "fakes.yml"
    patch.write_text(FAKES)
    assert main(["run", "audiomc", "--set", f"tasks.data={data}", "--set", "tasks.per_axis=null", "--patch", str(patch),
                 "--out", str(out), "--seeds", "0,1", "--limit", "2", "--tag", "minicpmo"]) == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("episodes.jsonl"))
    assert files == ["minicpmo/A/episodes.jsonl", "minicpmo/B/episodes.jsonl"]  # out/<tag>/<cond>/, as the script writes
    eps = {c: [json.loads(x) for x in (out / "minicpmo" / c / "episodes.jsonl").read_text().splitlines()] for c in "AB"}
    assert len(eps["A"]) == len(eps["B"]) == 4  # 2 tasks × 2 seeds
    m = eps["B"][0]["meta"]
    # every field the script's `report` reads, and the ones it wrote
    assert m["episode_id"].startswith("minicpmo/c") and m["episode_id"].endswith("/B") and m["run_id"] == "audiomc-ab"
    assert m["condition"] == "B" and m["agent_name"] == "minicpmo" and m["task_id"] in ("c0", "c1") and m["seed"] in (0, 1)
    assert m["agent"]["name"] == "agent" and m["controlled"] == "agent" and "wall_s" in m and m["config"]["digest"]
    assert {"duration_ms", "end_reason", "user", "task"} <= set(m)
    # the script's report on this output: judge replaced by a fake that passes every rubric item
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import audiomc_ab
    import fdb2_ab

    monkeypatch.setattr(audiomc_ab, "llm", lambda seed, temperature=0.0, max_tokens=800: FakeChat(['{"criteria_met": true}']))
    monkeypatch.setattr(fdb2_ab, "LLM_URL", "http://127.0.0.1:1/v1")

    from types import SimpleNamespace

    asyncio.run(audiomc_ab.report(SimpleNamespace(data=str(data), out=str(out), concurrency=2)))
    summary = (out / "summary.md").read_text()
    rows = json.loads((out / "rows.json").read_text())
    assert "| minicpmo | all | 4 |" in summary and len(rows) == 8 and all(r["rubric_rate"] == 1.0 for r in rows)
    # `ig report` (report/audiomc with a fake judge row) writes the same files as the script's report
    (out / "summary.md").rename(out / "summary.script.md")
    (out / "rows.json").rename(out / "rows.script.json")
    judge = tmp_path / "judge.yml"
    judge.write_text('- id: judge_llm\n  plugin: backend/fake_chat\n  config: {replies: [\'{"criteria_met": true}\']}\n')
    assert main(["report", "audiomc", "--set", f"tasks.data={data}", "--set", "tasks.per_axis=null", "--patch", str(patch), "--patch", str(judge),
                 "--out", str(out), "--tag", "minicpmo"]) == 0
    assert (out / "summary.md").read_text() == summary and (out / "rows.json").read_text() == (out / "rows.script.json").read_text()
    # resume: nothing left to run
    assert main(["run", "audiomc", "--set", f"tasks.data={data}", "--set", "tasks.per_axis=null", "--patch", str(patch),
                 "--out", str(out), "--seeds", "0,1", "--limit", "2", "--tag", "minicpmo"]) == 0
    assert "0 pairs to run (8 episodes done)" in capsys.readouterr().out


def test_arms_share_a_cache_and_the_episode_seed_reaches_the_user_llm(tmp_path):
    data = _data(tmp_path)
    seen = []

    class Spy(FakeChat):
        async def chat(self, messages, **kw):
            seen.append(kw.get("seed"))
            return await super().chat(messages, **kw)

    from interaction_gym.plugins import REGISTRY, plugin

    if "backend/spy_chat" not in {p.key for p in REGISTRY.all()}:
        plugin("backend", "spy_chat")(lambda ctx, kw: __import__("interaction_gym.plugins.builtin.backends", fromlist=["EpisodeSeeded"]).EpisodeSeeded(Spy(REPLAN)))
    caches = []

    def remember(ctx, kw):
        from interaction_gym.plugins.builtin.agents import Factory

        def make(ep):
            caches.append((ep.arm, ep.seed, id(ep.cache)))
            return CannedAgent(REPLIES, ctx["controlled"].spec, yield_after=160)
        return Factory(make)

    if "agent/remember" not in {p.key for p in REGISTRY.all()}:
        plugin("agent", "remember", requires=("controlled",))(remember)
    comp = bundle(data, "- id: user_llm\n  plugin: backend/spy_chat\n- id: agent\n  plugin: agent/remember\n  config: {}\n")
    run = compose(comp.resolve())
    asyncio.run(run.driver.run(run, out=tmp_path / "o", seeds=[3], limit=1, log=lambda *a, **k: None))
    assert seen == [3]  # arm B's one replanning call carried the episode seed
    assert len(caches) == 2 and caches[0][2] == caches[1][2] and [c[0] for c in caches] == ["A", "B"]  # one cache per (task, seed) pair


def test_retries_then_gives_up_and_output_mode_is_checked(tmp_path, capsys):
    data = _data(tmp_path)
    from interaction_gym.plugins import REGISTRY, plugin
    from interaction_gym.plugins.builtin.agents import Factory

    n = {"calls": 0}

    def flaky(ctx, kw):
        def make(ep):
            n["calls"] += 1
            if n["calls"] <= 2:
                raise ConnectionError("server not ready")
            return CannedAgent(REPLIES, ctx["controlled"].spec, yield_after=160)
        return Factory(make, audio_out=kw.get("audio_out", False))

    if "agent/flaky" not in {p.key for p in REGISTRY.all()}:
        plugin("agent", "flaky", requires=("controlled",))(flaky)
    comp = bundle(data, "- id: agent\n  plugin: agent/flaky\n  config: {}\n"
                        "- id: driver\n  config: {concurrency: 1, media: false, retries: 3, retry_wait_s: 0, retry_jitter_s: 0, arms: {A: []}}\n")
    run = compose(comp.resolve())
    logs = []
    eps = asyncio.run(run.driver.run(run, out=tmp_path / "o", seeds=[0], limit=1, log=lambda *a, **k: logs.append(a[0])))
    assert len(eps) == 1 and n["calls"] == 3 and sum("attempt" in x for x in logs) == 2
    first = json.loads((tmp_path / "o" / "agent" / "A" / "episodes.jsonl").read_text().splitlines()[0])["meta"]["agent"]
    assert first["audio_out"] is False and first["name"] == "agent"  # the factory's declaration reaches meta.agent
    # the file now holds text-only episodes: an audio-output agent must not append to it
    comp2 = bundle(data, "- id: agent\n  plugin: agent/flaky\n  config: {audio_out: true}\n- id: driver\n  config: {concurrency: 1, media: false, arms: {A: []}}\n")
    run2 = compose(comp2.resolve())
    with pytest.raises(ValueError, match="text-only agent episodes; this run is audio-output"):
        asyncio.run(run2.driver.run(run2, out=tmp_path / "o", seeds=[0], limit=1, log=lambda *a, **k: None))


def test_env_references_in_bundles_expand_and_fail_loud(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_AUDIOMC_DATA", str(tmp_path))
    monkeypatch.delenv("IG_LLM_URL", raising=False)
    res = Composition.bundle("audiomc").resolve()
    assert res.configs["tasks"].data == str(tmp_path) and res.configs["user_llm"].base_url == "http://127.0.0.1:8000/v1"
    c = Composition.bundle("audiomc")
    c.set("tasks.data=${IG_NO_SUCH_THING}")
    with pytest.raises(PluginError, match="IG_NO_SUCH_THING.* is not set and has no default"):
        c.resolve()
    c = Composition.bundle("audiomc")
    c.set("tasks.per_axis=${IG_PER_AXIS:-7}")
    assert c.resolve().configs["tasks"].per_axis == 7  # a whole-value reference keeps its type
    monkeypatch.delenv("IG_JUDGE_URL", raising=False)
    monkeypatch.setenv("IG_LLM_URL", "http://llm:1/v1")
    c = Composition.bundle("audiomc")
    c.set("tasks.data=${IG_JUDGE_URL:-${IG_LLM_URL:-http://x}}")
    assert c.resolve().configs["tasks"].data == "http://llm:1/v1"  # a default may hold another reference
