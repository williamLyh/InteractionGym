"""The fdb_baseline bundle against examples/fdb_baseline.py: each synthetic sample replayed exactly as
full_duplex_bench.make_env replays it (a silent agent, as the script's --limit smoke), the per-sample metrics, the
per-subset output layout, clean variants, and `ig report` writing summary.json / per_sample.json."""

import asyncio
import json

from interaction_gym import AgentSpec
from interaction_gym.agents import CannedAgent
from interaction_gym.benchmarks import full_duplex_bench as fdb
from interaction_gym.plugins import Composition, EpisodeContext, compose
from interaction_gym.plugins.builtin.drivers import run_episode
from interaction_gym.plugins.cli import main
from interaction_gym.traj import episode
from tests.test_benchmarks import root  # noqa: F401  (the synthetic FD-Bench fixture)

SR = 16000
SILENT = "- id: agent\n  plugin: agent/canned\n  config: {replies: []}\n- id: traces\n  disabled: true\n- id: driver\n  config: {concurrency: 2, media: false, retries: 1, run_id: fdb-minicpmo45, group_by: benchmark.subset}\n"


def bundle(data, extra: str = "") -> Composition:
    c = Composition.bundle("fdb_baseline")
    c.set(f"tasks.data={data}")
    c.apply(SILENT + extra, "fakes")
    return c


async def run_as_script(sample):
    spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)
    env, agent = fdb.make_env(sample, spec), CannedAgent([], spec)
    obs, done = await env.reset(sample.task, seed=0), False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs))
    ep = episode(env, sample.id)
    ep["eval"]["benchmark"] = fdb.sample_metrics(ep, None)
    return ep


def test_every_sample_is_replayed_as_the_script_replays_it(root):
    run_ = compose(bundle(root).resolve())
    have = {s for s in fdb.SUBSETS if (root / s).exists()}
    assert {t.scenario["benchmark"]["subset"] for t in run_.tasks} == have and all("/" in t.id or t.id.startswith("fdb-") for t in run_.tasks)
    for task in run_.tasks:
        b = task.scenario["benchmark"]
        sample = fdb.load_sample(root, b["subset"], b["sample_id"], sr=SR, clean=b["clean"])
        assert task.id == sample.id  # the script's episode id
        rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(task, 0, episode_id=sample.id)))
        ep = asyncio.run(run_as_script(sample))
        assert rec["turns"] == ep["turns"], task.id
        assert rec["meta"]["duration_ms"] == ep["meta"]["duration_ms"] == sample.duration_ms  # exactly input.wav long
        assert rec["eval"]["benchmark"] == ep["eval"]["benchmark"] and rec["eval"]["scores"] == ep["eval"]["scores"]
        assert fdb.bench_meta(rec) == fdb.bench_meta(ep) and rec.get("background") == ep.get("background")  # the residual input as a track
        assert env.end_idle_ms == sample.duration_ms + 1 and env.max_ms == sample.duration_ms


def test_output_per_subset_clean_variants_and_the_report(root, tmp_path, capsys):
    out = tmp_path / "run"
    patch = tmp_path / "fakes.yml"
    patch.write_text(SILENT)
    cmd = ["run", "fdb_baseline", "--set", f"tasks.data={root}", "--set", "tasks.clean=true", "--patch", str(patch), "--out", str(out)]
    assert main(cmd) == 0
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("episodes.jsonl"))
    assert files == ["v1.0/candor_turn_taking/episodes.jsonl", "v1.0/icc_backchannel/episodes.jsonl", "v1.0/synthetic_pause_handling/episodes.jsonl",
                     "v1.0/synthetic_user_interruption/episodes.jsonl", "v1.5/user_backchannel/episodes.jsonl"]
    eps = [json.loads(x) for x in (out / "v1.5/user_backchannel/episodes.jsonl").read_text().splitlines()]
    ids = sorted(e["meta"]["episode_id"] for e in eps)
    assert ids == sorted(["fdb-v1.5-user_backchannel-1/s0", "fdb-v1.5-user_backchannel-1-clean/s0", "fdb-v1.5-user_backchannel-2/s0", "fdb-v1.5-user_backchannel-2-clean/s0"])
    assert all(e["meta"]["run_id"] == "fdb-minicpmo45" and e["meta"]["group"] == "v1.5/user_backchannel" for e in eps)
    assert sum(fdb.bench_meta(e)["clean"] for e in eps) == 2
    # a rerun resumes from every subset's file
    assert main(cmd) == 0
    assert "0 pairs to run (" in [x for x in capsys.readouterr().out.splitlines() if x.startswith("ig:")][-1]
    # the report (no judge): official metrics per subset, the scores' summary, pages
    assert main(["report", "fdb_baseline", "--set", f"tasks.data={root}", "--set", "tasks.clean=true", "--patch", str(patch), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {s for s in fdb.SUBSETS if (root / s).exists()}
    assert summary["v1.5/user_backchannel"]["n"] == 2 and "scores" in summary["v1.5/user_backchannel"]  # noisy inputs only
    assert (out / "v1.5/user_backchannel/per_sample.json").exists() and (out / "index.html").exists()


def test_limit_and_spread_are_per_subset(root):
    run_ = compose(bundle(root, "- id: tasks\n  config: {data: %s, bench: fdb, limit: 1, spread: true}\n" % root).resolve())
    by = {}
    for t in run_.tasks:
        by.setdefault(t.scenario["benchmark"]["subset"], []).append(t.id)
    assert all(len(v) == 1 for v in by.values()) and "v1.5/user_backchannel" in by
