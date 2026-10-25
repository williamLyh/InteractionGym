"""The tau bundle against examples/tau_mock.py: the oracle on the mock domain's tasks gets the official full reward,
the tools are the domain's, slow tools work; and the openai_realtime bundle composes (no API call)."""

import asyncio
import os

import pytest

from interaction_gym.plugins import Composition, EpisodeContext, compose
from interaction_gym.plugins.builtin.drivers import run_episode

tau2 = pytest.importorskip("tau2")
from interaction_gym.integrations import tau  # noqa: E402


def test_the_oracle_gets_the_official_reward_through_the_bundle():
    run_ = compose(Composition.bundle("tau").resolve())
    assert [t.id for t in run_.tasks] == [t.id for t in tau.tasks("mock")] and set(run_.nodes) == {"user", "tools"}
    # the two tasks the oracle cannot solve by replaying gold actions (tests/test_tau.py) score 0 here as in the script
    unsolvable = {"update_task_with_history_and_env_assertions", "update_task_with_user_tools"}
    for task in run_.tasks:
        rec, env, _ = asyncio.run(run_episode(run_, EpisodeContext(task, 0, episode_id=task.id)))
        res = tau.evaluate(env.log, task, env.truncated)  # the script's scoring
        assert rec["eval"]["reward"] == {"total": res["reward"], "parts": res["breakdown"]}
        assert rec["eval"]["tau"]["termination"] == "user_stop"
        if task.scenario["tau_id"] not in unsolvable:
            assert res["nl_skipped"] or rec["eval"]["reward"]["total"] == 1.0, task.id
        assert env.initial_session(task).tools  # the domain's tools were offered
    # slow tools: the oracle waits for the results (the script's tool-1500ms case)
    c = Composition.bundle("tau")
    c.set("tools.latency_ms=1500")
    c.set("tasks.ids=[create_task_1]")
    slow = compose(c.resolve())
    rec, env, _ = asyncio.run(run_episode(slow, EpisodeContext(slow.tasks[0], 0, episode_id="x")))
    assert rec["eval"]["reward"]["total"] == 1.0
    calls = [c for t in rec["turns"] for c in t.get("tool_calls", [])]
    assert calls and all(c["result_time"] - c["call_time"] == 1500 for c in calls)


def test_the_openai_realtime_bundle_composes_without_calling_the_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    res = Composition.bundle("openai_realtime").resolve()
    assert res.configs["user_llm"].base_url is None and res.configs["agent"]["model"] == "gpt-realtime-mini"
    run_ = compose(res)
    assert run_.spec.chunk_ms == 100 and run_.spec.sr == 24000 and run_.spec.obs == ("user.speech", "tool.result")
    env = run_.make_env(EpisodeContext(run_.tasks[0], 0))
    user = env.nodes["user"]
    assert user.voice.clone is None and user.voice.voice == "coral"  # clone: false → no cloning TTS
    assert user.soundscape.background  # the demo's UserSim default (a background from the persona), not the benchmark user
