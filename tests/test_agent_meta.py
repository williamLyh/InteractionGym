"""meta.agent: runners record which agent an episode ran; agents describe themselves."""

import asyncio

import pytest

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.traj import episode
from interaction_gym.user import ReplayUser
from tests.test_traj import check


def run_canned():
    spec = AgentSpec()
    env = Env({"user": ReplayUser([{"t": 0, "text": "Hello there, anyone?"}])}, spec, max_ms=10_000)
    agent = CannedAgent(["Hi!"], spec)

    async def go():
        obs = await env.reset(Task(id="t", scenario={}))
        done = False
        while not done:
            obs, _, done = await env.step(agent.act(env.t, obs))
    asyncio.run(go())
    return env, agent


def test_agent_describes_itself_into_meta():
    env, agent = run_canned()
    ep = check(episode(env, "e", agent=agent))
    assert ep["meta"]["agent"] == {"name": "CannedAgent", "kind": "text", "replies": 1, "streaming": False}
    ep = check(episode(env, "e", agent={"name": "x", "kind": "cascaded", "model": "m"}))
    assert ep["meta"]["agent"]["kind"] == "cascaded"
    assert "agent" not in episode(env, "e")["meta"]


def test_vllm_omni_agent_description():
    pytest.importorskip("websockets")  # extra vllm-omni
    from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent

    a = VllmOmniDuplexAgent(AgentSpec(audio="user.audio", sr=24000), "ws://h:8010/v1/realtime?duplex=1", clock="input")
    d = a.describe()
    assert d["name"] == "MiniCPM-o 4.5" and d["kind"] == "full-duplex" and d["model"] == "openbmb/MiniCPM-o-4_5"
    assert d["clock"] == "input" and "ws://h:8010" in d["server"]
    assert d["output"] == "audio"
    t = VllmOmniDuplexAgent(AgentSpec(audio="user.audio", sr=16000), clock="input", audio_out=False).describe()
    assert t["audio_out"] is False and t["output"] == "text @ 11.3 chars/s"


def test_check_output_mode(tmp_path):
    pytest.importorskip("websockets")  # extra vllm-omni
    import json

    from interaction_gym.agents.vllm_omni import check_output_mode

    p = tmp_path / "episodes.jsonl"
    check_output_mode(p, False)  # no file yet
    p.write_text(json.dumps({"meta": {"agent": {"output": "audio"}}}) + "\n")
    check_output_mode(p, True)
    with pytest.raises(ValueError, match="audio-output"):
        check_output_mode(p, False)
    p.write_text(json.dumps({"meta": {"agent": {"audio_out": False}}}) + "\n")
    check_output_mode(p, False)
    with pytest.raises(ValueError, match="text-only"):
        check_output_mode(p, True)
    p.write_text(json.dumps({"meta": {"agent": {"kind": "cascaded"}}}) + "\n")
    check_output_mode(p, False)  # not a vLLM-Omni agent: nothing to compare
