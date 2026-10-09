"""τ-bench tasks running in InteractionGym (layer 1: plumbing only, no real models).

The user is an LLMSource with a *fake* LLM (canned replies), the agent is the
OracleAgent (replays the task's gold tool calls) or a lazy agent that never uses
tools. Episodes are scored with τ-bench's official evaluators.

    uv sync --extra tau
    uv run python examples/tau_mock.py --html runs/tau_mock.html
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from interaction_gym import AgentSpec, Chunk, Env, Frame, Segment
from interaction_gym.clients import FakeChat
from interaction_gym.integrations import tau
from interaction_gym.tools import RESULT, ToolWorld
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode
from interaction_gym.user import STOP, LLMSource, UserSim
from interaction_gym.viewer import export_html

SPEC = AgentSpec(chunk_ms=200, obs=("user.speech", RESULT["agent"]))


class LazyAgent:
    def __init__(self, *_):
        self.replied = False

    def act(self, t, obs):
        if not self.replied and any(isinstance(f.data, Chunk) and f.data.last for f in obs):
            self.replied = True
            return [Frame(SPEC.out, t, Segment("a0", t, 1500, "Sure, anything else I can help with?"))]
        return []


async def run(task, agent_cls, tool_latency_ms=0):
    user = UserSim(LLMSource(FakeChat(["Hi, I need some help with my tasks.", f"Great, thanks! {STOP}"])))
    env = Env({"user": user, "tools": ToolWorld(tau.TauBackend(task.scenario["domain"]), tool_latency_ms)}, SPEC, max_ms=120_000)
    agent = agent_cls(task, SPEC) if agent_cls is tau.OracleAgent else agent_cls()
    obs = await env.reset(task)
    done = False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs))
    return env


async def main(html: str | None):
    records = []
    cases = [(t, tau.OracleAgent, 0) for t in tau.tasks("mock")]
    first = tau.load_task("mock", "create_task_1")
    cases += [(first, tau.OracleAgent, 1500), (first, LazyAgent, 0)]
    for task, agent_cls, latency in cases:
        env = await run(task, agent_cls, latency)
        res = tau.evaluate(env.log, task, env.truncated)
        name = f"{task.id}/{'oracle' if agent_cls is tau.OracleAgent else 'lazy'}" + (f"/tool-{latency}ms" if latency else "")
        print(f"{name:60s} reward={res['reward']} parts={res['parts']} nl_skipped={res['nl_skipped']}")
        reward = {"total": res["reward"], "parts": res["breakdown"]}
        records.append(episode(env, name, media=MediaStore(Path(html or "runs/x").parent), reward=reward, meta={"tool_latency_ms": latency}))
    if html:
        print(f"{len(records)} cases -> {export_html(records, html)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html")
    asyncio.run(main(ap.parse_args().html))
