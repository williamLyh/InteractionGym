"""AutomationBench tasks in InteractionGym (plumbing only, no real models).

The user says the task's request (scripted, verbatim); the agent replays a hand-written correct call
sequence (the benchmark ships no reference solutions) or never touches a tool. Episodes are scored
with AutomationBench's official end-state rubric. Tool calls take simulated time (``rest_latency``).

Cases
  api/<task>             the official toolset: api_search + api_fetch over simulated REST endpoints
  zapier/<task>          the benchmark's search_tools / execute_tool meta-tools
  limited_zapier/<task>  only the task's own Zapier actions, one service per app
  apps/<task>            all 47 apps as services behind ToolWorld's discover mode
  api/<task>/lazy        talks, never acts: reward 0

    git clone --depth 1 https://github.com/zapier/AutomationBench third_party/automationbench
    uv sync --extra automationbench
    uv run python examples/automationbench_demo.py --html runs/automationbench_demo.html
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from interaction_gym import AgentSpec, Chunk, Env, Frame, Segment
from interaction_gym.integrations import automationbench as ab
from interaction_gym.media import MediaStore
from interaction_gym.tools import RESULT
from interaction_gym.traj import episode
from interaction_gym.user import ScriptSource, UserSim
from interaction_gym.viewer import export_html

SPEC = AgentSpec(chunk_ms=200, obs=("user.speech", RESULT["agent"]))


class LazyAgent:
    def __init__(self):
        self.replied = False

    def act(self, t, obs):
        if not self.replied and any(isinstance(f.data, Chunk) and f.data.last for f in obs):
            self.replied = True
            return [Frame(SPEC.out, t, Segment("a0", t, 1500, "Sure, I'll take care of that."))]
        return []


async def run(task, agent, toolset):
    env = Env({"user": UserSim(ScriptSource()), **ab.nodes(task, toolset, ab.rest_latency())}, SPEC, max_ms=600_000)
    obs = await env.reset(task)
    done = False
    while not done:
        obs, _, done = await env.step(agent.act(env.t, obs))
    return env


async def main(html: str | None):
    cases = [(name, toolset, ab.ReplayAgent(calls, say="All done.", say_first="On it, one moment.", spec=SPEC))
             for (name, toolset), calls in ab.HANDWRITTEN_SOLUTIONS.items()]
    cases.append(("sales.multi_hop_lookup", "api", LazyAgent()))
    media = MediaStore(Path(html or "runs/x").parent)
    records = []
    for name, toolset, agent in cases:
        task = ab.load_task(name)
        env = await run(task, agent, toolset)
        res = ab.evaluate(env, task, env.truncated)
        label = f"{toolset}/{name}" + ("/lazy" if isinstance(agent, LazyAgent) else "")
        n_calls = sum(f.stream == RESULT["agent"] for f in env.log)
        print(f"{label:55s} reward={res['reward']} partial_credit={res['partial_credit']:.2f} calls={n_calls} t={env.t / 1000:.1f}s")
        reward = {"total": res["reward"], "parts": res["parts"]}
        meta = {"toolset": toolset, "assertions": res["breakdown"]}
        records.append(episode(env, label, media=media, reward=reward, meta=meta))
    if html:
        print(f"{len(records)} cases -> {export_html(records, html)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html")
    asyncio.run(main(ap.parse_args().html))
