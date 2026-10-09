"""Minimal closed loop: a rule-based user and a scripted policy, with one barge-in.

The user asks a question. While the policy is answering, the user checks every
second what it has heard so far; once the answer reveals the wrong day, the user
interrupts with a correction (after a 300 ms reaction time). The policy yields,
waits for the user to finish, and answers again.

    uv run python examples/minimal.py                 # print the log
    uv run python examples/minimal.py --html out.html # write a viewer page with several variants
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from interaction_gym import REWARD, AgentSpec, Env, Frame, Node, Segment, Task
from interaction_gym.agents import CannedAgent
from interaction_gym.eval import response_latencies, yield_latencies
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode
from interaction_gym.viewer import export_html

WORDS_PER_SEC = 3.4  # speaking-rate estimate used by FDGym for text-only timelines


def say(sid: str, t: int, text: str) -> Segment:
    return Segment(sid, t, round(len(text.split()) / WORDS_PER_SEC * 1000), text)


class RuleUser(Node):
    reads = ("policy.speech",)
    check_every = 1000  # how often the user reconsiders barging in while the agent speaks (ms)
    reaction = 300

    def init_state(self, task, rng):
        return {"task": task, "phase": "start", "agent": None}

    def profile(self, task):  # what meta.user records about this user
        return {"mode": "semi_online", "voice": {"tts": None, "words_per_sec": WORDS_PER_SEC},
                "barge_in": {"type": "keyword", "words": [task.scenario["wrong_word"]]},
                "turn_taking": {"check_ms": self.check_every, "reaction_ms": self.reaction}}

    async def step(self, s, t, inbox):
        for f in inbox:
            s["agent"] = f.data
        agent: Segment | None = s["agent"]
        sc = s["task"].scenario

        if s["phase"] == "start":
            s["phase"] = "asked"
            return [Frame("user.speech", t, say("u0", t, sc["question"]))], None

        if s["phase"] == "asked" and agent is not None:
            if sc["wrong_word"] in agent.heard(t):
                s["phase"] = "corrected"
                t1 = t + self.reaction
                return [Frame("user.speech", t1, say("u1", t1, sc["correction"]))], None
            return [], t + self.check_every if agent.active(t) else None

        if s["phase"] == "corrected" and agent is not None and agent.id != "a0":
            if t >= agent.end:
                s["phase"] = "end"
                return [Frame(REWARD, t, 1.0)], None
            return [], agent.end

        return [], None


REPLIES = [
    "Tomorrow will be sunny with a high of twenty four degrees and a light breeze from the west.",
    "Got it, the day after tomorrow looks rainy, so bring an umbrella.",
]


TASK = Task(
    id="weather-correction",
    scenario={
        "question": "What's the weather like tomorrow?",
        "correction": "Sorry, I meant the day after tomorrow.",
        "wrong_word": "Tomorrow",
    },
)


async def run(chunk_ms: int = 200, yield_after: int | None = 160, streaming: bool = False, verbose: bool = False) -> Env:
    spec = AgentSpec(chunk_ms=chunk_ms)
    env = Env({"user": RuleUser()}, spec, max_ms=60_000)
    policy = CannedAgent(REPLIES, spec, streaming=streaming, yield_after=yield_after)
    obs = await env.reset(TASK, seed=0)
    done = False
    while not done:
        obs, _, done = await env.step(policy.act(env.t, obs))
    if verbose:
        for f in env.log:
            if isinstance(f.data, Segment):
                seg = f.data
                print(f"{f.t:6d}ms {f.stream:14s} [{seg.t0}-{seg.end}] {seg.transcript!r}")
            else:
                print(f"{f.t:6d}ms {f.stream:14s} {f.data!r}")
        print("reward:", env.sim.reward)
        print("response latencies (ms):", response_latencies(env.log))
        print("yield latencies (ms):", yield_latencies(env.log))
    return env


VARIANTS = {
    "chunk-200": dict(chunk_ms=200),
    "chunk-80": dict(chunk_ms=80),
    "chunk-1000": dict(chunk_ms=1000),
    "streaming-agent": dict(chunk_ms=200, streaming=True),
    "never-yields": dict(chunk_ms=200, yield_after=None),
}


async def main(html: str | None) -> None:
    if html is None:
        await run(verbose=True)
        return
    records = []
    for name, kw in VARIANTS.items():
        env = await run(**kw)
        records.append(episode(env, f"{TASK.id}/{name}", media=MediaStore(Path(html).parent), meta=kw))
    print(f"{len(records)} cases -> {export_html(records, html)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="write a viewer page instead of printing")
    asyncio.run(main(ap.parse_args().html))
