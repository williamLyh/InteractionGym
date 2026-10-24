"""Drivers: how a run's episodes are organised. ``driver/eval`` runs every (task, seed) once, concurrently up to
``concurrency``, scores each episode, hands it to the sinks and resumes a run from the jsonl sink's file.

An episode is the same loop every example runs: reset, then ``env.step(agent.act(t, obs))`` until the episode is done
and the agent has nothing pending; ``act`` may be a coroutine (server adapters) or plain (scripted agents).
"""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from pathlib import Path

from ...core import Env, Task
from ...traj import episode as make_episode
from ..compose import Run
from ..registry import plugin


async def run_episode(run: Run, task: Task, seed: int, episode_id: str, media=None) -> tuple[dict, Env, object]:
    env, agent = run.make_env(), run.agent(seed)
    wall = time.monotonic()
    try:
        obs = await env.reset(task, seed)
        while True:
            action = agent.act(env.t, obs)
            if inspect.isawaitable(action):
                action = await action
            obs, _, done = await env.step(action)
            if env.truncated or (done and not getattr(agent, "busy", False)):
                break
    finally:
        close = getattr(agent, "close", None)
        if close is not None:
            r = close()
            if inspect.isawaitable(r):
                await r
    meta = {**run.meta(), "wall_s": round(time.monotonic() - wall, 2)}
    ep = make_episode(env, episode_id, media=media, agent=agent if hasattr(agent, "describe") else None, meta=meta)
    for rid, scorer in run.scorers.items():
        ep["eval"][rid] = scorer(ep, env, agent)
    return ep, env, agent


@dataclass
class EvalConfig:
    concurrency: int = 4
    media: bool = True  # write audio next to the episodes (MediaStore in the output directory)


class EvalDriver:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    async def run(self, run: Run, *, out: str | Path = "runs/ig", seeds=(0,), limit: int = 0, ids=(), tag: str = "", log=print) -> list[dict]:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for sink in run.sinks.values():
            sink.out = out
        media = None
        if self.cfg.media:
            from ...media import MediaStore

            media = MediaStore(out)
        tasks = [t for t in run.tasks if not ids or t.id in ids]
        if limit:
            tasks = tasks[:limit]
        done: set[str] = set()
        for sink in run.sinks.values():
            if hasattr(sink, "done"):
                done |= sink.done()
        eid = lambda t, s: f"{tag + '/' if tag else ''}{t.id}/s{s}"  # noqa: E731
        jobs = [(t, s) for s in seeds for t in tasks if eid(t, s) not in done]
        log(f"ig: {len(tasks)} tasks × {len(seeds)} seeds, {len(jobs)} to run ({len(done)} done) -> {out}", flush=True)
        sem, eps, t0 = asyncio.Semaphore(self.cfg.concurrency), [], time.monotonic()

        async def one(task, seed):
            async with sem:
                ep, env, agent = await run_episode(run, task, seed, eid(task, seed), media)
                for sink in run.sinks.values():
                    sink.write(ep, env, agent)
                eps.append(ep)
                sc = ep["eval"]["scores"]["total"]
                log(f"  {ep['meta']['episode_id']}: {ep['meta']['duration_ms'] / 1000:.1f} s, scores {sc if sc is None else round(sc, 3)}, "
                    f"{len(eps)}/{len(jobs)} in {time.monotonic() - t0:.0f} s", flush=True)

        try:
            await asyncio.gather(*(one(t, s) for t, s in jobs))
        finally:
            for sink in run.sinks.values():
                sink.close()
        return eps


@plugin("driver", "eval", config=EvalConfig)
def eval_driver(ctx, cfg: EvalConfig) -> EvalDriver:
    """Every (task, seed) once, ``concurrency`` at a time; resumes from the jsonl sink."""
    return EvalDriver(cfg)
