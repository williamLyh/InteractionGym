"""Drivers: how a run's episodes are organised. ``driver/eval`` is the evaluation runner the benchmark scripts shared:
every (task, seed) once, ``concurrency`` at a time, each episode scored and handed to the sinks, a rerun resuming from
the jsonl sink's file.

With ``arms`` it is an A/B runner: each arm is a patch layer over the run's own configuration (e.g. arm B swaps the
user's source for the replanning one); one (task, seed) goes through the arms in order, sharing an ``EpisodeContext.cache``
(a cascaded agent memoizes its model calls there, so arms with the same prefix make the same calls); each arm writes
under ``<out>/<tag>/<arm>/`` and episode ids are ``<tag>/<task>/s<seed>/<arm>``, which is the layout the ``report``
commands of ``examples/*_ab.py`` read. ``retries`` re-run an episode that raised (a server hiccup) after
``retry_wait_s`` (+ up to 30 s of jitter); ``warmup`` runs one throw-away episode per agent server first (a duplex
server's first session after a start returns no audio); a run whose agent reports ``audio_out`` refuses to resume a
file of the other output mode (``agents.vllm_omni.check_output_mode``).

An episode is the loop every example runs: reset, then ``env.step(agent.act(t, obs))`` until the episode is done and
the agent has nothing pending; ``act`` may be a coroutine (server adapters) or plain (scripted agents).
"""

from __future__ import annotations

import asyncio
import inspect
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from ...core import Env, Task
from ...traj import episode as make_episode
from ..compose import Run, compose
from ..config import Composition
from ..registry import CURRENT_EPISODE, EpisodeContext, PluginError, plugin


async def run_episode(run: Run, ep: EpisodeContext, media=None) -> tuple[dict, Env, object]:
    token = CURRENT_EPISODE.set(ep)
    try:
        env, agent = run.make_env(), run.agent(ep)
        wall = time.monotonic()
        try:
            obs = await env.reset(ep.task, ep.seed)
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
    finally:
        CURRENT_EPISODE.reset(token)
    meta = {**run.meta(), "task_id": ep.task.id, "wall_s": round(time.monotonic() - wall, 2)}
    if ep.arm is not None:
        meta["condition"] = ep.arm
    record = make_episode(env, ep.episode_id, media=media, agent=agent if hasattr(agent, "describe") else None, meta=meta)
    for rid, scorer in run.scorers.items():
        record["eval"][rid] = scorer(record, env, agent)
    return record, env, agent


@dataclass
class EvalConfig:
    concurrency: int = 4
    media: bool = True  # write audio next to the episodes (MediaStore in the output directory)
    arms: dict | None = None  # {name: [patch operations]}: an A/B run; None: one arm, no subdirectory
    retries: int = 1  # attempts per episode
    retry_wait_s: float = 45.0  # before another attempt, plus up to retry_jitter_s at random
    retry_jitter_s: float = 30.0
    warmup: bool = False  # one throw-away episode per agent server before the run (duplex servers)
    run_id: str | None = None  # meta.run_id (default: the tag)


class EvalDriver:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg

    def arms(self, run: Run) -> dict[str, Run]:
        """The runs of the arms: the base configuration patched by each arm's operations (``{"": run}`` without arms)."""
        if not self.cfg.arms:
            return {"": run}
        out = {}
        for name, ops in self.cfg.arms.items():
            comp = Composition()
            comp.apply(run.resolved.dump(), "base")
            comp.apply(ops or [], f"arm {name}")
            out[name] = compose(comp.resolve())
        return out

    async def run(self, run: Run, *, out: str | Path = "runs/ig", seeds=(0,), limit: int = 0, ids=(), tag: str = "", log=print) -> list[dict]:
        cfg, out = self.cfg, Path(out)
        arms = self.arms(run)
        tag = tag or (run.agent_id if cfg.arms else "")
        run_id = cfg.run_id or tag or None
        media = None
        if cfg.media:
            from ...media import MediaStore

            media = MediaStore(out)
        done: dict[str, set[str]] = {}
        for arm, r in arms.items():
            d = out / tag / arm if cfg.arms else out
            d.mkdir(parents=True, exist_ok=True)
            done[arm] = set()
            for sink in r.sinks.values():
                sink.out = d
                if hasattr(sink, "done"):
                    done[arm] |= sink.done()
                    audio_out = getattr(r.agent_factory, "audio_out", None)
                    if audio_out is not None and hasattr(sink, "path"):
                        from ...agents.vllm_omni import check_output_mode

                        check_output_mode(sink.path, audio_out)
        tasks = [t for t in run.tasks if not ids or t.id in ids]
        if limit:
            tasks = tasks[:limit]
        eid = lambda t, s, a: "/".join(x for x in (tag, t.id, f"s{s}", a) if x)  # noqa: E731
        jobs = [(t, s) for s in seeds for t in tasks if any(eid(t, s, a) not in done[a] for a in arms)]
        n_done = sum(len(d) for d in done.values())
        log(f"ig: {len(tasks)} tasks × {len(seeds)} seeds × {len(arms)} arm(s), {len(jobs)} pairs to run ({n_done} episodes done) -> {out}", flush=True)
        if cfg.warmup and jobs:
            await self._warmup(arms, jobs[0], log)
        sem, eps, t0 = asyncio.Semaphore(cfg.concurrency), [], time.monotonic()

        async def pair(task: Task, seed: int):
            async with sem:
                cache: dict = {}
                for arm, r in arms.items():
                    e = eid(task, seed, arm)
                    if e in done[arm]:
                        continue
                    ep = EpisodeContext(task, seed, arm or None, cache, e)
                    for attempt in range(max(1, cfg.retries)):
                        try:
                            record, env, agent = await run_episode(r, ep, media)
                            break
                        except Exception as ex:  # noqa: BLE001 — a server hiccup: wait, retry, give up after `retries`
                            log(f"  {e} attempt {attempt + 1}: {type(ex).__name__}: {str(ex)[:200]}", flush=True)
                            if attempt + 1 >= max(1, cfg.retries):
                                log(f"  GAVE UP {e}", flush=True)
                                return
                            await asyncio.sleep(cfg.retry_wait_s + random.random() * cfg.retry_jitter_s)
                    if run_id:
                        record["meta"]["run_id"] = run_id
                    declared = {k: getattr(r.agent_factory, k) for k in ("audio_out",) if hasattr(r.agent_factory, k)}
                    record["meta"]["agent"] = {**declared, **record["meta"].get("agent", {}), "name": r.agent_id or ""}
                    record["meta"]["agent_name"] = tag or r.agent_id or ""
                    for sink in r.sinks.values():
                        sink.write(record, env, agent)
                    eps.append(record)
                    sc = record["eval"]["scores"]["total"]
                    users = [t for t in record["turns"] if t["role"] != run.controlled.name and t["text"]] if run.controlled else []
                    log(f"  {len(eps):4d} {e:45s} {record['meta']['duration_ms'] / 1000:6.1f}s sim {record['meta']['wall_s']:6.1f}s wall "
                        f"users={len(users)} scores {sc if sc is None else round(sc, 3)} {len(eps) / (time.monotonic() - t0) * 3600:.0f} ep/h", flush=True)

        try:
            await asyncio.gather(*(pair(t, s) for t, s in jobs))
        finally:
            for r in arms.values():
                for sink in r.sinks.values():
                    sink.close()
        return eps

    async def _warmup(self, arms: dict[str, Run], job, log) -> None:
        from ...agent_proxy import servers_behind

        first = next(iter(arms.values()))
        url = getattr(first.agent_factory, "url", None)
        n = servers_behind(url) if url else 1
        task, seed = job
        t0 = time.monotonic()
        warm = await asyncio.gather(*[run_episode(first, EpisodeContext(task, seed, None, {}, "warmup")) for _ in range(n)])
        log(f"ig: warm-up ({n} server{'s' * (n > 1)}): {warm[0][1].t / 1000:.1f}s sim in {time.monotonic() - t0:.1f}s wall", flush=True)


@plugin("driver", "eval", config=EvalConfig)
def eval_driver(ctx, cfg: EvalConfig) -> EvalDriver:
    """Every (task, seed) once, ``concurrency`` at a time, resuming from the jsonl sink; with ``arms``, an A/B runner."""
    if cfg.arms is not None and (not isinstance(cfg.arms, dict) or not cfg.arms):
        raise PluginError("driver/eval: arms is a mapping {name: [patch operations]} with at least one arm")
    return EvalDriver(cfg)
