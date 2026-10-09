"""Full-Duplex-Bench v1.0 / v1.5 (``--bench fdb``), the Easy Turn testset (``--bench easy_turn``) or HumDial-FDBench
(``--bench humdial``; ``--data`` = the unpacked dataset) with a vLLM-Omni duplex model (MiniCPM-o 4.5) as the agent.

    # run (resumable: episodes already in <out>/<subset>/episodes.jsonl are skipped)
    PYTHONPATH=src:. python examples/fdb_baseline.py run --data /path/to/full_duplex_bench --out runs/fdb \
        --agent-urls ws://127.0.0.1:8010/v1/realtime?duplex=1,ws://127.0.0.1:8011/v1/realtime?duplex=1 --sessions 4
    # official metrics + eval.scores summary (+ optional LLM judges) and a browsable subset
    PYTHONPATH=src:. python examples/fdb_baseline.py report --data … --out runs/fdb [--judge-url http://localhost:8000/v1 --judge-model M]

Each sample is replayed exactly as the benchmark streams it (``full_duplex_bench.make_env``): the agent's
microphone carries ``input.wav`` sample for sample, at 16 kHz, for exactly its length, in lockstep
(``clock="input"``), with no system prompt (as in the official inference scripts). v1.5 subsets also run
``clean_input.wav`` (``--clean``), which the v1.5 behaviour judge needs.

Agent output: text only, timed at ``speech_cps``, against a Thinker-only server (the default; docs/BENCHMARKS.md
"Agent output"), or the talker's speech with ``--audio-out`` (Thinker + Talker + Code2Wav). The official metrics take
the agent's words from its transcript either way; with audio, word times are spread over the voiced parts of each
turn (energy VAD on the agent's audio), text-only over the whole estimated turn. The two are not comparable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

from interaction_gym import AgentSpec
from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent, check_output_mode
from interaction_gym.benchmarks import easy_turn as et
from interaction_gym.benchmarks import full_duplex_bench as fdb
from interaction_gym.benchmarks import humdial as hd
from interaction_gym.envvars import getenv
from interaction_gym.eval import duplex, scores
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, load, save
from interaction_gym.viewer import export_run

MODEL = getenv("IG_AGENT_MODEL", "openbmb/MiniCPM-o-4_5")
REF_AUDIO = getenv("IG_AGENT_REF_AUDIO", str(Path(getenv("IG_MODEL_DIR", "models")) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
SR = 16000


BENCHES = {"fdb": ("Full-Duplex-Bench", fdb.SUBSETS), "easy_turn": ("Easy Turn", et.SUBSETS), "humdial": ("HumDial-FDBench", hd.SUBSETS)}


def _subsets(args) -> dict:
    return BENCHES[args.bench][1]


def _ids(args, subset: str) -> list[str]:
    ids = fdb._ids(Path(args.data) / subset) if _is_fdb(subset) else et.ids(args.data, subset) if subset in et.SUBSETS else hd.ids(args.data, subset)
    if args.spread and args.limit and len(ids) > args.limit:  # evenly spaced, so a small run covers all speakers
        step = len(ids) / args.limit
        return [ids[int(k * step)] for k in range(args.limit)]
    return ids[: args.limit]


def _eid(subset: str, sid: str, clean: bool) -> str:
    if subset in et.SUBSETS:
        return f"et-{et.SUBSETS[subset]}-{sid}"
    lang_cat = "-".join(hd.SUBSETS[subset]) if subset in hd.SUBSETS else None
    base = f"fdb-{subset.replace('/', '-')}-{sid}" if lang_cat is None else f"hd-{lang_cat}-{sid}"
    return base + ("-clean" if clean else "")


def _load(args, subset: str, sid: str, clean: bool, cache: dict):
    if _is_fdb(subset):
        return fdb.load_sample(args.data, subset, sid, sr=SR, clean=clean)
    if subset in et.SUBSETS:
        return et.load_sample(args.data, subset, sid, sr=SR, _cache=cache)
    return hd.load_sample(args.data, subset, sid, sr=SR, clean=clean)


def _is_fdb(subset: str) -> bool:
    return subset in fdb.SUBSETS


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["meta"]["episode_id"] for line in path.read_text().splitlines() if line.strip()}


async def run_one(sample: fdb.Sample, url: str, audio_out: bool = False) -> tuple:
    spec = AgentSpec(chunk_ms=200, audio="user.audio", sr=SR)
    env = fdb.make_env(sample, spec)
    agent = VllmOmniDuplexAgent(spec, url, model=MODEL, ref_audio=REF_AUDIO, out_sr=SR, clock="input", audio_out=audio_out,
                                trace_tokens=True)
    wall = time.monotonic()
    obs, done = await env.reset(sample.task, seed=0), False
    try:
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    return env, agent, time.monotonic() - wall


async def run(args) -> None:
    out = Path(args.out)
    media = MediaStore(out)
    urls = args.agent_urls.split(",")
    queue: asyncio.Queue = asyncio.Queue()
    subsets = args.subsets.split(",") if args.subsets else list(_subsets(args))
    et_cache: dict = {}
    for subset in subsets:
        d = out / subset
        d.mkdir(parents=True, exist_ok=True)
        check_output_mode(d / "episodes.jsonl", args.audio_out)  # never resume a run of the other output mode
        done = _done(d / "episodes.jsonl")
        ids = _ids(args, subset)
        variants = [False, True] if args.clean and _is_fdb(subset) and fdb.SUBSETS[subset][0] == "1.5" else [False]
        for sid in ids:
            for clean in variants:
                eid = _eid(subset, sid, clean)
                if eid not in done:
                    queue.put_nowait((subset, sid, clean))
    total = queue.qsize()
    print(f"{total} episodes to run on {len(urls)} server(s) x {args.sessions} sessions", flush=True)
    t_start, n_ok = time.monotonic(), 0

    async def worker(url: str, k: int) -> None:
        nonlocal n_ok
        await asyncio.sleep(k * 0.5)
        while not queue.empty():
            subset, sid, clean = queue.get_nowait()
            sample = _load(args, subset, sid, clean, et_cache)
            for attempt in range(args.retries):
                try:
                    env, agent, wall = await run_one(sample, url, args.audio_out)
                    break
                except Exception as e:  # server restarting (hourly kill) or stuck: wait and retry
                    print(f"[{url[-30:]}] {sample.id} attempt {attempt + 1}: {type(e).__name__}: {e}", flush=True)
                    await asyncio.sleep(60 + random.random() * 30)
            else:
                print(f"GAVE UP {sample.id}", flush=True)
                continue
            ep = episode(env, sample.id, media=media, run_id=args.run_id,
                         agent={**agent.describe(), "url": url}, meta={"wall_s": round(wall, 1)})
            ep["eval"]["benchmark"] = fdb.sample_metrics(ep, media) if _is_fdb(subset) else {}
            save([ep], out / subset / "episodes.jsonl", append=True)
            tr = agent.trace(sample.id)
            if tr:
                save([tr], out / subset / "agent_traces.jsonl", append=True)
            n_ok += 1
            rate = n_ok / (time.monotonic() - t_start)
            agent_ms = sum(t["end_time"] - t["start_time"] for t in ep["turns"] if t["role"] == "agent")
            print(f"{n_ok}/{total} {sample.id:52s} {env.t / 1000:5.1f}s sim {wall:5.1f}s wall agent={agent_ms / 1000:4.1f}s "
                  f"bench={ {k: v for k, v in ep['eval']['benchmark'].items() if not isinstance(v, (list, str))} } "
                  f"eta={(total - n_ok) / rate / 60:.0f}min", flush=True)

    await asyncio.gather(*(worker(u, k) for u in urls for k in range(args.sessions)))


async def report(args) -> None:
    out = Path(args.out)
    media = MediaStore(out)
    gt = json.loads(Path(args.gt).read_text()) if args.gt and Path(args.gt).exists() else None
    judge = None
    if args.judge_url:
        from interaction_gym.clients import OpenAIChat

        judge = OpenAIChat(args.judge_url, args.judge_model, max_tokens=1024, temperature=0)
    summary, pages = {}, []
    summary_prev = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for subset in _subsets(args):
        path = out / subset / "episodes.jsonl"
        if not path.exists():
            continue
        eps = load(path)
        for e in eps:  # re-score: episodes recorded before eval.scores knew the episode end (censoring)
            e["eval"]["duplex"] = duplex(e["turns"], end_ms=e["meta"]["duration_ms"])
            e["eval"]["scores"] = scores(e["turns"], end_ms=e["meta"]["duration_ms"])
        noisy = [e for e in eps if not fdb.bench_meta(e).get("clean")]
        clean = {fdb.bench_meta(e)["sample_id"]: e for e in eps if fdb.bench_meta(e).get("clean")}
        ratings = behaviours = None
        cached = json.loads((out / subset / "per_sample.json").read_text()) if (out / subset / "per_sample.json").exists() and not args.rejudge else {}
        if judge is None and any("rating" in v for v in cached.values()):  # reuse earlier judge verdicts
            ratings = {k: v["rating"] for k, v in cached.items() if v.get("rating") is not None}
        if judge is None and any("behaviour" in v for v in cached.values()):
            behaviours = {k: v["behaviour"] for k, v in cached.items() if v.get("behaviour")}
        if not _is_fdb(subset):
            m = {"subset": subset, "n": len(noisy)}
        if judge is not None and _is_fdb(subset) and fdb.SUBSETS[subset][1] == "user_interruption" and fdb.SUBSETS[subset][0] == "1.0":
            rs = await asyncio.gather(*(fdb.judge_interruption(e, judge, media) for e in noisy))
            ratings = {e["meta"]["episode_id"]: r for e, r in zip(noisy, rs) if r is not None}
        if judge is not None and _is_fdb(subset) and fdb.SUBSETS[subset][0] == "1.5" and clean:
            instr = fdb.behaviour_instruction(args.repo)
            pairs = [(e, clean[fdb.bench_meta(e)["sample_id"]]) for e in noisy if fdb.bench_meta(e)["sample_id"] in clean]
            tags = await asyncio.gather(*(fdb.judge_behaviour(e, c, judge, instr, media) for e, c in pairs))
            behaviours = {e["meta"]["episode_id"]: t or "C_UNKNOWN" for (e, _), t in zip(pairs, tags)}
        if _is_fdb(subset):
            m = fdb.official_metrics(noisy, media, gt_distribution=gt, ratings=ratings, behaviours=behaviours)
        per = m.pop("per_sample", None) or {e["meta"]["episode_id"]: {"text": [t.get("text") for t in e["turns"]],
                                                                    "scores": e["eval"]["scores"]["by_expectation"]} for e in noisy}
        by: dict[str, list[float]] = {}
        totals, outcomes = [], {}
        for e in noisy:
            sc = e["eval"]["scores"]
            if sc["total"] is not None:
                totals.append(sc["total"])
            for k, v in sc["by_expectation"].items():
                by.setdefault(k, []).append(v)
            for ev in sc["events"]:
                o = outcomes.setdefault(ev["expects"], {})
                o[ev["outcome"]] = o.get(ev["outcome"], 0) + 1
        m["scores"] = {"total": round(sum(totals) / len(totals), 4) if totals else None,
                       "by_expectation": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
                       "outcomes": {k: dict(sorted(v.items())) for k, v in sorted(outcomes.items())}}
        m["wall_s"] = round(sum(e["meta"].get("wall_s", 0) for e in eps), 1)
        m["sim_s"] = round(sum(e["meta"]["duration_ms"] for e in eps) / 1000, 1)
        if ratings is not None or behaviours is not None:
            m["judge"] = args.judge_model or (summary_prev.get(subset, {}).get("judge") if judge is None else None)
        summary[subset] = m
        (out / subset / "per_sample.json").write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items()} | ({"rating": ratings.get(k)} if ratings else {}) | ({"behaviour": behaviours.get(k)} if behaviours else {})
             for k, v in per.items()}, indent=1))
        pages += noisy[: args.pages] + [clean[fdb.bench_meta(e)["sample_id"]] for e in noisy[: args.pages] if fdb.bench_meta(e)["sample_id"] in clean]
        print(json.dumps({subset: {k: v for k, v in m.items()}}, indent=1), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    traces = []
    for subset in _subsets(args):
        p = out / subset / "agent_traces.jsonl"
        if p.exists():
            want = {e["meta"]["episode_id"] for e in pages}
            traces += [t for t in load(p) if t["episode_id"] in want]
    notes = {e["meta"]["episode_id"]: f"{fdb.bench_meta(e)['subset']} #{fdb.bench_meta(e)['sample_id']}"
             + (" (clean)" if fdb.bench_meta(e).get("clean") else "") + " · " + json.dumps(
                 {k: v for k, v in e["eval"].get("benchmark", {}).items() if not isinstance(v, (list, str))}) for e in pages}
    title = BENCHES[args.bench][0]
    print("->", export_run(pages, out, notes, title=f"{title} · MiniCPM-o 4.5", traces=traces))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--data", required=True)
    r.add_argument("--bench", choices=list(BENCHES), default="fdb")
    r.add_argument("--spread", action="store_true", help="with --limit: evenly spaced samples instead of the first ones")
    r.add_argument("--out", default="runs/fdb")
    r.add_argument("--subsets", default="")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--agent-urls", default="ws://127.0.0.1:8010/v1/realtime?duplex=1")
    r.add_argument("--sessions", type=int, default=4, help="concurrent episodes per server")
    r.add_argument("--clean", action="store_true", help="also run v1.5 clean inputs (for the behaviour judge)")
    r.add_argument("--retries", type=int, default=6)
    r.add_argument("--run-id", default="fdb-minicpmo45")
    r.add_argument("--audio-out", action="store_true", help="the agent speaks (Thinker + Talker + Code2Wav server); "
                   "default: text-only output timed at speech_cps, Thinker-only server")
    p = sub.add_parser("report")
    p.add_argument("--data", required=True)
    p.add_argument("--bench", choices=list(BENCHES), default="fdb")
    p.add_argument("--out", default="runs/fdb")
    p.add_argument("--repo", default="", help="checkout of the official repo (judge instructions)")
    p.add_argument("--gt", default="", help="icc_gt_distribution.json from the official repo")
    p.add_argument("--judge-url", default="")
    p.add_argument("--judge-model", default="")
    p.add_argument("--rejudge", action="store_true", help="without --judge-url, earlier verdicts in per_sample.json are reused unless this is set")
    p.add_argument("--pages", type=int, default=3, help="episodes per subset in the browsable run")
    a = ap.parse_args()
    asyncio.run(run(a) if a.cmd == "run" else report(a))
