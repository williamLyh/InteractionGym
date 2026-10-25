"""Service calls with an SOP, MiniCPM-o 4.5 (full duplex, speech only) as the agent.

Each scenario is a caller with a request, an SOP for the business (``interaction_gym.sop`` source format) and a case:
what the agent can see on its screen and the ground truth of every condition, which decides whether the request
should be done or declined. The agent gets the SOP rendered in ``--format`` (any registered SOP format) plus the
screen in its instructions; it has no tools, so it must ask the caller for the rest. After the call
``sop_task.judge`` scores the decision, the procedure and the details (an LLM reports what happened, rules on the
SOP tree score it).

On the GPU host (user LLM :8000, TTS :8001, clone TTS :8005, MiniCPM-o :8010):

    PYTHONPATH=src:. python examples/sop_calls.py --out runs/sop_calls --audio-out --seeds 0 1 2
    PYTHONPATH=src:. python examples/sop_calls.py --format list --only booking-mina      # another SOP format
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from interaction_gym import AgentSpec, Env, Task
from interaction_gym import sop_task as st
from interaction_gym.benchmarks import benchmark_user
from interaction_gym.clients import Cached, OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.media import MediaStore
from interaction_gym.soundscape import Soundscape
from interaction_gym.sop import formats, validate
from interaction_gym.traj import episode, save
from interaction_gym.user import QWEN3_TTS_VOICES, LLMInterrupt, LLMSource, ResponseDelay, TurnTaking, Voice
from interaction_gym.viewer import export_run

LLM_URL = getenv("IG_LLM_URL", "http://localhost:8000/v1")
LLM_MODEL = getenv("IG_LLM_MODEL", "Qwen3.8-27B")
JUDGE_URL = getenv("IG_JUDGE_URL", LLM_URL)
TTS_URL = getenv("IG_TTS_URL", "http://localhost:8001/v1")
CLONE_URL = getenv("IG_CLONE_URL", "http://localhost:8005/v1")
AGENT_URL = getenv("IG_AGENT_URL", "ws://127.0.0.1:8010/v1/realtime?duplex=1")
AGENT_MODEL = getenv("IG_AGENT_MODEL", "openbmb/MiniCPM-o-4_5")
REF_AUDIO = getenv("IG_AGENT_REF_AUDIO", str(Path(getenv("IG_MODEL_DIR", "models")) / "MiniCPM-o-4_5/assets/system_ref_audio.wav"))
TTS_MODEL, CLONE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SR = 24000

# ---------------------------------------------------------------- the scenarios: interaction_gym/data/sop_calls.json

SCENARIOS = st.scenarios()  # name -> caller, SOP (validated) and case; `ig run sop_calls` runs the same file
make_task = st.task_of

# Chinese SOPs ("lang": "zh": the formats' own words and the agent's frame are in Chinese too)
TODAY_ZH = "今天是 2026 年 10 月 16 日，星期五，15:20。"
PARCELS = (f"{TODAY_ZH}\n包裹信息：\n"
           "- 运单号 7731 0526 88，收件人李娜，手机尾号 3721，原地址杭州市拱墅区湖墅南路 186 号 2 幢 1101 室；状态：杭州转运中心，待派送（预计明天上午派送）。\n"
           "- 运单号 7731 0529 03，收件人周强，手机尾号 5580，原地址苏州市姑苏区人民路 512 号；状态：派送中（派送员王师傅，电话 138 0512 2246）。")
GUESTS = (f"今天是 2026 年 10 月 16 日，星期五，09:40。\n入住记录（今天都应在 12:00 前退房）：\n"
          "- 1608 房，孙丽，大床房，房价 598 元一晚，普通会员。\n"
          "- 2215 房，何建，双床房，房价 698 元一晚，金卡会员。\n"
          "今日预订：1608 房今晚无人入住；2215 房今晚 19:00 有新客人入住。")
ZH_NOTE = "全程说普通话，口语化，句子要短。"

# ---------------------------------------------------------------- scenarios (one case each)

async def run(name: str, sc: dict, fmt: str, *, audio_out: bool, seed: int, chunk_ms: int, clock: str = "input"):
    from interaction_gym.agents.vllm_omni import VllmOmniDuplexAgent  # needs websockets (extra vllm-omni)
    llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=80)
    voice = Voice(Cached(OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR)), voices=QWEN3_TTS_VOICES, clone=OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR))
    user = benchmark_user(LLMSource(llm), voice=voice, interrupt=LLMInterrupt(llm), timing=TurnTaking(response_delay=ResponseDelay()),
                          soundscape=Soundscape(background="background" in sc))
    task = make_task(name, sc)
    spec = AgentSpec(chunk_ms=chunk_ms, audio="user.audio", sr=SR)
    env = Env({"user": user}, spec, max_ms=sc.get("max_ms", 180_000))
    agent = VllmOmniDuplexAgent(spec, AGENT_URL, model=AGENT_MODEL, ref_audio=REF_AUDIO, clock=clock, audio_out=audio_out,
                                trace_tokens=clock == "input", session={"instructions": st.agent_instructions(task, fmt)})
    obs = await env.reset(task, seed=seed)
    wall, done = time.monotonic(), False
    try:
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
    finally:
        await agent.close()
    return task, env, agent, time.monotonic() - wall


async def score(ep: dict, task: Task, judge_llm) -> dict:
    try:
        res = await st.judge(ep, task, judge_llm)
    except Exception as e:  # noqa: BLE001  (keep the episode; it is scored as not computed)
        res = {"reward": {"total": None, "parts": {}}, "error": f"{e.__class__.__name__}: {e}"}
    ep["eval"]["reward"] = res["reward"]
    ep["meta"]["sop_judge"] = {k: v for k, v in res.items() if k != "reward"}
    return res


async def rejudge(out: Path, judge_llm) -> None:
    """Score the episodes in ``out`` again (e.g. after a judge change), in place."""
    from interaction_gym.traj import load
    eps = load(out / "episodes.jsonl")
    tasks = [Task(**{k: e["meta"]["task"].get(k) or {} for k in ("scenario", "initial_state", "criteria")}, id=e["meta"]["task"]["id"]) for e in eps]
    res = await asyncio.gather(*(score(e, t, judge_llm) for e, t in zip(eps, tasks)))
    for e, r in zip(eps, res):
        print(f"{e['meta']['episode_id']:22s} decision {r.get('decision')} reward {r['reward']['total']} {json.dumps(r['reward']['parts'])} {r.get('error', '')}")
    save(eps, out / "episodes.jsonl")
    (out / "judge.json").write_text(json.dumps({e["meta"]["episode_id"]: r for e, r in zip(eps, res)}, indent=1, ensure_ascii=False))


async def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    judge_llm = OpenAIChat(JUDGE_URL, LLM_MODEL, timeout=900, max_tokens=16000, temperature=0, chat_template_kwargs={"enable_thinking": True})
    if a.rejudge:
        return await rejudge(out, judge_llm)
    media = MediaStore(out)
    jobs = [(seed, name, sc) for seed in a.seeds for name, sc in SCENARIOS.items() if not a.only or name in a.only]
    sem = asyncio.Semaphore(a.parallel)  # the MiniCPM-o server takes a few sessions at once

    async def one(seed, name, sc):
        async with sem:
            eid = f"{name}-s{seed}"
            task, env, agent, wall = await run(name, sc, a.format, audio_out=a.audio_out, seed=seed, chunk_ms=a.chunk_ms)
            ep = episode(env, eid, media=media, agent=agent, meta={"wall_s": round(wall, 1), "policy": st.policy(task, a.format)},
                         verifier=st.VERIFIER)
            res = await score(ep, task, judge_llm)
            want = "do it" if st.should_do(task) else "decline"
            note = (f"SOP {sc['sop']['name']} · {task.scenario['sop_case']['action']} · should {want} · decision {res.get('decision')} · "
                    f"reward {res['reward']['total']} {json.dumps(res['reward']['parts'])} · {env.t / 1000:.0f}s" + (" · truncated" if env.truncated else ""))
            print(f"{eid:22s} {note}", flush=True)
            return eid, ep, agent.trace(eid), res, note

    done = await asyncio.gather(*(one(*j) for j in jobs))
    episodes = [d[1] for d in done]
    traces = [d[2] for d in done]
    results = {d[0]: d[3] for d in done}
    notes = {d[0]: d[4] for d in done}
    save(episodes, out / "episodes.jsonl")
    save([t for t in traces if t], out / "agent_traces.jsonl")
    (out / "notes.json").write_text(json.dumps(notes, indent=1))
    (out / "judge.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    (out / "qa.json").write_text("{}")
    print("->", export_run(episodes, out, notes, title=f"SOP service calls (format: {a.format})", traces=traces))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sop_calls")
    ap.add_argument("--format", default="structured", help="SOP format shown to the agent: " + ", ".join(sorted(formats())))
    ap.add_argument("--audio-out", action="store_true", help="MiniCPM-o speaks (Thinker + Talker + Code2Wav)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--chunk-ms", type=int, default=1000)
    ap.add_argument("--only", nargs="+", default=[])
    ap.add_argument("--parallel", type=int, default=4, help="concurrent calls (MiniCPM-o sessions)")
    ap.add_argument("--rejudge", action="store_true", help="only score the episodes already in --out again")
    asyncio.run(main(ap.parse_args()))
