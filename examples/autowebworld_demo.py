"""Co-browsing on AutoWebWorld sites (``integrations.autowebworld``): the caller asks the agent, by voice, to do
something on a website (buy something and check out, post in a team chat, add a task to a project) and watches it on a shared screen; the agent operates the site.

    # build the sites first (see integrations/autowebworld.py); AWW_SITES=<dir with <site>/dist and <site>/fsm.json>
    IG_LLM_URL=http://localhost:8000/v1 python examples/autowebworld_demo.py
    python examples/autowebworld_demo.py --tts --out runs/aww   # voices, and a browsable run

The caller runs on ``IG_LLM_*``; the agent on ``IG_VLM_*`` (a model that takes images, with tool calls parsed from its
text). Every LLM call takes a fixed simulated time (``--llm-ms``).

Licence: AutoWebWorld is MIT; its sites are built from your own checkout, not shipped here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.cascaded import ToolChat
from interaction_gym.agents.watcher import ScreenToolAgent
from interaction_gym.clients import OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.integrations import browsergym as bg
from interaction_gym.integrations import cobrowse as co
from interaction_gym.integrations import autowebworld as aw
from interaction_gym.media import MediaStore
from interaction_gym.tools import RESULT
from interaction_gym.traj import episode, save
from interaction_gym.user import TurnTaking, Voice
from interaction_gym.video import SCREEN

LLM_URL, LLM_MODEL = getenv("IG_LLM_URL", "http://localhost:8000/v1"), getenv("IG_LLM_MODEL", "Qwen3.8-27B")
VLM_URL, VLM_MODEL = getenv("IG_VLM_URL", LLM_URL), getenv("IG_VLM_MODEL", LLM_MODEL)
TTS_URL, TTS_MODEL = getenv("IG_TTS_URL", "http://localhost:8001/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CLONE_URL, CLONE_MODEL = getenv("IG_CLONE_URL", "http://localhost:8005/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SR = 24000

CASES = [  # (site, cue, seed)
    ("shopify", "none", 0),
    ("shopify", "staged", 2),
    ("shopify", "misspoke", 4),
    ("slack", "none", 0),
    ("slack", "misspoke", 1),
    ("asana", "none", 0),
    ("asana", "staged", 2),
    ("asana", "misspoke", 1),
]


async def run(case, args, llm, vlm, tts, clone, media):
    site, cue, seed = case
    task = aw.make_task(site, cue, seed)
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN, RESULT["agent"]), sr=SR)
    voice = Voice(tts, voice="serena", clone=clone) if tts else Voice()
    user = aw.SiteUser(co.CoSource(llm), voice=voice, timing=TurnTaking(respond_after_ms=600, nudge_after_ms=12000))
    env = Env({"user": user, "tools": aw.tool_world()}, spec, max_ms=args.max_s * 1000)
    agent = ScreenToolAgent(spec, vlm, args.llm_ms, args.look_ms, max_rounds=40, tts=tts, keep_results=3)
    try:
        obs, done = await env.reset(task, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
        res = aw.evaluate(env, task)
        parts = {"state_reward": res["reward"], "as_meant": float(res["as_meant"]), "late_correction": float(res["late_correction"])}
        ep = episode(env, task.id, media=media, agent=agent, verifier=aw.VERIFIER, reward={"total": float(res["success"]), "parts": parts},
                     meta={"benchmark": f"autowebworld.{site}", "goal": task.scenario["aww"]["what"], "cobrowse": res, "agent_rounds": agent.trace})
        return res, ep
    finally:
        bg.BrowserGymBackend.close(bg.browser_state(env))


def note(task, res) -> str:
    s, n = res["slip"], len(task.scenario["steps"])
    what = (f"the caller misspoke the {s['kind']} ({s['intended']!r} as {s['done']!r})" if s else
            f"asked in parts ({len(res['asks']) + 1} of {len({x['stage'] for x in task.scenario['steps']})})" if res["cue"] == "staged" else "all asked at once")
    k = len(res["corrections"])
    fix = "no corrections" if not k else f"{k} correction{'s' if k > 1 else ''}" + (" (one too late)" if res["late_correction"] else "")
    return (f"{task.scenario['aww']['what']} · {what} · {fix} · {res['steps']} actions · ends on {res['final_page']} · state reward {res['reward']:.0f} · as meant "
            f"{'yes' if res['as_meant'] else 'no'}" + ("" if res["as_meant"] else f" ({'; '.join(res['left'])})"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--llm-ms", type=int, default=800)
    ap.add_argument("--look-ms", type=int, default=1000)
    ap.add_argument("--max-s", type=int, default=240)
    ap.add_argument("--cases", help="comma-separated indices into CASES")
    ap.add_argument("--part", help="with --out: save this process's episodes as a part (episodes.<part>.jsonl), for --merge")
    ap.add_argument("--merge", action="store_true", help="with --out: build the run from the parts saved there")
    args = ap.parse_args()
    if args.merge:
        from interaction_gym.viewer import export_run
        eps, notes = [], {}
        for f in sorted(args.out.glob("episodes.*.jsonl")):
            eps += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
            notes.update(json.loads(f.with_name(f.name.replace("episodes.", "notes.").replace(".jsonl", ".json")).read_text()))
        save(eps, args.out / "episodes.jsonl")
        export_run(eps, args.out, notes=notes, title="Co-browsing on AutoWebWorld sites (shared screen): samples")
        print("run:", args.out / "index.html")
        return
    llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=150, temperature=0.7)
    vlm = ToolChat(VLM_URL, VLM_MODEL, native=False, max_tokens=400, temperature=0)
    tts = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR) if args.tts else None
    clone = OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR) if args.tts else None
    media = MediaStore(args.out) if args.out else None
    cases = [CASES[int(i)] for i in args.cases.split(",")] if args.cases else CASES
    eps, notes = [], {}
    for case in cases:
        task = aw.make_task(*case)
        res, ep = await run(case, args, llm, vlm, tts, clone, media)
        eid = ep["meta"]["episode_id"]
        eps.append(ep)
        notes[eid] = note(task, res)
        print(f"\n== {eid}: {notes[eid]}  [first wrong {res['first_wrong_t']} all done {res['all_done_t']}]")
        for t in ep["turns"]:
            if t["text"]:
                print(f"   {t['start_time'] / 1000:5.1f}s {t['role']:5s} {t['text']}")
        for r in ep["meta"]["agent_rounds"]:
            print(f"   {r['t'] / 1000:5.1f}s [{r['kind']}] {r['said']!r} {r['calls']}")
        print("   changes:", json.dumps(res["changes"])[:600])
    if args.out and args.part:
        save(eps, args.out / f"episodes.{args.part}.jsonl")
        (args.out / f"notes.{args.part}.json").write_text(json.dumps(notes, indent=1))
    elif args.out:
        from interaction_gym.viewer import export_run
        save(eps, args.out / "episodes.jsonl")
        (args.out / "notes.json").write_text(json.dumps(notes, indent=1))
        export_run(eps, args.out, notes=notes, title="Co-browsing on AutoWebWorld sites (shared screen): samples")
        print("\nrun:", args.out / "index.html")


if __name__ == "__main__":
    asyncio.run(main())
