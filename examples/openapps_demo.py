"""Co-browsing on OpenApps (``integrations.openapps``): the caller asks the agent, by voice, to do things in their apps
(calendar, to-dos, messages, map) and watches it on a shared screen; the agent operates the desktop and sees it too.

    # OpenApps (CC BY-NC 4.0) in its own environment: see integrations/openapps.py
    IG_LLM_URL=http://localhost:8000/v1 python examples/openapps_demo.py
    python examples/openapps_demo.py --tts --out runs/openapps   # voices, and a browsable run

The caller runs on ``IG_LLM_*``; the agent on ``IG_VLM_*`` (a model that takes images, with tool calls parsed from its
text). Every LLM call takes a fixed simulated time (``--llm-ms``).

Licence: OpenApps (Meta FAIR) is CC BY-NC 4.0, non-commercial use only; it is installed separately, not shipped here,
and the runs this script records (screenshots, task text) are under that licence too.
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
from interaction_gym.integrations import openapps as oa
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

CASES = [  # (OpenApps longer-horizon task, cue, seed)
    ("book_club_event_and_message_alice", "none", 0),
    ("book_club_event_and_message_alice", "misspoke", 0),
    ("mom_birthday_plan", "staged", 0),
    ("mom_birthday_plan", "misspoke", 1),
    ("conference_full_setup", "none", 0),
    ("conference_full_setup", "staged", 0),
]


async def run(case, args, llm, vlm, tts, clone, media):
    key, cue, seed = case
    task = oa.make_task(key, cue, seed)
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN, RESULT["agent"]), sr=SR)
    voice = Voice(tts, voice="serena", clone=clone) if tts else Voice()
    user = oa.AppUser(co.CoSource(llm), voice=voice, timing=TurnTaking(respond_after_ms=600, nudge_after_ms=12000))
    env = Env({"user": user, "tools": oa.tool_world()}, spec, max_ms=args.max_s * 1000)
    agent = ScreenToolAgent(spec, vlm, args.llm_ms, args.look_ms, max_rounds=40, tts=tts, keep_results=3)
    try:
        obs, done = await env.reset(task, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
        res = oa.evaluate(env, task)
        parts = {"openapps_reward": res["reward"], "as_meant": float(res["as_meant"]), "late_correction": float(res["late_correction"])}
        ep = episode(env, task.id, media=media, agent=agent, verifier=oa.VERIFIER, reward={"total": float(res["success"]), "parts": parts},
                     meta={"benchmark": f"openapps.{key}", "goal": task.scenario["openapps"]["goal"], "cobrowse": res, "agent_rounds": agent.trace})
        return res, ep
    finally:
        bg.BrowserGymBackend.close(bg.browser_state(env))


def note(task, res) -> str:
    s, n = res["slip"], len(task.scenario["steps"])
    what = (f"the caller misspoke {'the date' if s['kind'] == 'date' else 'who the message is for'} "
            f"({s['intended']!r} as {s['done']!r})" if s else
            f"asked one step at a time ({len(res['asks']) + 1} of {n} asked)" if res["cue"] == "staged" else f"all {n} steps asked at once")
    k = len(res["corrections"])
    fix = "no corrections" if not k else f"{k} correction{'s' if k > 1 else ''}" + (" (one too late)" if res["late_correction"] else "")
    return (f"{n} steps · {what} · {fix} · {res['steps']} actions · OpenApps reward {res['reward']:.0f} · as meant "
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
        export_run(eps, args.out, notes=notes, title="Co-browsing on OpenApps (shared screen): samples")
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
        task = oa.make_task(*case)
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
        export_run(eps, args.out, notes=notes, title="Co-browsing on OpenApps (shared screen): samples")
        print("\nrun:", args.out / "index.html")


if __name__ == "__main__":
    asyncio.run(main())
