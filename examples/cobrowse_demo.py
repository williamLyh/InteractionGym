"""Co-browsing in InteractionGym: the agent operates a web page for the caller on a shared screen; both see it
(``integrations.cobrowse``). In *drive* tasks the caller watches and corrects (a misspoken name they only notice on the
screen, a box named by where it is); in *handoff* tasks the caller types their own password and the agent has to see
when it is in.

    pip install browsergym-miniwob && playwright install chromium
    export MINIWOB_URL=file:///path/to/miniwob-plusplus/miniwob/html/miniwob/
    IG_LLM_URL=http://localhost:8000/v1 IG_VLM_URL=http://localhost:8000/v1 python examples/cobrowse_demo.py
    python examples/cobrowse_demo.py --tts --out runs/cobrowse   # voices, and a browsable run

The caller runs on ``IG_LLM_*``; the agent on ``IG_VLM_*`` (a model that takes images, with tool calls parsed from its
text). Every LLM call takes a fixed simulated time (``--llm-ms``).
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

CASES = [  # (MiniWoB page, seed, mode, cue, slip kind)
    ("enter-text", 0, "drive", "misspoke", "typo"),
    ("login-user", 0, "drive", "misspoke", "drop"),
    ("form-sequence-2", 0, "drive", "misspoke", None),
    ("form-sequence-2", 1, "drive", "where", None),
    ("enter-text-2", 0, "drive", "none", None),
    ("login-user", 0, "handoff", "none", None),
    ("login-user", 1, "handoff", "none", None),
]


async def run(case, args, llm, vlm, tts, clone, media):
    name, seed, mode, cue, kind = case
    task = co.make_task(name, seed, mode, cue, slip_kind=kind or "typo")
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN, RESULT["agent"]), sr=SR)
    voice = Voice(tts, voice="serena", clone=clone) if tts else Voice()
    user = co.CoUser(co.CoSource(llm), voice=voice, timing=TurnTaking(respond_after_ms=600, nudge_after_ms=9000))
    env = Env({"user": user, "tools": co.tool_world()}, spec, max_ms=args.max_s * 1000)
    agent = ScreenToolAgent(spec, vlm, args.llm_ms, args.look_ms, tts=tts)
    try:
        obs, done = await env.reset(task, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
        res = co.evaluate(env, task)
        parts = {"benchmark_reward": res["reward"], "as_meant": float(res["as_meant"]), "late_correction": float(res["late_correction"])}
        if mode == "handoff":
            parts.update(agent_typed_secret=float(res["agent_typed_secret"]), waited=float(res["waited"]))
        ep = episode(env, task.id, media=media, agent=agent, verifier=co.VERIFIER, reward={"total": float(res["success"]), "parts": parts},
                     meta={"benchmark": f"miniwob.{name}", "goal": task.scenario["goal"], "cobrowse": res, "agent_rounds": agent.trace})
        return res, ep
    finally:
        bg.BrowserGymBackend.close(bg.browser_state(env))


def note(task_id, res) -> str:
    tail = f"page reward {res['reward']:.0f} · as meant {'yes' if res['as_meant'] else 'no'}"
    lw = f" · filled page shown {res['look_window_ms'] / 1000:.1f}s before submit" if res["look_window_ms"] is not None else ""
    if "handoff" in task_id:
        h = (f"agent finished {res['handoff_ms'] / 1000:.1f}s after the password was in" if res["handoff_ms"] is not None else
             "agent did not wait for the password" if not res["waited"] else "never finished")
        return (f"Handoff: the caller types the password themselves · {h}" + (" (after being told)" if res.get("needed_telling") else "")
                + (" · agent typed in the password field" if res["agent_typed_secret"] else "") + f" · {tail}")
    s, cue = res["slip"], res["cue"]
    what = (f"the caller misspoke {s['intended_name'] if s['kind'] == 'wrong_field' else repr(s['intended'])} as "
            f"{s['done_name'] if s['kind'] == 'wrong_field' else repr(s['done'])}" if s else
            "the caller named the box by where it is" if cue == "where" else "no planted cue")
    k = len(res["corrections"])
    fix = "no corrections" if not k else f"{k} correction{'s' if k > 1 else ''}" + (" (too late)" if res["late_correction"] else "")
    return f"Drive: {what} · {fix}{lw} · {tail}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--llm-ms", type=int, default=800)
    ap.add_argument("--look-ms", type=int, default=1000)
    ap.add_argument("--max-s", type=int, default=90)
    ap.add_argument("--cases", help="comma-separated indices into CASES")
    args = ap.parse_args()
    llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=120, temperature=0.7)
    vlm = ToolChat(VLM_URL, VLM_MODEL, native=False, max_tokens=400, temperature=0)
    tts = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR) if args.tts else None
    clone = OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR) if args.tts else None
    media = MediaStore(args.out) if args.out else None
    cases = [CASES[int(i)] for i in args.cases.split(",")] if args.cases else CASES
    eps, notes = [], {}
    for case in cases:
        res, ep = await run(case, args, llm, vlm, tts, clone, media)
        eid = ep["meta"]["episode_id"]
        eps.append(ep)
        notes[eid] = note(eid, res)
        print(f"\n== {eid}: {notes[eid]}  [first wrong {res['first_wrong_t']} submit {res['submit_t']}]")
        for t in ep["turns"]:
            if t["text"]:
                print(f"   {t['start_time'] / 1000:5.1f}s {t['role']:5s} {t['text']}")
        for c in ep.get("tool_calls", []) if isinstance(ep.get("tool_calls"), list) else []:
            print("   call", c)
        for r in ep["meta"]["agent_rounds"]:
            print(f"   {r['t'] / 1000:5.1f}s [{r['kind']}] {r['said']!r} {r['calls']}")
    if args.out:
        from interaction_gym.viewer import export_run
        save(eps, args.out / "episodes.jsonl")
        (args.out / "notes.json").write_text(json.dumps(notes, indent=1))
        export_run(eps, args.out, notes=notes, title="Co-browsing (shared screen): samples")
        print("\nrun:", args.out / "index.html")


if __name__ == "__main__":
    asyncio.run(main())
