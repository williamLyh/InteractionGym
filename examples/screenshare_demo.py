"""Screen sharing in InteractionGym: the caller fills in a web page themselves; the agent watches their screen (video)
and listens, and has to catch the caller's slip before they submit (``integrations.screenshare``).

    pip install browsergym-miniwob && playwright install chromium
    export MINIWOB_URL=file:///path/to/miniwob-plusplus/miniwob/html/miniwob/
    IG_LLM_URL=http://localhost:8000/v1 IG_VLM_URL=http://localhost:8000/v1 python examples/screenshare_demo.py
    python examples/screenshare_demo.py --agent silent          # nobody warns: the slips go through
    python examples/screenshare_demo.py --tts --out runs/screen  # voices, and a browsable run in runs/screen/site

The user (speech and hands) runs on ``IG_LLM_*``; the agent on ``IG_VLM_*``, a model that takes images. Every LLM call
takes a fixed simulated time (``--llm-ms``), so how fast the server answers does not change the episode.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.watcher import ScreenWatchAgent
from interaction_gym.clients import OpenAIChat, OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.integrations import browsergym as bg
from interaction_gym.integrations import screenshare as ss
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode, save
from interaction_gym.user import TurnTaking, Voice
from interaction_gym.video import SCREEN

LLM_URL, LLM_MODEL = getenv("IG_LLM_URL", "http://localhost:8000/v1"), getenv("IG_LLM_MODEL", "Qwen3.8-27B")
VLM_URL, VLM_MODEL = getenv("IG_VLM_URL", LLM_URL), getenv("IG_VLM_MODEL", LLM_MODEL)
TTS_URL, TTS_MODEL = getenv("IG_TTS_URL", "http://localhost:8001/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CLONE_URL, CLONE_MODEL = getenv("IG_CLONE_URL", "http://localhost:8005/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SR = 24000

CASES = [  # (MiniWoB page, seed, slip)
    ("enter-text", 0, "typo"),
    ("login-user", 0, "drop"),
    ("form-sequence-2", 0, "wrong_field"),
    ("enter-text-2", 0, "case"),
    ("login-user", 1, "none"),
]


class Silent:
    """Says nothing (the floor: every slip goes through)."""

    def describe(self):
        return {"name": "silent", "kind": "text"}

    async def act(self, t, obs):
        return []


async def run(case, args, llm, vlm, tts, clone, media):
    name, seed, slip = case
    task = ss.make_task(name, seed, slip, review_ms=args.review_ms)
    spec = AgentSpec(chunk_ms=200, obs=("user.speech", SCREEN), sr=SR)
    voice = Voice(tts, voice="serena", clone=clone) if tts else Voice()
    user = ss.ScreenUser(ss.ScreenShareSource(llm), voice=voice, timing=TurnTaking(respond_after_ms=600, nudge_after_ms=9000))
    env = Env({"user": user, "tools": ss.tool_world()}, spec, max_ms=args.max_s * 1000)
    agent = Silent() if args.agent == "silent" else ScreenWatchAgent(spec, vlm, args.llm_ms, args.look_ms, tts=tts)
    try:
        obs, done = await env.reset(task, seed=0), False
        while not done:
            obs, _, done = await env.step(await agent.act(env.t, obs))
        res = ss.evaluate(env, task)
        parts = {"benchmark_reward": res["reward"], "false_alarm": float(res["false_alarms"] > 0)}
        if res["caught"] is not None:
            parts["caught"] = float(res["caught"])
        ep = episode(env, task.id, media=media, agent=agent, verifier=ss.VERIFIER,
                     reward={"total": float(res["success"]), "parts": parts},
                     meta={"benchmark": f"miniwob.{name}", "goal": task.scenario["goal"], "screenshare": res,
                           "agent_rounds": getattr(agent, "trace", None)})
        return res, ep
    finally:
        bg.BrowserGymBackend.close(bg.browser_state(env))


def note(res) -> str:
    s = res["slip"]
    if s is None:
        return f"Control (no slip) · page reward {res['reward']:.0f} · false alarms {res['false_alarms']}"
    what = {"typo": "two letters swapped", "drop": "a letter left out", "case": "wrong capitalisation", "wrong_field": "the wrong element"}[s["kind"]]
    caught = f"caught {res['catch_ms'] / 1000:.1f}s after it showed" if res["caught"] else "not caught"
    pair = (s["intended_name"], s["done_name"]) if s["kind"] == "wrong_field" else (repr(s["intended"]), repr(s["done"]))
    return f"Slip: {what} ({pair[0]} → {pair[1]}) · {caught} · page reward {res['reward']:.0f}"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=("vlm", "silent"), default="vlm")
    ap.add_argument("--tts", action="store_true", help="voices for the user (cloned) and the agent")
    ap.add_argument("--out", type=Path, help="save episodes.jsonl and a browsable run (site/) here")
    ap.add_argument("--llm-ms", type=int, default=800)
    ap.add_argument("--look-ms", type=int, default=1000)
    ap.add_argument("--review-ms", type=int, default=3000, help="the user's pause between typing and submitting")
    ap.add_argument("--max-s", type=int, default=90)
    ap.add_argument("--cases", help="comma-separated indices into CASES")
    args = ap.parse_args()
    llm = OpenAIChat(LLM_URL, LLM_MODEL, max_tokens=120, temperature=0.7)
    vlm = OpenAIChat(VLM_URL, VLM_MODEL, max_tokens=80, temperature=0)
    tts = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR) if args.tts else None
    clone = OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR) if args.tts else None
    media = MediaStore(args.out) if args.out else None
    cases = [CASES[int(i)] for i in args.cases.split(",")] if args.cases else CASES
    eps, notes = [], {}
    for case in cases:
        res, ep = await run(case, args, llm, vlm, tts, clone, media)
        eps.append(ep)
        notes[ep["meta"]["episode_id"]] = note(res)
        h = ep["meta"]["screenshare"]
        print(f"\n== {ep['meta']['episode_id']}: {note(res)}  [visible {h['visible_t']} submit {h['submit_t']}]")
        for t in ep["turns"]:
            if t["text"]:
                print(f"   {t['start_time'] / 1000:5.1f}s {t['role']:5s} {t['text']}")
        for r in ep["meta"].get("agent_rounds") or []:
            if r["said"]:
                print(f"   {r['t'] / 1000:5.1f}s ({r['kind']}) {r['said']}")
    if args.out:
        from interaction_gym.viewer import export_run
        save(eps, args.out / "episodes.jsonl")
        (args.out / "notes.json").write_text(json.dumps(notes, indent=1))
        export_run(eps, args.out, notes=notes, title="Screen sharing (rendered video): samples")
        print("\nrun:", args.out / "index.html")


if __name__ == "__main__":
    asyncio.run(main())
