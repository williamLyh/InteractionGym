"""Computer use in InteractionGym: a caller asks, an LLM agent operates a real browser (BrowserGym MiniWoB++).

The simulated user says the page's instruction; the agent hears it, calls click / fill / ... on the page
and talks back; the benchmark's own reward scores the episode. The agent hears the user's transcript (no ASR);
with ``--tts`` it speaks through a TTS, else its text is timed at speaking rate. Each LLM call takes ``--llm-ms``
of simulated time (the env waits for the LLM, so wall-clock speed does not change the episode).

    pip install browsergym-miniwob && playwright install chromium
    export MINIWOB_URL=file:///path/to/miniwob-plusplus/miniwob/html/miniwob/
    IG_LLM_URL=http://localhost:8000/v1 IG_LLM_MODEL=Qwen3.8-27B python examples/browsergym_demo.py click-button enter-text
    python examples/browsergym_demo.py --record runs/cu      # the two example episodes of the demo site (TTS voices)
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from interaction_gym import AgentSpec, Env
from interaction_gym.agents.cascaded import ToolChat
from interaction_gym.agents.transcript import TranscriptToolAgent
from interaction_gym.clients import OpenAISpeech
from interaction_gym.envvars import getenv
from interaction_gym.integrations import browsergym as bg
from interaction_gym.media import MediaStore
from interaction_gym.tools import RESULT, tool_log
from interaction_gym.traj import episode, save
from interaction_gym.user import ReplayUser, ScriptSource, TurnTaking, UserSim, Voice

LLM_URL, LLM_MODEL = getenv("IG_LLM_URL", "http://localhost:8000/v1"), getenv("IG_LLM_MODEL", "Qwen3.8-27B")
TTS_URL, TTS_MODEL = getenv("IG_TTS_URL", "http://localhost:8001/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CLONE_URL, CLONE_MODEL = getenv("IG_CLONE_URL", "http://localhost:8005/v1"), "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SR = 24000
SPEC = AgentSpec(chunk_ms=200, obs=("user.speech", RESULT["agent"]), sr=SR)


async def play(env: Env, agent: TranscriptToolAgent, task) -> Env:
    obs, done = await env.reset(task, seed=0), False
    while not done:
        obs, _, done = await env.step(await agent.act(env.t, obs))
    return env


async def run(name: str, seed: int, llm: ToolChat, llm_ms: int) -> dict:
    task = bg.make_task(f"browsergym/miniwob.{name}", seed)
    user = UserSim(ScriptSource(), timing=TurnTaking(respond_after_ms=600))
    env = Env({"user": user, "tools": bg.tool_world(screenshots=False)}, SPEC, max_ms=90_000)
    try:
        await play(env, TranscriptToolAgent(SPEC, llm, llm_ms), task)
        res = bg.evaluate(env)
        ep = episode(env, task.id)
        res.update(task=task.id, goal=task.scenario["goal"], ms=env.t, turns=[f"{u['role']}: {u['text']}" for u in ep["turns"]],
                   calls=[(t, c.name, c.arguments, "ERR" if r and r.error else "ok") for t, _, c, r in tool_log(env.log)])
        return res
    finally:
        bg.BrowserGymBackend.close(bg.browser_state(env))


# ---------------------------------------------------------------- the demo site's two examples

EXAMPLES = [
    # (episode id, MiniWoB task, seed, user turns: (text, seconds after the previous turn ends | None = at t=0), note)
    ("cu-login-cierra", "login-user", 0,
     [("Hi, could you log me in? The username is cierra, C I E R R A, and the password is 11L.", None)],
     "Computer use (BrowserGym MiniWoB++ login-user): the caller dictates the credentials; the agent reads the page, fills both fields and clicks Login while telling the caller what it does."),
    ("cu-correction-myron", "enter-text", 0,
     [("Hey, can you type Myra into the text field and press submit?", None),
      ("Oh sorry, wait, not Myra. Myron. M Y R O N.", 0.4)],
     "Computer use with a correction (MiniWoB++ enter-text): the caller gives the wrong name and corrects it while the agent is already working; only the corrected name scores."),
]


async def record(out: Path, llm: ToolChat, llm_ms: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    media = MediaStore(out)
    tts, clone = OpenAISpeech(TTS_URL, TTS_MODEL, sr=SR), OpenAISpeech(CLONE_URL, CLONE_MODEL, sr=SR)
    episodes, notes = [], {}
    for eid, name, seed, script, note in EXAMPLES:
        task = bg.make_task(f"browsergym/miniwob.{name}", seed)
        task.id, task.scenario["voice"] = eid, "serena"
        task.scenario["persona"] = "A caller who asks an assistant to operate a web page for them."
        turns, t = [], 0
        for i, (text, gap) in enumerate(script):  # pre-timed turns: the first is synthesized here to time the next
            audio = await tts.synth(text, "serena") if i == 0 else None
            if i:
                t = t + round(gap * 1000)
            turns.append({"t": t, "text": text, **({"audio": audio} if audio is not None else {})})
            if audio is not None:
                t += audio.dur_ms
        env = Env({"user": ReplayUser(turns, voice=Voice(tts, voice="serena", clone=clone)), "tools": bg.tool_world()}, SPEC, max_ms=90_000)
        agent = TranscriptToolAgent(SPEC, llm, llm_ms, tts=tts, voice="aiden")
        try:
            await play(env, agent, task)
            res = bg.evaluate(env)
            reward = {"total": res["reward"], "parts": {"benchmark_reward": res["reward"], "page_ended_task": float(res["terminated"])}}
            episodes.append(episode(env, eid, media=media, agent=agent, reward=reward, meta={"benchmark": f"miniwob.{name}", "goal": task.scenario["goal"]},
                                    verifier=bg.VERIFIER))
            notes[eid] = f"{note} · reward {res['reward']:.2f}"
            print(eid, res)
        finally:
            bg.BrowserGymBackend.close(bg.browser_state(env))
    save(episodes, out / "episodes.jsonl")
    (out / "agent_traces.jsonl").write_text("")
    (out / "qa.json").write_text("{}")
    (out / "notes.json").write_text(json.dumps(notes, indent=1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", help="MiniWoB task names, e.g. click-button enter-text")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--llm-ms", type=int, default=700)
    ap.add_argument("--record", type=Path, help="record the demo site's examples into this run directory")
    args = ap.parse_args()
    llm = ToolChat(LLM_URL, LLM_MODEL, native=False, max_tokens=400, temperature=0)
    if args.record:
        return await record(args.record, llm, args.llm_ms)
    wins = n = 0
    for name in args.tasks:
        for seed in range(args.seeds):
            res = await run(name, seed, llm, args.llm_ms)
            n += 1
            wins += res["success"]
            print(f"\n== {res['task']}  reward {res['reward']:.2f}  success {res['success']}  steps {res['steps']}  {res['ms'] / 1000:.1f}s")
            print("   goal:", res["goal"])
            for line in res["turns"]:
                print("  ", line)
            for t, name_, a, ok in res["calls"]:
                print(f"   {t / 1000:6.1f}s  {name_}({json.dumps(a)})  {ok}")
    print(f"\nsuccess {wins}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
