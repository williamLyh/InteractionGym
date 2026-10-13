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

# ---------------------------------------------------------------- SOPs (source format: interaction_gym.sop)

RESTAURANT_SOP = validate({
    "name": "Trattoria Roma: table bookings",
    "rules": ["Never confirm a booking you have not checked against the free tables.", "Keep replies short."],
    "actions": {"book_table": {
        "description": "Book a table for the caller.",
        "requires": {"chain": [
            {"check": "request_details", "text": "The caller has told you the day, the time and the number of people.", "by": "ask"},
            {"all": [
                {"check": "opening_hours", "text": "The time is within opening hours (Tuesday to Sunday, 17:00 to 22:00; last booking 21:00).",
                 "by": "look_up", "on_fail": "Suggest the nearest time within opening hours."},
                {"any": [
                    {"check": "party_of_8_or_fewer", "text": "The party is 8 people or fewer.", "by": "look_up"},
                    {"check": "deposit_agreed", "text": "For 9 people or more: the caller agrees to a deposit of 20 euros per person.", "by": "ask",
                     "on_fail": "Give them the events line, 020 7946 0321, for large groups."}]},
                {"check": "table_free", "text": "A table for that many people is free at that time (see today's free tables).", "by": "look_up",
                 "on_fail": "Offer the nearest free time."}]},
            {"check": "name_and_phone", "text": "The caller has given a name and a phone number for the booking.", "by": "ask"}]},
        "then": ["Read the booking back (day, time, number of people, name) and confirm it."],
        "otherwise": "Explain which condition is not met, decline, and offer the alternative if there is one."}}})

CLINIC_SOP = validate({
    "name": "Greenfield Medical Clinic: appointments",
    "rules": ["Never give medical advice.", "Speak clearly and keep replies short."],
    "actions": {"book_appointment": {
        "description": "Book a doctor's appointment for the caller.",
        "requires": {"chain": [
            {"check": "identity", "text": "The caller has given their full name and date of birth, and they match a registered patient (see the patient list).",
             "by": "ask", "on_fail": "Ask them to register at the front desk with photo ID first."},
            {"check": "not_emergency", "text": "The problem is not an emergency (no chest pain, trouble breathing, heavy bleeding or sudden weakness).",
             "by": "ask", "on_fail": "Tell them to call 999 or go to A&E now."},
            {"check": "open_slot", "text": "The time is an open slot in a doctor's schedule (see the open slots).", "by": "look_up",
             "on_fail": "Offer the nearest open slot."}]},
        "then": ["Confirm the doctor, the day and the time.", "Ask them to arrive 10 minutes early."],
        "otherwise": "Decline and say what to do instead."}}})

SHOP_SOP = validate({
    "name": "VoltMart online shop: returns and refunds",
    "rules": ["Keep replies short.", "Do not promise anything the policy does not allow."],
    "actions": {"refund_return": {
        "description": "Take back an item for a full refund.",
        "requires": {"chain": [
            {"check": "order_verified", "text": "The caller has given the order number and the email address on the order, and they match an order (see the orders).",
             "by": "ask"},
            {"check": "within_30_days", "text": "The order was delivered no more than 30 days ago.", "by": "look_up",
             "on_fail": "Offer a free repair under the one-year warranty instead."},
            {"any": [
                {"check": "faulty", "text": "The item is faulty (does not work as described).", "by": "ask"},
                {"check": "unopened", "text": "The item is unopened, in its original packaging.", "by": "ask"}]}]},
        "then": ["Tell them a return label is emailed and the refund arrives 5 to 7 working days after the item does."],
        "otherwise": "Explain why a refund is not possible, decline it, and offer the alternative."}}})

TODAY = "Today is Friday 16 October 2026, 15:20."

# ---------------------------------------------------------------- scenarios (one case each)

SCENARIOS = {
    "booking-alex": dict(  # should be done
        agent_role="You are the phone booking assistant of Trattoria Roma, an Italian restaurant.",
        sop=RESTAURANT_SOP,
        sop_case={"action": "book_table",
                  "facts": f"{TODAY}\nFree tables tonight: for 2 at 18:30, 19:00 and 20:30; for 4 at 20:00; for 6 at 21:00.\nSaturday: fully booked.",
                  "truth": {"request_details": True, "opening_hours": True, "party_of_8_or_fewer": True, "deposit_agreed": False,
                            "table_free": True, "name_and_phone": True},
                  "expect": {"day": "tonight", "time": "19:00", "people": 2, "name": "Alex Chen"}},
        persona="You are calling a restaurant to book a table.",
        profile={"name": "Alex Chen", "gender": "female", "age": 34, "language": "en", "occupation": "product manager",
                 "speaking_style": "friendly and quick, a little rushed", "traits": "busy, to the point"},
        goal=("Book a table for two people tonight at around 7pm (7pm exactly is best; 6:30 is fine if 7 is not free). Your name is "
              "Alex Chen and your phone number is 07700 900123; give them when asked. Correct the agent if it gets a detail wrong. "
              "Once the booking is confirmed, thank them and say goodbye."),
        first_turn="Hi, I'd like to book a table for tonight."),
    "booking-mina": dict(  # should be declined: 10 people and no deposit
        agent_role="You are the phone booking assistant of Trattoria Roma, an Italian restaurant.",
        sop=RESTAURANT_SOP,
        sop_case={"action": "book_table",
                  "facts": f"{TODAY}\nFree tables tomorrow, Saturday: for 4 at 19:30; for 10 at 20:00 (the back room).\nTonight: fully booked.",
                  "truth": {"request_details": True, "opening_hours": True, "party_of_8_or_fewer": False, "deposit_agreed": False,
                            "table_free": True, "name_and_phone": True}},
        persona="You are calling a restaurant from a noisy cafe.",
        profile={"name": "Mina Park", "gender": "female", "age": 45, "language": "en", "native_language": "ko", "occupation": "architect",
                 "speaking_style": "warm, measured", "traits": "speaks English with a Korean accent, double-checks details"},
        goal=("Book a table for ten people, your team's dinner, tomorrow (Saturday) at 8pm. Your name is Mina Park, phone 07700 900456. "
              "You cannot pay a deposit: the company only pays on the night. If they ask for a deposit, say so and do not agree. "
              "If they cannot book it, ask what you can do instead, note it, thank them and say goodbye."),
        first_turn="Hello, I would like to make a reservation for Saturday, please.",
        background=-12),  # as the chat set's noisy cafe
    "clinic-frank": dict(  # should be done
        agent_role="You are the receptionist of Greenfield Medical Clinic.",
        sop=CLINIC_SOP,
        sop_case={"action": "book_appointment",
                  "facts": (f"{TODAY}\nPatient list (name, date of birth): Frank Li, 3 March 1954; Mary Jones, 12 July 1961; Omar Haddad, 30 January 1990.\n"
                            "Open slots next week: Dr Patel, Tuesday 20 October 9:30 and Wednesday 21 October 11:00; Dr Okafor, Thursday 22 October 14:00."),
                  "truth": {"identity": True, "not_emergency": True, "open_slot": True},
                  "expect": {"appointment": "Dr Patel on Tuesday 20 October at 9:30, or Dr Patel on Wednesday 21 October at 11:00"}},
        persona="You are calling your local clinic.",
        profile={"name": "Frank Li", "gender": "male", "age": 72, "language": "en", "occupation": "retired teacher",
                 "speaking_style": "slow and deliberate, with pauses", "traits": "polite, a little hard of hearing, asks to repeat things"},
        goal=("Book an appointment with a doctor about your knee pain (it aches when you climb stairs, for two weeks now; nothing "
              "else is wrong), preferably on a weekday morning. Your full name is Frank Li, date of birth 3 March 1954; give them "
              "when asked. Once the appointment is confirmed, thank them and say goodbye."),
        first_turn="Hello? Is this the clinic? I'd like to see a doctor."),
    "return-jake": dict(  # should be declined: delivered 44 days ago; the alternative is a warranty repair
        agent_role="You are the customer service assistant of VoltMart, an online electronics shop.",
        sop=SHOP_SOP,
        sop_case={"action": "refund_return",
                  "facts": (f"{TODAY}\nOrders: #4471, jake.miller@example.com, AirBeat wireless headphones, delivered 2 September 2026 (44 days ago); "
                            "#4502, s.ng@example.com, USB-C charger, delivered 9 October 2026."),
                  "truth": {"order_verified": True, "within_30_days": False, "faulty": True, "unopened": False}},
        persona="You are calling an online shop about a broken purchase.",
        profile={"name": "Jake Miller", "gender": "male", "age": 24, "language": "en", "accent": "american",
                 "speaking_style": "energetic, talks fast", "traits": "impatient, interrupts when the agent rambles"},
        goal=("You want a full refund for wireless headphones (order 4471, email jake.miller@example.com) that stopped charging last "
              "week; you got them about six weeks ago. Give the order number and email when asked. Push once for the refund if "
              "they refuse; if they offer something else, accept it if it fixes the problem, then thank them and say goodbye."),
        first_turn="Hey, I need to return some headphones, they're already broken.",
        turn_taking={"yield_after_ms": None}),  # never stops when talked over
}


def make_task(name: str, sc: dict) -> Task:
    keep = ("persona", "profile", "first_turn", "turn_taking", "background", "agent_role", "sop", "sop_case")
    return Task(id=name, scenario={k: sc[k] for k in keep if k in sc} | {"instructions": sc["goal"], "categories": ["sop"]})


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
