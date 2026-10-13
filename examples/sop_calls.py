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

# Chinese SOPs ("lang": "zh": the formats' own words and the agent's frame are in Chinese too)
PARCEL_SOP = validate({
    "name": "顺达快递：修改收件地址",
    "lang": "zh",
    "rules": ["回答简短。", "不要透露来电者以外的任何包裹信息。"],
    "actions": {"change_address": {
        "description": "为来电者修改一个包裹的收件地址。",
        "requires": {"chain": [
            {"check": "parcel_verified", "text": "来电者报出了运单号和收件人手机号后四位，并且和系统里的一个包裹一致（见包裹信息）。", "by": "ask",
             "on_fail": "请对方核对运单号后再来电。"},
            {"check": "not_out_for_delivery", "text": "包裹还没有开始派送（状态不是「派送中」或「已签收」）。", "by": "look_up",
             "on_fail": "派送中的包裹不能改地址：把派送员的电话告诉对方，请对方直接联系派送员；或者拒收，包裹退回后再重新寄。"},
            {"all": [
                {"check": "new_address_given", "text": "来电者说出了完整的新地址（区、街道门牌、楼栋和房间号）。", "by": "ask"},
                {"check": "same_city", "text": "新地址和原地址（见包裹信息）在同一个城市。", "by": "look_up",
                 "on_fail": "跨城市不能改地址：可以拒收，包裹退回后由寄件人重新寄到新地址。"}]}]},
        "then": ["把新地址完整复述一遍，对方确认后，告诉对方修改已提交，送达会晚一天左右。"],
        "otherwise": "说明哪一条不满足，婉拒，并给出替代办法。"}}})

HOTEL_SOP = validate({
    "name": "云栖酒店：延迟退房",
    "lang": "zh",
    "rules": ["回答简短、礼貌。"],
    "actions": {"late_checkout": {
        "description": "为住客把今天的退房时间延后。",
        "requires": {"chain": [
            {"check": "guest_verified", "text": "来电者报出了房号和入住人姓名，并且和入住记录一致（见入住记录）。", "by": "ask"},
            {"all": [
                {"check": "room_not_booked", "text": "这个房间今天没有下一位客人入住（见今日预订）。", "by": "look_up",
                 "on_fail": "只能免费延到 13:00；也可以帮对方把行李免费寄存在前台。"},
                {"check": "until_18", "text": "要延到的时间不晚于 18:00。", "by": "ask",
                 "on_fail": "18:00 以后要按一晚房费收，可以建议对方续住一晚。"},
                {"any": [
                    {"check": "gold_member", "text": "住客是金卡或以上会员（见入住记录），延迟退房免费。", "by": "look_up"},
                    {"check": "pays_half_day", "text": "住客同意支付半天房费（该房一晚房价的一半）。", "by": "ask",
                     "on_fail": "不付费的话，只能免费延到 13:00。"}]}]}]},
        "then": ["复述新的退房时间和费用（如有），对方确认后告诉对方已经办好。"],
        "otherwise": "说明哪一条不满足，婉拒，并给出替代办法。"}}})

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
    "zh-parcel-li": dict(  # should be done
        agent_role="你是顺达快递的电话客服。",
        sop=PARCEL_SOP,
        sop_case={"action": "change_address", "facts": PARCELS,
                  "truth": {"parcel_verified": True, "not_out_for_delivery": True, "new_address_given": True, "same_city": True},
                  "expect": {"new_address": "杭州市西湖区文三路 90 号 3 幢 502 室"}},
        persona="你在公司，打电话给快递客服。", voice="vivian",
        profile={"name": "李娜", "gender": "female", "age": 31, "language": "zh", "occupation": "程序员",
                 "speaking_style": "干脆、语速偏快", "traits": "怕麻烦，喜欢一次说清楚"},
        goal=("你有个包裹寄到了公司，可明天周六公司没人，想改寄到家里。运单号 7731 0526 88，收件人手机尾号 3721，被问到再说。"
              "新地址：杭州市西湖区文三路 90 号 3 幢 502 室（和公司在同一个城市）。客服复述地址时如果说错了就纠正。改好后道谢、说再见。" + ZH_NOTE),
        first_turn="喂，你好，我想改一下快递的收件地址。"),
    "zh-parcel-zhou": dict(  # should be declined: already out for delivery; the alternative is the courier's number
        agent_role="你是顺达快递的电话客服。",
        sop=PARCEL_SOP,
        sop_case={"action": "change_address", "facts": PARCELS,
                  "truth": {"parcel_verified": True, "not_out_for_delivery": False, "new_address_given": True, "same_city": True}},
        persona="你走在马路边，打电话给快递客服。", voice="eric", background=None,  # the street's own ambience
        profile={"name": "周强", "gender": "male", "age": 45, "language": "zh", "occupation": "装修公司老板",
                 "speaking_style": "嗓门大、有点急", "traits": "急性子，被拒绝会再争取一次", "surroundings": "street"},
        goal=("你的包裹寄到了苏州市姑苏区人民路 512 号的店里，但你今天在工地，想改送到苏州市吴中区东吴北路 88 号工地项目部 2 楼办公室。"
              "运单号 7731 0529 03，收件人手机尾号 5580，被问到再说。如果客服说改不了，坚持争取一次；如果客服给你别的办法"
              "（比如派送员的电话），记下来，道谢、说再见。" + ZH_NOTE),
        first_turn="喂？我那个快递，能不能改个地址啊？"),
    "zh-hotel-sun": dict(  # should be done: not a member, agrees to pay half a day
        agent_role="你是云栖酒店的前台。",
        sop=HOTEL_SOP,
        sop_case={"action": "late_checkout", "facts": GUESTS,
                  "truth": {"guest_verified": True, "room_not_booked": True, "until_18": True, "gold_member": False, "pays_half_day": True},
                  "expect": {"checkout_time": "16:00", "fee": "半天房费, or 299 元"}},
        persona="你在酒店房间里，用房间电话打给前台。", voice="serena",
        profile={"name": "孙丽", "gender": "female", "age": 38, "language": "zh", "occupation": "销售经理",
                 "speaking_style": "客气、说话清楚", "traits": "出差中，下午要开线上会"},
        goal=("你住 1608 房，入住人孙丽。下午 2 点到 3 点半要在房间开线上会，想把退房延到下午 4 点。你不是金卡会员。"
              "如果要收半天房费，你可以接受。确认好时间和费用后，道谢、说再见。" + ZH_NOTE),
        first_turn="你好，前台吗？我想问一下能不能晚点退房。"),
    "zh-hotel-he": dict(  # should be declined: the room is booked tonight (a gold member, but that does not help)
        agent_role="你是云栖酒店的前台。",
        sop=HOTEL_SOP,
        sop_case={"action": "late_checkout", "facts": GUESTS,
                  "truth": {"guest_verified": True, "room_not_booked": False, "until_18": True, "gold_member": True, "pays_half_day": False}},
        persona="你在酒店房间里，用手机打给前台。", voice="dylan",
        profile={"name": "何建", "gender": "male", "age": 52, "language": "zh", "occupation": "工程师",
                 "speaking_style": "慢条斯理", "traits": "常住酒店，很看重会员待遇"},
        goal=("你住 2215 房，入住人何建，是酒店的金卡会员。想把退房延到下午 5 点。如果被拒绝，提一句自己是金卡会员，争取一次；"
              "如果还是不行，接受客服给的替代办法，道谢、说再见。你不想为延迟退房付钱。" + ZH_NOTE),
        first_turn="喂，你好，我是 2215 的客人，想延迟退房。"),
}


def make_task(name: str, sc: dict) -> Task:
    keep = ("persona", "profile", "first_turn", "turn_taking", "background", "voice", "agent_role", "sop", "sop_case")
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
