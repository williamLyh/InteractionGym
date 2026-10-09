"""Full-Duplex-Bench v3 port and the cascaded agent: no models, no data (synthetic recordings, fake clients)."""

import asyncio
import json
import math
from array import array

from interaction_gym import AgentSpec, Env, Task
from interaction_gym.agents.cascaded import CascadedAgent, Endpointing, LatencyModel, parse_tool_calls
from interaction_gym.audio import Audio
from interaction_gym.benchmarks import fdb3
from interaction_gym.clients import FakeChat, FakeSpeech
from interaction_gym.tools import RESULT, ToolCall, ToolWorld
from interaction_gym.traj import episode
from interaction_gym.user import LLMSource, ReplayUser, ResponseDelay, TurnTaking, UserSim, Voice

SR = 16000


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- mock APIs and scoring


def test_mock_apis_match_official_values():
    b = fdb3.MockAPIBackend()
    st = b.reset(Task(), None)

    def call(name, **args):
        return json.loads(run(b.call(st, "agent", ToolCall("c", name, args))).content)

    assert call("search_flights", destination="London", date="2026-08-20") == {
        "status": "success", "flights": [{"flight_id": "FL123", "destination": "London", "date": "2026-08-20", "price": 450.0}]}
    assert call("get_exchange_rate", amount="100", from_currency="EUR", to_currency="USD")["converted_amount"] == 110.00000000000001
    assert call("search_apartments", city="Austin", bedrooms="2", max_price="2000")["results"] == [{"id": "APT1", "price": 1900.0, "beds": 2}]
    assert call("search_products", query="mouse")["products"][0]["price"] == 99.99
    assert call("add_to_cart", product_id="PROD1") == {"status": "success", "product_id": "PROD1", "quantity": 1, "cart_total": 99.99}
    assert call("update_identity_doc", doc_type="passport", doc_number="X1234567")["masked_number"] == "4567"
    assert call("calculate_commute", origin_address="a", destination_address="b")["mode"] == "driving"
    assert len(st["calls"]) == 7
    bad = run(b.call(st, "agent", ToolCall("c", "cancel_order", {})))
    assert bad.error and "Unknown function" in bad.content
    assert {s.name for s in b.tools()} == set(fdb3.FUNCTIONS)


def test_pass_at_1_official_dry_run_cases():
    exp = [{"function": "search_flights", "args": {"destination": "London", "date": "August 20"}},
           {"function": "book_flight", "args": {"passenger_name": "Alice"}}]
    ok = [{"function": "search_flights", "args": {"destination": "London", "date": "August 20"}},
          {"function": "book_flight", "args": {"passenger_name": "alice"}}]
    assert run(fdb3.pass_at_1(exp, ok))["passed"]
    r = run(fdb3.pass_at_1(exp, ok[:1]))
    assert not r["passed"] and "Missing tools" in r["failure_reason"]
    assert not run(fdb3.pass_at_1(exp, []))["passed"]
    r = run(fdb3.pass_at_1(exp, ok + [{"function": "cancel_flight", "args": {}}]))
    assert not r["passed"] and "Unexpected tools" in r["failure_reason"]
    r = run(fdb3.pass_at_1(exp, [{"function": "search_flights", "args": {"destination": "Paris", "date": "August 20"}}, ok[1]]))
    assert not r["passed"] and "Wrong arguments" in r["failure_reason"]
    # dynamic references are not checked by exact match; an LLM judge decides when given
    ref = [{"function": "add_to_cart", "args": {"product_id": "$RESULT_0.products[0].product_id", "quantity": 1}}]
    assert run(fdb3.pass_at_1(ref, [{"function": "add_to_cart", "args": {"product_id": "PROD1", "quantity": 1}}]))["passed"]
    judge = FakeChat(['```json\n{"correct": false, "explanation": "no"}\n```'])
    assert not run(fdb3.pass_at_1(ref, [{"function": "add_to_cart", "args": {"product_id": "PROD1", "quantity": 1}}], judge))["passed"]


def test_effective_calls_keep_corrections_and_extras():
    exp = [{"function": "search_products"}, {"function": "add_to_cart"}]
    calls = [{"function": "search_products", "args": {"query": "running shoes"}}, {"function": "search_products", "args": {"query": "hiking boots"}},
             {"function": "add_to_cart", "args": {}}, {"function": "track_order", "args": {}}]
    eff = fdb3.effective_calls(calls, exp)
    assert [c["function"] for c in eff] == ["search_products", "add_to_cart", "track_order"]
    assert eff[0]["args"]["query"] == "hiking boots"


def test_user_card_has_corrected_value_only():
    meta = {"id": "x", "dialogue": [{"user": "search for running shoes, actually no, hiking boots", "ai": "Searching hiking boots."}],
            "expected_tool_calls": [{"function": "search_products", "args": {"query": "hiking boots"}},
                                    {"function": "add_to_cart", "args": {"product_id": "$RESULT_0.products[0].product_id"}}],
            "state_rollback_details": {"original_param": {"query": "running shoes"}, "corrected_param": {"query": "hiking boots"}},
            "acting_notes": "Quickly correct the search item."}
    card = fdb3.user_card(meta)
    assert card["facts"] == {"search_products.query": "hiking boots"}
    assert "result of step 1" in card["goal"] and "corrected yourself" in card["goal"]
    assert "running shoes" not in card["goal"].split("What you want done")[1]
    assert "Quickly correct" in card["persona"]


def test_parse_tool_calls_both_formats():
    text = ("Sure.\n<tool_call>\n<function=search_apartments>\n<parameter=city>\nAustin\n</parameter>\n<parameter=bedrooms>\n2\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>{\"name\": \"track_order\", \"arguments\": {\"order_id\": \"ABC123\"}}</tool_call>")
    spoken, calls = parse_tool_calls(text)
    assert spoken == "Sure."
    assert calls == [{"name": "search_apartments", "arguments": {"city": "Austin", "bedrooms": 2}},
                     {"name": "track_order", "arguments": {"order_id": "ABC123"}}]
    assert parse_tool_calls("<think>hmm</think>Hello!") == ("Hello!", [])


# ---------------------------------------------------------------- synthetic recordings


def tone(ms, amp=4000, f=220):
    n = round(ms * SR / 1000)
    return Audio(array("h", (int(amp * math.sin(2 * math.pi * f * i / SR)) for i in range(n))), SR)


def silence(ms):
    return Audio.silence(ms, SR)


def test_request_end_rule():
    a = tone(1000) + silence(500) + tone(1000) + silence(2500) + tone(300) + silence(1000)
    assert abs(fdb3.request_end_ms(a) - 2500) <= 20
    assert abs(fdb3.request_end_ms(tone(800) + silence(1000)) - 800) <= 20


# ---------------------------------------------------------------- cascaded agent in the env


class FakeASR:
    model = "fake-asr"

    def __init__(self):
        self.calls = []

    async def transcribe(self, audio, language=None):
        self.calls.append(audio.dur_ms)
        return "track my order " + ("A B C one two three" if audio.dur_ms > 2000 else "um")


class FakeToolLLM:
    """Round 1: a tool call (with a filler); round 2: the answer; for a user turn after a tool result: 'You're welcome.'"""

    model = "fake-llm"
    defaults: dict = {}

    def __init__(self):
        self.calls = []

    async def complete(self, messages, tools, **kw):
        self.calls.append(messages)
        last = messages[-1]
        if last["role"] == "tool":
            return {"content": "Your order is out for delivery.", "tool_calls": [], "completion_tokens": 8, "wall_ms": 1}
        if "ABC" in last["content"] or "A B C" in last["content"]:
            return {"content": "", "tool_calls": [{"name": "track_order", "arguments": {"order_id": "ABC123"}}], "completion_tokens": 20, "wall_ms": 1}
        return {"content": "Sorry, could you repeat that?", "tool_calls": [], "completion_tokens": 7, "wall_ms": 1}

    def describe(self):
        return {"model": self.model}


SPEC = AgentSpec(chunk_ms=100, obs=("user.speech", RESULT["agent"]), audio="user.audio", sr=SR)


def make_task(audio):
    meta = {"id": "ecommerce_01", "domain": "ecommerce_support", "title": "t", "difficulty": "easy",
            "dialogue": [{"user": "um... track my order ABC123", "ai": "I will check order ABC123."}], "acting_notes": "",
            "disfluency_features": ["PAUSE"], "expected_tool_calls": [{"function": "track_order", "args": {"order_id": "ABC123"}}],
            "state_rollback_test": False, "latency_profile": "normal"}
    rec = fdb3.Recording("ecommerce_01_spk", meta, audio, fdb3.request_end_ms(audio), audio.dur_ms)
    return fdb3.task_of(rec)


async def run_episode(task, user, cache, seed=0):
    env = Env({"user": user, "tools": ToolWorld(fdb3.MockAPIBackend(), fdb3.latency_ms(task))}, SPEC, max_ms=60_000, end_idle_ms=3000)
    agent = CascadedAgent(SPEC, FakeASR(), FakeToolLLM(), FakeSpeech(sr=SR), instructions=fdb3.AGENT_INSTRUCTIONS, seed=seed, cache=cache)
    obs = await env.reset(task, seed)
    while True:
        obs, _, done = await env.step(await agent.act(env.t, obs))
        if env.truncated or (done and not agent.busy):
            break
    return env, agent


def test_cascaded_agent_open_loop_episode():
    audio = tone(2500) + silence(300)
    task = make_task(audio)
    env, agent = run(run_episode(task, ReplayUser(), {}))
    ep = episode(env, "a")
    calls = fdb3.agent_calls(ep)
    assert [c["function"] for c in calls] == ["track_order"]
    lm, ept = LatencyModel(), Endpointing()
    user_end = task.scenario["benchmark"]["user_end_ms"]
    endpoint = [e for e in agent.events if e["kind"] == "endpoint"][0]
    assert endpoint["t"] == endpoint["speech_end"] + ept.silence_ms
    # call at endpoint + ASR + LLM(20 tokens); speech after the tool result (500 ms latency) + LLM + TTS
    t_call = endpoint["t"] + lm.asr(endpoint["audio_ms"]) + lm.llm(20)
    assert calls[0]["time"] == t_call
    speech = fdb3.agent_speech(ep)
    assert len(speech) == 1 and speech[0]["text"] == "Your order is out for delivery."
    assert speech[0]["start_time"] == t_call + 500 + lm.llm(8) + lm.tts()
    tm = fdb3.timing(ep, user_end)
    assert tm["turn_taken"] and not tm["interrupted"] and tm["first_response_ms"] > ept.silence_ms
    assert ep["meta"]["end_reason"] == "idle"
    assert run(fdb3.pass_at_1(task.criteria["expected_tool_calls"], calls))["passed"]


def test_cascaded_agent_cuts_in_on_a_long_pause_and_yields():
    # "um" (300 ms), a 1.5 s pause (> 800 ms endpointing), then the request: the agent answers the "um" while the
    # user is still going, and stops when talked over
    audio = tone(400) + silence(1500) + tone(4000) + silence(300)
    task = make_task(audio)
    env, agent = run(run_episode(task, ReplayUser(), {}))
    ep = episode(env, "a")
    eps = [e for e in agent.events if e["kind"] == "endpoint"]
    assert len(eps) == 2 and eps[0]["speech_end"] < task.scenario["benchmark"]["user_end_ms"]
    speech = fdb3.agent_speech(ep)
    assert speech[0]["start_time"] < 1900 + 4000  # replied to the fragment while the user was still going
    assert "unsaid" in speech[0] and ("Sorry, could you repeat that?").startswith(speech[0]["text"])  # and yielded
    assert speech[0]["end_time"] - 1900 <= Endpointing().barge_in_ms + SPEC.chunk_ms + 100
    tm = fdb3.timing(ep, task.scenario["benchmark"]["user_end_ms"])
    assert tm["interrupted"]
    assert [c["function"] for c in fdb3.agent_calls(ep)] == ["track_order"]


def test_open_and_closed_loop_identical_until_the_users_second_turn():
    audio = tone(2500) + silence(300)
    task = make_task(audio)
    cache: dict = {}
    env_a, _ = run(run_episode(task, ReplayUser(), cache, seed=3))
    user_llm = FakeChat(["Great, thanks! ###STOP###"])
    user_b = UserSim(LLMSource(user_llm), voice=Voice(FakeSpeech(sr=SR)),
                     timing=TurnTaking(response_delay=ResponseDelay(), yield_after_ms=None, nudge_after_ms=10_000))
    env_b, _ = run(run_episode(task, user_b, cache, seed=3))
    a, b = episode(env_a, "a"), episode(env_b, "b")
    users = fdb3.user_turns(b)
    assert len(users) == 2 and users[1]["text"] == "Great, thanks!"
    u1 = users[1]["start_time"]
    sig = lambda ep, cut: ([(t["start_time"], t["end_time"], t["text"]) for t in fdb3.agent_speech(ep, cut)],  # noqa: E731
                           [(c["function"], c["args"], c["time"]) for c in fdb3.agent_calls(ep, cut)])
    assert sig(a, u1) == sig(b, u1)
    ue = task.scenario["benchmark"]["user_end_ms"]
    assert fdb3.timing(a, ue, u1) == fdb3.timing(b, ue, u1)
    assert user_llm.calls and "track_order" not in json.dumps(user_llm.calls[0][1:])  # the user hears speech, not tool calls


def test_reply_cancelled_before_its_tool_call_went_out_does_not_hang():
    # the first part already names the order; the user goes on after a 900 ms pause, before the call is sent
    audio = tone(2500) + silence(900) + tone(2500) + silence(300)
    task = make_task(audio)
    env, agent = run(run_episode(task, ReplayUser(), {}))
    ep = episode(env, "a")
    assert any(e["kind"] == "cancel" and "call" in e["dropped"] for e in agent.events)
    assert ep["meta"]["end_reason"] == "idle" and not agent.busy
    assert [c["function"] for c in fdb3.agent_calls(ep)] == ["track_order"]
    assert fdb3.agent_speech(ep)[-1]["text"] == "Your order is out for delivery."


def test_speakable_strips_markdown_and_emoji():
    from interaction_gym.agents.cascaded import speakable

    assert speakable("Here:\n\n- **Flight** FL123 ✅\n- `B789`") == "Here: Flight FL123 B789"


def test_user_source_is_told_about_silence():
    audio = tone(1000) + silence(300)
    task = make_task(audio)
    llm = FakeChat(["Hello? Are you there?"])
    src = fdb3.UserSource(llm, max_turns=3)
    from interaction_gym.user import Utterance

    first = run(src.next(0, task, []))
    assert first.audio is audio and first.text == task.scenario["first_turn"]["text"]
    turn = run(src.next(1, task, [Utterance("user", first.text, 0, False)]))
    assert turn.text == "Hello? Are you there?"
    msgs = llm.calls[0]
    assert msgs[-1]["role"] == "user" and "not said anything" in msgs[-1]["content"]
    assert run(src.next(3, task, [])) is None


# ---------------------------------------------------------------- spoken fulfilment (agents without tools)


def _ep(turns, expected, rollback=None, script="", sid="x"):
    bench = {"state_rollback_details": rollback, "script": script, "id": sid}
    return {"meta": {"task": {"scenario": {"benchmark": bench}, "criteria": {"expected_tool_calls": expected}}},
            "turns": [{"role": r, "text": t, "start_time": s, "end_time": s + 1000} for r, t, s in turns]}


def test_spoken_slots_final_values_stated_only():
    from interaction_gym.benchmarks import fdb3_spoken as sp

    exp = [{"function": "search_apartments", "args": {"city": "Austin", "bedrooms": 2, "max_price": 1600}},
           {"function": "calculate_commute", "args": {"origin_address": "$RESULT_0.apartments[0].address", "destination_address": "the gym",
                                                      "mode": "driving"}},
           {"function": "update_search_filter", "args": {"filter_name": "max_price", "value": 1600}}]
    rb = {"original_param": {"bedrooms": 1, "max_price": 1200}, "corrected_param": {"bedrooms": 2, "max_price": 1600}}
    script = "I was thinking a 1-bedroom under 1200... no wait, we need a 2-bedroom, up to 1600 a month, and the commute to the gym."
    slots = sp.task_slots(exp, rb, script, "housing_11")
    keys = [s["key"] for s in slots]
    # $RESULT references and values the caller never said (Austin, the default mode) are not slots; 1600 twice is one slot
    assert keys == ["search_apartments.bedrooms", "search_apartments.max_price", "calculate_commute.destination_address",
                    "update_search_filter.filter_name"]
    by = {s["key"]: s for s in slots}
    assert by["search_apartments.bedrooms"]["superseded"] == [1] and by["search_apartments.max_price"]["superseded"] == [1200]
    assert len(sp.task_slots(exp, rb)) == 6  # without the script: every distinct literal value


def test_spoken_mentions_rules():
    from interaction_gym.benchmarks import fdb3_spoken as sp

    amount = {"type": "amount"}
    assert sp.mentions(amount, 150, "convert 1 5 0 euros") and sp.mentions(amount, 150, "1 50 euros")
    assert sp.mentions(amount, 1500, "up to fifteen hundred a month") and sp.mentions(amount, 1500, "$1,500")
    assert not sp.mentions(amount, 150, "100 euros")
    date = {"type": "monthday"}
    assert sp.mentions(date, "October 7", "Miami on October 7 th") and sp.mentions(date, "October 7", "the seventh of October")
    assert not sp.mentions(date, "October 7", "October 6th")
    code = {"type": "code"}
    assert sp.mentions(code, "BOB12", "order B O B one two") and sp.mentions(code, "K2", "product K-2")
    assert not sp.mentions(code, "K2", "I'll book 2 seats")
    cur = sp.task_slots([{"function": "get_exchange_rate", "args": {"amount": 50, "from_currency": "CAD", "to_currency": "USD"}}])
    assert sp.mentions(cur[1], "CAD", "fifty Canadian dollars") and not sp.mentions(cur[2], "USD", "fifty Canadian dollars")
    count = sp.task_slots([{"function": "search_products", "args": {"query": "tablet"}},
                           {"function": "add_to_cart", "args": {"product_id": "$RESULT_0.products[0].product_id", "quantity": 2}}])[1]
    assert sp.mentions(count, 2, "adding two tablets to your cart") and not sp.mentions(count, 2, "you have 2 orders pending; adding one tablet")


def test_spoken_outcome_last_mention_and_fallback():
    from interaction_gym.benchmarks import fdb3_spoken as sp

    exp = [{"function": "search_flights", "args": {"destination": "Miami", "date": "October 7"}},
           {"function": "book_flight", "args": {"passenger_name": "Casey Lee"}}]
    rb = {"original_param": {"date": "October 5"}, "corrected_param": {"date": "October 7"}}
    script = "Flights to Miami on October 5th, wait, October 7th, for Casey Lee."
    turns = [("user", script, 0), ("agent", "Flights to Miami on October 5th for Casey, right?", 9000),
             ("user", "No, the 7th.", 12000), ("agent", "Sorry, October 7th, not the 5th.", 15000)]
    ep = _ep(turns, exp, rb, script)
    judge = FakeChat(['{"book_flight.passenger_name": "confirmed"}'])
    first = run(sp.outcome(ep, judge, before_ms=12000))
    st = {s["key"]: (s["status"], s["source"]) for s in first["slots"]}
    assert st == {"search_flights.destination": ("correct", "rule"), "search_flights.date": ("wrong", "rule"),
                  "book_flight.passenger_name": ("correct", "llm")}
    assert not first["fulfilled"] and first["outcome"] == round(2 / 3, 4)
    assert "Casey Lee" in judge.calls[0][0]["content"] and "No, the 7th" not in judge.calls[0][0]["content"]
    final = run(sp.outcome(ep, FakeChat(['{"book_flight.passenger_name": "not_confirmed"}'])))
    st = {s["key"]: s["status"] for s in final["slots"]}
    assert st["search_flights.date"] == "correct" and st["book_flight.passenger_name"] == "missing"
    silent = run(sp.outcome(_ep([("user", script, 0)], exp, rb, script), judge))
    assert silent["outcome"] == 0 and all(s["status"] == "missing" for s in silent["slots"])
