"""What an online user does besides its turns: model-decided backchannels / barge-ins at the agent's phrase
boundaries, LLM-placed mid-thought pauses, random (non-subjective) events, the background track — every one a
labelled user turn."""

import asyncio
import dataclasses
import math
import random
import re

import pytest

from interaction_gym import AgentSpec, Env, Segment, Task
from interaction_gym.clients import FakeChat, FakeSpeech
from interaction_gym.core import Frame
from interaction_gym.eval import scores
from interaction_gym.soundscape import (DEFAULTS, EVENT_RATES, EVENT_TYPES, JITTER_DB, QUIET_FLOOR_DBFS, SPEECH_DBFS,
                                              EventProcess, Soundscape, SoundBank, active_rms, background_spec, colored_noise,
                                              event_rates, mean_burst, render_background, synth_event)
from interaction_gym.traj import episode
from interaction_gym.user import (SITUATIONS, AsideWriter, ASIDES, BACKCHANNELING, BACKCHANNELS, HESITANCY, PERSONA, SURROUNDINGS, Behaviors,
                                        BARGE_IN_NOTE, KeywordInterrupt, LLMInterrupt, LLMListener, LLMSource, ScriptSource, TurnTaking, UserSim,
                                        Utterance, Voice, _pause, _spoken, n_words, parse_decision, persona_levels, phrase_boundaries,
                                        parse_intent, pieces)
from tests.test_traj import check

AGENT = ("Sure, I can help with that. First, I need your booking reference, which is on the email we sent. "
         "Then I will check the availability for Saturday, and after that I can confirm the new time for you.")


def bc_at_sentence_end(messages):
    """A fake listener: a backchannel right after a sentence ends, else listen."""
    p = messages[-1]["content"]
    heard = re.findall(r"AGENT: (.*) \[CURRENTLY SPEAKING, INCOMPLETE\]", p)
    if "BACKCHANNEL" in p.split("The options:")[1] and heard and heard[-1].rstrip().endswith("."):
        return "BACKCHANNEL: mm-hmm"
    return "LISTEN"


def run(replies, listener=bc_at_sentence_end, interrupt="llm", behaviors=Behaviors(), seed=0, agent_text=AGENT, agent_ms=12_000,
        speech=None, scenario=None, soundscape=None, spec=None, max_ms=60_000, reply_after=500):
    """One episode: the user's first turn, one long agent turn, then whatever follows."""
    async def main():
        sp = spec or AgentSpec(chunk_ms=100)
        lst = LLMListener(FakeChat(listener)) if listener is not None else None
        kw = {"interrupt": lst} if interrupt == "llm" else {"interrupt": interrupt, "listener": lst}
        user = UserSim(LLMSource(FakeChat(replies)), Voice(speech) if speech else Voice(), behaviors=behaviors, soundscape=soundscape, **kw)
        env = Env({"user": user}, sp, max_ms=max_ms)
        await env.reset(Task(id="t", scenario={"instructions": "Move my booking to Saturday.", **(scenario or {})}), seed=seed)
        n = 0
        while not env.done:
            act = []
            users = [t for t in episode(env, "x")["turns"] if t["role"] == "user" and t.get("kind") is None]
            if n == 0 and users and env.t >= users[-1]["end_time"] + reply_after:
                act, n = [Frame(sp.out, env.t, Segment("a0", env.t, agent_ms, agent_text))], 1
            await env.step(act)
        return env, check(episode(env, "e"))
    return asyncio.run(main())


def user_turns(ep, kind="any"):
    return [t for t in ep["turns"] if t["role"] == "user" and (kind == "any" or t.get("kind") == kind)]


# ---------------------------------------------------------------- decision points


def test_decision_points_are_the_agents_phrase_boundaries():
    seg = Segment("a", 1000, 2000, "Okay, so. Then 3.5 more!")
    b = phrase_boundaries(seg)
    assert len(b) == 2  # after "Okay," and "so." — not inside "3.5", not the final "!"
    for t in b:
        assert re.search(r"[,.]$", seg.heard_text(t))  # at each, the listener has just heard the mark
    assert phrase_boundaries(Segment("z", 0, 1000, "你好，我想预约。好的")) == [math.ceil(1000 * 3 / 10), math.ceil(1000 * 8 / 10)]


def test_listener_decides_at_boundaries_with_a_fallback_and_min_words():
    seen = []

    def rec(messages):
        p = messages[-1]["content"]
        seen.append(("end of a phrase" in p, re.findall(r"AGENT: (.*) \[CURRENTLY", p)[-1]))
        return "LISTEN"

    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=rec)
    assert seen and all(not boundary or heard.rstrip()[-1] in ",." for boundary, heard in seen)
    assert all(n_words(h) >= 5 for _, h in seen)  # never asked before 5 words were heard
    d = ep["meta"]["user"]["decisions"]
    assert d["llm_calls"] == len(seen) == d["LISTEN"] and d["boundary"] >= 4 and d["too_few_words"] >= 1  # "Sure," came too early
    # an agent that never pauses: decisions every max_decision_gap_ms
    seen.clear()
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=rec, agent_text="blah " * 40, agent_ms=12_000)
    d = ep["meta"]["user"]["decisions"]
    assert d["boundary"] == 0 and d["fallback"] == 12_000 // TurnTaking().max_decision_gap_ms - 1 + (12_000 % 3000 > 0)
    assert all(not b for b, _ in seen)


def test_backchannels_come_from_the_model_at_phrase_ends_and_are_labelled():
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"])
    bcs = user_turns(ep, "backchannel")
    a0 = next(t for t in ep["turns"] if t["id"] == "a0")
    assert bcs and bcs[0]["text"] == "mm-hmm" and all(t["expects"] == "ignore" for t in bcs)  # the model's sound,
    assert all(a["text"] != b["text"] and b["text"] in BACKCHANNELS["en"] for a, b in zip(bcs, bcs[1:]))  # never twice in a row
    seg = Segment("a0", a0["start_time"], a0["end_time"] - a0["start_time"], AGENT)
    assert all(seg.heard_text(t["start_time"]).endswith(".") for t in bcs)  # where the model put them
    ev = [e for e in ep["eval"]["scores"]["events"] if e["turn"] in {t["id"] for t in bcs}]
    assert ev and {e["expects"] for e in ev} == {"ignore"}
    assert ep["meta"]["user"]["decisions"]["BACKCHANNEL"] == len(bcs)


def test_safety_caps_bound_a_runaway_listener():
    # a model that always wants to backchannel: min_gap_ms and the per-turn cap hold it
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=lambda m: "BACKCHANNEL: yeah",
                  agent_text="One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve, thirteen, fourteen. " * 2,
                  agent_ms=20_000)
    bcs = user_turns(ep, "backchannel")
    starts = [t["start_time"] for t in bcs]
    assert len(bcs) == Behaviors().max_backchannels_per_turn and all(b - a >= Behaviors().min_gap_ms for a, b in zip(starts, starts[1:]))
    assert ep["meta"]["user"]["decisions"]["capped"] > 0


def test_options_follow_the_persona_and_the_barge_in_setting():
    prompts = []

    def rec(messages):
        prompts.append(messages[-1]["content"])
        return "LISTEN"

    run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=rec, scenario={"profile": {"name": "Ann", "backchanneling": "rare"}})
    p = prompts[0]
    assert BACKCHANNELING["rare"] in p and "Ann" in p and "Move my booking to Saturday." in p and '"mm-hmm"' in p
    assert "- INTERRUPT" in p and "Do NOT repeatedly interrupt" in p  # τ-Voice's considerations
    prompts.clear()  # never barges in: the listener only decides backchannels
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=rec, interrupt=None)
    assert prompts and all("- INTERRUPT" not in p and "- BACKCHANNEL" in p for p in prompts)
    assert ep["meta"]["user"]["barge_in"] == {"type": "never"} and ep["meta"]["user"]["listening"]["options"] == ["LISTEN", "BACKCHANNEL"]
    prompts.clear()  # a silent listener who never barges in: no model calls at all
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=rec, interrupt=None, scenario={"profile": {"backchanneling": "none"}})
    assert not prompts and ep["meta"]["user"]["decisions"]["llm_calls"] == 0
    prompts.clear()  # Chinese: the language's sounds are offered
    run(["你好，我想改一下预约。", "再见 ###STOP###"], listener=rec, scenario={"profile": {"language": "zh"}})
    assert '"嗯"' in prompts[0]


def test_barge_in_intent_is_chosen_by_the_user_llm_writing_the_line():
    def decide(messages):  # the listener only decides to cut in; anything after INTERRUPT is ignored
        p = messages[-1]["content"]
        assert "INTERRUPT: <" not in p and "correction | question" not in p
        return "INTERRUPT: stop" if "Saturday" in p.split("<conversation_history>")[1] else "LISTEN"

    def user(messages):
        last = messages[-1]["content"]
        if BARGE_IN_NOTE in last:  # the barge-in line: told it is cutting in, it picks why while writing it
            return "INTENT: correction\nNo, Sunday, sorry."
        return "Bye ###STOP###" if any(m["content"].startswith("No, Sunday") for m in messages) else "Hi, I need to move a booking."

    env, ep = run(user, listener=decide)
    barge = next(t for t in user_turns(ep) if t["text"] == "No, Sunday, sorry.")
    a0 = next(t for t in ep["turns"] if t["id"] == "a0")
    assert a0["start_time"] < barge["start_time"] < a0["end_time"]
    assert barge["expects"] == "yield" and barge["intent"] == "correction"  # the user LLM's, not the listener's "stop"
    assert {e["expects"] for e in ep["eval"]["scores"]["events"] if e["turn"] == barge["id"]} == {"yield"}  # one event per turn
    calls = env.nodes["user"].source.llm.calls
    barges = [t for t in user_turns(ep) if t.get("expects") == "yield"]
    assert all(t["intent"] == "correction" for t in barges)
    assert sum(BARGE_IN_NOTE in c[-1]["content"] for c in calls) == len(barges)  # one generation call each, none extra for the intent


def test_parse_intent():
    assert parse_intent("INTENT: question\nWhat time is it?") == ("question", "What time is it?")
    assert parse_intent("**Intent:** stop\nOkay, that's enough.") == ("stop", "Okay, that's enough.")
    assert parse_intent("INTENT: question What time?") == ("question", "What time?")
    assert parse_intent("intent: enough\nStop, please.")[0] == "stop"
    assert parse_intent("INTENT: rant\nUgh.") == ("other", "Ugh.")
    assert parse_intent("No, Sunday.") == ("other", "No, Sunday.")  # no tag: other, the text untouched


def test_rule_interrupts_still_work_and_never_means_never():
    env, ep = run(["Hi, I need to move a booking.", "Wait, what reference?", "Bye ###STOP###"], listener=None,
                  interrupt=KeywordInterrupt("reference"))
    barge = next(t for t in user_turns(ep) if t["text"] == "Wait, what reference?")
    assert barge["expects"] == "yield" and ep["meta"]["user"]["decisions"]["rule_interrupts"] >= 1
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None)
    assert not user_turns(ep, "backchannel") and ep["meta"]["user"]["decisions"]["points"] == 0


def test_parse_decision():
    opts = ("LISTEN", "BACKCHANNEL", "INTERRUPT")
    assert parse_decision("BACKCHANNEL: mm-hmm", opts).token == "mm-hmm"
    assert parse_decision("backchannel - \"Okay.\"", opts).token == "Okay"
    assert parse_decision("BACKCHANNEL: I see what you are saying there", opts).token is None  # too long: the vocabulary's
    assert parse_decision("INTERRUPT: Correction (wrong day)", opts).intent is None  # the user LLM picks the intent
    assert parse_decision("INTERRUPT", opts).choice == "INTERRUPT"
    assert parse_decision("YES", opts).choice == "INTERRUPT" and parse_decision("NO", opts).choice == "LISTEN"  # τ-Voice's answers
    assert parse_decision("INTERRUPT: question", ("LISTEN", "BACKCHANNEL")).choice == "LISTEN"  # not offered
    assert parse_decision("I would keep quiet", opts).choice == "LISTEN" and parse_decision("", opts).choice == "LISTEN"


def test_decisions_are_greedy_memoised_and_reproducible():
    llm = FakeChat(["BACKCHANNEL: yeah", "INTERRUPT: stop"])
    lst = LLMListener(llm)
    task = Task(scenario={"instructions": "x"})
    convo = [Utterance("agent", "Here is a long explanation of it.", 0, True)]
    a, b = (asyncio.run(lst.decide(task, convo)) for _ in range(2))
    assert a == b and a.choice == "BACKCHANNEL" and len(llm.calls) == lst.calls == 1  # the same prompt is asked once
    assert lst.params == {"temperature": 0.0, "max_tokens": 16}
    assert asyncio.run(lst(task, convo)) is True  # τ-Voice use: barge in now? (a new prompt: LISTEN / INTERRUPT)
    assert LLMInterrupt is LLMListener
    e1, e2 = run(["Hi, I need to move a booking.", "Bye ###STOP###"], seed=4)[1], run(["Hi, I need to move a booking.", "Bye ###STOP###"], seed=4)[1]
    assert e1["turns"] == e2["turns"]


# ---------------------------------------------------------------- mid-thought pauses


def test_pause_markers_split_a_turn():
    assert pieces("I'd like, (pause) hmm, Saturday.") == [("I'd like...", "short"), ("hmm, Saturday.", None)]
    assert pieces("(pause) Well (long pause) okay (pause)") == [("Well...", "long"), ("okay", None)]
    assert pieces("a (pause) b (pause) c (pause) d", max_pauses=2) == [("a...", "short"), ("b...", "short"), ("c d", None)]
    assert pieces("我想订，（停顿）周六吧。") == [("我想订...", "short"), ("周六吧。", None)]
    assert _spoken("It's for, (long pause) six people.", keep_pauses=True) == "It's for, (long pause) six people."
    assert _spoken("It's for, (pause) six people.") == "It's for, six people."
    assert _pause("（停顿）嗯，我想想") == "short" and _pause("(long pause) 稍等") == "long" and _pause("（长时间停顿）好") == "long"
    assert _spoken("（停顿）嗯，我想想") == "嗯，我想想" and _spoken("好的（停顿）是周三") == "好的是周三"


def test_llm_placed_pause_becomes_turns_with_expects():
    b = dataclasses.replace(Behaviors(), pause_ms=(1500, 1500))
    env, ep = run(["I need to move, (pause) let me see, my Saturday booking.", "Bye ###STOP###"], listener=None, interrupt=None,
                  behaviors=b, agent_ms=1000, agent_text="Sure.", reply_after=2000)
    first, rest = user_turns(ep)[:2]
    assert first["text"] == "I need to move..." and first["expects"] == "wait"
    assert rest["text"] == "let me see, my Saturday booking." and "expects" not in rest
    assert rest["start_time"] - first["end_time"] == 1500
    exp = {e["turn"]: e["expects"] for e in ep["eval"]["scores"]["events"] if e["expects"] in ("wait", "respond")}
    assert exp[first["id"]] == "wait" and exp[rest["id"]] == "respond"
    llm_msgs = env.nodes["user"].source.llm.calls[-1]  # the user LLM later sees it as one line, without markers
    assert any(m["content"] == "I need to move... let me see, my Saturday booking." for m in llm_msgs)


def test_hesitancy_reaches_the_user_prompt():
    src = LLMSource(FakeChat(["x"]))
    for level in ("low", "high"):
        msgs = src.messages(Task(scenario={"profile": {"hesitancy": level}}), [])
        assert HESITANCY[level] in msgs[0]["content"] and "(pause)" in msgs[0]["content"]
    assert HESITANCY["high"] in src.messages(Task(scenario={"profile": {"traits": "hesitates, pauses mid-sentence"}}), [])[0]["content"]


def test_deprecated_random_pause_still_works():
    with pytest.warns(DeprecationWarning):
        b = Behaviors(pause_p=1.0, pause_ms=(1500, 1500))
    env, ep = run(["I would like to book a table for two people tonight", "Bye ###STOP###"], listener=None, interrupt=None, behaviors=b,
                  agent_ms=1000, agent_text="Sure.")
    first, rest = user_turns(ep)[:2]
    assert first["text"].endswith("...") and first["expects"] == "wait" and rest["start_time"] - first["end_time"] == 1500
    assert Behaviors().pause_p == 0.0 and PERSONA.pause_p == 0.0  # off by default


# ---------------------------------------------------------------- random events


def test_random_events_by_surroundings_are_turns_with_labels():
    b = dataclasses.replace(PERSONA, max_asides_per_turn=None)
    for place in ("quiet", "street", "home"):
        n = {"noise": 0, "aside": 0}
        for seed in range(4):
            env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, seed=seed,
                          behaviors=dataclasses.replace(b, noise_per_min=12.0, aside_per_min=None), agent_ms=40_000,
                          scenario={"profile": {"surroundings": place}})
            for t in user_turns(ep):
                if t.get("kind") == "noise":
                    n["noise"] += 1
                    assert t["text"] == "" and t["expects"] == "ignore" and t["label"] in EVENT_RATES[place]  # the place's sounds
                elif t.get("kind") == "aside":
                    n["aside"] += 1
                    assert t["expects"] == "ignore" and t["text"] in ASIDES["en"][place]  # the fake LLM writes no usable line
        assert n["noise"] > 0 and (place != "quiet" or n["aside"] == 0)
    assert SURROUNDINGS["quiet"][0] == 0
    assert {k: v[1] for k, v in SURROUNDINGS.items()} == {k: round(sum(v.values()), 4) for k, v in EVENT_RATES.items()}
    assert SURROUNDINGS["street"][1] > SURROUNDINGS["home"][1] > SURROUNDINGS["quiet"][1]


def test_event_process_rates_bursts_and_determinism():
    rates = {"horn": 0.6, "dog_bark": 0.3, "door_slam": 0.1}
    onsets, labels, sizes = 0, {}, {}
    seeds, minutes = 3000, 2
    for seed in range(seeds):
        p = EventProcess(rates, seed)
        while (ev := p.peek()).t < minutes * 60000:
            p.pop()
            onsets += 1
            labels[ev.label] = labels.get(ev.label, 0) + 1
            sizes.setdefault(ev.label, []).append(ev.n)
            lo, hi = EVENT_TYPES[ev.label][4]
            assert len(ev.gaps_ms) == ev.n - 1 and all(lo <= g <= hi for g in ev.gaps_ms)
            assert abs(ev.level_db - EVENT_TYPES[ev.label][1]) <= JITTER_DB + 1e-9
    assert abs(onsets / (seeds * minutes) - 1.0) < 0.05  # the empirical rate is the configured one
    for lab, r in rates.items():
        assert abs(labels[lab] / (seeds * minutes) - r) < 0.05 * r + 0.01
        _, _, bp, bmax, _ = EVENT_TYPES[lab]
        assert max(sizes[lab]) <= bmax and abs(sum(sizes[lab]) / len(sizes[lab]) - mean_burst(bp, bmax)) < 0.08
    assert set(sizes["door_slam"]) == {1} and max(sizes["dog_bark"]) > 2  # bursty sources cluster, others do not
    a, b = EventProcess(rates, 7), EventProcess(rates, 7)
    assert [a.pop() for _ in range(20)] == [b.pop() for _ in range(20)] != [EventProcess(rates, 8).pop() for _ in range(20)]
    assert event_rates("street", 1.6) == pytest.approx({k: 2 * v for k, v in EVENT_RATES["street"].items()})
    assert event_rates("street", rates={"siren": 1}) == {"siren": 1.0} and EventProcess({}, 0).peek() is None


def test_a_burst_is_one_clip_repeated_up_to_a_length_cap(tmp_path):
    from interaction_gym.soundscape import MAX_BURST_S, NoiseEvent, event_audio

    (tmp_path / "events" / "horn").mkdir(parents=True)
    colored_noise("white", 16000, -20.0, seconds=2).write_wav(tmp_path / "events" / "horn" / "h.wav")
    ev = NoiseEvent(0, "horn", 4, (500, 500, 500), -10.0)
    audio, source, n = event_audio(ev, 16000, random.Random(0), SoundBank(tmp_path))
    assert source.startswith("file:") and n == 2 and audio.dur_ms == 4500 <= MAX_BURST_S["horn"] * 1000
    assert abs(20 * math.log10(active_rms(audio.samples, 16000) / 32768) - (SPEECH_DBFS - 10.0)) < 1.0


def test_noise_overlaps_speech_and_is_labelled_apart():
    # a coughing user: noises fire whatever is going on, even mid-sentence, as separate turns that expect to be ignored
    b = Behaviors(noise_rates={"cough": 40.0})
    env, ep = run(["Hi, I need to move my booking from Friday to Saturday, if that is possible at all.", "Bye ###STOP###"],
                  listener=None, interrupt=None, behaviors=b, speech=FakeSpeech(), seed=1)
    noises = user_turns(ep, "noise")
    own = [t for t in user_turns(ep) if t.get("kind") is None]
    over = [n for n in noises if any(u["start_time"] <= n["start_time"] < u["end_time"] for u in own)]
    agent = [t for t in ep["turns"] if t["role"] == "agent"]
    assert over and any(any(a["start_time"] <= n["start_time"] < a["end_time"] for a in agent) for n in noises)
    log = ep["meta"]["user"]["noise_events"]
    assert len(log) == len(noises) and sum(e["over_user_speech"] for e in log) == len(over) == ep["meta"]["user"]["random_events"]["noise_over_user"]
    assert all(e["expects"] == "ignore" for e in ep["eval"]["scores"]["events"] if e["turn"] in {n["id"] for n in noises})
    # the user's own lines are not held up by a noise: same lines, same times as without noise
    env2, ep2 = run(["Hi, I need to move my booking from Friday to Saturday, if that is possible at all.", "Bye ###STOP###"],
                    listener=None, interrupt=None, behaviors=Behaviors(), speech=FakeSpeech(), seed=1)
    assert [(t["start_time"], t["text"]) for t in own] == [(t["start_time"], t["text"]) for t in user_turns(ep2) if t.get("kind") is None]
    prof = ep["meta"]["user"]["behaviors"]["noise_process"]
    assert prof["labels"]["cough"]["rate_per_min"] == 40.0 and prof["labels"]["cough"]["source"] == "user"


def test_a_cough_overlapping_the_users_question_is_not_answered():
    T = lambda i, r, a, b, text="", **k: {"id": i, "role": r, "start_time": a, "end_time": b, "text": text, **k}  # noqa: E731
    turns = [T("u0", "user", 0, 3000, "When do you open?"), T("n1", "user", 2500, 3200, kind="noise", expects="ignore", label="cough"),
             T("a0", "agent", 3600, 5000, "At nine.")]
    ev = {e["turn"]: e for e in scores(turns)["events"]}
    assert ev["n1"]["outcome"] == "ignored" and ev["n1"]["score"] == 1.0 and ev["u0"]["outcome"] == "responded"


def test_random_events_differ_by_task_with_the_same_seed():
    def first(task_id):
        env = Env({"user": UserSim(ScriptSource(["Hi."]), behaviors=Behaviors(noise_per_min=1.0))}, AgentSpec())
        asyncio.run(env.reset(Task(id=task_id, scenario={}), seed=0))
        return env.sim.states["user"]["noise"].peek()

    assert first("a") == first("a") != first("b")


def test_random_events_do_not_depend_on_the_agent():
    b = Behaviors(noise_per_min=20)
    starts = []
    for text in (AGENT, "blah " * 40):
        env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, behaviors=b, agent_text=text, seed=2)
        starts.append([t["start_time"] for t in user_turns(ep, "noise")][:2])
    assert starts[0] and starts[0] == starts[1]


def test_noise_audio_is_audible_at_its_level_and_from_a_bank_when_given(tmp_path):
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, behaviors=Behaviors(noise_per_min=30),
                  speech=FakeSpeech())
    segs = [s for s in env.sim.states["user"]["mine"].values() if s.kind == "noise"]
    assert segs and all(max(abs(x) for x in s.data.samples) > 300 for s in segs)
    for s, e in zip(segs, ep["meta"]["user"]["noise_events"]):  # active-part level = speech level + the event's level
        assert abs(20 * math.log10(active_rms(s.data.samples, 16000) / 32768) - (SPEECH_DBFS + e["level_db"])) < 1.0
    clip = synth_event("phone_ring", 16000, random.Random(0))
    (tmp_path / "events" / "cough").mkdir(parents=True)
    clip.write_wav(tmp_path / "events" / "cough" / "a.wav")
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None,
                  behaviors=Behaviors(noise_rates={"cough": 30.0}), speech=FakeSpeech(), soundscape=Soundscape(bank=tmp_path))
    log = ep["meta"]["user"]["noise_events"]
    assert log and all(e["clip"] == f"file:{tmp_path / 'events' / 'cough' / 'a.wav'}" for e in log)
    for e in log:  # the real clip's length, repeated n times with the burst's gaps
        assert abs(e["dur_ms"] - e["n"] * clip.dur_ms) <= e["n"] * 2 + sum(EVENT_TYPES["cough"][4][1] for _ in range(e["n"] - 1))


def test_being_called_away():
    b = Behaviors(aside_per_min=20, away_p=1.0, away_ms=(4000, 4000))
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, behaviors=b, agent_ms=20_000,
                  scenario={"profile": {"surroundings": "home"}})
    seq = [t for t in user_turns(ep) if t.get("label") == "called_away"][:4]
    hold, aside, away, back = seq
    assert back.get("kind") is None and back.get("expects") == "yield"  # back while the agent still talks: a normal barge-in
    assert (hold.get("kind"), hold["expects"]) == (None, "wait") and (aside["kind"], aside["expects"]) == ("aside", "ignore")
    assert (away["kind"], away["expects"], away["text"]) == ("away", "wait", "") and away["end_time"] - away["start_time"] == 4000
    assert hold["start_time"] < aside["start_time"] < away["start_time"] < back["start_time"]
    a0 = next(t for t in ep["turns"] if t["id"] == "a0")
    assert a0["start_time"] < hold["start_time"] < a0["end_time"]  # called away while the agent talks
    exp = {(e["turn"], e["expects"]): e["outcome"] for e in ep["eval"]["scores"]["events"]}
    assert exp[(hold["id"], "wait")] == "kept_talking" and exp[(away["id"], "wait")] in ("took_floor", "waited")
    assert ep["meta"]["user"]["random_events"]["called_away"] >= 1


def test_aside_and_called_away_lines_come_from_the_user_llm():
    prompts = []

    def llm(messages):
        p = messages[-1]["content"]
        if p.startswith("You write what a person on a phone call says"):
            prompts.append(p)
            if "HOLD:" in p.split("Answer in exactly this format:")[1]:
                return "HOLD: Oh, one sec, someone's at the door.\nASIDE: Coming! Just leave it there, thanks.\nBACK: Sorry, that was the post. Go on."
            return "ASIDE: Not now, love, I'm on the phone."
        return "Bye ###STOP###" if len(messages) > 3 else "Hi, I need to move a booking."

    b = Behaviors(aside_per_min=20, away_p=1.0, away_ms=(3000, 3000))
    env, ep = run(llm, listener=None, interrupt=None, behaviors=b, agent_ms=20_000, scenario={"profile": {"surroundings": "home"}})
    hold, aside, away, back = [t for t in user_turns(ep) if t.get("label") == "called_away"][:4]
    assert (hold["text"], aside["text"], back["text"]) == ("Oh, one sec, someone's at the door.", "Coming! Just leave it there, thanks.",
                                                          "Sorry, that was the post. Go on.")
    assert (hold["expects"], aside["kind"], away["kind"]) == ("wait", "aside", "away")  # labels unchanged
    p = prompts[0]
    assert "at home" in p and "English" in p and "Hi, I need to move a booking." in p and any(x in p for x in SITUATIONS["home"])
    u = ep["meta"]["user"]
    assert u["behaviors"]["aside_lines"]["type"] == "llm" and u["random_events"]["lines_llm"] >= 3
    # an aside while the agent is silent: one line; Chinese persona: told the language
    b = Behaviors(aside_per_min=30, away_p=0.0)
    env, ep = run(llm, listener=None, interrupt=None, behaviors=b, agent_ms=1000, agent_text="Sure.", max_ms=30_000,
                  scenario={"profile": {"surroundings": "home"}})
    assert {t["text"] for t in user_turns(ep, "aside")} == {"Not now, love, I'm on the phone."}
    w = AsideWriter(FakeChat(["ASIDE: 我在打电话呢，等会儿。"]))
    zh = Task(scenario={"profile": {"language": "zh"}})
    assert "Chinese" in w.prompt(zh, [], "home", "x", False)
    assert asyncio.run(w.write(zh, [], "home", "x", False)) == asyncio.run(w.write(zh, [], "home", "x", False)) == {"ASIDE": "我在打电话呢，等会儿。"}
    assert w.calls == 1  # memoised
    # unusable answers, or no model: the fixed lines
    w = AsideWriter(FakeChat(["Sure! Here you go: " + "very " * 30]))
    assert asyncio.run(w.write(zh, [], "home", "x", True)) == {}
    env, ep = run(lambda m: "nonsense", listener=None, interrupt=None, behaviors=Behaviors(aside_per_min=30), agent_ms=1000,
                  agent_text="Sure.", max_ms=20_000, scenario={"profile": {"surroundings": "home"}})
    assert all(any(a.startswith(t["text"]) for a in ASIDES["en"]["home"]) for t in user_turns(ep, "aside"))  # (the last may be cut)
    assert ep["meta"]["user"]["random_events"]["lines_fixed"]
    user = UserSim(ScriptSource(["Hi."]), behaviors=Behaviors(aside_per_min=1.0))
    assert user.aside_writer is None and user.profile(Task(scenario={}))["behaviors"]["aside_lines"] == {"type": "fixed"}


def test_away_scoring_and_check_in_rule():
    T = lambda i, r, a, b, text="", **k: {"id": i, "role": r, "start_time": a, "end_time": b, "text": text, **k}  # noqa: E731
    away = T("u1", "user", 1000, 9000, kind="away", expects="wait")
    for agent, outcome, score in (([], "waited", 1.0), ([T("a1", "agent", 4000, 5500, "Still there?")], "checked_in", 1.0),
                                  ([T("a1", "agent", 1500, 3000, "So")], "took_floor", 0.0),
                                  ([T("a1", "agent", 4000, 5000, "Hello?"), T("a2", "agent", 6000, 7000, "Hello??")], "took_floor", 0.0)):
        ev = [e for e in scores([T("u0", "user", 0, 900, "Hi."), away, *agent])["events"] if e["turn"] == "u1"]
        assert [(e["expects"], e["outcome"], e["score"]) for e in ev] == [("wait", outcome, score)]


# ---------------------------------------------------------------- persona levels


def test_persona_levels_are_inferred_from_the_profile():
    lv, inferred = persona_levels({"speaking_style": "warm, talkative", "traits": "says 'mm-hmm' while listening"})
    assert (lv["backchanneling"], lv["surroundings"], lv["hesitancy"]) == ("frequent", "quiet", "normal")
    assert inferred == {"backchanneling", "surroundings", "hesitancy"}
    lv, _ = persona_levels({"speaking_style": "shy, quiet", "traits": "hesitates, pauses mid-sentence"})
    assert (lv["backchanneling"], lv["hesitancy"]) == ("rare", "high")
    lv, _ = persona_levels({"speaking_style": "gruff, short answers", "traits": "calls from a noisy road"})
    assert (lv["backchanneling"], lv["surroundings"]) == ("rare", "street")
    assert persona_levels({"traits": "talks to her kids in between", "attentiveness": "low"})[0]["surroundings"] == "home"
    assert persona_levels({"attentiveness": "low"})[0]["backchanneling"] == "rare"
    assert persona_levels({}, background=True)[0]["surroundings"] == "cafe"
    assert persona_levels({"speaking_style": "fast, businesslike"})[0]["hesitancy"] == "low"
    lv, inferred = persona_levels({"speaking_style": "chatty", "backchanneling": "rare", "surroundings": "car", "hesitancy": "high"})
    assert (lv["backchanneling"], lv["surroundings"], lv["hesitancy"]) == ("rare", "car", "high") and not inferred
    with pytest.raises(ValueError):
        persona_levels({"surroundings": "moon"})


def test_persona_drives_event_rates_and_overrides_win():
    prof = {"backchanneling": "frequent", "surroundings": "home", "hesitancy": "high", "attentiveness": "low"}
    b, levels = PERSONA.resolve(1000, prof)
    assert (b.aside_per_min, b.noise_per_min) == (2 * SURROUNDINGS["home"][0], SURROUNDINGS["home"][1])  # low attentiveness: x2
    assert {k: levels[k] for k in ("backchanneling", "surroundings", "hesitancy", "attentiveness")} == prof
    user = UserSim(LLMSource(FakeChat(["x"])), behaviors=dataclasses.replace(PERSONA, noise_per_min=4.0))
    task = Task(scenario={"profile": prof, "behaviors": {"aside_per_min": 1.0}})
    habits = user.profile(task)["behaviors"]
    assert (habits["aside_per_min"], habits["noise_per_min"]) == (1.0, 4.0) and habits["persona"]["surroundings"] == "home"
    with pytest.warns(DeprecationWarning):  # the old backchannel rate: the nearest listening style
        old = Behaviors(backchannel_per_min=0.5)
    assert old.resolve(1000, prof)[1]["backchanneling"] == "rare"
    zh = UserSim(ScriptSource([]))._behaviors(Task(scenario={"profile": {"language": "zh", "surroundings": "cafe"}}))
    assert zh.backchannels == BACKCHANNELS["zh"] and zh.asides == ASIDES["zh"]["cafe"]


# ---------------------------------------------------------------- background (context, not a turn)


def test_background_defaults_from_surroundings_and_levels():
    assert background_spec(None, "quiet") == {"type": "pink", "level_dbfs": QUIET_FLOOR_DBFS, "snr_db": SPEECH_DBFS - QUIET_FLOOR_DBFS,
                                              "source": "synthetic:pink", "from": "surroundings", "surroundings": "quiet"}  # a mic floor
    assert background_spec(False, "quiet")["type"] == "silence" and background_spec("silence", "home")["source"] == "none"
    s = background_spec(None, "cafe")
    assert s["type"] == "ambience:cafe" and s["source"] == "synthetic:pink" and s["level_dbfs"] == DEFAULTS["cafe"][1]
    assert s["snr_db"] == round(SPEECH_DBFS - s["level_dbfs"], 1)
    assert background_spec({"type": "brown", "snr_db": 10}, "quiet")["level_dbfs"] == SPEECH_DBFS - 10
    assert background_spec(-12, "street")["level_dbfs"] == round(20 * math.log10(900 / 32768) - 12, 1)  # the old numeric form
    assert background_spec(-12, "quiet")["type"] == "white"
    with pytest.raises(ValueError):
        background_spec("ambience:moon", "quiet")
    for color in ("white", "pink", "brown"):
        a = colored_noise(color, 8000, -40.0, seconds=2)
        rms = math.sqrt(sum(x * x for x in a.samples) / len(a.samples))
        assert abs(20 * math.log10(rms / 32768) + 40) < 0.5
    t1, t2 = render_background(background_spec(None, "car"), 8000, 3), render_background(background_spec(None, "car"), 8000, 3)
    assert t1.offset_ms == t2.offset_ms and t1.audio == t2.audio and t1.loop
    assert render_background(background_spec(None, "car"), 8000, 4).offset_ms != t1.offset_ms


def test_background_from_a_bank(tmp_path):
    (tmp_path / "ambience" / "street").mkdir(parents=True)
    colored_noise("white", 8000, -10.0, seconds=1).write_wav(tmp_path / "ambience" / "street" / "s.wav")
    spec = background_spec(None, "street", SoundBank(tmp_path))
    assert spec["source"] == "bank"
    bg = render_background(spec, 8000, 0, SoundBank(tmp_path))
    rms = math.sqrt(sum(x * x for x in bg.audio.samples) / len(bg.audio.samples))
    assert spec["source"].startswith("file:") and abs(20 * math.log10(rms / 32768) - DEFAULTS["street"][1]) < 0.5


def test_background_is_recorded_and_mixed(tmp_path):
    from interaction_gym.media import MediaStore
    from interaction_gym.traj import episode as ep_of

    sp = AgentSpec(chunk_ms=100, audio="user.audio", sr=8000)
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, spec=sp, speech=FakeSpeech(sr=8000),
                  scenario={"profile": {"surroundings": "street"}})
    (bg,) = ep["meta"]["env"]["background"]
    assert bg["type"] == "ambience:street" and bg["from"] == "surroundings"
    assert not any(t.get("kind") == "background" for t in ep["turns"])  # context, never a turn
    full = check(ep_of(env, "e", media=MediaStore(tmp_path)))
    assert full["background"][0]["spec"]["type"] == "ambience:street"
    env, ep = run(["Hi.", "Bye ###STOP###"], listener=None, interrupt=None, scenario={"profile": {"surroundings": "street"}},
                  soundscape=Soundscape(background=False))
    assert "background" not in ep["meta"]["env"] and not env.background


def test_meta_user_records_how_decisions_were_made():
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"])
    u = ep["meta"]["user"]
    assert u["listening"]["decision_points"] == "phrase boundaries" and u["listening"]["model"]["params"]["temperature"] == 0.0
    assert u["decisions"]["points"] == u["decisions"]["boundary"] + u["decisions"]["fallback"]
    assert set(u["random_events"]) >= {"noise", "aside", "called_away"}



def test_viewer_shows_background_and_labels():
    from interaction_gym.viewer import render_html

    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], behaviors=Behaviors(noise_per_min=30),
                  scenario={"profile": {"surroundings": "cafe"}})
    page = render_html([ep])
    assert "function bgSpecs" in page and "'background'" in page and '"label": "' in page and '"ambience:cafe"' in page
