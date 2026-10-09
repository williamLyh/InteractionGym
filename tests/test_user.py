import asyncio

from examples.user_modes import run as run_modes
from interaction_gym import AgentSpec, Env, Frame, Segment, Task
from interaction_gym.audio import Audio
from interaction_gym.clients import Cached, FakeChat, FakeSpeech
from interaction_gym.eval import turns
from interaction_gym.media import MediaStore
from interaction_gym.traj import episode
from interaction_gym.user import STOP, KeywordInterrupt, LLMSource, ReplayUser, ScriptSource, TurnTaking, UserSim


def run(coro):
    return asyncio.run(coro)


def say(sid, t, text, wps=3.4):
    return Segment(sid, t, round(len(text.split()) / wps * 1000), text)


async def drive(env, task, actions: dict[int, list[Frame]], until: int):
    """Step the env, injecting scripted agent frames at the given step times."""
    await env.reset(task)
    while env.t < until and not env.done:
        await env.step(actions.pop(env.t, []))
    return env


def test_audio_segment_keeps_transcript_aligned():
    seg = Segment("s", 0, 1000, Audio.silence(1000, 16000), text="abcdefghij")
    cut = seg.cut(500)
    assert len(cut.data) == 8000 and cut.text == "abcde" and cut.heard_text(250) == seg.heard_text(250)


def test_cached_speech_synthesizes_once():
    fake = FakeSpeech()
    tts = Cached(fake)
    a = run(tts.synth("hello there"))
    b = run(tts.synth("hello there"))
    assert a == b and fake.calls == 1 and a.dur_ms == round(2 / 3.4 * 1000)


def test_replay_user_ignores_agent():
    task = Task(scenario={"turns": [{"t": 0, "text": "one two"}, {"t": 1000, "text": "three"}]})
    env = Env({"user": ReplayUser()}, AgentSpec(chunk_ms=100))
    env = run(drive(env, task, {0: [Frame("policy.speech", 0, say("a0", 0, "talking over everything " * 5))]}, 10_000))
    assert [(u.planned.t0, u.planned.text) for u in turns(env.log, "user.speech")] == [(0, "one two"), (1000, "three")]
    assert env.done and env.t < 10_000


def test_reactive_user_waits_for_silence_then_replies():
    task = Task(scenario={"turns": ["hi", "second"]})
    env = Env({"user": UserSim(ScriptSource())}, AgentSpec(chunk_ms=100))
    env = run(drive(env, task, {1000: [Frame("policy.speech", 1000, say("a0", 1000, "hello how can I help"))]}, 20_000))
    u = turns(env.log, "user.speech")
    a0_end = 1000 + say("a0", 1000, "hello how can I help").dur
    assert u[0].planned.t0 == 0 and u[1].planned.t0 == a0_end + TurnTaking().respond_after_ms
    assert env.done  # script exhausted: the user stops talking and the episode winds down


def test_barge_in_uses_only_heard_text():
    llm = FakeChat(["No, the other one."])
    user = UserSim(LLMSource(llm, first="Book it."), interrupt=KeywordInterrupt("tuesday"))
    long = "Okay I will book it for Tuesday morning at nine and send you a confirmation email right away"
    env = Env({"user": user}, AgentSpec(chunk_ms=100))
    env = run(drive(env, Task(), {1000: [Frame("policy.speech", 1000, say("a0", 1000, long))]}, 6_000))
    barge = turns(env.log, "user.speech")[1].planned
    check_t = barge.t0 - TurnTaking().reaction_ms
    assert (check_t - 1000) % TurnTaking().max_decision_gap_ms == 0  # no phrase boundary in it: decided at a fallback point
    agent_msg, note = llm.calls[0][-1]["content"].split("\n\n", 1)
    assert agent_msg.endswith("[CURRENTLY SPEAKING, INCOMPLETE]") and note.startswith("(You are cutting in now")
    heard = say("a0", 1000, long).heard_text(check_t)
    assert agent_msg.startswith(heard) and "confirmation" not in agent_msg  # nothing unheard leaks


def test_user_yields_when_agent_talks_over():
    task = Task(scenario={"turns": ["this is a very long sentence that goes on and on and on for quite a while indeed"]})
    env = Env({"user": UserSim(ScriptSource())}, AgentSpec(chunk_ms=100))
    env = run(drive(env, task, {500: [Frame("policy.speech", 500, say("a0", 500, "sorry to cut in here but"))]}, 3_000))
    (u0,) = turns(env.log, "user.speech")
    assert u0.was_cut and u0.actual.end == 500 + TurnTaking().yield_after_ms


def test_llm_stop_ends_episode():
    llm = FakeChat([f"Great, bye! {STOP}"])
    env = Env({"user": UserSim(LLMSource(llm, first="Hi."))}, AgentSpec(chunk_ms=100))
    env = run(drive(env, Task(), {800: [Frame("policy.speech", 800, say("a0", 800, "hello"))]}, 20_000))
    u = turns(env.log, "user.speech")
    assert [x.planned.text for x in u] == ["Hi.", "Great, bye!"]
    assert env.done and not env.truncated and not u[-1].was_cut  # the last turn is spoken in full
    assert 0 <= env.t - (u[-1].actual.end + env.end_idle_ms) < env.chunk_ms  # then ends after a quiet spell (checked per step)


def test_user_modes_example():
    offline, script, online = (run(run_modes(m)) for m in ("offline", "script", "online"))
    for env in (offline, script, online):
        assert env.done
    # open loop: the offline user's second turn overlaps the agent's question and cuts it
    assert turns(offline.log, "policy.speech")[0].was_cut
    # closed loop: the correction arrives as a barge-in on the wrong booking
    for env in (script, online):
        a1 = turns(env.log, "policy.speech")[1]
        assert a1.was_cut and "four" in a1.actual.data

def test_user_audio_is_stored_as_media(tmp_path):
    ep = episode(run(run_modes("online")), "online", media=MediaStore(tmp_path))
    u0 = ep["turns"][0]
    assert u0["role"] == "user" and u0["text"].startswith("Hi") and (tmp_path / u0["media"]["uri"]).exists()


def test_llm_messages_always_have_a_user_message():
    from interaction_gym.user import Utterance
    src = LLMSource(FakeChat(["hi"]))
    task = Task(id="t", scenario={"persona": "caller"})
    msgs = src.messages(task, [Utterance("user", "Hello, I'd like to book a table.", 0, False)])  # the agent never answered
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]  # never ends on the simulator's own line
    assert msgs[-1]["content"].startswith("(Silence")
    assert [m["role"] for m in src.messages(task, [])] == ["system", "user"]


def test_user_waits_for_a_streamed_agent_turn_to_finish():
    """A text agent streams its turn in chunks; the user decides its reply on the whole turn, not on the part
    played when a chunk boundary is reached (the env runs that instant before the agent's next chunk arrives)."""
    from interaction_gym.core import Chunk
    from interaction_gym.user import ResponseDelay, UserTurn

    heard = []

    class Recorder:
        def profile(self):
            return {"mode": "online"}

        async def next(self, i, task, convo):
            heard.append([u.text for u in convo if u.role == "agent"])
            return UserTurn("hi there") if i == 0 else UserTurn("thanks", final=True)

    text = "Okay, I can help you with that today."
    chunks = {t: [Frame("policy.speech", t, Chunk("a0", t, 100, text[(t - 1000) // 25:(t - 900) // 25], text[(t - 1000) // 25:(t - 900) // 25],
                                                 first=t == 1000, last=t == 1500))] for t in range(1000, 1600, 100)}
    env = Env({"user": UserSim(Recorder(), timing=TurnTaking(response_delay=ResponseDelay()))}, AgentSpec(chunk_ms=100))
    run(drive(env, Task(scenario={}), chunks, 5_000))
    assert heard[1] == [text[:24]]  # all the chunks: 6 x 4 characters


def test_llm_messages_skip_own_sounds_and_merge_pieces():
    """The user LLM sees its own turns, not the Behaviors' backchannels / asides / noises, and a turn said in two
    pieces (a mid-thought pause) as one message."""
    from interaction_gym.user import Utterance
    src = LLMSource(FakeChat(["hi"]))
    task = Task(id="t", scenario={"persona": "caller"})
    convo = [Utterance("user", "Hi, I need", 0, False), Utterance("user", "a table for two.", 1500, False),
             Utterance("agent", "Sure, for when?", 3000, False), Utterance("user", "mm-hmm", 3500, False, "backchannel"),
             Utterance("user", "hold on a second", 4000, False, "aside"), Utterance("agent", "And your name?", 5000, True)]
    msgs = src.messages(task, convo)
    assert [(m["role"], m["content"]) for m in msgs[1:]] == [
        ("assistant", "Hi, I need a table for two."),
        ("user", "Sure, for when? And your name? [CURRENTLY SPEAKING, INCOMPLETE]")]
