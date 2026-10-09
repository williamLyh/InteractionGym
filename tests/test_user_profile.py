"""meta.user: each user component reports who the simulated user is and how it was produced."""

import asyncio
import dataclasses
import json
from pathlib import Path

import examples.minimal as minimal
import examples.user_modes as um
from interaction_gym import AgentSpec, Env
from interaction_gym.agents import CannedAgent
from interaction_gym.clients import Cached, FakeChat, FakeSpeech, OpenAIChat, OpenAISpeech, describe
from interaction_gym.traj import episode
from interaction_gym.user import NEUTRAL_STYLE, LLMSource, UserSim, Voice
from tests.test_traj import VALIDATOR, check


def run(coro):
    return asyncio.run(coro)


def user_of(mode):
    return check(episode(run(um.run(mode)), "e"))["meta"]["user"]


def test_online_user_records_persona_voice_models_and_policies():
    u = user_of("online")
    assert u["component"] == "user" and u["mode"] == "online"
    assert u["persona"] == um.TASK.scenario["persona"] and u["goal"] == um.TASK.scenario["instructions"]
    v = u["voice"]  # FakeSpeech clones by itself: the defaults (neutral style, first-turn clone, leveling, seed) apply
    assert {k: v[k] for k in ("speaker", "chosen_by", "tts")} == {"speaker": "default", "chosen_by": "default", "tts": {"model": "FakeSpeech", "sr": 16000}}  # looked through Cached
    assert v["style"] == "neutral" and v["instructions"] == NEUTRAL_STYLE and v["clone"]["reference"] == "first turn"
    assert v["leveling"]["level_dbfs"] == -23.0 and isinstance(v["seed"], int)
    assert u["llm"] == {"model": "FakeChat"} and u["barge_in"]["type"] == "llm" and u["barge_in"]["min_words"] == 5
    assert u["turn_taking"]["respond_after_ms"] == 1000 and "system_prompt" not in u


def test_semi_online_and_offline_users():
    script = user_of("script")
    assert script["mode"] == "semi_online" and script["barge_in"] == {"type": "keyword", "words": ["four"]} and "llm" not in script
    offline = user_of("offline")
    assert offline["mode"] == "offline" and "barge_in" not in offline and "turn_taking" not in offline


def test_text_only_user_and_no_user():
    assert check(episode(run(minimal.run()), "e"))["meta"]["user"]["barge_in"] == {"type": "keyword", "words": ["Tomorrow"]}
    env = Env({}, AgentSpec(), max_ms=1000)
    run(env.reset(um.TASK))
    assert "user" not in episode(env, "e")["meta"]  # no simulated user component
    env = Env({"user": UserSim(LLMSource(FakeChat(["Hi."])))}, AgentSpec(), max_ms=5000)
    run(env.reset(dataclasses.replace(um.TASK, scenario={})))
    assert check(episode(env, "e"))["meta"]["user"]["voice"] == {"tts": None, "words_per_sec": 3.4}


def test_task_picks_the_voice():
    task = dataclasses.replace(um.TASK, scenario={**um.TASK.scenario, "voice": "ethan"})
    speech = FakeSpeech()
    voices = []
    orig = speech.synth

    async def synth(text, voice="default", instructions=None, language=None):
        voices.append(voice)
        return await orig(text, voice, instructions)

    speech.synth = synth

    async def main():
        spec = AgentSpec()
        env = Env({"user": UserSim(LLMSource(FakeChat(["Hi there.", "Bye. ###STOP###"])), Voice(speech, voice="vivian", clone=False))}, spec, max_ms=30_000)
        agent = CannedAgent(["Hello!"], spec)
        obs = await env.reset(task)
        done = False
        while not done:
            obs, _, done = await env.step(agent.act(env.t, obs))
        return env

    ep = check(episode(run(main()), "e"))
    assert voices and set(voices) == {"ethan"} and ep["meta"]["user"]["voice"]["speaker"] == "ethan"


def test_describe_real_clients():
    assert describe(Cached(OpenAIChat("http://x/v1", "Qwen3.8-27B", max_tokens=80))) == {"model": "Qwen3.8-27B", "params": {"max_tokens": 80}}
    assert describe(OpenAISpeech("http://x/v1", "Qwen3-TTS", sr=24000)) == {"model": "Qwen3-TTS", "sr": 24000}
    assert describe(None) is None


def test_format_example_validates():
    ep = json.loads((Path(__file__).parents[1] / "docs/format_example.json").read_text())
    assert not list(VALIDATOR.iter_errors(ep)) and ep["meta"]["user"]["mode"] == "online"


def test_voice_follows_the_profile():
    from interaction_gym.user import QWEN3_TTS_VOICES, age_group, pick_voice

    pick = lambda **p: pick_voice(p, QWEN3_TTS_VOICES)["speaker"]  # noqa: E731
    assert pick(name="Frank", gender="male", age=68, language="zh") == "uncle_fu"
    assert pick(name="Jake", gender="male", age=25, language="en", accent="american") == "aiden"
    assert pick(name="Min", gender="female", age=40, language="ko") == "sohee"
    assert pick(name="Alex", gender="female", age=30, language="en") in ("vivian", "serena", "ono_anna")  # no English female: any young one
    assert pick(name="Alex", gender="female", age=30, language="en") == pick(name="Alex", gender="female", age=30, language="en")  # stable
    assert {pick(name=f"user{i}", gender="female", age=25) for i in range(30)} >= {"vivian", "serena"}  # a population spreads out
    assert [age_group(a) for a in (8, 20, 45, 70, "senior", None)] == ["child", "young", "adult", "senior", "senior", None]


def test_profile_reaches_the_user_llm_the_tts_and_the_trajectory():
    from interaction_gym.user import QWEN3_TTS_VOICES

    profile = {"name": "Frank Li", "gender": "male", "age": 68, "language": "en", "occupation": "retired teacher",
               "speaking_style": "slow and deliberate, a little hard of hearing", "traits": "polite but insists on details"}
    task = dataclasses.replace(um.TASK, scenario={**um.TASK.scenario, "profile": profile})
    llm, said = FakeChat(["Hello? Is this the restaurant?", "Thank you. ###STOP###"]), []
    speech = FakeSpeech()
    orig = speech.synth

    async def synth(text, voice="default", instructions=None, language=None):
        said.append((voice, instructions, language))
        return await orig(text, voice, instructions)

    speech.synth = synth

    async def main():
        spec = AgentSpec()
        env = Env({"user": UserSim(LLMSource(llm), Voice(speech, voices=QWEN3_TTS_VOICES, clone=False, style="persona"))}, spec, max_ms=30_000)
        agent = CannedAgent(["Yes, how can I help?"], spec)
        obs = await env.reset(dataclasses.replace(task, scenario={**task.scenario, "first_turn": None}))
        done = False
        while not done:
            obs, _, done = await env.step(agent.act(env.t, obs))
        return env

    ep = check(episode(run(main()), "e"))
    system = llm.calls[0][0]["content"]
    assert "Frank Li, male, 68 (senior)" in system and "retired teacher" in system and "insists on details" in system
    assert "Always speak English." in system
    assert said and all(v == "uncle_fu" and i == profile["speaking_style"] and lang == "English" for v, i, lang in said)  # older male; style; language
    u = ep["meta"]["user"]
    assert u["profile"] == profile and u["voice"]["speaker"] == "uncle_fu" and u["voice"]["chosen_by"] == "profile"
    assert u["voice"]["instructions"] == profile["speaking_style"]


def test_a_user_who_never_yields():
    from interaction_gym import Segment
    from interaction_gym.core import Frame
    from interaction_gym.user import ScriptSource

    async def main(overrides):
        task = dataclasses.replace(um.TASK, scenario={"turns": ["I would like to book a table for six people tomorrow evening please"],
                                                       **({"turn_taking": overrides} if overrides else {})})
        spec = AgentSpec(chunk_ms=100)
        env = Env({"user": UserSim(ScriptSource())}, spec, max_ms=20_000)
        obs = await env.reset(task)
        sent = False
        while not env.done:
            act = []
            if env.t == 1000 and not sent:  # the agent talks over the user from 1 s
                act, sent = [Frame(spec.out, 1000, Segment("a0", 1000, 3000, "Sure, one moment please", text="Sure, one moment please"))], True
            obs, _, _ = await env.step(act)
        return check(episode(env, "e"))

    polite, stubborn = run(main(None)), run(main({"yield_after_ms": None}))
    u_p = next(t for t in polite["turns"] if t["role"] == "user")
    u_s = next(t for t in stubborn["turns"] if t["role"] == "user")
    assert u_p["end_time"] == 2000 and "unsaid" in u_p  # talked over for 1 s: stops
    assert "unsaid" not in u_s and u_s["end_time"] > 2000  # keeps talking to the end
    assert stubborn["meta"]["user"]["turn_taking"]["yield_after_ms"] is None


class Recording(FakeSpeech):
    def __init__(self, name):
        super().__init__(sr=16000)
        self.name, self.log = name, []

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None):
        self.log.append((text, ref_text, None if ref_audio is None else len(ref_audio)))
        return await super().synth(text, voice, instructions)


def test_one_voice_per_episode_cloned_from_the_first_turn():
    from interaction_gym.user import ReplayUser

    speech, clone = Recording("custom"), Recording("clone")
    turns = [{"t": 0, "text": "mm-hmm", "kind": "backchannel"},  # too short to be the reference
             {"t": 1000, "text": "I would like to book a table"},  # the reference
             {"t": 4000, "text": "for two people please"},
             {"t": 7000, "text": "", "kind": "noise", "dur": 300}]
    env = Env({"user": ReplayUser(turns, voice=Voice(speech, clone=clone))}, AgentSpec(), max_ms=20_000)
    run(env.reset(um.TASK))
    assert [x[0] for x in speech.log] == ["mm-hmm", "I would like to book a table"]
    (cloned,) = clone.log
    assert cloned[0] == "for two people please" and cloned[1] == "I would like to book a table" and cloned[2] > 0
    u = check(episode(env, "e"))["meta"]["user"]
    assert u["voice"]["clone"] == {"tts": {"model": "Recording", "sr": 16000}, "reference": "first turn"}


def test_a_recorded_first_turn_is_the_reference():
    from interaction_gym.audio import Audio
    from interaction_gym.user import ReplayUser

    clone = Recording("clone")
    rec = Audio.silence(1500, 16000)
    env = Env({"user": ReplayUser([{"t": 0, "text": "hello there my friend", "audio": rec}, {"t": 3000, "text": "are you there"}],
                                  voice=Voice(Recording("custom"), clone=clone))}, AgentSpec(), max_ms=10_000)
    run(env.reset(um.TASK))
    assert clone.log == [("are you there", "hello there my friend", len(rec))]


def test_openai_speech_sends_the_reference_for_cloning(monkeypatch):
    import base64 as b64
    import io
    import wave

    import interaction_gym.clients as C
    from interaction_gym.audio import Audio

    sent = []
    monkeypatch.setattr(C, "_post", lambda url, payload, key: sent.append(payload) or b"\x00\x00" * 2400)
    ref = Audio.silence(500, 24000)
    run(OpenAISpeech("http://x/v1", "Qwen3-TTS-Base", sr=24000).synth("hi there", "vivian", "calm", ref_audio=ref, ref_text="hello"))
    (p,) = sent
    assert p["task_type"] == "Base" and p["ref_text"] == "hello" and "voice" not in p and p["instructions"] == "calm"
    with wave.open(io.BytesIO(b64.b64decode(p["ref_audio"].split(",", 1)[1]))) as w:
        assert w.getframerate() == 24000 and w.getnframes() == len(ref)


def test_a_runaway_tts_turn_is_retried_then_cut_to_size():
    from interaction_gym.audio import Audio
    from interaction_gym.user import ReplayUser

    class Runaway(FakeSpeech):
        def __init__(self, good_after):
            super().__init__(sr=16000)
            self.n, self.good_after = 0, good_after

        async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None):
            self.n += 1
            return Audio.silence(37_000 if self.n <= self.good_after else 600, 16000)

    for good_after, expect in ((1, 600), (5, None)):  # recovers on the retry / keeps running away
        tts = Runaway(good_after)
        voice = Voice(tts)
        env = Env({"user": ReplayUser([{"t": 0, "text": "mm-hmm", "kind": "backchannel"}], voice=voice)}, AgentSpec(), max_ms=60_000)

        async def play(env=env):
            await env.reset(um.TASK)
            while not env.done:
                await env.step([])

        run(play())
        (u,) = [t for t in check(episode(env, "e"))["turns"] if t["role"] == "user"]
        assert tts.n == 2 and u["end_time"] - u["start_time"] == (expect or voice.max_ms("mm-hmm"))


def test_an_empty_llm_reply_is_not_spoken():
    from interaction_gym import Segment
    from interaction_gym.core import Frame

    async def main():
        llm = FakeChat(["Hello, is anyone there?", "", "", "", "", "", "", "", ""])
        spec = AgentSpec(chunk_ms=100)
        env = Env({"user": UserSim(LLMSource(llm))}, spec, max_ms=30_000)
        obs = await env.reset(dataclasses.replace(um.TASK, scenario={}))
        sent = False
        while not env.done:  # the agent talks for a long time; the user's nudges come back empty
            act = []
            if env.t == 2000 and not sent:
                act, sent = [Frame(spec.out, 2000, Segment("a0", 2000, 20_000, "blah " * 60, text="blah " * 60))], True
            obs, _, _ = await env.step(act)
        return check(episode(env, "e"))

    users = [t for t in run(main())["turns"] if t["role"] == "user"]
    assert [t["text"] for t in users] == ["Hello, is anyone there?"]  # no empty zero-length turns
