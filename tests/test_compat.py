"""User-simulator compatibility with different LLM / TTS servers (found by running a matrix of real models)."""

import asyncio
import io
import json
import wave
from array import array

import interaction_gym.clients as C
from interaction_gym import AgentSpec, Env, Segment, Task
from interaction_gym.audio import Audio
from interaction_gym.clients import FakeChat, OpenAIChat, OpenAISpeech
from interaction_gym.core import _mix
from interaction_gym.user import QWEN3_TTS_VOICES, LLMSource, pick_voice


def run(coro):
    return asyncio.run(coro)


def chat_reply(monkeypatch, message):
    monkeypatch.setattr(C, "_post", lambda url, payload, key: json.dumps({"choices": [{"message": message}]}).encode())
    return run(OpenAIChat("http://x/v1", "m").chat([{"role": "user", "content": "hi"}]))


def test_reasoning_never_reaches_the_user(monkeypatch):
    assert chat_reply(monkeypatch, {"content": "<think>they want a table</think>Two people, please."}) == "Two people, please."
    assert chat_reply(monkeypatch, {"content": "<think>still thinking when max_tokens ran out"}) == ""  # cut mid-thought
    assert chat_reply(monkeypatch, {"content": None, "reasoning_content": "long reasoning"}) == ""  # parser server, budget used up
    assert chat_reply(monkeypatch, {"content": "Sure."}) == "Sure."


def test_tts_audio_is_taken_at_its_own_rate(monkeypatch):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:  # a 48 kHz model answering with a WAV
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(array("h", [1000] * 48000).tobytes())
    monkeypatch.setattr(C, "_post", lambda url, payload, key: buf.getvalue())
    audio = run(OpenAISpeech("http://x/v1", "voxcpm", sr=24000).synth("hello"))
    assert audio.sr == 24000 and audio.dur_ms == 1000  # 1 s stays 1 s
    monkeypatch.setattr(C, "_post", lambda url, payload, key: array("h", [0] * 22050).tobytes())
    raw = run(OpenAISpeech("http://x/v1", "m", sr=24000, response_format="pcm", native_sr=22050).synth("hello"))
    assert raw.sr == 24000 and raw.dur_ms == 1000


def test_a_voice_at_another_rate_is_mixed_into_the_microphone():
    spec = AgentSpec(audio="user.audio", sr=24000)
    env = Env({}, spec)
    seg = Segment("u0", 0, 1000, Audio(array("h", [1000] * 48000), 48000), text="hello")
    mic = _mix(env, [seg], 0, 200)
    assert mic.sr == 24000 and len(mic) == 4800 and set(mic.samples) == {1000}


def test_what_the_user_llm_writes_is_cleaned_to_what_a_person_says():
    replies = ['"Two people, please." [CURRENTLY SPEAKING, INCOMPLETE]', "[SCRIPT OFF]", "Around seven."]
    src, task = LLMSource(FakeChat(replies)), Task(id="t", scenario={})
    assert run(src.next(0, task, [])).text == "Two people, please."
    assert run(src.next(1, task, [])).text == "Around seven."  # nothing sayable: asked again


def test_the_voice_follows_the_native_language():
    mina = {"name": "Mina Park", "gender": "female", "age": 45, "language": "en", "native_language": "ko"}
    assert pick_voice(mina, QWEN3_TTS_VOICES)["speaker"] == "sohee"
