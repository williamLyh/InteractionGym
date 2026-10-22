"""The model clients against the OpenAI API: a local stand-in that refuses what the hosted API refuses (local-server
request fields, ``max_tokens``, a reasoning model's temperature, TTS seeds and cloning), rate-limits, and counts."""

import asyncio
import json
import threading
import urllib.error
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from interaction_gym.agents.cascaded import ToolChat
from interaction_gym.audio import Audio
from interaction_gym.clients import LOCAL_ONLY_PARAMS, Cached, OpenAIChat, OpenAISpeech, OpenAITranscribe, api_of, describe, is_reasoning
from interaction_gym.user import OPENAI_TTS_VOICES, Voice, pick_voice

KEY = "sk-test"


class FakeOpenAI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self, code: int, body: bytes, ctype="application/json", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, msg: str, kind="invalid_request_error", headers=None):
        self._reply(code, json.dumps({"error": {"message": msg, "type": kind, "code": kind}}).encode(), headers=headers)

    def do_POST(self):
        srv = self.server
        data = self.rfile.read(int(self.headers["Content-Length"]))
        if self.headers.get("Authorization") != f"Bearer {KEY}":
            return self._error(401, "Incorrect API key provided")
        if srv.limit:
            srv.limit -= 1
            return self._error(429, "Rate limit reached", "rate_limit_exceeded", {"Retry-After": "0"})
        if srv.quota:
            return self._error(429, "You exceeded your current quota", "insufficient_quota")
        if self.path.endswith("/audio/transcriptions"):
            srv.seen.append({"multipart": len(data)})
            return self._reply(200, json.dumps({"text": " hello there "}).encode())
        req = json.loads(data)
        srv.seen.append(req)
        if self.path.endswith("/chat/completions"):
            bad = sorted(k for k in req if k in LOCAL_ONLY_PARAMS)
            if bad:
                return self._error(400, f"Unrecognized request argument supplied: {bad[0]}")
            if "max_tokens" in req and is_reasoning(req["model"]):
                return self._error(400, "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens' instead.")
            reasons = req["model"] == "gpt-5-mini" or req.get("reasoning_effort", "none") != "none"  # as gpt-5.4-mini
            if reasons and req.get("temperature", 1) != 1:
                return self._error(400, "Unsupported value: 'temperature' does not support 0 with this model. "
                                        "Only the default (1) value is supported.")
            msg = {"role": "assistant", "content": "Two people, please."}
            if req.get("tools"):
                msg = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "book", "arguments": json.dumps({"n": 2})}}]}
            usage = {"prompt_tokens": 20, "completion_tokens": 30, "completion_tokens_details": {"reasoning_tokens": 25}}
            return self._reply(200, json.dumps({"choices": [{"message": msg}], "usage": usage}).encode())
        if self.path.endswith("/audio/speech"):
            for k in ("seed", "language", "ref_audio", "task_type"):
                if k in req:
                    return self._error(400, f"Unrecognized request argument supplied: {k}")
            assert req["response_format"] == "pcm"
            return self._reply(200, array("h", [500] * 12000).tobytes(), "audio/pcm")  # 0.5 s at 24 kHz
        self._error(404, "no such route")


@pytest.fixture
def api(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    srv.seen, srv.limit, srv.quota = [], 0, False
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    yield srv
    srv.shutdown()


def run(coro):
    return asyncio.run(coro)


def test_the_hosted_api_is_recognised_by_its_host():
    assert api_of("https://api.openai.com/v1") == "openai"
    assert api_of("http://127.0.0.1:8000/v1") == "compat"
    assert api_of("http://my-proxy/v1", "openai") == "openai"
    assert is_reasoning("gpt-5-mini") and is_reasoning("o4-mini") and not is_reasoning("gpt-5-chat-latest")
    assert not is_reasoning("gpt-4.1-mini") and not is_reasoning("Qwen/Qwen3-8B")


def test_a_hosted_client_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIChat.openai("gpt-5-mini")
    OpenAIChat("http://127.0.0.1:8000/v1", "m")  # a local server needs none


def test_local_server_fields_are_not_sent_and_max_tokens_is_renamed(api):
    llm = OpenAIChat(api.url, "gpt-4.1-mini", api="openai", max_tokens=80, temperature=0.7,
                     chat_template_kwargs={"enable_thinking": False}, top_k=20)
    with pytest.warns(UserWarning, match="chat_template_kwargs"):
        assert run(llm.chat([{"role": "user", "content": "hi"}])) == "Two people, please."
    req = api.seen[-1]
    assert req["max_completion_tokens"] == 80 and req["temperature"] == 0.7 and "max_tokens" not in req
    assert describe(llm)["provider"] == "openai" and describe(llm)["not_sent"] == ["chat_template_kwargs", "top_k"]
    assert describe(llm)["params"]["max_tokens"] == 80  # what was asked for stays recorded


def test_a_reasoning_model_gets_headroom_and_loses_temperature_only_when_refused(api):
    llm = OpenAIChat(api.url, "gpt-5-mini", api="openai", reasoning_tokens=1000)
    with pytest.warns(UserWarning, match="temperature"):
        run(llm.chat([{"role": "user", "content": "yes or no?"}], temperature=0.0, max_tokens=16))  # the listener's call
    refused, sent = api.seen[-2:]
    assert refused["temperature"] == 0.0 and sent["max_completion_tokens"] == 1016 and "temperature" not in sent
    assert llm.usage == {"requests": 1, "prompt_tokens": 20, "completion_tokens": 30, "reasoning_tokens": 25}
    n = len(api.seen)
    run(llm.chat([{"role": "user", "content": "again"}], temperature=0.0))
    assert len(api.seen) == n + 1 and "temperature" not in api.seen[-1]  # remembered: not tried again
    assert describe(llm)["not_sent"] == ["temperature"]
    off = OpenAIChat(api.url, "gpt-5.4-mini", api="openai")  # reasoning off by default: temperature 0 is kept
    run(off.chat([{"role": "user", "content": "yes or no?"}], temperature=0.0))
    assert api.seen[-1]["temperature"] == 0.0 and not off.dropped
    with pytest.warns(UserWarning, match="temperature"):
        run(off.chat([{"role": "user", "content": "think"}], temperature=0.0, reasoning_effort="low"))


def test_rate_limits_are_retried_but_an_empty_quota_is_not(api):
    api.limit = 2
    llm = OpenAIChat(api.url, "gpt-4.1-mini", api="openai")
    assert run(llm.chat([{"role": "user", "content": "hi"}])) == "Two people, please."
    api.quota = True
    with pytest.raises(urllib.error.HTTPError, match="exceeded your current quota"):
        run(llm.chat([{"role": "user", "content": "hi"}]))
    no_retry = OpenAIChat(api.url, "gpt-4.1-mini", api="openai", retries=0)
    api.quota, api.limit = False, 1
    with pytest.raises(urllib.error.HTTPError, match="Rate limit"):
        run(no_retry.chat([{"role": "user", "content": "hi"}]))


def test_hosted_tts_takes_preset_voices_only(api):
    tts = OpenAISpeech(api.url, "gpt-4o-mini-tts", sr=16000, api="openai")
    with pytest.warns(UserWarning, match="seed"):
        audio = run(tts.synth("Two people, please.", "coral", "calm", language="English", seed=7))
    assert audio.sr == 16000 and audio.dur_ms == 500  # raw 24 kHz PCM, resampled
    assert api.seen[-1]["instructions"] == "calm" and api.seen[-1]["voice"] == "coral"
    assert tts.usage == {"requests": 1, "characters": 19}
    with pytest.raises(ValueError, match="cannot clone"):
        run(tts.synth("again", ref_audio=audio))


def test_a_voice_on_hosted_tts_uses_the_preset_catalogue(api):
    tts = OpenAISpeech(api.url, "gpt-4o-mini-tts", api="openai")
    with pytest.raises(ValueError):  # cloning is the default, and this TTS cannot
        Voice(tts, voices=OPENAI_TTS_VOICES)
    Voice(tts, voices=OPENAI_TTS_VOICES, clone=False)
    assert pick_voice({"gender": "female", "age": 30}, OPENAI_TTS_VOICES)["speaker"] in {"coral", "nova", "sage", "shimmer"}
    assert pick_voice({"gender": "male", "age": 70}, OPENAI_TTS_VOICES)["speaker"] == "onyx"


def test_hosted_asr_and_tool_calls(api):
    asr = OpenAITranscribe(api.url, "gpt-4o-transcribe", api="openai")
    assert run(asr.transcribe(Audio.silence(1500, 16000))) == "hello there"
    assert asr.usage == {"requests": 1, "audio_s": 1.5}
    llm = ToolChat(api.url, "gpt-5-mini", api="openai", max_tokens=300)
    assert llm.native  # the hosted API renders no chat template: tools are always native
    tools = [{"type": "function", "function": {"name": "book", "parameters": {"type": "object", "properties": {}}}}]
    out = run(llm.complete([{"role": "user", "content": "book it"}], tools, seed=3))
    assert out["tool_calls"] == [{"name": "book", "arguments": {"n": 2}}] and out["completion_tokens"] == 30
    req = api.seen[-1]
    assert req["tool_choice"] == "auto" and req["max_completion_tokens"] == 2348 and req["seed"] == 3
    assert llm.describe()["provider"] == "openai"


def test_cached_hosted_calls_are_made_once(api):
    llm = Cached(OpenAIChat(api.url, "gpt-4.1-mini", api="openai"))
    msgs = [{"role": "user", "content": "hi"}]
    run(llm.chat(msgs))
    run(llm.chat(msgs))
    assert llm.inner.usage["requests"] == 1
