"""Model clients used by simulated users and judges.

The environment never hosts models: text generation and speech synthesis are
separate services reached through small interfaces. Real clients speak the
OpenAI-compatible HTTP APIs (vLLM for chat, any server exposing
``/v1/audio/speech`` for TTS); fakes make everything runnable offline; ``Cached``
wraps either for reproducible replays and to avoid re-synthesizing repeated text.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import urllib.request
import wave
from typing import Callable, Protocol

from .audio import Audio

Messages = list[dict]


class TextGen(Protocol):
    async def chat(self, messages: Messages, **kw) -> str: ...


class Speech(Protocol):
    async def synth(self, text: str, voice: str = "default", instructions: str | None = None,
                    ref_audio: Audio | None = None, ref_text: str | None = None, language: str | None = None,
                    seed: int | None = None) -> Audio: ...


def _post(url: str, payload: dict, api_key: str | None) -> bytes:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


class OpenAIChat:
    """Chat completions against an OpenAI-compatible endpoint (e.g. a vLLM server)."""

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, **defaults):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.defaults = defaults

    async def chat(self, messages: Messages, **kw) -> str:
        """The reply's text, without any reasoning: servers with a reasoning parser put it in
        ``reasoning_content`` (and may return no ``content`` at all if it used up ``max_tokens``);
        servers without one leave ``<think>…</think>`` in the text — an unterminated one was cut off
        mid-thought and says nothing."""
        payload = {"model": self.model, "messages": messages, **self.defaults, **kw}
        raw = await asyncio.to_thread(_post, self.url, payload, self.api_key)
        text = json.loads(raw)["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return text.split("<think>")[0].strip()


class OpenAISpeech:
    """TTS against an OpenAI-compatible ``/audio/speech`` endpoint. Audio comes back at ``sr``: WAV
    responses say their own rate and are resampled; raw PCM is taken to be at ``native_sr`` (default
    ``sr``). ``extra`` adds model-specific request fields."""

    def __init__(self, base_url: str, model: str, sr: int = 24000, api_key_env: str | None = None, *,
                 response_format: str = "wav", native_sr: int | None = None, extra: dict | None = None):
        self.url = base_url.rstrip("/") + "/audio/speech"
        self.model = model
        self.sr = sr
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.response_format, self.native_sr, self.extra = response_format, native_sr, extra or {}

    async def synth(self, text: str, voice: str = "default", instructions: str | None = None,
                    ref_audio: Audio | None = None, ref_text: str | None = None, language: str | None = None,
                    seed: int | None = None) -> Audio:
        payload = {"model": self.model, "input": text, "voice": voice, "response_format": self.response_format, **self.extra}
        if language:
            payload["language"] = language
        if seed is not None:  # sampling seed (vLLM-Omni's /audio/speech takes one)
            payload["seed"] = seed
        if instructions:  # how to say it (style, emotion, pace), for TTS models that take it
            payload["instructions"] = instructions
        if ref_audio is not None:  # voice cloning (vLLM-Omni Qwen3-TTS "Base"): sound like this reference
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(ref_audio.sr)
                w.writeframes(ref_audio.samples.tobytes())
            payload.update(task_type="Base", ref_audio="data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode())
            payload.pop("voice")
            if ref_text:
                payload["ref_text"] = ref_text
        raw = await asyncio.to_thread(_post, self.url, payload, self.api_key)
        if raw[:4] == b"RIFF":  # a WAV says its own rate (some TTS models are 16, 22.05, 44.1 or 48 kHz)
            with wave.open(io.BytesIO(raw)) as w:
                audio = Audio.from_pcm16(w.readframes(w.getnframes()), w.getframerate())
        else:
            audio = Audio.from_pcm16(raw, self.native_sr or self.sr)
        return audio.resample(self.sr)


class Transcriber(Protocol):
    async def transcribe(self, audio: Audio, language: str | None = None) -> str: ...


def _wav_bytes(audio: Audio) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sr)
        w.writeframes(audio.samples.tobytes())
    return buf.getvalue()


class OpenAITranscribe:
    """Speech recognition against an OpenAI-compatible ``/audio/transcriptions`` endpoint (e.g. vLLM serving
    Qwen3-ASR or Whisper). Returns the text."""

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, language: str | None = None):
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.model = model
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.language = language

    def _post(self, audio: Audio, language: str | None) -> bytes:
        boundary = "----dig" + hashlib.sha256(audio.samples.tobytes()[:4096]).hexdigest()[:16]
        fields = {"model": self.model, **({"language": language} if language else {})}
        body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields.items())
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n").encode() + _wav_bytes(audio) + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with urllib.request.urlopen(urllib.request.Request(self.url, body, headers), timeout=120) as r:
            return r.read()

    async def transcribe(self, audio: Audio, language: str | None = None) -> str:
        raw = await asyncio.to_thread(self._post, audio, language or self.language)
        return json.loads(raw).get("text", "").strip()


class FakeChat:
    """Replies from a list (in order) or a function of the messages; records every call."""

    def __init__(self, replies: list[str] | Callable[[Messages], str]):
        self.replies = replies
        self.calls: list[Messages] = []

    async def chat(self, messages: Messages, **kw) -> str:
        self.calls.append(messages)
        if callable(self.replies):
            return self.replies(messages)
        return self.replies[min(len(self.calls), len(self.replies)) - 1]


class FakeSpeech:
    """Silence whose length follows a speaking rate, so timing behaves like real TTS. It "clones" too (ignores
    ``ref_audio``), so it can be a ``Voice``'s own clone TTS."""

    clones = True

    def __init__(self, words_per_sec: float = 3.4, sr: int = 16000):
        self.wps = words_per_sec
        self.sr = sr
        self.calls = 0

    async def synth(self, text: str, voice: str = "default", instructions: str | None = None,
                    ref_audio: Audio | None = None, ref_text: str | None = None, language: str | None = None,
                    seed: int | None = None) -> Audio:
        self.calls += 1
        return Audio.silence(round(len(text.split()) / self.wps * 1000), self.sr)


class Cached:
    """Memoizes ``chat`` / ``synth`` of a wrapped client by their arguments."""

    def __init__(self, inner):
        self.inner = inner
        self.store: dict[str, object] = {}

    @staticmethod
    def _key(*args, **kw) -> str:
        return hashlib.sha256(json.dumps([args, kw], sort_keys=True, default=str).encode()).hexdigest()

    async def chat(self, messages: Messages, **kw) -> str:
        k = self._key("chat", messages, **kw)
        if k not in self.store:
            self.store[k] = await self.inner.chat(messages, **kw)
        return self.store[k]

    async def synth(self, text: str, voice: str = "default", instructions: str | None = None,
                    ref_audio: Audio | None = None, ref_text: str | None = None, language: str | None = None,
                    seed: int | None = None) -> Audio:
        ref = None if ref_audio is None else hashlib.sha256(ref_audio.samples.tobytes()).hexdigest()
        kw = {"seed": seed} if seed is not None else {}
        k = self._key("synth", text, voice, instructions, ref, ref_text, language, **kw)
        if k not in self.store:
            self.store[k] = await self.inner.synth(text, voice, instructions, ref_audio, ref_text, language, **kw)
        return self.store[k]


def describe(client) -> dict | None:
    """What a trajectory records about a model client: its model name (the class name for fakes),
    sampling defaults and output rate. Wrappers exposing ``inner`` (e.g. ``Cached``) are looked through."""
    if client is None:
        return None
    while hasattr(client, "inner"):
        client = client.inner
    out: dict = {"model": getattr(client, "model", type(client).__name__)}
    if getattr(client, "defaults", None):
        out["params"] = dict(client.defaults)
    if hasattr(client, "sr"):
        out["sr"] = client.sr
    return out
