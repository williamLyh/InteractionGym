"""Model clients used by simulated users and judges.

The environment never hosts models: text generation and speech synthesis are
separate services reached through small interfaces. Real clients speak the
OpenAI-compatible HTTP APIs (vLLM for chat, any server exposing
``/v1/audio/speech`` for TTS); fakes make everything runnable offline; ``Cached``
wraps either for reproducible replays and to avoid re-synthesizing repeated text.

**The OpenAI API itself.** The same clients reach ``api.openai.com`` (``OpenAIChat.openai("gpt-5-mini")``, or a
``base_url`` on that host, e.g. ``IG_LLM_URL=https://api.openai.com/v1``). Pointed there they switch to the hosted
API (``api="openai"``; pass it explicitly for a proxy of the same API):

- the key comes from ``$OPENAI_API_KEY`` unless ``api_key_env`` names another variable; it is never recorded;
- rate limits (429, except an exhausted quota), 5xx and dropped connections are retried with backoff (``retries``,
  default 6), honouring ``Retry-After``;
- request fields only local servers take (``chat_template_kwargs``, ``top_k``, …; TTS ``seed`` / ``language``) are
  not sent, ``max_tokens`` becomes ``max_completion_tokens``, and reasoning models (o-series, gpt-5*) get
  ``reasoning_tokens`` of headroom on top of it — a 16-token decision would otherwise be spent on reasoning; a field
  the API refuses for a model (``temperature`` while it reasons) is dropped and the request sent again. What was not
  sent is recorded by ``describe`` (a listener asked for temperature 0 may then sample at the default);
- ``usage`` counts requests and tokens (TTS: characters, ASR: seconds of audio) for every client, local or hosted.

OpenAI's TTS cannot clone a voice: with it a ``Voice`` uses preset speakers (``user.OPENAI_TTS_VOICES``,
``clone=False``), or keeps a local cloning TTS for every turn after the first.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
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


def _send(url: str, data: bytes, headers: dict, timeout: float = 120, retries: int = 0) -> bytes:
    """POST ``data``. A server error (5xx), a rate limit (429, unless the quota is used up) or a dropped connection is
    retried ``retries`` times, after ``Retry-After`` if the server says, else 1, 2, 4 … (at most 30) s. A request
    error (4xx) is raised with the server's explanation in its message."""
    for k in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data, headers), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            body = e.read()[:2000]
            retry = e.code >= 500 or (e.code == 429 and b"insufficient_quota" not in body)
            if not retry or k == retries:
                e.msg = f"{e.msg}: {body.decode('utf-8', 'replace')}" if body else e.msg
                raise e
            after = e.headers.get("Retry-After") if e.headers else None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == retries:
                raise
            after = None
        try:
            delay = float(after) if after is not None else min(30.0, 2.0 ** k)
        except ValueError:
            delay = min(30.0, 2.0 ** k)
        time.sleep(delay)
    raise AssertionError("unreachable")


def _post(url: str, payload: dict, api_key: str | None, timeout: float = 120, retries: int = 0) -> bytes:
    """POST JSON (``_send``)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return _send(url, json.dumps(payload).encode(), headers, timeout, retries)


OPENAI_URL = "https://api.openai.com/v1"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
HOSTED_RETRIES = 6
# Request fields that vLLM / SGLang take and the OpenAI API refuses ("Unrecognized request argument").
LOCAL_ONLY_PARAMS = {"chat_template_kwargs", "top_k", "min_p", "repetition_penalty", "skip_special_tokens",
                     "spaces_between_special_tokens", "include_stop_str_in_output", "ignore_eos", "min_tokens",
                     "guided_json", "guided_regex", "guided_choice", "guided_grammar", "structured_outputs",
                     "truncate_prompt_tokens", "extra_body"}
_REASONING = re.compile(r"^(o\d|gpt-5)(?!.*-chat)")


def api_of(base_url: str, api: str | None = None) -> str:
    """``"openai"`` for the hosted OpenAI API (its host, or ``api="openai"`` for a proxy of it), else ``"compat"``
    (a local OpenAI-compatible server)."""
    if api is not None:
        assert api in ("openai", "compat"), api
        return api
    return "openai" if urllib.parse.urlparse(base_url).hostname == "api.openai.com" else "compat"


def is_reasoning(model: str) -> bool:
    """OpenAI reasoning models (o-series, gpt-5 but not its ``-chat`` variants): no temperature, and their reasoning
    tokens count against ``max_completion_tokens``."""
    return bool(_REASONING.match(model))


def _client_http(self, base_url: str, api: str | None, api_key_env: str | None, retries: int | None) -> None:
    """The connection settings every client shares: ``api``, ``api_key``, ``retries``, ``usage``, ``dropped``."""
    self.api = api_of(base_url, api)
    hosted = self.api == "openai"
    env = api_key_env or (OPENAI_KEY_ENV if hosted else None)
    self.api_key = os.environ.get(env) if env else None
    if hosted and not self.api_key:
        raise RuntimeError(f"{type(self).__name__}: the OpenAI API needs a key in ${env}")
    self.retries = (HOSTED_RETRIES if hosted else 0) if retries is None else retries
    self.usage: dict[str, int | float] = {"requests": 0}
    self.dropped: set[str] = set()


def _count(usage: dict, **n) -> None:
    usage["requests"] += 1
    for k, v in n.items():
        if v:
            usage[k] = usage.get(k, 0) + v


def _count_tokens(usage: dict, u: dict | None) -> None:
    u = u or {}
    _count(usage, prompt_tokens=u.get("prompt_tokens", 0), completion_tokens=u.get("completion_tokens", 0),
           reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))


def _drop(client, payload: dict, keys) -> None:
    for k in [k for k in keys if k in payload]:
        payload.pop(k)
        if k not in client.dropped:
            client.dropped.add(k)
            warnings.warn(f"{type(client).__name__}({client.model}): the OpenAI API does not take {k!r}; not sent", stacklevel=3)


def openai_chat_payload(client, payload: dict, reasoning_tokens: int) -> dict:
    """A chat-completions request as the OpenAI API takes it (see the module docstring): no local-server fields nor
    any field the API already refused for this client, ``max_completion_tokens``, and reasoning headroom."""
    payload = dict(payload)
    _drop(client, payload, LOCAL_ONLY_PARAMS | client.dropped)
    if "max_tokens" in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
    if is_reasoning(payload["model"]) and "max_completion_tokens" in payload:
        payload["max_completion_tokens"] += reasoning_tokens
    return payload


_UNSUPPORTED = re.compile(r"Unsupported (?:value|parameter): '(\w+)'")


def openai_post(client, payload: dict, timeout: float) -> dict:
    """POST a chat request to the OpenAI API. A field it refuses for this model (e.g. ``temperature`` once a reasoning
    model reasons: gpt-5-mini always, gpt-5.4-mini only with a ``reasoning_effort``) is dropped and the request sent
    again; the client remembers it and does not send it again."""
    while True:
        try:
            return json.loads(_post(client.url, payload, client.api_key, timeout, client.retries))
        except urllib.error.HTTPError as e:
            m = _UNSUPPORTED.search(e.msg or "") if e.code == 400 else None
            if not m or m.group(1) not in payload or m.group(1) in ("model", "messages"):
                raise
            payload = dict(payload)
            _drop(client, payload, [m.group(1)])


class OpenAIChat:
    """Chat completions against an OpenAI-compatible endpoint (e.g. a vLLM server), or the OpenAI API (``openai``)."""

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, timeout: float = 120, *,
                 api: str | None = None, retries: int | None = None, reasoning_tokens: int = 2048, **defaults):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        _client_http(self, base_url, api, api_key_env, retries)
        self.timeout = timeout
        self.reasoning_tokens = reasoning_tokens
        self.defaults = defaults

    @classmethod
    def openai(cls, model: str, **kw) -> "OpenAIChat":
        return cls(OPENAI_URL, model, **kw)

    async def chat(self, messages: Messages, **kw) -> str:
        """The reply's text, without any reasoning: servers with a reasoning parser put it in
        ``reasoning_content`` (and may return no ``content`` at all if it used up ``max_tokens``);
        servers without one leave ``<think>…</think>`` in the text — an unterminated one was cut off
        mid-thought and says nothing."""
        payload = {"model": self.model, "messages": messages, **self.defaults, **kw}
        if self.api == "openai":
            out = await asyncio.to_thread(openai_post, self, openai_chat_payload(self, payload, self.reasoning_tokens), self.timeout)
        else:
            http = (self.timeout,) + ((self.retries,) if self.retries else ())
            out = json.loads(await asyncio.to_thread(_post, self.url, payload, self.api_key, *http))
        _count_tokens(self.usage, out.get("usage"))
        text = out["choices"][0]["message"].get("content") or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return text.split("<think>")[0].strip()


class OpenAISpeech:
    """TTS against an OpenAI-compatible ``/audio/speech`` endpoint, or the OpenAI API (``openai``). Audio comes back
    at ``sr``: WAV responses say their own rate and are resampled; raw PCM is taken to be at ``native_sr`` (default
    ``sr``). ``extra`` adds model-specific request fields. The OpenAI API is asked for raw PCM (24 kHz) and gets no
    ``seed`` or ``language``; it cannot clone (``ref_audio`` raises)."""

    def __init__(self, base_url: str, model: str, sr: int = 24000, api_key_env: str | None = None, *,
                 response_format: str | None = None, native_sr: int | None = None, extra: dict | None = None,
                 api: str | None = None, retries: int | None = None):
        self.url = base_url.rstrip("/") + "/audio/speech"
        self.model = model
        self.sr = sr
        _client_http(self, base_url, api, api_key_env, retries)
        if self.api == "openai" and response_format is None:
            response_format, native_sr = "pcm", native_sr or 24000
        self.response_format, self.native_sr, self.extra = response_format or "wav", native_sr, extra or {}

    @classmethod
    def openai(cls, model: str = "gpt-4o-mini-tts", **kw) -> "OpenAISpeech":
        return cls(OPENAI_URL, model, **kw)

    async def synth(self, text: str, voice: str = "default", instructions: str | None = None,
                    ref_audio: Audio | None = None, ref_text: str | None = None, language: str | None = None,
                    seed: int | None = None) -> Audio:
        payload = {"model": self.model, "input": text, "voice": voice, "response_format": self.response_format, **self.extra}
        hosted = self.api == "openai"
        if hosted and ref_audio is not None:
            raise ValueError(f"OpenAISpeech({self.model}): the OpenAI API cannot clone a voice; use preset voices "
                             "(Voice(..., voices=OPENAI_TTS_VOICES, clone=False)) or a local cloning TTS as clone=")
        if language:
            payload["language"] = language
        if seed is not None:  # sampling seed (vLLM-Omni's /audio/speech takes one)
            payload["seed"] = seed
        if instructions and not (hosted and self.model.startswith("tts-1")):  # how to say it (style, emotion, pace)
            payload["instructions"] = instructions
        if hosted:
            _drop(self, payload, ("language", "seed"))
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
        http = (120, self.retries) if self.retries else ()
        for k in range(3):  # a server error is retried; a seeded TTS that ran away (no end-of-speech) would again: new seed
            try:
                raw = await asyncio.to_thread(_post, self.url, payload, self.api_key, *http)
                break
            except urllib.error.HTTPError as e:
                if e.code < 500 or k == 2:
                    raise
                if "seed" in payload:
                    payload["seed"] += 1
        _count(self.usage, characters=len(text))
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
    Qwen3-ASR or Whisper), or the OpenAI API (``openai``). Returns the text."""

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, language: str | None = None, *,
                 api: str | None = None, retries: int | None = None):
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.model = model
        _client_http(self, base_url, api, api_key_env, retries)
        self.language = language

    @classmethod
    def openai(cls, model: str = "gpt-4o-transcribe", **kw) -> "OpenAITranscribe":
        return cls(OPENAI_URL, model, **kw)

    def _post(self, audio: Audio, language: str | None) -> bytes:
        boundary = "----dig" + hashlib.sha256(audio.samples.tobytes()[:4096]).hexdigest()[:16]
        fields = {"model": self.model, **({"language": language} if language else {})}
        body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields.items())
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n"
                 "Content-Type: audio/wav\r\n\r\n").encode() + _wav_bytes(audio) + f"\r\n--{boundary}--\r\n".encode()
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return _send(self.url, body, headers, 120, self.retries)

    async def transcribe(self, audio: Audio, language: str | None = None) -> str:
        raw = await asyncio.to_thread(self._post, audio, language or self.language)
        _count(self.usage, audio_s=round(audio.dur_ms / 1000, 3))
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


class CloneWithFallback:
    """A cloning TTS with a safety net: a line the cloning model refuses is retried with a final period, then spoken
    by ``plain`` in a preset ``voice`` (one bad line does not lose the episode). Used by the benchmark runners."""

    clones = True

    def __init__(self, clone, plain, voice: str = "ryan"):
        self.inner, self.plain, self.voice = clone, plain, voice
        self.model, self.sr = getattr(clone, "model", type(clone).__name__), clone.sr

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None, **kw):
        for t in (text, text.rstrip(".!?") + "."):
            try:
                return await self.inner.synth(t, voice, instructions, ref_audio, ref_text, language, **kw)
            except Exception:  # noqa: BLE001 — the cloning server refused this line; try the next form, then the plain voice
                continue
        return await self.plain.synth(text, self.voice, None, language=language)


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
    """What a trajectory records about a model client: its model name (the class name for fakes), its provider
    when hosted, sampling defaults (and those the hosted API did not take) and output rate. Wrappers exposing ``inner`` (e.g. ``Cached``) are looked through."""
    if client is None:
        return None
    while hasattr(client, "inner"):
        client = client.inner
    out: dict = {"model": getattr(client, "model", type(client).__name__)}
    if getattr(client, "api", None) == "openai":
        out["provider"] = "openai"
    if getattr(client, "defaults", None):
        out["params"] = dict(client.defaults)
    if getattr(client, "dropped", None):
        out["not_sent"] = sorted(client.dropped)  # request fields the hosted API does not take
    if hasattr(client, "sr"):
        out["sr"] = client.sr
    return out
