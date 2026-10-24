"""Backends: model clients that roles and agents name by row id. ``base_url`` is an OpenAI-compatible server; with
no ``base_url`` the hosted OpenAI API is used and the key comes from ``$OPENAI_API_KEY``. ``cached`` wraps the
client in ``clients.Cached`` (same input → same output within the process).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...clients import Cached, FakeChat, FakeSpeech, OpenAIChat, OpenAISpeech, OpenAITranscribe
from ..registry import plugin


@dataclass
class ChatConfig:
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api: str | None = None  # "openai" | "vllm" | None (by host)
    timeout: float = 120
    retries: int | None = None
    reasoning_tokens: int = 2048
    cached: bool = False


def _wrap(client, cached: bool):
    return Cached(client) if cached else client


@plugin("backend", "chat", config=ChatConfig)
def chat(ctx, cfg: ChatConfig):
    """A chat-completions LLM (``OpenAIChat``): the user simulator's writer, listener, judges."""
    kw = dict(api_key_env=cfg.api_key_env, timeout=cfg.timeout, retries=cfg.retries, reasoning_tokens=cfg.reasoning_tokens)
    client = OpenAIChat.openai(cfg.model, **kw) if cfg.base_url is None else OpenAIChat(cfg.base_url, cfg.model, api=cfg.api, **kw)
    return _wrap(client, cfg.cached)


@plugin("backend", "tool_chat", config=ChatConfig)
def tool_chat(ctx, cfg: ChatConfig):
    """A chat LLM with tool calling (``agents.cascaded.ToolChat``), for the cascaded agent."""
    from ...agents.cascaded import ToolChat

    base = cfg.base_url or "https://api.openai.com/v1"
    return ToolChat(base, cfg.model, api_key_env=cfg.api_key_env, api=cfg.api, retries=cfg.retries, timeout=cfg.timeout,
                    reasoning_tokens=cfg.reasoning_tokens)


@dataclass
class SpeechConfig:
    model: str = "gpt-4o-mini-tts"
    base_url: str | None = None
    sr: int = 24000
    api_key_env: str | None = None
    api: str | None = None
    cached: bool = False


@plugin("backend", "tts", config=SpeechConfig)
def tts(ctx, cfg: SpeechConfig):
    """A text-to-speech backend (``OpenAISpeech``): voices roles; a cloning model (Qwen3-TTS Base) can be the ``clone``."""
    client = OpenAISpeech.openai(cfg.model, sr=cfg.sr, api_key_env=cfg.api_key_env) if cfg.base_url is None \
        else OpenAISpeech(cfg.base_url, cfg.model, sr=cfg.sr, api_key_env=cfg.api_key_env, api=cfg.api)
    return _wrap(client, cfg.cached)


@dataclass
class TranscribeConfig:
    model: str = "gpt-4o-transcribe"
    base_url: str | None = None
    api_key_env: str | None = None
    language: str | None = None
    api: str | None = None


@plugin("backend", "asr", config=TranscribeConfig)
def asr(ctx, cfg: TranscribeConfig):
    """A speech-to-text backend (``OpenAITranscribe``), for the cascaded agent."""
    if cfg.base_url is None:
        return OpenAITranscribe.openai(cfg.model, api_key_env=cfg.api_key_env, language=cfg.language)
    return OpenAITranscribe(cfg.base_url, cfg.model, api_key_env=cfg.api_key_env, language=cfg.language, api=cfg.api)


@dataclass
class FakeChatConfig:
    replies: list


@plugin("backend", "fake_chat", config=FakeChatConfig)
def fake_chat(ctx, cfg: FakeChatConfig):
    """Fixed replies in order, no server: tests and dry runs."""
    return FakeChat(list(cfg.replies))


@dataclass
class FakeSpeechConfig:
    words_per_sec: float = 3.4
    sr: int = 16000


@plugin("backend", "tts_fake", config=FakeSpeechConfig)
def tts_fake(ctx, cfg: FakeSpeechConfig):
    """Silence of the length the words would take, no server: tests and dry runs."""
    return FakeSpeech(cfg.words_per_sec, cfg.sr)
