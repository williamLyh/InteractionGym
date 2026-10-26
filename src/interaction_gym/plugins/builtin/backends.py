"""Backends: model clients that roles and agents name by row id. ``base_url`` is an OpenAI-compatible server; with
no ``base_url`` the hosted OpenAI API is used and the key comes from ``$OPENAI_API_KEY``. ``cached`` wraps the
client in ``clients.Cached`` (same input → same output within the process). ``seed: episode`` makes every chat call
carry the running episode's seed (what the runners did by building one client per episode), so a seed reproduces the
user's lines on a server that honours it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...clients import Cached, CloneWithFallback, FakeChat, FakeSpeech, OpenAIChat, OpenAISpeech, OpenAITranscribe
from ..registry import CURRENT_EPISODE, PluginError, plugin


class EpisodeSeeded:
    """A chat client whose calls carry the current episode's seed (``CURRENT_EPISODE``) plus ``offset`` unless the
    call gives one (an offset keeps two clients of one episode on different sampling streams, e.g. an examiner and the
    reference assistant it talks to)."""

    def __init__(self, inner, offset: int = 0):
        self.inner, self.offset = inner, offset

    async def chat(self, messages, **kw):
        ep = CURRENT_EPISODE.get()
        if ep is not None:
            kw.setdefault("seed", ep.seed + self.offset)
        return await self.inner.chat(messages, **kw)

    def describe(self) -> dict | None:
        """Inside an episode: the seed the calls carry (what a runner building one client per episode recorded);
        outside: ``"episode"`` (+ offset)."""
        from ...clients import describe

        d = describe(self.inner)
        if not d:
            return d
        ep = CURRENT_EPISODE.get()
        seed = ep.seed + self.offset if ep is not None else "episode" + (f"+{self.offset}" if self.offset else "")
        return {**d, "params": {**d.get("params", {}), "seed": seed}}

    def __getattr__(self, name):
        return getattr(self.inner, name)


@dataclass
class ChatConfig:
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api: str | None = None  # "openai" | "vllm" | None (by host)
    timeout: float = 120
    retries: int | None = None
    reasoning_tokens: int = 2048
    defaults: dict = field(default_factory=dict)  # request fields sent with every call: temperature, max_tokens, ...
    seed: str | int | None = None  # "episode": the running episode's seed; an int: fixed; None: unseeded
    seed_offset: int = 0  # with seed: episode, added to the episode's seed
    cached: bool = False


def _seeded(client, seed, cached: bool, offset: int = 0):
    if isinstance(seed, int) and not isinstance(seed, bool):
        client.defaults["seed"] = seed + offset
    elif seed == "episode":
        client = EpisodeSeeded(client, offset)
    elif seed is not None:
        raise PluginError(f"backend: seed is 'episode', an int or null, got {seed!r}")
    return Cached(client) if cached else client


@plugin("backend", "chat", config=ChatConfig)
def chat(ctx, cfg: ChatConfig):
    """A chat-completions LLM (``OpenAIChat``): the user simulator's writer, listener, judges."""
    kw = dict(api_key_env=cfg.api_key_env, timeout=cfg.timeout, retries=cfg.retries, reasoning_tokens=cfg.reasoning_tokens, **cfg.defaults)
    client = OpenAIChat.openai(cfg.model, **kw) if cfg.base_url is None else OpenAIChat(cfg.base_url, cfg.model, api=cfg.api, **kw)
    return _seeded(client, cfg.seed, cfg.cached, cfg.seed_offset)


@dataclass
class ToolChatConfig(ChatConfig):
    native: bool = False  # native tool calling (forced on the hosted API) or prompted


@plugin("backend", "tool_chat", config=ToolChatConfig)
def tool_chat(ctx, cfg: ToolChatConfig):
    """A chat LLM with tool calling (``agents.cascaded.ToolChat``), for the cascaded agent."""
    from ...agents.cascaded import ToolChat

    base = cfg.base_url or "https://api.openai.com/v1"
    client = ToolChat(base, cfg.model, cfg.native, api_key_env=cfg.api_key_env, api=cfg.api, retries=cfg.retries, timeout=cfg.timeout,
                      reasoning_tokens=cfg.reasoning_tokens, **cfg.defaults)
    if cfg.seed == "episode":
        raise PluginError("backend/tool_chat: the cascaded agent seeds its own calls per episode; use seed: null")
    return _seeded(client, cfg.seed, cfg.cached)


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
    return Cached(client) if cfg.cached else client


@dataclass
class CloneFallbackConfig:
    clone: str  # row id of the cloning TTS
    plain: str  # row id of the preset-voice TTS used when cloning fails
    voice: str = "ryan"  # the plain TTS's speaker


@plugin("backend", "clone_fallback", config=CloneFallbackConfig, refs=("clone", "plain"))
def clone_fallback(ctx, cfg: CloneFallbackConfig) -> CloneWithFallback:
    """Voice cloning with a fallback: a line the cloning TTS refuses is retried with a final period, then spoken by
    the plain TTS in a preset voice (so one bad line does not lose the episode)."""
    return CloneWithFallback(ctx[cfg.clone], ctx[cfg.plain], cfg.voice)


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


class EpisodeFakeChat(FakeChat):
    """``FakeChat`` whose reply cursor restarts for every episode (``CURRENT_EPISODE``): what a script that built one
    client per episode saw, with one shared row."""

    def __init__(self, replies):
        super().__init__(replies)
        self._cursors: dict[int, list] = {}  # id(episode) -> [episode, calls so far]

    async def chat(self, messages, **kw):
        ep = CURRENT_EPISODE.get()
        if ep is None:
            return await super().chat(messages, **kw)
        slot = self._cursors.setdefault(id(ep), [ep, 0])
        slot[1] += 1
        self.calls.append(messages)
        return self.replies[min(slot[1], len(self.replies)) - 1]


@dataclass
class FakeChatConfig:
    replies: list
    seed: str | int | None = None
    seed_offset: int = 0
    per_episode: bool = False  # the reply sequence restarts for every episode


@plugin("backend", "fake_chat", config=FakeChatConfig)
def fake_chat(ctx, cfg: FakeChatConfig):
    """Fixed replies in order, no server: tests and dry runs."""
    client = EpisodeFakeChat(list(cfg.replies)) if cfg.per_episode else FakeChat(list(cfg.replies))
    return _seeded(client, cfg.seed, False, cfg.seed_offset) if cfg.seed == "episode" else client


@dataclass
class FakeSpeechConfig:
    words_per_sec: float = 3.4
    sr: int = 16000


@plugin("backend", "tts_fake", config=FakeSpeechConfig)
def tts_fake(ctx, cfg: FakeSpeechConfig):
    """Silence of the length the words would take, no server: tests and dry runs."""
    return FakeSpeech(cfg.words_per_sec, cfg.sr)
