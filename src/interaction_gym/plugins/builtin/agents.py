"""Agents: how an external policy is attached to the controlled role. Each returns a factory ``(EpisodeContext) ->
agent``; the driver makes one agent per episode. Every agent row requires ``controlled`` (the ``role/controlled`` row).

The server adapters take their constructors' keyword arguments unchanged (``session``, ``clock``, ``voice``, ...):
their signatures are the contract, see ``agents/vllm_omni.py`` and ``agents/openai_realtime.py``. A factory may
carry ``url`` (the driver warms up one session per server behind it) and ``audio_out`` (the driver refuses to resume
a run of the other output mode).
"""

from __future__ import annotations

from dataclasses import dataclass

from ...agents import CannedAgent
from ..registry import EpisodeContext, plugin


class Factory:
    """``(EpisodeContext) -> agent`` with attributes the driver may read (``url``, ``audio_out``)."""

    def __init__(self, make, **attrs):
        self._make = make
        self.__dict__.update(attrs)

    def __call__(self, ep: EpisodeContext):
        return self._make(ep)


@dataclass
class CannedConfig:
    replies: list
    streaming: bool = False
    yield_after: int | None = 160
    reply_after: int = 300
    words_per_sec: float = 3.4
    min_words: int = 0


@plugin("agent", "canned", requires=("controlled",), config=CannedConfig)
def canned(ctx, cfg: CannedConfig):
    """Fixed replies in order; yields after ``yield_after`` ms of being talked over (None: never)."""
    spec = ctx["controlled"].spec
    return Factory(lambda ep: CannedAgent(list(cfg.replies), spec, streaming=cfg.streaming, yield_after=cfg.yield_after,
                                          reply_after=cfg.reply_after, words_per_sec=cfg.words_per_sec, min_words=cfg.min_words))


@plugin("agent", "vllm_omni", requires=("controlled",))
def vllm_omni(ctx, kw: dict):
    """A full-duplex model served by vLLM-Omni (``VllmOmniDuplexAgent(spec, url, **kw)``). ``preset`` (``minicpmo``,
    ``gander``, ``personaplex``, ...) fills the model's settings (``agents.vllm_omni.duplex_model``), explicit keys
    win; ``instructions`` is the session's system prompt."""
    from ...agents.vllm_omni import VllmOmniDuplexAgent, duplex_model

    spec, kw = ctx["controlled"].spec, dict(kw)
    url = kw.pop("url", "ws://127.0.0.1:8010/v1/realtime?duplex=1")
    preset = kw.pop("preset", None)
    if preset:
        kw = {**duplex_model(preset), **kw}
    instructions = kw.pop("instructions", None)
    if instructions is not None:
        kw["session"] = {**kw.get("session", {}), "instructions": instructions}
    return Factory(lambda ep: VllmOmniDuplexAgent(spec, url, **kw), url=url, audio_out=kw.get("audio_out", True), preset=preset)


@plugin("agent", "openai_realtime", requires=("controlled",))
def openai_realtime(ctx, kw: dict):
    """An OpenAI Realtime model (``OpenAIRealtimeAgent(spec, model, **kw)``): wall-clock, audio-paced, not reproducible."""
    from ...agents.openai_realtime import OpenAIRealtimeAgent

    spec, kw = ctx["controlled"].spec, dict(kw)
    model = kw.pop("model", "gpt-realtime-mini")
    return Factory(lambda ep: OpenAIRealtimeAgent(spec, model, **kw))


@plugin("agent", "cascaded", requires=("controlled",), refs=("asr", "llm", "tts"))
def cascaded(ctx, kw: dict):
    """VAD + ASR + LLM + TTS (``CascadedAgent``); ``asr`` / ``llm`` / ``tts`` name backend rows (llm: a
    backend/tool_chat); ``endpointing`` / ``latency`` are dicts of ``Endpointing`` / ``LatencyModel`` fields. The
    episode's seed seeds it and the arms of one (task, seed) share its model-call cache."""
    from ...agents.cascaded import CascadedAgent, Endpointing, LatencyModel

    spec, kw = ctx["controlled"].spec, dict(kw)
    asr, llm, tts = (ctx[kw.pop(k)] for k in ("asr", "llm", "tts"))
    if isinstance(kw.get("endpointing"), dict):
        kw["endpointing"] = Endpointing(**kw["endpointing"])
    if isinstance(kw.get("latency"), dict):
        kw["latency"] = LatencyModel(**kw["latency"])
    return Factory(lambda ep: CascadedAgent(spec, asr, llm, tts, seed=ep.seed, cache=ep.cache, **kw))
