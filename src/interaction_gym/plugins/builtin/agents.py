"""Agents: how an external policy is attached to the controlled role. Each returns ``(seed) -> agent``; the driver
makes one agent per episode. Every agent row requires ``controlled`` (the ``role/controlled`` row).

The server adapters take their constructors' keyword arguments unchanged (``session``, ``clock``, ``voice``, ...):
their signatures are the contract, see ``agents/vllm_omni.py`` and ``agents/openai_realtime.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...agents import CannedAgent
from ..registry import plugin


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
    return lambda seed: CannedAgent(list(cfg.replies), spec, streaming=cfg.streaming, yield_after=cfg.yield_after,
                                    reply_after=cfg.reply_after, words_per_sec=cfg.words_per_sec, min_words=cfg.min_words)


@plugin("agent", "vllm_omni", requires=("controlled",))
def vllm_omni(ctx, kw: dict):
    """A full-duplex model served by vLLM-Omni (``VllmOmniDuplexAgent(spec, url, **kw)``)."""
    from ...agents.vllm_omni import VllmOmniDuplexAgent

    spec, kw = ctx["controlled"].spec, dict(kw)
    url = kw.pop("url", "ws://127.0.0.1:8010/v1/realtime?duplex=1")
    return lambda seed: VllmOmniDuplexAgent(spec, url, **kw)


@plugin("agent", "openai_realtime", requires=("controlled",))
def openai_realtime(ctx, kw: dict):
    """An OpenAI Realtime model (``OpenAIRealtimeAgent(spec, model, **kw)``): wall-clock, audio-paced, not reproducible."""
    from ...agents.openai_realtime import OpenAIRealtimeAgent

    spec, kw = ctx["controlled"].spec, dict(kw)
    model = kw.pop("model", "gpt-realtime-mini")
    return lambda seed: OpenAIRealtimeAgent(spec, model, **kw)


@plugin("agent", "cascaded", requires=("controlled",), refs=("asr", "llm", "tts"))
def cascaded(ctx, kw: dict):
    """VAD + ASR + LLM + TTS (``CascadedAgent``); ``asr`` / ``llm`` / ``tts`` name backend rows (llm: a backend/tool_chat)."""
    from ...agents.cascaded import CascadedAgent

    spec, kw = ctx["controlled"].spec, dict(kw)
    asr, llm, tts = (ctx[kw.pop(k)] for k in ("asr", "llm", "tts"))
    return lambda seed: CascadedAgent(spec, asr, llm, tts, seed=seed, **kw)
