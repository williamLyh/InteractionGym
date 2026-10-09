"""Agents the env can step: scripted stand-ins, and adapters to model servers.

- ``CannedAgent``: scripted replies (no model), for tests and demos.
- ``VllmOmniDuplexAgent`` (``agents.vllm_omni``): a full-duplex model served by vLLM-Omni.

An agent has ``act(t, obs) -> frames`` (async for server adapters); one adapter per server protocol.
"""

from .canned import CannedAgent

__all__ = ["CannedAgent", "speech_rate"]


def speech_rate(episodes: list[dict], role: str = "agent") -> float:
    """Speaking rate in characters per second, pooled over the complete turns (not cut, with audio)
    of ``role`` in schema-v1 episodes — to time a text-only agent like the real one sounded."""
    chars = ms = 0
    for ep in episodes:
        for t in ep["turns"]:
            if t["role"] == role and t["text"] and "unsaid" not in t and "media" in t:
                chars += len(t["text"])
                ms += t["end_time"] - t["start_time"]
    if not ms:
        raise ValueError(f"no complete {role} turns with audio to calibrate from")
    return chars / ms * 1000
