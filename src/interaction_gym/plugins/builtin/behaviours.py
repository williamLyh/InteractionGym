"""Behaviours: what a simulated role does besides its turns (``interaction_gym.behaviours``). A ``role/user`` row
names them in order (``behaviours: [noise, asides]``); the default library is both; ``[]`` is a user that only
speaks. Their rates still come from the resolved ``Behaviors`` (``role/user`` ``behaviors`` and the persona's
surroundings); a behaviour plugin of your own implements ``behaviours.Behaviour``."""

from __future__ import annotations

from dataclasses import dataclass

from ...behaviours import Asides, Listening, NoiseEvents
from ..registry import plugin


@plugin("behaviour", "noise")
def noise(ctx, kw: dict) -> NoiseEvents:
    """Noise events on the call clock (cough, door, phone, third-party speech): turns of kind ``noise``, to be ignored."""
    return NoiseEvents()


@plugin("behaviour", "asides")
def asides(ctx, kw: dict) -> Asides:
    """Being addressed by someone nearby: a remark to them (``aside``), or being called away ("hold on", the remark,
    an absence, "I'm back")."""
    return Asides()


@dataclass
class ListeningConfig:
    listener: str | None = None  # row id of a backend/chat: the model deciding LISTEN / BACKCHANNEL / INTERRUPT at the agent's pauses
    min_words: int = 3  # no decision before this many words of the agent's turn were heard
    barge_in: bool = True  # the model may INTERRUPT (else it only listens and backchannels)


@plugin("behaviour", "listening", config=ListeningConfig, refs=("listener",))
def listening(ctx, cfg: ListeningConfig) -> Listening:
    """What the user does while the agent talks: at each of the agent's pauses the listener model chooses to listen,
    backchannel or barge in; the calibration barge-ins (Behaviors.barge_in_per_min / per_turn) force cut-ins."""
    from ...user import LLMListener

    model = LLMListener(ctx[cfg.listener], min_words=cfg.min_words) if cfg.listener else None
    return Listening(interrupt=model if cfg.barge_in else None, listener=model)
