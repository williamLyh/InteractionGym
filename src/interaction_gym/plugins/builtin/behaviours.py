"""Behaviours: what a simulated role does besides its turns (``interaction_gym.behaviours``). A ``role/user`` row
names them in order (``behaviours: [noise, asides]``); the default library is both; ``[]`` is a user that only
speaks. Their rates still come from the resolved ``Behaviors`` (``role/user`` ``behaviors`` and the persona's
surroundings); a behaviour plugin of your own implements ``behaviours.Behaviour``."""

from __future__ import annotations

from ...behaviours import Asides, NoiseEvents
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
