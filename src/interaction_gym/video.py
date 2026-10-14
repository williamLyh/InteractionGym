"""Video streams: what a camera or a shared screen shows, as timestamped frames.

A video stream (e.g. ``user.screen``) carries ``VideoFrame``s. It is variable-frame-rate: a frame holds until the next
one on the same stream, so a rendered world only sends a frame when its picture changes (a still screen costs nothing).
An agent that wants a fixed rate samples the latest frame at that rate (``latest``).

Rendered worlds produce their frames themselves: a ``ToolWorld`` with ``screen=`` publishes the picture of every
backend that can ``render`` it, at the start and after every action that changes it (``tools.ToolWorld``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import Frame

SCREEN = "user.screen"  # the user's shared screen


@dataclass(frozen=True)
class VideoFrame:
    """One picture (PNG bytes) of a video stream, shown from the frame's time until the next frame."""

    png: bytes
    source: str = ""  # what it shows, e.g. the service whose screen it is
    note: str = ""  # what changed, for logs and the viewer (e.g. "user: fill")
    view: dict | None = field(default=None, compare=False, hash=False)  # what it shows, as data (``ToolBackend.view``):
    # how a simulated person who looks at the picture reads it, so what they see and what the agent sees cannot differ


def latest(frames: list[Frame], t: int, stream: str = SCREEN) -> Frame | None:
    """The frame of ``stream`` shown at time ``t`` (the newest one at or before ``t``)."""
    shown = [f for f in frames if f.stream == stream and isinstance(f.data, VideoFrame) and f.t <= t]
    return max(shown, key=lambda f: f.t) if shown else None
