"""Existing duplex benchmarks as InteractionGym test sets (see docs/BENCHMARKS.md).

- ``full_duplex_bench``: Full-Duplex-Bench v1.0 / v1.5 → ``Task`` + ``ReplayUser`` turns, and the official metrics.
- ``easy_turn``: the Easy Turn testset (turn-state clips, Mandarin) → short labelled episodes.
- ``humdial``: HumDial-FDBench (real en + zh recordings: interruptions, backchannels, asides, third-party speech, pauses).
- ``clip_bank``: reusable clips (backchannels, asides, background speech, interruptions, openings), human-behaviour
  statistics, and ``behaviors()`` to seed an online user from them.
- ``wav``: reading any WAV, resampling, an energy VAD.

Benchmark users are silent apart from what they say: ``benchmark_user`` builds a closed-loop ``UserSim`` with no
background track (``Soundscape(background=False)``, no room-tone / -60 dBFS floor) and no random noise events or
asides, so the agent's microphone in a closed-loop (B) condition carries exactly what the replayed open-loop (A)
condition plays until the user's second turn (docs/BENCHMARKS.md, "No background in evaluations"). The env's
general default (a background from the persona's surroundings) is unchanged for other simulations.
"""

from __future__ import annotations


def benchmark_soundscape():
    """The acoustic setting of every benchmark user: no background track, no sound bank (no recorded noises)."""
    from ..soundscape import Soundscape

    return Soundscape(bank=None, background=False)


def benchmark_behaviors():
    """The habits of every benchmark user besides its turns: no random noise events, no asides (being addressed by
    someone nearby / called away). Explicit, although the ``Behaviors`` defaults are the same."""
    from ..user import Behaviors

    return Behaviors(aside_per_min=0.0, noise_per_min=0.0, noise_rates=None)


def benchmark_user(source, **kw):
    """A closed-loop ``UserSim`` for evaluations: ``benchmark_soundscape()`` and ``benchmark_behaviors()`` unless
    given; every other keyword goes to ``UserSim``."""
    from ..user import UserSim

    kw.setdefault("soundscape", benchmark_soundscape())
    kw.setdefault("behaviors", benchmark_behaviors())
    return UserSim(source, **kw)
