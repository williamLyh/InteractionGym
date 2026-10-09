"""Where the user is calling from, as sound: an always-on background track and clips for discrete noise events.

- ``background_spec`` / ``render_background``: the episode's background (``silence | white | pink | brown |
  ambience:<kind>``) at a level, from ``task.scenario["background"]`` or, by default, from the persona's
  ``surroundings``. It is context, not a turn: nobody expects a reaction to it (docs/FORMAT.md §4.2).
- ``EventProcess``: when discrete noise events happen: a seeded compound Poisson process on the call clock, with
  per-label rates by surroundings (``EVENT_RATES``) and per-label bursts, levels and jitter (``EVENT_TYPES``).
- ``event_clip`` / ``event_audio``: the audio of a noise event (a cough, a door slam, a phone ring ...), from a sound
  bank directory when one is configured, else a synthetic stand-in; a burst is one clip repeated with gaps.
- ``SoundBank``: a directory of recordings, laid out as ``ambience/<kind>/*.wav`` and ``events/<label>/*.wav`` (mono
  16-bit WAV). ``Soundscape()`` uses the default bank (``noisebank.default_bank``): DEMAND ambience, fetched once on
  first use into a user cache (``$IG_SOUNDBANK``; ``synthetic`` to turn it off); ``scripts/fetch_noise_banks.py``
  builds a full one from DEMAND (ambience) and MUSAN (events), see docs/FORMAT.md §4.2.

Everything is seeded: the same seed gives the same track, offset and clips.
"""

from __future__ import annotations

import math
import random
from array import array
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .audio import Audio
from .core import Background

SPEECH_DBFS = -20.0  # nominal RMS level of the user's speech (TTS output): the reference of ``snr_db``
LEGACY_BED_DBFS = 20 * math.log10(900 / 32768)  # the examples' old gaussian "cafe noise" bed (std 900) at 0 dB gain
LOOP_S = 20  # synthetic noise is rendered as a seeded loop of this length

COLORS = ("white", "pink", "brown")
AMBIENCE = ("home", "office", "car", "cafe", "street")
QUIET_FLOOR_DBFS = -60.0  # a quiet room is never digital silence: a microphone's self-noise and room tone, ~40 dB below speech
# surroundings -> (type, RMS level in dBFS). Speech is ~-20 dBFS, so these are SNRs of ~40 (quiet) to ~16 dB (street).
DEFAULTS = {"quiet": ("pink", QUIET_FLOOR_DBFS), "home": ("ambience:home", -52.0), "office": ("ambience:office", -48.0),
            "car": ("ambience:car", -40.0), "cafe": ("ambience:cafe", -40.0), "street": ("ambience:street", -36.0)}
# without recordings, an ambience is stood in for by coloured noise: low-frequency rumble (brown) for engines and
# traffic, pink for rooms full of broadband sound
STAND_IN = {"home": "pink", "office": "pink", "car": "brown", "cafe": "pink", "street": "brown"}
# ---- noise events: what can happen where, how often, how loud, in bursts or not (docs/FORMAT.md §4.1.2)
# label -> (source, level_db, burst_p, burst_max, gap_ms). ``source``: "user" (an involuntary sound of the caller; it
# may overlap its own speech) or "surroundings" (independent of who is talking). ``level_db``: the occurrence's
# active-part RMS relative to nominal speech (SPEECH_DBFS), jittered by +-JITTER_DB per event. A burst: after each
# occurrence, one more with probability ``burst_p`` (at most ``burst_max`` in all), after a silent gap drawn from
# ``gap_ms`` (a dog barking again, a horn sequence, a phone ringing on).
EVENT_TYPES = {
    "cough": ("user", -2.0, 0.4, 3, (250, 700)),
    "sneeze": ("user", 0.0, 0.15, 2, (400, 1200)),
    "throat_clear": ("user", -8.0, 0.0, 1, (0, 0)),
    "door_slam": ("surroundings", -10.0, 0.0, 1, (0, 0)),
    "dog_bark": ("surroundings", -12.0, 0.45, 4, (300, 1500)),
    "phone_ring": ("surroundings", -14.0, 0.7, 5, (1500, 3000)),
    "dishes": ("surroundings", -16.0, 0.4, 4, (300, 1500)),
    "cup": ("surroundings", -18.0, 0.3, 3, (400, 2000)),
    "keyboard": ("surroundings", -20.0, 0.3, 3, (300, 1500)),
    "horn": ("surroundings", -12.0, 0.45, 4, (150, 900)),
    "siren": ("surroundings", -18.0, 0.0, 1, (0, 0)),
    "indicator": ("surroundings", -22.0, 0.0, 1, (0, 0)),
}
JITTER_DB = 6.0
# the longest a burst gets (seconds): later occurrences that would pass it are dropped
MAX_BURST_S = {"cough": 4.0, "sneeze": 3.0, "door_slam": 3.0, "dog_bark": 8.0, "phone_ring": 20.0, "dishes": 6.0, "cup": 5.0,
               "keyboard": 8.0, "horn": 5.0, "siren": 10.0}
# surroundings -> {label: event onsets per minute of the call}. Totals (0.1 quiet ... 0.8 street) are the earlier
# single noise rates; bursts add occurrences, not onsets.
EVENT_RATES = {
    "quiet": {"cough": 0.06, "throat_clear": 0.03, "sneeze": 0.01},
    "home": {"cough": 0.04, "sneeze": 0.01, "door_slam": 0.04, "dog_bark": 0.04, "phone_ring": 0.03, "dishes": 0.04},
    "office": {"cough": 0.05, "sneeze": 0.01, "phone_ring": 0.05, "door_slam": 0.03, "keyboard": 0.06},
    "car": {"cough": 0.05, "horn": 0.15, "indicator": 0.15, "siren": 0.05},
    "cafe": {"cough": 0.05, "dishes": 0.15, "cup": 0.2, "door_slam": 0.05, "phone_ring": 0.05},
    "street": {"cough": 0.05, "horn": 0.35, "siren": 0.12, "dog_bark": 0.18, "door_slam": 0.1},
}
EVENTS = {k: tuple(v) for k, v in EVENT_RATES.items()}  # the labels that can happen in each place


def dbfs(level: float) -> float:
    """A linear int16 RMS for a level in dBFS."""
    return 32768 * 10 ** (level / 20)


class SoundBank:
    """Recordings under ``root``: ``ambience/<kind>/*.wav`` (long room / street / car tones) and
    ``events/<label>/*.wav`` (short noises: ``cough``, ``door_slam``, ``phone_ring``, ``dog_bark``, ``horn`` ...).
    Mono 16-bit WAV; any sample rate (resampled where mixed). Missing folders just mean "no recording" and the
    synthetic stand-in is used."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    def files(self, group: str, name: str) -> list[Path]:
        d = self.root / group / name
        return sorted(d.glob("*.wav")) if d.is_dir() else []

    def pick(self, group: str, name: str, rng: random.Random) -> Path | None:
        fs = self.files(group, name)
        return fs[rng.randrange(len(fs))] if fs else None


DEFAULT_BANK = "default"  # ``Soundscape(bank=...)``: the default bank (noisebank.default_bank)
SYNTHETIC = "synthetic"  # ``Soundscape(bank=...)``: no recordings, the synthetic stand-ins (as ``None``)


def _bank(bank, fetch: bool = True) -> SoundBank | None:
    """A ``SoundBank`` for ``bank``: a ``SoundBank``, a directory, ``"default"`` (the default bank: fetched first if
    ``fetch``, else only if already there; None if unavailable) or None / ``"synthetic"`` (None: no recordings)."""
    if bank is None or isinstance(bank, SoundBank):
        return bank
    if isinstance(bank, str) and bank in (DEFAULT_BANK, SYNTHETIC):
        if bank == SYNTHETIC:
            return None
        from .noisebank import default_bank

        root = default_bank(fetch=fetch)
        return SoundBank(root) if root is not None else None
    return SoundBank(bank)


# ---------------------------------------------------------------- background track


def background_spec(scenario_bg, surroundings: str, bank=None) -> dict:
    """The episode's background, resolved: ``{"type", "level_dbfs", "snr_db", "source", "from"}``.

    ``scenario_bg`` (``task.scenario["background"]``) may be: absent (None) → the persona's ``surroundings`` default
    (``DEFAULTS``); a number → the old form, the gain in dB of the examples' gaussian noise bed (kept at the same
    loudness, with the surroundings' type); a type string (``"pink"``, ``"ambience:cafe"``, ``"silence"``); or a dict
    ``{"type", "level_dbfs" | "snr_db", "file"}`` (``file``: a recording to use instead of the bank / generator).
    ``source`` says what is actually played: ``synthetic:<color>``, ``file:<path>``, or ``none``."""
    origin = "scenario"
    if scenario_bg is None or scenario_bg is True:
        typ, level = DEFAULTS[surroundings]
        spec, origin = {"type": typ, "level_dbfs": level}, "surroundings"
    elif scenario_bg is False:
        spec = {"type": "silence"}
    elif isinstance(scenario_bg, (int, float)):
        typ = DEFAULTS[surroundings][0]
        spec = {"type": "white" if surroundings == "quiet" else typ, "level_dbfs": round(LEGACY_BED_DBFS + scenario_bg, 1)}
    elif isinstance(scenario_bg, str):
        spec = {"type": scenario_bg}
    else:
        spec = dict(scenario_bg)
    typ = spec.get("type", "silence")
    if typ not in ("silence", *COLORS) and not (typ.startswith("ambience:") and typ.split(":", 1)[1] in AMBIENCE):
        raise ValueError(f"background type {typ!r}: one of silence, {', '.join(COLORS)}, ambience:<{'|'.join(AMBIENCE)}>")
    out = {"type": typ}
    if typ != "silence":
        if spec.get("snr_db") is not None:
            level = SPEECH_DBFS - spec["snr_db"]
        elif spec.get("level_dbfs") is not None:
            level = spec["level_dbfs"]
        else:  # a type without a level: the surroundings' level, else a moderate one
            level = DEFAULTS.get(typ.split(":", 1)[-1], (None, None))[1] or DEFAULTS[surroundings][1] or -45.0
        out.update(level_dbfs=round(level, 1), snr_db=round(SPEECH_DBFS - level, 1))
    if typ == "silence":
        out["source"] = "none"
    elif spec.get("file"):
        out["source"] = f"file:{spec['file']}"
    elif typ.startswith("ambience:"):
        kind = typ.split(":", 1)[1]
        b = _bank(bank)
        files = b.files("ambience", kind) if b is not None else []
        out["source"] = "bank" if files else f"synthetic:{STAND_IN[kind]}"
    else:
        out["source"] = f"synthetic:{typ}"
    out["from"] = origin
    if origin == "surroundings" or typ.startswith("ambience:"):
        out["surroundings"] = surroundings
    return out


def render_background(spec: dict, sr: int, seed: int, bank=None) -> Background | None:
    """The track for a resolved spec (``background_spec``), looped from a seeded offset; None for silence."""
    if spec["type"] == "silence":
        return None
    rng = random.Random(f"background:{seed}")
    src = spec["source"]
    if src.startswith("synthetic:"):
        audio = colored_noise(src.split(":", 1)[1], sr, spec["level_dbfs"])
    else:
        if src == "bank":
            path = _bank(bank).pick("ambience", spec["type"].split(":", 1)[1], rng)
            spec["source"] = f"file:{path}"
        else:
            path = src.split(":", 1)[1]
        audio = _normalized(str(path), spec["level_dbfs"])
    offset = rng.randrange(max(audio.dur_ms, 1))
    return Background(audio, loop=True, offset_ms=offset, spec=spec)


@lru_cache(maxsize=32)
def _normalized(path: str, level: float) -> Audio:
    a = Audio.read_wav(path)
    rms = math.sqrt(sum(x * x for x in a.samples) / max(len(a.samples), 1)) or 1.0
    g = dbfs(level) / rms
    return Audio(array("h", (max(-32768, min(32767, int(x * g))) for x in a.samples)), a.sr)


@lru_cache(maxsize=32)
def colored_noise(color: str, sr: int, level: float, seconds: int = LOOP_S) -> Audio:
    """``seconds`` of white / pink / brown noise at an RMS of ``level`` dBFS (fixed seed: the loop is the same
    everywhere; episodes differ by their offset into it). Pink: Paul Kellet's filter; brown: a leaky integrator."""
    rng = random.Random(f"{color}:{sr}")
    n = seconds * sr
    out = [0.0] * n
    if color == "white":
        for i in range(n):
            out[i] = rng.gauss(0, 1)
    elif color == "pink":
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
        for i in range(n):
            w = rng.gauss(0, 1)
            b0 = 0.99886 * b0 + w * 0.0555179
            b1 = 0.99332 * b1 + w * 0.0750759
            b2 = 0.96900 * b2 + w * 0.1538520
            b3 = 0.86650 * b3 + w * 0.3104856
            b4 = 0.55000 * b4 + w * 0.5329522
            b5 = -0.7616 * b5 - w * 0.0168980
            out[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
            b6 = w * 0.115926
    elif color == "brown":
        y = 0.0
        for i in range(n):
            y = 0.998 * y + rng.gauss(0, 1) * 0.1
            out[i] = y
    else:
        raise ValueError(f"noise color {color!r}: one of {', '.join(COLORS)}")
    rms = math.sqrt(sum(x * x for x in out) / n) or 1.0
    g = dbfs(level) / rms
    return Audio(array("h", (max(-32768, min(32767, int(x * g))) for x in out)), sr)


# ---------------------------------------------------------------- noise events


@dataclass(frozen=True)
class NoiseEvent:
    """One event of the process: ``t`` (ms on the call clock), ``label``, ``n`` occurrences with ``gaps_ms`` (n - 1
    silent gaps between them), its ``level_db`` (relative to nominal speech, jitter included) and ``source``."""

    t: int
    label: str
    n: int = 1
    gaps_ms: tuple[int, ...] = ()
    level_db: float = 0.0
    source: str = "surroundings"


class EventProcess:
    """When noise events happen: a homogeneous compound Poisson process on the call clock (docs/FORMAT.md §4.1.2).

    Onsets of each label are Poisson with that label's rate (``rates``: {label: onsets per minute}); the superposition
    is drawn as one Poisson stream of the total rate, each onset's label picked in proportion to the rates. Each onset
    spawns a burst: after every occurrence one more follows with probability ``burst_p`` (at most ``burst_max``),
    after a gap from ``gap_ms`` (``EVENT_TYPES``). The event's level is its label's ``level_db`` plus a uniform jitter of
    +-``jitter_db``. Seeded: the same seed gives the same events, whatever the agent or the user does."""

    def __init__(self, rates: dict[str, float], seed, jitter_db: float = JITTER_DB, types: dict | None = None):
        self.rates = {k: float(v) for k, v in rates.items() if v and v > 0}
        self.types = {**EVENT_TYPES, **(types or {})}
        self.jitter_db = jitter_db
        self.total = sum(self.rates.values())
        self.rng = random.Random(f"events:{seed}")
        self.labels = sorted(self.rates)
        self._next: NoiseEvent | None = None
        self._t = 0

    def _type(self, label: str) -> tuple:
        return self.types.get(label, ("surroundings", -12.0, 0.0, 1, (0, 0)))

    def peek(self) -> NoiseEvent | None:
        """The next event (not consumed); None if the rates are all zero."""
        if self._next is None and self.total > 0:
            r = self.rng
            self._t += max(1, round(r.expovariate(self.total / 60000)))
            x, label = r.random() * self.total, self.labels[-1]
            for lab in self.labels:
                x -= self.rates[lab]
                if x < 0:
                    label = lab
                    break
            source, level, p, nmax, gap = self._type(label)
            n = 1
            while n < nmax and r.random() < p:
                n += 1
            gaps = tuple(r.randint(*gap) for _ in range(n - 1))
            jit = r.uniform(-self.jitter_db, self.jitter_db) if self.jitter_db else 0.0
            self._next = NoiseEvent(self._t, label, n, gaps, round(level + jit, 1), source)
        return self._next

    def pop(self) -> NoiseEvent | None:
        ev = self.peek()
        self._next = None
        return ev

    def params(self) -> dict:
        """The process parameters, for the trajectory's ``meta.user.behaviors.noise_process``."""
        labels = {}
        for lab in self.labels:
            source, level, p, nmax, gap = self._type(lab)
            labels[lab] = {"rate_per_min": round(self.rates[lab], 4), "source": source, "level_db": level, "burst_p": p,
                           "burst_max": nmax, "gap_ms": list(gap), "mean_burst": round(mean_burst(p, nmax), 3)}
        return {"model": "compound Poisson (Poisson onsets per label, geometric bursts)", "onsets_per_min": round(self.total, 4),
                "jitter_db": self.jitter_db, "labels": labels}


def mean_burst(p: float, nmax: int) -> float:
    """Expected occurrences per burst: 1 + p + p^2 + ... + p^(nmax-1)."""
    return sum(p ** k for k in range(max(nmax, 1)))


def event_rates(surroundings: str, total: float | None = None, rates: dict | None = None) -> dict[str, float]:
    """{label: onsets per minute}: ``rates`` if given, else the surroundings' table (``EVENT_RATES``), scaled to
    ``total`` onsets per minute when given."""
    if rates:
        return {k: float(v) for k, v in rates.items()}
    table = EVENT_RATES[surroundings]
    if total is None:
        return dict(table)
    s = sum(table.values())
    return {k: v * total / s for k, v in table.items()} if s > 0 and total > 0 else {}


def event_clip(label: str, sr: int, rng: random.Random, bank=None) -> tuple[Audio, str]:
    """(audio, source) of one noise occurrence: a recording from the bank's ``events/<label>/`` if there is one
    (``file:<path>``), else a synthetic stand-in (``synthetic``)."""
    b = _bank(bank, fetch=False)  # events never trigger the DEMAND download (it has none): a bank already there only
    path = b.pick("events", label, rng) if b is not None else None
    if path is not None:
        return _clip(str(path)), f"file:{path}"
    return synth_event(label, sr, rng), "synthetic"


@lru_cache(maxsize=256)
def _clip(path: str) -> Audio:
    return Audio.read_wav(path)


def active_rms(samples, sr: int, frame_ms: int = 20, rel_db: float = -30.0) -> float:
    """RMS over the frames within ``rel_db`` of the loudest one: the level of a sound, not of the silence around it."""
    f = max(1, sr * frame_ms // 1000)
    frames = [samples[i:i + f] for i in range(0, max(len(samples) - f + 1, 1), f)]
    ms = [sum(x * x for x in fr) / max(len(fr), 1) for fr in frames]
    if not ms or max(ms) <= 0:
        return 0.0
    thr = max(ms) * 10 ** (rel_db / 10)
    on = [m for m in ms if m >= thr]
    return math.sqrt(sum(on) / len(on))


def event_audio(ev: NoiseEvent, sr: int, rng: random.Random, bank=None) -> tuple[Audio, str, int]:
    """(audio, source, occurrences) of a whole event: one clip (the same dog, the same phone) repeated ``ev.n`` times
    with the event's gaps — but no further once the burst would pass ``MAX_BURST_S`` (a recording may already hold
    several barks or rings) — at the event's level (active-part RMS = ``SPEECH_DBFS + ev.level_db``)."""
    clip, source = event_clip(ev.label, sr, rng, bank)
    if clip.sr != sr:
        clip = clip.resample(sr)
    level = active_rms(clip.samples, sr)
    g = dbfs(SPEECH_DBFS + ev.level_db) / level if level > 0 else 0.0
    out, n, cap = array("h"), 0, MAX_BURST_S.get(ev.label, 8.0) * sr
    for k in range(ev.n):
        gap = round(ev.gaps_ms[k - 1] * sr / 1000) if k else 0
        if k and len(out) + gap + len(clip.samples) > cap:
            break
        out.extend([0] * gap)
        out.extend(max(-32768, min(32767, int(x * g))) for x in clip.samples)
        n += 1
    return Audio(out, sr), source, n


def _env(i: int, n: int, attack: float = 0.05) -> float:
    """A quick attack, exponential-ish decay envelope over n samples."""
    x = i / max(n, 1)
    return x / attack if x < attack else math.exp(-4 * (x - attack))


def synth_event(label: str, sr: int, rng: random.Random) -> Audio:
    """A crude, recognisable-enough stand-in for a noise event (used when no recording is configured)."""
    parts: list[tuple[float, float, str, float]] = []  # (start s, length s, shape, peak)
    if label in ("cough", "throat_clear"):  # one occurrence: repeats come from the event's burst
        parts.append((0.0, rng.uniform(0.15, 0.25), "burst", 9000))
    elif label == "sneeze":
        parts += [(0.0, 0.25, "burst", 6000), (0.3, 0.2, "burst", 12000)]
    elif label == "cup":
        for j in range(rng.randint(1, 3)):
            parts.append((j * rng.uniform(0.2, 0.5), 0.12, "clink", rng.uniform(5000, 9000)))
    elif label == "door_slam":
        parts.append((0.0, 0.35, "thud", 16000))
    elif label == "dishes":
        for j in range(rng.randint(3, 6)):
            parts.append((j * rng.uniform(0.06, 0.15), 0.05, "click", rng.uniform(5000, 11000)))
    elif label == "keyboard":
        for j in range(rng.randint(5, 10)):
            parts.append((j * rng.uniform(0.08, 0.16), 0.02, "click", rng.uniform(2500, 4000)))
    elif label == "dog_bark":
        parts.append((0.0, 0.18, "bark", 12000))
    elif label == "phone_ring":
        parts += [(0.0, 0.4, "ring", 7000), (0.6, 0.4, "ring", 7000)]
    elif label == "horn":
        parts.append((0.0, rng.uniform(0.3, 0.7), "horn", 9000))
    elif label == "indicator":
        for j in range(4):
            parts.append((j * 0.4, 0.03, "click", 3000))
    elif label == "siren":
        parts.append((0.0, 1.5, "siren", 6000))
    else:
        parts.append((0.0, 0.4, "burst", 8000))
    total = max(s + d for s, d, _, _ in parts) + 0.05
    n = round(total * sr)
    acc = [0.0] * n
    lp = 0.0
    for start, length, shape, peak in parts:
        i0, m = round(start * sr), round(length * sr)
        for i in range(m):
            if i0 + i >= n:
                break
            t, e = i / sr, _env(i, m)
            if shape == "burst":
                v = rng.gauss(0, 0.5)
            elif shape == "thud":
                lp = 0.9 * lp + 0.1 * rng.gauss(0, 1)
                v = 3 * lp + 0.2 * rng.gauss(0, 1)
            elif shape == "click":
                v = rng.gauss(0, 1) * math.exp(-i / (0.004 * sr))
                e = 1.0
            elif shape == "clink":  # a ceramic / glass tap: a decaying high partial
                v = math.sin(2 * math.pi * 2900 * t) * math.exp(-i / (0.03 * sr)) + 0.2 * rng.gauss(0, 1) * math.exp(-i / (0.003 * sr))
                e = 1.0
            elif shape == "bark":
                v = 0.6 * math.sin(2 * math.pi * 550 * t) * (1 + 0.5 * math.sin(2 * math.pi * 30 * t)) + 0.3 * rng.gauss(0, 1)
            elif shape == "ring":
                v = 0.5 * (math.sin(2 * math.pi * 440 * t) + math.sin(2 * math.pi * 480 * t))
                e = 1.0 if 0.01 * sr < i < m - 0.01 * sr else 0.5
            elif shape == "horn":
                v = sum(math.sin(2 * math.pi * f * t) / k for k, f in enumerate((415, 830, 1245), 1)) * 0.6
                e = 1.0 if 0.02 * sr < i < m - 0.02 * sr else 0.5
            else:  # siren: a slow sweep
                f = 700 + 300 * math.sin(2 * math.pi * 0.8 * t)
                v = math.sin(2 * math.pi * f * t)
                e = 1.0
            acc[i0 + i] += max(-1.0, min(1.0, v)) * e * peak
    return Audio(array("h", (max(-32768, min(32767, int(v))) for v in acc)), sr)


@dataclass(frozen=True)
class Soundscape:
    """A simulated user's acoustic setting: ``bank`` supplies recordings, where it has them (else synthetic stand-ins
    are used); ``background`` lays the surroundings' always-on track under the episode.

    ``bank``: ``"default"`` (the default: ``$IG_SOUNDBANK`` or the user cache, with the DEMAND ambience fetched there
    on first use, falling back to synthetic if that fails; ``noisebank``), a ``SoundBank`` or its directory (used as
    it is, never fetched), or None / ``"synthetic"`` (synthetic stand-ins only)."""

    bank: object = DEFAULT_BANK
    background: bool = True
