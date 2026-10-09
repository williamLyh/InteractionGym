"""Mono 16-bit PCM audio that slices like a sequence, so it can be a Segment's data."""

from __future__ import annotations

import math
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, eq=False)
class Audio:
    samples: array  # typecode "h" (int16)
    sr: int

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, s: slice) -> Audio:
        return Audio(self.samples[s], self.sr)

    def __add__(self, other: Audio) -> Audio:
        assert self.sr == other.sr
        return Audio(self.samples + other.samples, self.sr)

    def __eq__(self, other) -> bool:
        return isinstance(other, Audio) and self.sr == other.sr and self.samples == other.samples

    @property
    def dur_ms(self) -> int:
        return round(len(self.samples) * 1000 / self.sr)

    def resample(self, sr: int) -> Audio:
        """Linear-interpolation resampling (enough for speech fed to a model)."""
        if sr == self.sr:
            return self
        src, n = self.samples, round(len(self.samples) * sr / self.sr)
        out = array("h", bytes(2 * n))
        last = len(src) - 1
        for i in range(n):
            x = i * self.sr / sr
            j = min(int(x), last)
            frac = x - j
            nxt = src[j + 1] if j < last else src[j]
            out[i] = int(src[j] + (nxt - src[j]) * frac)
        return Audio(out, sr)

    @classmethod
    def silence(cls, ms: int, sr: int = 16000) -> Audio:
        return cls(array("h", bytes(2 * round(ms * sr / 1000))), sr)

    @classmethod
    def from_pcm16(cls, data: bytes, sr: int) -> Audio:
        a = array("h")
        a.frombytes(data[: len(data) // 2 * 2])
        return cls(a, sr)

    @classmethod
    def read_wav(cls, path: str | Path) -> Audio:
        with wave.open(str(path)) as w:
            assert w.getsampwidth() == 2 and w.getnchannels() == 1, "expects mono 16-bit wav"
            return cls.from_pcm16(w.readframes(w.getnframes()), w.getframerate())

    def write_wav(self, path: str | Path) -> None:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(self.samples.tobytes())


# ---------------------------------------------------------------- speech levels (pure Python, no numpy)

FRAME_MS = 20


def frame_levels(audio: Audio, frame_ms: int = FRAME_MS) -> list[float]:
    """RMS of each ``frame_ms`` frame, in dBFS (-120 for digital silence)."""
    n = max(1, round(audio.sr * frame_ms / 1000))
    s = audio.samples
    out = []
    for i in range(0, len(s), n):
        fr = s[i:i + n]
        ms = sum(x * x for x in fr) / max(1, len(fr))
        out.append(10 * math.log10(ms / 32768 ** 2) if ms > 0 else -120.0)
    return out


def speech_span(audio: Audio, floor_db: float = -40.0, abs_floor_dbfs: float = -60.0) -> tuple[int, int] | None:
    """(first, last+1) sample of the speech in ``audio``: frames within ``floor_db`` of the loudest frame and above
    ``abs_floor_dbfs``. None when there is no speech at all."""
    lv = frame_levels(audio)
    if not lv:
        return None
    thr = max(max(lv) + floor_db, abs_floor_dbfs)
    on = [i for i, v in enumerate(lv) if v >= thr]
    if not on:
        return None
    n = max(1, round(audio.sr * FRAME_MS / 1000))
    return on[0] * n, min(len(audio.samples), (on[-1] + 1) * n)


def trim_silence(audio: Audio, margin_ms: int = 120, floor_db: float = -40.0) -> Audio:
    """``audio`` without its leading / trailing silence, keeping ``margin_ms`` on each side (TTS clips often carry
    1-3 s of trailing silence that would otherwise count as the user still speaking)."""
    span = speech_span(audio, floor_db)
    if span is None:
        return audio
    m = round(audio.sr * margin_ms / 1000)
    return audio[max(0, span[0] - m): min(len(audio.samples), span[1] + m)]


def active_level(audio: Audio, floor_db: float = -30.0) -> float | None:
    """Active-speech level in dBFS: the RMS over the frames within ``floor_db`` of the loudest one (pauses and
    silence do not pull it down). None for silence."""
    lv = [v for v in frame_levels(audio) if v > -100]
    if not lv:
        return None
    act = [v for v in lv if v >= max(lv) + floor_db]
    return 10 * math.log10(sum(10 ** (v / 10) for v in act) / len(act))


def gain(audio: Audio, db: float, peak_dbfs: float = -1.0) -> Audio:
    """``audio`` scaled by ``db`` dB, with the gain lowered so that the peak stays at or below ``peak_dbfs``."""
    peak = max((abs(x) for x in audio.samples), default=0)
    g = 10 ** (db / 20)
    if peak and peak * g > 32767 * 10 ** (peak_dbfs / 20):
        g = 32767 * 10 ** (peak_dbfs / 20) / peak
    if abs(g - 1) < 1e-4:
        return audio
    return Audio(array("h", (max(-32768, min(32767, round(x * g))) for x in audio.samples)), audio.sr)


def normalize_level(audio: Audio, level_dbfs: float = -23.0, peak_dbfs: float = -1.0) -> Audio:
    """``audio`` brought to an active-speech level of ``level_dbfs`` (peak-limited by ``peak_dbfs``)."""
    lv = active_level(audio)
    return audio if lv is None else gain(audio, level_dbfs - lv, peak_dbfs)
