"""Audio helpers for benchmark data: reading any common WAV, decent resampling, and an energy VAD.

Pure Python by default (the package has no dependencies); ``numpy`` is used when installed to make
resampling band-limited and fast. Benchmark files come as PCM16, PCM24, PCM32 or float WAV at 16, 24
or 48 kHz; everything is turned into the package's mono 16-bit ``Audio``.
"""

from __future__ import annotations

import math
import struct
from array import array
from pathlib import Path

from ..audio import Audio


def read_wav(path: str | Path) -> Audio:
    """Any PCM (8/16/24/32-bit) or IEEE-float (32/64-bit) WAV, mixed down to mono 16-bit."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a RIFF/WAVE file")
    pos, fmt, pcm = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the real tag opens the sub-format GUID
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            pcm = body
        pos += 8 + size + (size & 1)
    if fmt is None or pcm is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    tag, ch, sr, bits = fmt
    width = bits // 8
    pcm = pcm[: len(pcm) // (width * ch) * width * ch]
    if tag == 3:  # float
        x = array("f" if bits == 32 else "d")
        x.frombytes(pcm)
        vals = [max(-32768, min(32767, round(v * 32768))) for v in x]
    elif tag == 1 and bits == 16:
        x = array("h")
        x.frombytes(pcm)
        vals = x
    elif tag == 1 and bits == 24:
        vals = [int.from_bytes(pcm[i : i + 3], "little", signed=True) >> 8 for i in range(0, len(pcm), 3)]
    elif tag == 1 and bits == 32:
        x = array("i")
        x.frombytes(pcm)
        vals = [v >> 16 for v in x]
    elif tag == 1 and bits == 8:
        vals = [(b - 128) << 8 for b in pcm]
    else:
        raise ValueError(f"{path}: unsupported WAV format tag={tag} bits={bits}")
    if ch > 1:
        vals = [round(sum(vals[i : i + ch]) / ch) for i in range(0, len(vals), ch)]
    return Audio(vals if isinstance(vals, array) and vals.typecode == "h" else array("h", vals), sr)


def resample(audio: Audio, sr: int) -> Audio:
    """Band-limited resampling with numpy (windowed-sinc low-pass before decimation, then linear
    interpolation); falls back to ``Audio.resample`` (linear, no low-pass) without numpy."""
    if audio.sr == sr:
        return audio
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is optional
        return audio.resample(sr)
    x = np.frombuffer(audio.samples.tobytes(), dtype=np.int16).astype(np.float64)
    if sr < audio.sr:  # low-pass at the new Nyquist (Hann-windowed sinc, 0.9 * cutoff)
        fc = 0.9 * 0.5 * sr / audio.sr
        n = np.arange(-64, 65)
        h = 2 * fc * np.sinc(2 * fc * n) * np.hanning(len(n))
        x = np.convolve(x, h / h.sum(), mode="same")
    m = round(len(x) * sr / audio.sr)
    y = np.interp(np.arange(m) * audio.sr / sr, np.arange(len(x)), x)
    return Audio(array("h", np.clip(np.round(y), -32768, 32767).astype(np.int16).tobytes()), sr)


def frame_rms(audio: Audio, frame_ms: int = 10) -> list[float]:
    n = max(1, round(audio.sr * frame_ms / 1000))
    s = audio.samples
    return [math.sqrt(sum(v * v for v in s[i : i + n]) / len(s[i : i + n])) for i in range(0, len(s) - n + 1, n)]


def speech_segments(audio: Audio, *, rms_threshold: float = 300.0, frame_ms: int = 10, min_speech_ms: int = 250,
                    min_silence_ms: int = 100, pad_ms: int = 30) -> list[tuple[float, float]]:
    """Voiced stretches ``[(start_s, end_s)]`` of ``audio`` by frame energy.

    A stand-in for Silero-VAD's ``get_speech_timestamps`` with its defaults (``min_speech_duration_ms=250``,
    ``min_silence_duration_ms=100``, ``speech_pad_ms=30``), which FD-Bench runs on the model's output; the
    energy threshold replaces the neural speech probability. Model output is clean synthetic speech with
    digital silence between utterances, where an energy detector and Silero agree closely."""
    rms = frame_rms(audio, frame_ms)
    segs, start, quiet = [], None, 0
    for i, r in enumerate(rms):
        if r >= rms_threshold:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * frame_ms >= min_silence_ms:
                segs.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        segs.append((start, len(rms) - quiet))
    dur = audio.dur_ms / 1000
    out = []
    for a, b in segs:
        if (b - a) * frame_ms < min_speech_ms:
            continue
        s, e = max(0.0, a * frame_ms / 1000 - pad_ms / 1000), min(dur, b * frame_ms / 1000 + pad_ms / 1000)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return [(round(s, 3), round(e, 3)) for s, e in out]
