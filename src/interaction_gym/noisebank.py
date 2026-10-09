"""The default sound bank: DEMAND ambience recordings, fetched on first use into a user cache (nothing is shipped).

``Soundscape()`` (the ``UserSim`` default) resolves its bank here (``default_bank``):

- ``$IG_SOUNDBANK``: the bank directory (a ``soundscape.SoundBank`` root), or ``synthetic`` (also ``none`` / ``off``
  / ``0``) for the synthetic stand-ins only. Default: ``$XDG_CACHE_HOME/interaction_gym/soundbank`` (``~/.cache/...``).
- If that directory has no ambience recordings yet, the DEMAND part is downloaded and built there once (~80 MB: one
  channel of 11 DEMAND environments, fetched with HTTP range requests from Zenodo), with one log line naming the
  dataset, its licence and the directory. ``$IG_SOUNDBANK_FETCH=0`` turns the download off (an existing bank is
  still used).
- If the download fails (offline, a sandbox), a warning is given once per process and the synthetic stand-ins are
  used: an episode never fails for want of recordings.

DEMAND: J. Thiemann, N. Ito, E. Vincent, "The Diverse Environments Multi-channel Acoustic Noise Database (DEMAND)",
ICA 2013, doi:10.5281/zenodo.1227121; CC BY-SA 3.0, so a bank built from it is ShareAlike (THIRD_PARTY.md). Noise
*events* (MUSAN, an 11 GB archive) are never downloaded here: ``scripts/fetch_noise_banks.py`` builds a full bank
(ambience + events), used by pointing ``$IG_SOUNDBANK`` at it.

    python -m interaction_gym.noisebank        # pre-build the default bank now (e.g. on a host before a run)

Standard library only.
"""

from __future__ import annotations

import io
import json
import math
import os
import shutil
import sys
import threading
import time
import urllib.request
import warnings
import wave
import zipfile
from array import array
from pathlib import Path

from .envvars import getenv

SR = 16000
AMBIENCE_DBFS = -30.0

DEMAND_RECORD = "https://zenodo.org/api/records/1227121"
DEMAND_LICENSE = "CC BY-SA 3.0 (DEMAND, Thiemann, Ito & Vincent 2013; doi:10.5281/zenodo.1227121)"
# DEMAND environment -> our surroundings kind (docs/FORMAT.md §4.2). The 15 DEMAND environments are DKITCHEN,
# DLIVING, DWASHING (domestic), NFIELD, NPARK, NRIVER (nature), OHALLWAY, OMEETING, OOFFICE (office), PCAFETER,
# PRESTO, PSTATION (public), SCAFE, SPSQUARE, STRAFFIC (street), TBUS, TCAR, TMETRO (transport); SCAFE has no
# 16 kHz version. Not used: nature (no such surroundings), PSTATION / TBUS / TMETRO (stations and public transport
# are not "car").
DEMAND_MAP = {"DKITCHEN": "home", "DLIVING": "home", "DWASHING": "home",
              "OOFFICE": "office", "OMEETING": "office", "OHALLWAY": "office",
              "PCAFETER": "cafe", "PRESTO": "cafe",
              "STRAFFIC": "street", "SPSQUARE": "street",
              "TCAR": "car"}
OFF = ("synthetic", "none", "off", "0", "false", "")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- download


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file over HTTP range requests (for ``zipfile``), with retries."""

    def __init__(self, url: str, size: int, block: int = 1 << 20, retries: int = 8, log=log):
        self.url, self.size, self.pos, self.block, self.cache = url, size, 0, block, {}
        self.retries, self.log = retries, log

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _blk(self, i: int) -> bytes:
        if i not in self.cache:
            lo, hi = i * self.block, min((i + 1) * self.block, self.size) - 1
            for attempt in range(self.retries):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    if len(data) == hi - lo + 1:
                        break
                except Exception as e:  # noqa: BLE001
                    self.log("range retry", attempt, repr(e))
                time.sleep(5 * (attempt + 1))
            else:
                raise IOError(f"range {lo}-{hi} of {self.url} failed")
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[i] = data
        return self.cache[i]

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            i, o = divmod(self.pos, self.block)
            chunk = self._blk(i)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def fetch_json(url: str, retries: int = 6, timeout: int = 60, log=log):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            log("retry", url, repr(e))
            if attempt + 1 < retries:
                time.sleep(10 * (attempt + 1))
    raise IOError(url)


def download_demand(dl: Path, retries: int = 8, log=log) -> None:
    """``dl/demand/<ENV>_ch01.wav`` for every environment of ``DEMAND_MAP`` (resumable: present files are kept)."""
    d = dl / "demand"
    d.mkdir(parents=True, exist_ok=True)
    rec = fetch_json(DEMAND_RECORD, retries=max(1, min(retries, 6)), log=log)
    (d / "record.json").write_text(json.dumps({k: rec[k] for k in ("doi", "links", "metadata") if k in rec}, indent=1))
    files = {f["key"]: f for f in rec["files"]}

    def one(env: str):
        out = d / f"{env}_ch01.wav"
        if out.exists():
            return
        f = files[f"{env}_16k.zip"]
        log("DEMAND", env, f"{f['size'] / 1e6:.0f} MB zip, fetching ch01 only")
        with zipfile.ZipFile(RangeFile(f["links"]["self"], f["size"], block=1 << 18, retries=retries, log=log)) as z:
            name = next(n for n in z.namelist() if n.endswith("ch01.wav"))
            data = z.read(name)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(out)
        log("DEMAND", env, "ok", len(data))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(len(DEMAND_MAP)) as ex:  # the server is slow per connection: all environments at once
        list(ex.map(one, DEMAND_MAP))


# ---------------------------------------------------------------- audio helpers


def read_wav(p: Path) -> tuple[array, int]:
    with wave.open(str(p)) as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"{p}: {8 * sw}-bit")
    a = array("h")
    a.frombytes(raw)
    if sys.byteorder == "big":
        a.byteswap()
    if ch > 1:
        a = array("h", (int(sum(a[i:i + ch]) / ch) for i in range(0, len(a), ch)))
    return a, sr


def resample(a: array, sr: int, to: int = SR) -> array:
    """Linear interpolation after a moving-average low-pass (enough for noise at a 16 kHz target)."""
    if sr == to:
        return a
    k = max(1, round(sr / to))
    if k > 1:
        acc, s = [], 0
        for i, x in enumerate(a):
            s += x - (a[i - k] if i >= k else 0)
            acc.append(s / k)
    else:
        acc = list(a)
    n = int(len(acc) * to / sr)
    out = array("h")
    for j in range(n):
        x = j * sr / to
        i = int(x)
        f = x - i
        v = acc[i] * (1 - f) + acc[min(i + 1, len(acc) - 1)] * f
        out.append(max(-32768, min(32767, int(v))))
    return out


def write_wav(p: Path, a: array, sr: int = SR) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    b = array("h", a)
    if sys.byteorder == "big":
        b.byteswap()
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b.tobytes())


def rms(xs) -> float:
    return math.sqrt(sum(x * x for x in xs) / max(len(xs), 1))


def scale(a: array, g: float) -> array:
    return array("h", (max(-32768, min(32767, int(x * g))) for x in a))


# ---------------------------------------------------------------- build


def build_ambience(dl: Path, out_root: Path, log=log) -> list[dict]:
    """``out_root/ambience/<kind>/<env>.wav`` from the DEMAND downloads (mono 16 kHz, RMS ``AMBIENCE_DBFS``); the
    manifest entries."""
    files = []
    for env, kind in DEMAND_MAP.items():
        src = dl / "demand" / f"{env}_ch01.wav"
        if not src.exists():
            log("missing", src)
            continue
        a, sr = read_wav(src)
        a = resample(a, sr)
        a = scale(a, 32768 * 10 ** (AMBIENCE_DBFS / 20) / (rms(a) or 1))
        out = out_root / "ambience" / kind / f"{env.lower()}.wav"
        write_wav(out, a)
        files.append({"path": str(out.relative_to(out_root)), "group": "ambience", "name": kind, "dataset": "DEMAND",
                      "source": f"{env}_16k.zip:{env}/ch01.wav", "license": DEMAND_LICENSE, "dur_s": round(len(a) / SR, 2)})
    return files


def build_demand_bank(bank: Path, retries: int = 2, log=lambda *a: None) -> Path:
    """Download DEMAND (ch01 of ``DEMAND_MAP``) and build the ambience-only bank at ``bank`` (atomic: built in
    ``<bank>.tmp``, then renamed; the downloads are deleted). Raises on failure."""
    bank = Path(bank)
    bank.parent.mkdir(parents=True, exist_ok=True)
    dl, tmp = bank.with_name(bank.name + ".downloads"), bank.with_name(bank.name + ".tmp")
    download_demand(dl, retries=retries, log=log)
    if tmp.exists():
        shutil.rmtree(tmp)
    files = build_ambience(dl, tmp, log=log)
    if len(files) != len(DEMAND_MAP):
        raise IOError(f"DEMAND: {len(files)} of {len(DEMAND_MAP)} environments built")
    man = {"sr": SR, "ambience_dbfs": AMBIENCE_DBFS,
           "sources": {"DEMAND": {"license": DEMAND_LICENSE, "url": DEMAND_RECORD, "map": DEMAND_MAP}},
           "files": files, "counts": {"ambience": {k: sum(f["name"] == k for f in files) for k in set(DEMAND_MAP.values())},
                                      "events": {}},
           "note": "ambience only (DEMAND); noise events need the full bank: scripts/fetch_noise_banks.py"}
    man["bytes"] = sum((tmp / f["path"]).stat().st_size for f in files)
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1))
    if bank.exists():  # an empty / partial directory (no ambience): replaced
        shutil.rmtree(bank)
    tmp.rename(bank)
    shutil.rmtree(dl, ignore_errors=True)
    return bank


# ---------------------------------------------------------------- the default bank


def default_bank_dir() -> Path | None:
    """``$IG_SOUNDBANK``, else ``$XDG_CACHE_HOME/interaction_gym/soundbank``; None if ``$IG_SOUNDBANK`` is ``synthetic``."""
    v = getenv("IG_SOUNDBANK")
    if v is not None:
        return None if v.strip().lower() in OFF else Path(v).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(cache) / "interaction_gym" / "soundbank"


def fetch_enabled() -> bool:
    return (getenv("IG_SOUNDBANK_FETCH") or "1").strip().lower() not in ("0", "false", "no", "off")


def bank_ready(root: Path) -> bool:
    """Has ``root`` any ambience recordings (a bank built here or by scripts/fetch_noise_banks.py)?"""
    return any(Path(root).glob("ambience/*/*.wav"))


_lock = threading.Lock()
_resolved: dict[tuple, Path | None] = {}


def _locked_build(root: Path) -> None:
    """Build the DEMAND bank at ``root`` unless another process did meanwhile (a file lock: parallel workers fetch
    it once)."""
    root.parent.mkdir(parents=True, exist_ok=True)
    with open(root.with_name(root.name + ".lock"), "w") as lf:
        try:
            import fcntl

            fcntl.flock(lf, fcntl.LOCK_EX)
        except ImportError:  # no fcntl (Windows): no cross-process lock
            pass
        if not bank_ready(root):
            print(f"interaction_gym: fetching background ambience from DEMAND (Thiemann, Ito & Vincent 2013; "
                  f"CC BY-SA 3.0) once, ~80 MB, into {root} (IG_SOUNDBANK=synthetic to skip)", file=sys.stderr, flush=True)
            build_demand_bank(root)


def default_bank(fetch: bool = True) -> Path | None:
    """The default bank's directory, fetching the DEMAND ambience into it first if it has none and ``fetch`` (and
    ``$IG_SOUNDBANK_FETCH`` allows); None (synthetic stand-ins) if it is disabled, missing or the fetch failed.
    Resolved once per process and setting."""
    root = default_bank_dir()
    if root is None:
        return None
    fetch = fetch and fetch_enabled()
    key = (str(root), fetch)
    with _lock:
        if key in _resolved:
            return _resolved[key]
        if bank_ready(root):
            out = root
        elif not fetch:
            out = None
        else:
            try:
                _locked_build(root)
                out = root if bank_ready(root) else None
            except Exception as e:  # noqa: BLE001  (offline, a sandbox, a server error: never fail the episode)
                warnings.warn(f"interaction_gym: could not fetch the DEMAND ambience into {root} ({e!r}); using "
                              f"synthetic background noise. Set IG_SOUNDBANK=synthetic to silence this.", RuntimeWarning, stacklevel=2)
                out = None
        _resolved[key] = out
        if out is not None:
            _resolved[(str(root), False)] = out
        return out


def reset() -> None:
    """Forget the resolved default bank (tests, or after building one)."""
    with _lock:
        _resolved.clear()


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Pre-build the default sound bank (DEMAND ambience) in the cache.")
    ap.add_argument("--dir", help="bank directory (default: $IG_SOUNDBANK or the user cache)")
    a = ap.parse_args(argv)
    root = Path(a.dir).expanduser() if a.dir else default_bank_dir()
    if root is None:
        sys.exit("IG_SOUNDBANK is set to synthetic: nothing to build")
    if bank_ready(root):
        log("bank ready:", root)
        return
    log("fetching DEMAND (CC BY-SA 3.0) into", root)
    build_demand_bank(root, retries=8, log=log)
    log("bank ready:", root)


if __name__ == "__main__":
    main()
