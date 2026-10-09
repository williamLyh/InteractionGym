"""Download DEMAND and MUSAN (noise) and build a local sound bank for ``soundscape.SoundBank``.

    python scripts/fetch_noise_banks.py --out ~/dig_soundbank            # download + build
    python scripts/fetch_noise_banks.py --out DIR --stage download       # just download (resumable)
    python scripts/fetch_noise_banks.py --out DIR --stage build          # build from what was downloaded

The bank (``DIR/bank``) is laid out as the env expects (docs/FORMAT.md §4.2):

- ``ambience/<kind>/<DEMAND env>.wav``: one channel (ch01) of each mapped DEMAND recording (``DEMAND_MAP``), 5 min,
  mono 16 kHz 16-bit, RMS-normalised to ``AMBIENCE_DBFS``;
- ``events/<label>/<musan id>.wav``: MUSAN noise clips with a clear label (``EVENT_RULES`` on a sound-bible clip's
  title; for the untitled free-sound clips the reviewed ``scripts/noise_labels.json``; anything else is left out), trimmed to their active part (at most ``MAX_EVENT_S[label]``), mono
  16 kHz 16-bit, normalised so the active part has an RMS of ``EVENT_DBFS`` (the env sets the level per label);
- ``manifest.json``: every file with its source file, dataset, license (and the clip's own license and attribution),
  the rule that mapped it, plus per-label / per-kind counts.

Nothing here is redistributed by the repository: you download the datasets yourself and must follow their licenses
(DEMAND: CC BY-SA 3.0; MUSAN: CC BY 4.0, with per-clip attribution in its ANNOTATIONS files; see THIRD_PARTY.md).

DEMAND comes from its Zenodo record; only ``ch01.wav`` of each zip is fetched (HTTP range requests on the zip, so
~7 MB per environment instead of ~100 MB). MUSAN is one 11 GB archive from OpenSLR (SLR17) or a mirror; only
``musan/noise/`` is extracted. Standard library only.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import wave
import zipfile
from array import array
from pathlib import Path

SR = 16000
AMBIENCE_DBFS = -30.0
EVENT_DBFS = -20.0  # the active part of an event clip, the same as nominal speech (soundscape.SPEECH_DBFS)

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

MUSAN_PATH = "resources/17/musan.tar.gz"
MUSAN_MIRRORS = ["https://openslr.elda.org", "https://openslr.magicdatatech.com", "https://www.openslr.org",
                 "https://us.openslr.org", "https://openslr.trmal.net"]
# the same archive mirrored on Hugging Face (reachable through hf-mirror.com where huggingface.co is not)
MUSAN_EXTRA = ["https://hf-mirror.com/datasets/huseinzol05/musan-mirror/resolve/main/musan.tar.gz",
               "https://huggingface.co/datasets/huseinzol05/musan-mirror/resolve/main/musan.tar.gz"]
MUSAN_LICENSE = "CC BY 4.0 (MUSAN, Snyder, Chen & Povey 2015; OpenSLR SLR17); per-clip sources in musan/noise/*/ANNOTATIONS"

# MUSAN noise clip -> event label. ``noise/sound-bible`` clips have titles (in its LICENSE file): the first rule whose
# pattern matches the title (lower-cased, - and _ as spaces) gives the label, and a title matching EXCLUDE or several
# labels is not used. ``noise/free-sound`` clips have no metadata: their labels come from a reviewed, model-assisted
# file (``--labels``, made by scripts/label_musan_noise.py); unlabelled clips are left out.
EVENT_RULES = [
    ("siren", r"\bsiren|ambulance|^police$|woop woop"),
    ("horn", r"\b(car|truck|bus|air|bike|bicycle|vehicle) horns?\b|\bhonk"),
    ("dog_bark", r"\bdogs?\b.*\bbark|\bbark(s|ing)?\b|\bwoof"),
    ("phone_ring", r"\b(tele|home |cell ?)?phones? ring|telephone ring"),
    ("door_slam", r"\bdoors? (slam|bang|shut)|slam(s|ming)? (the )?door|door slam"),
    ("cough", r"\bcough"),
    ("sneeze", r"\bsneez"),
    ("throat_clear", r"throat clear|clear(s|ing)? (his |her |the )?throat"),
    ("dishes", r"\bdish(es)?\b(?! ?washer)|\bplates?\b|cutlery|silverware|clinking"),
    ("keyboard", r"\bkeyboard|\btyping\b"),
    ("cup", r"\bcups?\b|\bmugs?\b|teaspoon"),
    ("indicator", r"\b(turn signal|indicator|blinker)"),
]
EXCLUDE = r"\bmusic|\bsong|\bspeech|\bvoices?\b|talking|crowd|applause|laugh|ringtone|vibrat|alarm"
MAX_EVENT_S = {"siren": 8.0, "horn": 3.0, "dog_bark": 3.0, "phone_ring": 6.0, "door_slam": 2.0, "cough": 2.5,
               "sneeze": 2.0, "throat_clear": 2.0, "dishes": 3.0, "keyboard": 4.0, "cup": 3.0, "indicator": 4.0}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- download


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file over HTTP range requests (for ``zipfile``), with retries."""

    def __init__(self, url: str, size: int, block: int = 1 << 20):
        self.url, self.size, self.pos, self.block, self.cache = url, size, 0, block, {}

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
            for attempt in range(8):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    if len(data) == hi - lo + 1:
                        break
                except Exception as e:  # noqa: BLE001
                    log("range retry", attempt, repr(e))
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


def _json(url: str):
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            log("retry", url, repr(e))
            time.sleep(10 * (attempt + 1))
    raise IOError(url)


def download_demand(dl: Path) -> None:
    d = dl / "demand"
    d.mkdir(parents=True, exist_ok=True)
    rec = _json(DEMAND_RECORD)
    (d / "record.json").write_text(json.dumps({k: rec[k] for k in ("doi", "links", "metadata") if k in rec}, indent=1))
    files = {f["key"]: f for f in rec["files"]}

    def one(env: str):
        out = d / f"{env}_ch01.wav"
        if out.exists():
            return
        f = files[f"{env}_16k.zip"]
        log("DEMAND", env, f"{f['size'] / 1e6:.0f} MB zip, fetching ch01 only")
        with zipfile.ZipFile(RangeFile(f["links"]["self"], f["size"], block=1 << 18)) as z:
            name = next(n for n in z.namelist() if n.endswith("ch01.wav"))
            data = z.read(name)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.rename(out)
        log("DEMAND", env, "ok", len(data))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(len(DEMAND_MAP)) as ex:  # the server is slow per connection: all environments at once
        list(ex.map(one, DEMAND_MAP))


def _size(url: str) -> int | None:
    try:
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            return int(cr.rsplit("/", 1)[1]) if "/" in cr else None
    except Exception:  # noqa: BLE001
        return None


def _get(url: str, lo: int, hi: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def segmented_download(urls: list[str], out: Path, size: int, parts: int = 16) -> None:
    """``out`` from ``urls`` (the same file on several servers), in ``parts`` parallel HTTP range downloads, each
    resumable (``out.part<i>`` keeps what arrived); retried, rotating through the servers."""
    import threading

    step = -(-size // parts)
    errors = []

    def worker(i: int):
        lo, hi = i * step, min((i + 1) * step, size) - 1
        part = out.with_name(f"{out.name}.part{i}")
        for attempt in range(200):
            have = part.stat().st_size if part.exists() else 0
            if lo + have > hi:
                return
            url = urls[(i + attempt) % len(urls)]
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={lo + have}-{hi}"})
                with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as f:
                    if r.status != 206:
                        raise IOError(f"status {r.status}")
                    t0, n0 = time.monotonic(), 0
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        n0 += len(chunk)
                        if time.monotonic() - t0 > 60:  # a slow server: try another one
                            if n0 / (time.monotonic() - t0) < 100_000:
                                raise IOError(f"slow ({n0 / (time.monotonic() - t0) / 1e3:.0f} kB/s)")
                            t0, n0 = time.monotonic(), 0
            except Exception as e:  # noqa: BLE001
                log(f"part {i} retry {attempt} ({url.split('/')[2]}): {e!r}")
                time.sleep(min(60, 5 + attempt))
        errors.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(parts)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(60)
        got = sum(out.with_name(f"{out.name}.part{i}").stat().st_size for i in range(parts)
                  if out.with_name(f"{out.name}.part{i}").exists())
        log(f"{out.name}: {got / 1e9:.2f} / {size / 1e9:.2f} GB")
    if errors:
        raise IOError(f"parts {errors} failed")
    tmp = out.with_suffix(".cat")
    with open(tmp, "wb") as f:
        for i in range(parts):
            p = out.with_name(f"{out.name}.part{i}")
            with open(p, "rb") as src:
                shutil.copyfileobj(src, f, 1 << 24)
    if tmp.stat().st_size != size:
        raise IOError(f"{out}: {tmp.stat().st_size} bytes, expected {size}")
    tmp.rename(out)
    for i in range(parts):
        out.with_name(f"{out.name}.part{i}").unlink()


def download_musan(dl: Path, mirrors=MUSAN_MIRRORS, extra=MUSAN_EXTRA) -> Path:
    """The MUSAN archive (segmented, resumable) from the OpenSLR mirrors (plus ``extra`` copies of the same file,
    used only if their size matches OpenSLR's and spot checks of their bytes match), then ``musan/noise`` extracted."""
    d = dl / "musan"
    d.mkdir(parents=True, exist_ok=True)
    tgz, noise = d / "musan.tar.gz", d / "musan" / "noise"
    if (d / "noise.done").exists():
        return noise
    if not tgz.exists():
        official = [f"{m}/{MUSAN_PATH}" for m in mirrors]
        sizes = {u: _size(u) for u in official}
        ref = next((u for u, n in sizes.items() if n), None)
        if ref is None:
            raise IOError("no OpenSLR mirror reachable")
        size = sizes[ref]
        urls = [u for u, n in sizes.items() if n == size]
        for u in extra:  # e.g. a Hugging Face copy: only if it is byte-identical where we look
            if _size(u) == size and all(_get(u, o, o + 65535) == _get(ref, o, o + 65535)
                                        for o in (0, size // 3, size - 65536)):
                urls.insert(0, u)
                log("MUSAN: verified copy", u)
        log("MUSAN:", size, "bytes from", [u.split("/")[2] for u in urls])
        (d / "sources.json").write_text(json.dumps({"size": size, "reference": ref, "urls": urls}, indent=1))
        segmented_download(urls, tgz, size)
    log("MUSAN: extracting musan/noise")
    r = subprocess.run(["tar", "-xzf", str(tgz), "-C", str(d), "--wildcards", "musan/noise/*", "musan/LICENSE", "musan/README"])
    if not noise.is_dir():
        raise IOError(f"MUSAN extract failed ({r.returncode})")
    (d / "noise.done").write_text("")
    return noise


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


def active_part(a: array, sr: int, max_s: float, frame_ms: int = 20, rel_db: float = -30.0) -> array | None:
    """The clip from its first to last frame within ``rel_db`` of its loudest frame, at most ``max_s`` long (from
    the onset), with 20 ms margins and 10 ms fades; None for a silent clip."""
    f = sr * frame_ms // 1000
    levels = [rms(a[i:i + f]) for i in range(0, len(a) - f + 1, f)]
    if not levels or max(levels) < 30:
        return None
    thr = max(levels) * 10 ** (rel_db / 20)
    on = [i for i, v in enumerate(levels) if v >= thr]
    lo = max(0, on[0] * f - f)
    hi = min(len(a), (on[-1] + 2) * f, lo + int(max_s * sr))
    out = array("h", a[lo:hi])
    fade = min(sr // 100, len(out) // 4)
    for i in range(fade):
        out[i] = int(out[i] * i / fade)
        out[-1 - i] = int(out[-1 - i] * i / fade)
    return out


# ---------------------------------------------------------------- build


def musan_titles(noise: Path) -> dict[str, str]:
    """{clip id: title} from musan/noise/sound-bible/LICENSE (blocks: id, Title:, License:, Recorded by, URL)."""
    return {cid: m["title"] for cid, m in musan_meta(noise).items() if m.get("title")}


def musan_meta(noise: Path) -> dict[str, dict]:
    """{clip id: {title, license, author, url}} for the sound-bible clips; free-sound clips are public domain (its LICENSE)."""
    out, cur = {}, None
    lic = noise / "sound-bible" / "LICENSE"
    for line in (lic.read_text(errors="replace").splitlines() if lic.exists() else []):
        line = line.strip()
        if line.startswith("noise-"):
            cur = out.setdefault(line, {})
        elif cur is not None and line.startswith("Title:"):
            cur["title"] = line[6:].strip()
        elif cur is not None and line.startswith("License:"):
            cur["license"] = line[8:].strip()
        elif cur is not None and line.startswith("Recorded by"):
            cur["author"] = line[11:].strip()
        elif cur is not None and line.startswith("http"):
            cur["url"] = line
    return out


def label_of(text: str) -> tuple[str | None, str | None]:
    """(label, matching rule) for a clip title, or (None, why not)."""
    t = re.sub(r"[-_/+%.]+", " ", text.lower())
    t = re.sub(r"\b\d+\b", " ", t)
    if re.search(EXCLUDE, t):
        return None, "excluded"
    hits = [(lab, rx) for lab, rx in EVENT_RULES if re.search(rx, t)]
    if not hits:
        return None, "no rule"
    if len({h[0] for h in hits}) > 1:  # e.g. "dog barking and car horn": not a clear single label
        return None, "ambiguous:" + "+".join(sorted({h[0] for h in hits}))
    return hits[0][0], hits[0][1]


def build(dl: Path, bank: Path, overrides: dict[str, str | None] | None = None) -> dict:
    """Write ``bank`` from the downloads; ``overrides`` ({musan id: label or None}, e.g. a reviewed labels file) win
    over the rules."""
    tmp = bank.with_name(bank.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    man = {"sr": SR, "ambience_dbfs": AMBIENCE_DBFS, "event_dbfs": EVENT_DBFS,
           "sources": {"DEMAND": {"license": DEMAND_LICENSE, "url": DEMAND_RECORD, "map": DEMAND_MAP},
                       "MUSAN": {"license": MUSAN_LICENSE, "url": f"https://www.openslr.org/17/"}},
           "files": [], "skipped": {}}
    for env, kind in DEMAND_MAP.items():
        src = dl / "demand" / f"{env}_ch01.wav"
        if not src.exists():
            log("missing", src)
            continue
        a, sr = read_wav(src)
        a = resample(a, sr)
        a = scale(a, 32768 * 10 ** (AMBIENCE_DBFS / 20) / (rms(a) or 1))
        out = tmp / "ambience" / kind / f"{env.lower()}.wav"
        write_wav(out, a)
        man["files"].append({"path": str(out.relative_to(tmp)), "group": "ambience", "name": kind, "dataset": "DEMAND",
                             "source": f"{env}_16k.zip:{env}/ch01.wav", "license": DEMAND_LICENSE, "dur_s": round(len(a) / SR, 2)})
    noise = dl / "musan" / "musan" / "noise"
    meta = musan_meta(noise) if noise.is_dir() else {}
    overrides = overrides or {}
    for p in sorted(noise.glob("*/*.wav")) if noise.is_dir() else []:
        cid = p.stem
        m = meta.get(cid, {})
        text = m.get("title", "")
        if cid in overrides:  # a reviewed label (null: drop the clip)
            label, rule = overrides[cid], "labels file"
            if label is None:
                rule = "dropped in review"
        elif text:
            label, rule = label_of(text)
        else:
            label, rule = None, "untitled, unlabelled"
        if label is None:
            man["skipped"][rule] = man["skipped"].get(rule, 0) + 1
            continue
        a, sr = read_wav(p)
        a = active_part(resample(a, sr), SR, MAX_EVENT_S.get(label, 4.0))
        if a is None or len(a) < SR // 10:
            man["skipped"]["silent"] = man["skipped"].get("silent", 0) + 1
            continue
        act = [x for x in a if abs(x) > 0]
        a = scale(a, 32768 * 10 ** (EVENT_DBFS / 20) / (rms(act) or 1))
        out = tmp / "events" / label / f"{cid}.wav"
        write_wav(out, a)
        man["files"].append({"path": str(out.relative_to(tmp)), "group": "events", "name": label, "dataset": "MUSAN",
                             "source": f"musan/noise/{p.parent.name}/{p.name}", "title": text, "rule": rule,
                             "license": MUSAN_LICENSE, "clip_license": m.get("license", "Public Domain"),
                             "attribution": " ".join(x for x in (m.get("author") and f"by {m['author']}", m.get("url")) if x)
                             or "Freesound (public domain, per MUSAN)", "dur_s": round(len(a) / SR, 2)})
    counts: dict[str, dict[str, int]] = {"ambience": {}, "events": {}}
    for f in man["files"]:
        counts[f["group"]][f["name"]] = counts[f["group"]].get(f["name"], 0) + 1
    man["counts"] = counts
    man["bytes"] = sum((tmp / f["path"]).stat().st_size for f in man["files"])
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    if bank.exists():
        shutil.rmtree(bank)
    tmp.rename(bank)
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="working directory: downloads/ and bank/ go here")
    ap.add_argument("--stage", choices=("download", "demand", "musan", "build", "all"), default="all")
    ap.add_argument("--labels", default=str(Path(__file__).with_name("noise_labels.json")),
                    help="JSON {musan clip id: label or null}: reviewed labels for untitled clips / overrides (default: "
                         "scripts/noise_labels.json, if present)")
    a = ap.parse_args(argv)
    root = Path(os.path.expanduser(a.out))
    dl = root / "downloads"
    if a.stage in ("download", "demand", "all"):
        download_demand(dl)
    if a.stage in ("download", "musan", "all"):
        download_musan(dl)
    if a.stage in ("build", "all"):
        over = json.loads(Path(a.labels).read_text()) if a.labels and Path(a.labels).exists() else None
        man = build(dl, root / "bank", over)
        log("bank:", json.dumps(man["counts"]), f"{man['bytes'] / 1e6:.0f} MB", "skipped:", json.dumps(man["skipped"]))


if __name__ == "__main__":
    main()
