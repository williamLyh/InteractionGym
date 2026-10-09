"""HumDial-FDBench (ICASSP 2026 HumDial Challenge, full-duplex track) as InteractionGym test episodes.

Data: arXiv 2604.21406, Hugging Face ``ASLP-lab/HumDial-FDBench`` (Apache-2.0), one zip
(``Humdial-Track2-Test.zip``, 1.8 GB, 6.9 GB unpacked) → ``test/{en,cn}_test_nondev/<category>/``. Per sample:
``<id>.wav`` (the user's channel, 16 kHz mono, human-recorded), ``<id>.json`` (``speech_segments``: ``xmin``,
``xmax``, ``text``; ``final_duration``), ``<id>_timestamp.json`` (word / character timestamps) and, for most
categories, ``clean_<id>.*`` without the overlap event. The release has the user side only (the paper's
dual-channel conversations are not in it) and no reference answers.

Each sample is laid out like FD-Bench v1.5: a request, then 5 s after it ends — while the agent answers — an
event; the episode lasts the wav (≈10 s after the last segment). Mapping (category → the event turn):

===========================  ======================================================================================
ask / deny / repeat / shift  a follow-up / correction / "say that again" / topic change: ``expects: "yield"``
                             (then answered: ``respond``)
wait                         "I'm busy, talk later", "hold on": ``expects: "wait"`` (``yield``, then stay quiet)
backchannel                  "Oh, I see." ``kind: "backchannel"`` (``ignore``)
talk_to_others               request, follow-up (normal turn), then a remark to someone else ``kind: "aside"``
others_talk_to_user_after    request, then a third person talking to the user ``kind: "noise"``
others_talk_to_user_before   a third person talking to the user first (``kind: "noise"``), then the request
pause                        one request with a mid-sentence ``[break]``, split at the longest gap between its
                             word timestamps; the first half ``expects: "wait"``
===========================  ======================================================================================

Everything outside the turns stays as a background track (``full_duplex_bench._build``), so the agent hears
the wav sample for sample. ``clean=True`` plays ``clean_<id>.wav`` (no event) where it exists.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core import Task
from .full_duplex_bench import Sample, _build, _ms, _trim, make_env  # noqa: F401  (make_env: one runner for all)
from .full_duplex_bench import split_of as _split_of
from .wav import read_wav, resample

NAME = "HumDial-FDBench"
HF = "ASLP-lab/HumDial-FDBench"
LICENSE = "Apache-2.0"
LANGS = {"en": "en_test_nondev", "zh": "cn_test_nondev"}
INTERRUPT = ("ask", "deny", "repeat", "shift")
CATEGORIES = (*INTERRUPT, "wait", "backchannel", "talk_to_others", "others_talk_to_user_after",
              "others_talk_to_user_before", "pause")
SUBSETS = {f"humdial/{lang}/{c}": (lang, c) for lang in LANGS for c in CATEGORIES}
PERSONA = "Pre-recorded HumDial-FDBench user (offline replay)."


def _dir(root: str | Path, subset: str) -> Path:
    lang, cat = SUBSETS[subset]
    return Path(root) / "test" / LANGS[lang] / cat


def split_of(lang: str, sample_id: str, material_frac: float = 0.0) -> str:
    """Ids are ``<speaker>_<item>``: split by speaker, so no voice is in both material (``clip_bank``) and evaluation."""
    return _split_of(f"humdial/{lang}", sample_id.split("_")[0], material_frac)


def ids(root: str | Path, subset: str, split: str | None = None, material_frac: float = 0.0) -> list[str]:
    """Sample ids of ``subset``; ``split="eval"`` keeps the speakers held out of the clip bank (same ``material_frac``)."""
    d = _dir(root, subset)
    out = sorted(p.stem for p in d.glob("*.json")
                 if not p.stem.endswith("_timestamp") and not p.stem.startswith("clean_") and (d / f"{p.stem}.wav").exists())
    return out if split is None else [i for i in out if split_of(SUBSETS[subset][0], i, material_frac) == split]


def _split_pause(seg: dict, words: list[dict]) -> list[dict]:
    """``[break]`` → two spans, cut in the longest gap between consecutive word timestamps inside the segment."""
    ws = sorted((c["timestamp"] for c in words if seg["xmin"] - 0.05 <= c["timestamp"][0] <= seg["xmax"] + 0.05), key=lambda t: t[0])
    parts = [p.strip() for p in seg["text"].split("[break]", 1)]
    if len(ws) < 2 or len(parts) < 2:
        return [seg]
    gap, k = max((ws[i + 1][0] - ws[i][1], i) for i in range(len(ws) - 1))
    cut0, cut1 = ws[k][1], ws[k + 1][0]
    return [{"xmin": seg["xmin"], "xmax": cut0, "text": parts[0], "expects": "wait", "pause_s": round(gap, 3)},
            {"xmin": cut1, "xmax": seg["xmax"], "text": parts[1]}]


def _roles(cat: str, segs: list[dict]) -> list[dict]:
    """Label the segments of one sample (see the module table)."""
    out = [dict(s) for s in segs]
    if cat in INTERRUPT and len(out) >= 2:
        out[1]["expects"] = "yield"
    elif cat == "wait" and len(out) >= 2:
        out[1]["expects"] = "wait"
    elif cat == "backchannel" and len(out) >= 2:
        out[1]["kind"] = "backchannel"
    elif cat == "talk_to_others" and len(out) >= 3:
        out[2]["kind"] = "aside"
    elif cat == "others_talk_to_user_after":
        for s in out[1:]:
            s["kind"] = "noise"
    elif cat == "others_talk_to_user_before" and len(out) >= 2:
        out[0]["kind"] = "noise"
    return out


def load_sample(root: str | Path, subset: str, sample_id: str, *, sr: int = 16000, exact: bool = True,
                clean: bool = False) -> Sample:
    lang, cat = SUBSETS[subset]
    d = _dir(root, subset)
    stem = f"clean_{sample_id}" if clean else sample_id
    if clean and not (d / f"{stem}.wav").exists():
        raise ValueError(f"{subset} has no clean input for {sample_id}")
    src = read_wav(d / f"{stem}.wav")
    inp = resample(src, sr)
    dur = inp.dur_ms
    ann = json.loads((d / f"{stem}.json").read_text(encoding="utf-8")) if (d / f"{stem}.json").exists() else \
        json.loads((d / f"{sample_id}.json").read_text(encoding="utf-8"))
    segs = sorted(ann["speech_segments"], key=lambda s: s["xmin"])
    if clean and not (d / f"{stem}.json").exists():
        segs = segs[:1]  # no clean annotation: the request alone
    ts = d / f"{sample_id}_timestamp.json"
    words = json.loads(ts.read_text(encoding="utf-8")).get("chunks", []) if ts.exists() else []
    if cat == "pause":
        segs = [p for s in segs for p in _split_pause(s, words)]
    spans = []
    for s in (_roles(cat, segs) if not clean else [dict(x) for x in segs]):
        a, b = _trim(inp, max(0, _ms(s["xmin"])), min(dur, _ms(s["xmax"])))
        spans.append({"start_ms": a, "end_ms": b, "text": s["text"].replace("[break]", "").strip(),
                      "kind": s.get("kind"), "expects": s.get("expects")})
    turns, bg = _build(inp, spans, exact)
    bench = {"name": NAME, "version": "test_nondev", "subset": subset, "task": cat, "language": lang,
             "sample_id": sample_id, "clean": clean, "source": str(d / f"{stem}.wav"), "source_sr": src.sr,
             "duration_ms": dur, "license": LICENSE, "annotation": {"segments": segs}}
    scenario = {"persona": PERSONA, "language": lang, "benchmark": bench,
                "turns": [{k: v for k, v in t.items() if k != "audio"} | {"dur": t["audio"].dur_ms} for t in turns]}
    tid = f"hd-{lang}-{cat}-{sample_id}" + ("-clean" if clean else "")
    return Sample(tid, subset, Task(id=tid, scenario=scenario, criteria={"benchmark": NAME, "task": cat}), turns, dur, bg)


def load(root: str | Path, subset: str, *, limit: int | None = None, **kw) -> list[Sample]:
    return [load_sample(root, subset, i, **kw) for i in ids(root, subset)[:limit]]
