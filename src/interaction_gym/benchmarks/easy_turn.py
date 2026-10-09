"""Easy Turn testset as InteractionGym test episodes.

Data: Li et al., "Easy Turn" (arXiv 2509.23938), Hugging Face ``ASLP-lab/Easy-Turn-Testset`` (Apache-2.0):
800 isolated Mandarin utterances labelled with the turn state a full-duplex system should infer at their end —
``complete`` 300, ``incomplete`` 300, ``backchannel`` 100, ``wait`` 100 (each half real recordings, half
CosyVoice 2). Layout: ``<root>/testset/<state>/*.list`` (JSON lines: ``key``, ``wav`` relative to ``testset/``,
``txt`` with a trailing state tag, ``speaker``, ``duration``) and ``<root>/testset/<state>/{real,synthetic}/*.wav``.

The benchmark is a classifier test (one clip → one label). Here each clip becomes a short episode whose
``expects`` encode the label, scored by ``eval.scores``:

============  ==================================================================================================
complete      the clip alone, then ``tail_ms`` of silence: a normal turn (``respond``: answered, not talked over)
incomplete    the clip alone (it stops mid-clause: "因为小时候…"), then ``wait_ms``: ``expects: "wait"``
              (``wait``: the agent should not take the floor)
backchannel   composed: an opening (a ``complete`` clip) and, ``gap_ms`` after it ends — while the agent is
              answering — the backchannel clip with ``kind: "backchannel"`` (``ignore``), as in FD-Bench v1.5
wait          composed the same way; the clip ("别说了", "立即静音") ``expects: "wait"`` (``wait``: if the agent is
              talking it should stop, and then stay quiet)
============  ==================================================================================================

The composed openings come from the ``complete`` set (same speaker when one exists, else a stable pick of the
same real / synthetic kind), so an opening clip also appears as a ``complete`` episode of its own. The episode
length is fixed (``make_env`` from ``full_duplex_bench``), so replies are never cut by an early end.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..core import Task
from .full_duplex_bench import Sample, make_env  # noqa: F401  (make_env re-exported: one runner for both)
from .wav import read_wav, resample, speech_segments

NAME = "Easy Turn"
HF = "ASLP-lab/Easy-Turn-Testset"
LICENSE = "Apache-2.0"
STATES = ("complete", "incomplete", "backchannel", "wait")
SUBSETS = {f"easy_turn/{s}": s for s in STATES}
PERSONA = "Pre-recorded Easy Turn speaker (offline replay)."
TAG = re.compile(r"<[^>]*>")


def records(root: str | Path, state: str) -> list[dict]:
    """The list entries of one state, in file order, with ``wav`` resolved and ``text`` without tags."""
    d = Path(root) / "testset" / state
    out = []
    for lst in sorted(d.glob("*.list")):
        for line in lst.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.append(r | {"wav": str((d.parent / r["wav"]).resolve()), "text": TAG.sub("", r["txt"]).strip(),
                                "synthetic": "real" not in r["key"]})
    return out


def _voiced(path: str, sr: int):
    """The clip at ``sr``, cut to its voiced part (turn ends where the voice stops, not the file)."""
    a = resample(read_wav(path), sr)
    v = speech_segments(a, rms_threshold=150.0, min_speech_ms=60, pad_ms=30)
    return a if not v else a[round(v[0][0] * a.sr) : round(v[-1][1] * a.sr)]


def opening_for(rec: dict, openings: list[dict]) -> dict:
    """A ``complete`` clip to open a composed episode with: same speaker if possible, else a stable pick
    among clips of the same kind (real / synthetic)."""
    same = [o for o in openings if o.get("speaker") not in (None, "<NONE>") and o.get("speaker") == rec.get("speaker")]
    pool = same or [o for o in openings if o["synthetic"] == rec["synthetic"]] or openings
    return pool[int(hashlib.sha256(rec["key"].encode()).hexdigest()[:8], 16) % len(pool)]


def load_sample(root: str | Path, subset: str, key: str, *, sr: int = 16000, lead_ms: int = 500, gap_ms: int = 4000,
                tail_ms: int = 5000, wait_ms: int = 3000, _cache: dict | None = None) -> Sample:
    """One Easy Turn clip (``key``) of ``subset`` (``easy_turn/<state>``) as a ``Sample`` for ``make_env``."""
    state = SUBSETS[subset]
    cache = _cache if _cache is not None else {}
    recs = cache.get(state) or cache.setdefault(state, records(root, state))
    rec = next(r for r in recs if r["key"] == key)
    clip = _voiced(rec["wav"], sr)
    turns, ann = [], {"key": key, "text": rec["txt"], "speaker": rec.get("speaker"), "synthetic": rec["synthetic"]}
    t = lead_ms
    if state in ("backchannel", "wait"):
        op = opening_for(rec, cache.get("complete") or cache.setdefault("complete", records(root, "complete")))
        oa = _voiced(op["wav"], sr)
        turns.append({"t": t, "text": op["text"], "audio": oa})
        ann["opening"] = {"key": op["key"], "text": op["txt"], "speaker": op.get("speaker")}
        t += oa.dur_ms + gap_ms
    turn = {"t": t, "text": rec["text"], "audio": clip}
    if state == "incomplete":
        turn["expects"] = "wait"
    elif state == "backchannel":
        turn["kind"] = "backchannel"
    elif state == "wait":
        turn["expects"] = "wait"
    turns.append(turn)
    end = t + clip.dur_ms
    dur = end + (wait_ms if state in ("incomplete", "wait") else tail_ms)
    bench = {"name": NAME, "version": "testset", "subset": subset, "task": state, "sample_id": key, "clean": False,
             "source": rec["wav"], "duration_ms": dur, "license": LICENSE, "annotation": ann}
    scenario = {"persona": PERSONA, "language": "zh", "benchmark": bench,
                "turns": [{k: v for k, v in x.items() if k != "audio"} | {"dur": x["audio"].dur_ms} for x in turns]}
    tid = f"et-{state}-{key}"
    return Sample(tid, subset, Task(id=tid, scenario=scenario, criteria={"benchmark": NAME, "task": state}), turns, dur)


def ids(root: str | Path, subset: str) -> list[str]:
    return [r["key"] for r in records(root, SUBSETS[subset])]


def load(root: str | Path, subset: str, *, limit: int | None = None, **kw) -> list[Sample]:
    cache: dict = {}
    return [load_sample(root, subset, k, _cache=cache, **kw) for k in ids(root, subset)[:limit]]
