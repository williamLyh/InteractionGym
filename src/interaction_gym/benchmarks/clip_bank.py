"""A clip bank from benchmark data: content-independent sounds (backchannels, asides, background speech) and
task-agnostic openings as audio files + a JSONL manifest, plus human-behaviour statistics to calibrate the
user simulator with.

Manifest record (one per clip, ``manifest.jsonl``)::

    {"id": "fdb15-bc-12", "kind": "backchannel", "text": "right yeah", "dur_ms": 640, "sr": 16000,
     "audio": "clips/backchannel/fdb15-bc-12.wav", "use": "clip",            # "clip" | "opening" | "interruption"
     "split": "material",                                                    # "material" | "eval" (see below)
     "source": {"benchmark": "Full-Duplex-Bench", "subset": "v1.5/user_backchannel", "sample_id": "12",
                "file": "backchannel.wav", "license": "MIT", "synthetic": true}}

``kind`` follows docs/FORMAT.md §4.4 (``backchannel`` / ``aside`` / ``noise``; ``null`` for directed speech).

Contamination: every clip carries the ``split`` of the benchmark sample it was cut from
(``full_duplex_bench.split_of(subset, id, material_frac)``; samples sharing content across subsets share a split),
and a material opening / interruption is dropped when its words also occur in any evaluation-split sample (v1.5
reuses request texts; content-independent clips like "yeah" are kept — their audio is per sample). Only ``material`` clips should feed training
scenarios; evaluation should then use ``full_duplex_bench.load(…, split="eval", material_frac=…)`` with the
same fraction, and say so (published numbers are over all samples). ``extract`` writes eval-split clips too
only when asked (``include_eval=True``, e.g. to inspect them), still labelled.

HumDial-FDBench (``extract_humdial``, Apache-2.0, real human recordings, en + zh) adds the same kinds of clips plus
interruptions with an ``intent``; it is split by speaker. Its records carry ``lang``.

Licensing: only the MIT subsets (v1.5, v1.0 synthetic) are extracted by default; the Candor / ICC recordings
(CC BY-NC 4.0 + upstream terms) are used for statistics only unless ``allow_noncommercial=True``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from .full_duplex_bench import NAME, SUBSETS, _ids, _words, split_of
from .wav import read_wav, resample, speech_segments

# subset -> [(file, kind, use, metadata text key)]
FDB_CLIPS = {
    "v1.5/user_backchannel": [("backchannel.wav", "backchannel", "clip", "backchannel_text"), ("context.wav", None, "opening", "context_text")],
    "v1.5/talking_to_other": [("current_turn.wav", "aside", "clip", "current_turn_text"), ("context.wav", None, "opening", "context_text")],
    "v1.5/background_speech": [("background.wav", "noise", "clip", "background_text"), ("context.wav", None, "opening", "context_text")],
    "v1.5/user_interruption": [("interrupt.wav", None, "interruption", "current_turn_text"), ("context.wav", None, "opening", "context_text")],
}
SHORT = {"backchannel": "bc", "aside": "aside", "noise": "bg", None: "speech"}


def _trimmed(audio):
    voiced = speech_segments(audio, rms_threshold=150.0, min_speech_ms=60, pad_ms=30)
    if not voiced:
        return audio
    a, b = round(voiced[0][0] * audio.sr), round(voiced[-1][1] * audio.sr)
    return audio[a:b]


def extract(root: str | Path, out_dir: str | Path, *, material_frac: float = 0.5, include_eval: bool = False,
            sr: int = 16000, subsets: list[str] | None = None, dedupe_text: bool = True) -> Path:
    """Cut the clips of ``FDB_CLIPS`` from the unpacked Full-Duplex-Bench data at ``root`` into ``out_dir``
    (``clips/<kind>/<id>.wav`` + ``manifest.jsonl``). Openings repeat across v1.5 subsets (the same request
    texts are reused); ``dedupe_text`` keeps one per text. Returns the manifest path."""
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    held_out = _eval_texts(root, material_frac)
    seen, recs = set(), []
    for subset in subsets or list(FDB_CLIPS):
        lic = SUBSETS[subset][2]
        for sid in _ids(root / subset):
            split = split_of(subset, sid, material_frac)
            if split == "eval" and not include_eval:
                continue
            d = root / subset / sid
            meta = json.loads((d / "metadata.json").read_text()) if (d / "metadata.json").exists() else {}
            for fname, kind, use, key in FDB_CLIPS[subset]:
                if not (d / fname).exists():
                    continue
                text = meta.get(key, "")
                if split == "material" and use != "clip" and text.lower().strip() in held_out:  # an eval sample says the same
                    continue
                if dedupe_text and use != "clip" and (use, text.lower().strip()) in seen:
                    continue
                seen.add((use, text.lower().strip()))
                audio = _trimmed(resample(read_wav(d / fname), sr))
                cid = f"fdb15-{SHORT[kind] if use == 'clip' else use}-{subset.split('/')[1]}-{sid}"
                rel = Path("clips") / (kind or use) / f"{cid}.wav"
                (out / rel).parent.mkdir(parents=True, exist_ok=True)
                audio.write_wav(out / rel)
                recs.append({"id": cid, "kind": kind, "text": text, "dur_ms": audio.dur_ms, "sr": audio.sr, "audio": str(rel),
                             "use": use, "split": split,
                             "source": {"benchmark": NAME, "subset": subset, "sample_id": sid, "file": fname, "license": lic,
                                        "synthetic": True}})
    path = out / "manifest.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    return path


HUMDIAL_CLIPS = {  # category -> [(segment index, kind, use)]; index -1: every segment after the first
    "backchannel": [(0, None, "opening"), (1, "backchannel", "clip")],
    "talk_to_others": [(0, None, "opening"), (2, "aside", "clip")],
    "others_talk_to_user_after": [(0, None, "opening"), (-1, "noise", "clip")],
    "others_talk_to_user_before": [(0, "noise", "clip"), (1, None, "opening")],
    **{c: [(0, None, "opening"), (1, None, "interruption")] for c in ("ask", "deny", "repeat", "shift", "wait")},
}


def humdial_split(lang: str, sample_id: str, material_frac: float) -> str:
    """HumDial ids are ``<speaker>_<item>``: split by speaker (``humdial.split_of``)."""
    from .humdial import split_of as hd_split

    return hd_split(lang, sample_id, material_frac)


def extract_humdial(root: str | Path, out_dir: str | Path, *, material_frac: float = 0.5, include_eval: bool = False,
                    sr: int = 16000, langs: tuple[str, ...] = ("en", "zh"), append: bool = True) -> Path:
    """Real human clips from HumDial-FDBench (Apache-2.0, see ``humdial``): backchannels, asides, third-party
    speech, interruptions (``intent``: ask / deny / repeat / shift / wait) and openings, cut at the annotated
    segments (trimmed to the voice). Split by speaker (``humdial_split``); appended to ``manifest.jsonl``
    unless ``append=False``. Records also carry ``lang``."""
    from . import humdial as hd

    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen, recs = set(), []
    for lang in langs:
        for cat, picks in HUMDIAL_CLIPS.items():
            subset = f"humdial/{lang}/{cat}"
            d = hd._dir(root, subset)
            if not d.exists():
                continue
            for sid in hd.ids(root, subset):
                split = humdial_split(lang, sid, material_frac)
                if split == "eval" and not include_eval:
                    continue
                segs = sorted(json.loads((d / f"{sid}.json").read_text(encoding="utf-8"))["speech_segments"], key=lambda x: x["xmin"])
                wav = None
                for idx, kind, use in picks:
                    chosen = segs[1:] if idx == -1 else segs[idx : idx + 1]
                    for j, seg in enumerate(chosen):
                        text = seg["text"].strip()
                        if use == "opening" and (lang, text.lower()) in seen:
                            continue
                        seen.add((lang, text.lower()))
                        if wav is None:
                            wav = resample(read_wav(d / f"{sid}.wav"), sr)
                        a, b = round(seg["xmin"] * wav.sr), round(seg["xmax"] * wav.sr)
                        audio = _trimmed(wav[a:b])
                        tag = SHORT[kind] if use == "clip" else use
                        cid = f"hd-{lang}-{tag}-{cat}-{sid}" + (f"-{j}" if idx == -1 and len(chosen) > 1 else "")
                        rel = Path("clips") / (kind or use) / f"{cid}.wav"
                        (out / rel).parent.mkdir(parents=True, exist_ok=True)
                        audio.write_wav(out / rel)
                        rec = {"id": cid, "kind": kind, "text": text, "dur_ms": audio.dur_ms, "sr": audio.sr, "audio": str(rel),
                               "use": use, "split": split, "lang": lang,
                               "source": {"benchmark": hd.NAME, "subset": subset, "sample_id": sid, "file": f"{sid}.wav",
                                          "segment": [seg["xmin"], seg["xmax"]], "license": hd.LICENSE, "synthetic": False}}
                        if use == "interruption":
                            rec["intent"] = cat
                        recs.append(rec)
    path = out / "manifest.jsonl"
    with path.open("a" if append else "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    return path


def _eval_texts(root: Path, material_frac: float) -> set[str]:
    """Every user text of an evaluation-split sample (v1.5 requests are reused across subsets, and v1.0's
    synthetic interruptions share v1.5's): a material clip saying the same words would leak into evaluation."""
    out = set()
    for subset in SUBSETS:
        if not (root / subset).exists():
            continue
        for sid in _ids(root / subset):
            if split_of(subset, sid, material_frac) != "eval":
                continue
            d = root / subset / sid
            for name in ("metadata.json", "interrupt.json"):
                if (d / name).exists():
                    m = json.loads((d / name).read_text())
                    m = m[0] if isinstance(m, list) else m
                    out |= {v.lower().strip() for k, v in m.items() if isinstance(v, str)}
    return out


def load_manifest(path: str | Path, *, split: str | None = "material", kind: str | None = None, use: str | None = None,
                  lang: str | None = None) -> list[dict]:
    """Manifest records (by default only ``material`` ones), with ``audio`` resolved to an absolute path.
    ``lang`` filters by language (records without one are Full-Duplex-Bench English)."""
    path = Path(path)
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if (split is None or r["split"] == split) and (kind is None or r["kind"] == kind) and (use is None or r["use"] == use) \
                and (lang is None or r.get("lang", "en") == lang):
            out.append(r | {"audio": str(path.parent / r["audio"])})
    return out


# ---------------------------------------------------------------- human-behaviour statistics


def _q(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    xs = sorted(xs)
    pick = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3), "median": round(pick(0.5), 3), "p10": round(pick(0.1), 3),
            "p90": round(pick(0.9), 3)}


def pause_stats(root: str | Path, subset: str = "v1.0/candor_pause_handling", material_frac: float = 0.0,
                split: str | None = None) -> dict:
    """Mid-utterance pauses (annotated ``[PAUSE]`` spans): duration, time and words since the speaker
    (re)started, and the word before the pause (filled pauses / conjunctions vs. clause ends)."""
    root = Path(root)
    durs, since, words_before, prev_words = [], [], [], {}
    for sid in _ids(root / subset):
        if split is not None and split_of(subset, sid, material_frac) != split:
            continue
        d = root / subset / sid
        pauses = sorted(p["timestamp"] for p in json.loads((d / "pause.json").read_text()))
        words = _words(d / "transcription.json")
        start = words[0][1] if words else 0.0
        for p0, p1 in pauses:
            durs.append(p1 - p0)
            since.append(p0 - start)
            before = [w for w in words if start <= w[1] < p0]
            words_before.append(len(before))
            if before:
                w = before[-1][0].lower().strip(".,?!")
                prev_words[w] = prev_words.get(w, 0) + 1
            start = p1
    top = sorted(prev_words.items(), key=lambda kv: -kv[1])[:15]
    return {"subset": subset, "pause_s": _q(durs), "speech_before_pause_s": _q(since), "words_before_pause": _q(words_before),
            "word_before_pause_top": top}


def backchannel_stats(root: str | Path, subset: str = "v1.0/icc_backchannel") -> dict:
    """Human listener backchannels in ICC (``aggregated_all_data.json``: per sample, the backchannel intervals
    pooled over the ICC annotators): duration, rate per minute of speaker audio (pooled annotations, so an
    upper bound), and placement relative to the speaker — inside a speaker pause vs. over speech, and the
    delay from the end of the speaker's last word."""
    root = Path(root)
    agg = json.loads((root / subset / "aggregated_all_data.json").read_text())
    durs, in_pause, delay, rates = [], 0, [], []
    for sid, ivs in agg.items():
        d = root / subset / sid
        if not (d / "input.wav").exists():
            continue
        words = _words(d / "transcription.json")
        total = read_wav(d / "input.wav").dur_ms / 1000
        rates.append(len(ivs) / total * 60)
        for s, e in ivs:
            durs.append(e - s)
            over = any(ws < e and we > s for _, ws, we in words)
            in_pause += not over
            prev = [we for _, _, we in words if we <= s]
            if prev:
                delay.append(s - prev[-1])
    n = len(durs)
    return {"subset": subset, "duration_s": _q(durs), "per_minute_pooled": _q(rates),
            "fraction_in_speaker_pause": round(in_pause / n, 3) if n else None, "delay_after_last_word_s": _q(delay)}


def interruption_stats(root: str | Path, subset: str = "v1.0/synthetic_user_interruption") -> dict:
    """Where the (synthetic) interruptions sit: onset after the end of the context question. FD-Bench places
    them at a fixed offset of the user's input, not relative to the agent's speech — so they say nothing about
    where real users cut in."""
    root = Path(root)
    offs, durs = [], []
    for sid in _ids(root / subset):
        d = root / subset / sid
        ts = (json.loads((d / "interrupt.json").read_text())[0]["timestamp"] if (d / "interrupt.json").exists()
              else json.loads((d / "metadata.json").read_text())["timestamps"])
        ctx = read_wav(d / "context.wav").dur_ms / 1000
        offs.append(ts[0] - ctx)
        durs.append(ts[1] - ts[0])
    return {"subset": subset, "onset_after_context_end_s": _q(offs), "interruption_s": _q(durs)}


def behaviors(manifest: str | Path, stats: dict | str | Path | None = None, lang: str | None = None, **overrides):
    """Seed an online user's ``Behaviors`` from the bank: its backchannel and aside wordings from the ``material``
    clips (deduplicated, lower-cased; rendered by the user's TTS, so only the wording is reused) and, given
    ``human_stats`` (dict or JSON path), the mid-thought pause length from the Candor p10–p90 range instead of the
    default 1.2–2.5 s. Rates (``backchannel_per_min`` …) and ``pause_p`` stay at their defaults (0) unless passed in
    ``overrides`` (``user.PERSONA``'s None = from the user's profile)."""
    from ..user import Behaviors

    def texts(kind):
        seen = dict.fromkeys(r["text"].strip().lower().rstrip(".") for r in load_manifest(manifest, kind=kind, lang=lang) if r["text"].strip())
        return tuple(seen)

    kw: dict = {}
    if bc := texts("backchannel"):
        kw["backchannels"] = bc
    if asides := texts("aside"):
        kw["asides"] = asides
    if stats is not None:
        st = json.loads(Path(stats).read_text()) if isinstance(stats, (str, Path)) else stats
        p = st.get("pauses", {}).get("v1.0/candor_pause_handling", {}).get("pause_s", {})
        if p.get("n"):
            kw["pause_ms"] = (round(p["p10"] * 1000), round(p["p90"] * 1000))
    return Behaviors(**(kw | overrides))


def human_stats(root: str | Path) -> dict:
    root = Path(root)
    out = {"pauses": {s: pause_stats(root, s) for s in ("v1.0/candor_pause_handling", "v1.0/synthetic_pause_handling")
                      if (root / s).exists()}}
    if (root / "v1.0/icc_backchannel/aggregated_all_data.json").exists():
        out["backchannels"] = backchannel_stats(root)
    out["interruptions"] = {s: interruption_stats(root, s) for s in ("v1.0/synthetic_user_interruption", "v1.5/user_interruption")
                            if (root / s).exists()}
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract a clip bank + human statistics from Full-Duplex-Bench data")
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--material-frac", type=float, default=0.5)
    ap.add_argument("--include-eval", action="store_true")
    ap.add_argument("--humdial", default="", help="also add HumDial-FDBench clips from this unpacked dataset")
    a = ap.parse_args()
    print(extract(a.root, a.out, material_frac=a.material_frac, include_eval=a.include_eval))
    if a.humdial:
        print(extract_humdial(a.humdial, a.out, material_frac=a.material_frac, include_eval=a.include_eval))
    (Path(a.out) / "human_stats.json").write_text(json.dumps(human_stats(a.root), indent=2))
