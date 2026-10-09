"""Full-Duplex-Bench v1.0 / v1.5 as InteractionGym test sets.

Benchmark: Lin et al., "Full-Duplex-Bench" (arXiv 2503.04721) and "Full-Duplex-Bench v1.5" (arXiv 2507.23159);
code github.com/DanielLin94144/Full-Duplex-Bench (CC BY-NC 4.0). Data: the authors' Google Drive folder, mirrored
on Hugging Face as ``Ssshangfu/Full-Duplex-Bench-Data`` (one zip per subset, 705 MB). Data licenses (dataset
README): v1.0 Candor/ICC subsets CC BY-NC 4.0 plus the upstream corpora's terms; v1.0 synthetic subsets and all
of v1.5 MIT. The official metric rules and judge prompts ported from the CC BY-NC repository live in
the optional component ``interaction-gym-fdbench`` (``interaction_gym_fdbench.v1``, CC BY-NC 4.0,
non-commercial; ``extras/fdbench`` in the repository), loaded on first use and re-exported here; this module itself is
Apache-2.0 and imports without it.

How the benchmark runs a model: it streams one ``input.wav`` (the user's side, pre-timed) into the model and
records a time-synchronous ``output.wav`` of the same length; metrics come from ASR word timestamps
(parakeet-tdt-0.6b-v2) and Silero-VAD on the output. Here each sample becomes a ``Task`` played by a
``ReplayUser`` whose turns are slices of ``input.wav`` at their original times; everything outside the turns
(room tone, breaths) is a per-episode background track, so the agent's microphone carries ``input.wav``
sample for sample. The episode runs exactly as long as ``input.wav`` (``make_env``), like the official output.

Mapping (subset → turns):

==============================  ============================================================================
v1.0 candor/synthetic pause     speech between annotated ``[PAUSE]`` spans; every turn followed by a pause
handling                        gets ``expects: "wait"``, the last one is a normal turn
v1.0 candor turn taking         one normal turn ending at the ``[TURN-TAKING]`` timestamp (the agent should
                                take the turn → ``respond``)
v1.0 ICC backchannel            the speaker's monologue split at word gaps ≥ ``icc_split_gap_s``; all but the
                                last turn ``expects: "wait"`` (the listener should backchannel, not take over)
v1.0 synthetic user             context question (normal) + the interruption (``expects: "yield"``)
interruption
v1.5 user_interruption          context (normal) + interruption (``expects: "yield"``)
v1.5 user_backchannel           context + ``kind: "backchannel"``
v1.5 talking_to_other           context + ``kind: "aside"`` (the user addresses someone else)
v1.5 background_speech          context + ``kind: "noise"`` (a third party talking in the background)
==============================  ============================================================================

Each task keeps the benchmark's own annotation in ``task.scenario["benchmark"]``.

Official metrics (``sample_metrics`` / ``official_metrics``) are faithful ports of ``v1_v1.5/evaluation``
(file and function cited at each port). What differs, inherently: the official scripts take word timestamps
from ASR of ``output.wav`` and speech spans from Silero-VAD; from an episode we take the agent's own transcript
(the model's text stream) with words spread linearly over the voiced parts of each agent turn, and voiced
parts from an energy VAD on the agent's audio (``wav.speech_segments``) — or the turn span when no audio is
stored. The LLM-judged parts (v1.0 interruption relevance rated by GPT-4-turbo, v1.5 behaviour classes by
GPT-4o) take any ``TextGen`` as judge, with the official prompts; numbers are only comparable when the judge is
the official model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from ..audio import Audio
from ..core import AgentSpec, Background, Env, Task
from ..user import ReplayUser
from ._fdbench import Lazy, reexport
from .wav import read_wav, resample, speech_segments

# Full-Duplex-Bench v1 / v1.5 metric rules and judge prompts (CC BY-NC 4.0): the optional component
# interaction-gym-fdbench, loaded on first use and re-exported here (``from ...full_duplex_bench import take_turn``)
_v1 = Lazy("v1")
__getattr__ = reexport(__name__, "v1", {
    "BC_EPS", "BC_TIME_THRESHOLD", "BC_WINDOW", "INTERRUPTION_JUDGE", "MODEL_MERGE_GAP", "TURN_DURATION_THRESHOLD",
    "TURN_NUM_WORDS_THRESHOLD", "USER_MERGE_GAP", "backchannel_histogram", "backchannel_tor", "behaviour_input",
    "interruption_user_message", "overlaps", "parse_behaviour", "parse_interruption_rating", "response_gaps", "take_turn",
}, {"_merge": "merge", "_tor": "take_turn"})

NAME = "Full-Duplex-Bench"
HF_MIRROR = "Ssshangfu/Full-Duplex-Bench-Data"
REPO = "https://github.com/DanielLin94144/Full-Duplex-Bench"

# subset -> (version, official task, data license)
SUBSETS: dict[str, tuple[str, str, str]] = {
    "v1.0/candor_pause_handling": ("1.0", "pause_handling", "CC BY-NC 4.0 + Candor terms"),
    "v1.0/synthetic_pause_handling": ("1.0", "pause_handling", "MIT"),
    "v1.0/candor_turn_taking": ("1.0", "smooth_turn_taking", "CC BY-NC 4.0 + Candor terms"),
    "v1.0/icc_backchannel": ("1.0", "backchannel", "CC BY-NC 4.0 + ICC terms"),
    "v1.0/synthetic_user_interruption": ("1.0", "user_interruption", "MIT"),
    "v1.5/user_interruption": ("1.5", "user_interruption", "MIT"),
    "v1.5/user_backchannel": ("1.5", "user_backchannel", "MIT"),
    "v1.5/talking_to_other": ("1.5", "talking_to_other", "MIT"),
    "v1.5/background_speech": ("1.5", "background_speech", "MIT"),
}
V15_OVERLAP = {  # v1.5 subset -> (metadata text key, kind, expects) of the overlap event
    "user_interruption": ("current_turn_text", None, "yield"),
    "user_backchannel": ("backchannel_text", "backchannel", None),
    "talking_to_other": ("current_turn_text", "aside", None),
    "background_speech": ("background_text", "noise", None),
}
# subsets whose sample k is the same content (v1.5 user_interruption k reuses v1.0 synthetic interruption k's
# texts and audio): they share a split, so a sample is never material in one and evaluation in the other
SPLIT_GROUP = {"v1.0/synthetic_user_interruption": "user_interruption", "v1.5/user_interruption": "user_interruption"}
PERSONA = "Pre-recorded Full-Duplex-Bench user (offline replay)."

@dataclass
class Sample:
    """One benchmark item, ready to play: ``turns`` for ``ReplayUser`` (``{"t", "text", "audio", "kind",
    "expects"}``), the residual ``background`` and the episode length (= ``input.wav``)."""

    id: str
    subset: str
    task: Task
    turns: list[dict]
    duration_ms: int
    background: list[Background] = field(default_factory=list)


def split_of(subset: str, sample_id: str, material_frac: float = 0.0) -> str:
    """``"material"`` or ``"eval"``, by a stable hash of (subset, id): the samples whose clips or statistics
    may feed our own scenarios vs. those held out for evaluation. With ``material_frac=0`` (the default)
    everything is evaluation data, which is what published numbers are computed on. Samples with the same
    content in two subsets (``SPLIT_GROUP``) always land on the same side."""
    group = SPLIT_GROUP.get(subset, subset)
    h = int(hashlib.sha256(f"{group}/{sample_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "material" if h < material_frac else "eval"


# ---------------------------------------------------------------- loading


def _ids(d: Path) -> list[str]:
    ids = [p.name for p in d.iterdir() if p.is_dir() and (p / "input.wav").exists()]
    return sorted(ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))


def _ms(s: float) -> int:
    return round(s * 1000)


def _trim(audio: Audio, a_ms: int, b_ms: int) -> tuple[int, int]:
    """Tighten [a, b) to the speech inside it (turn ends should be where the voice stops, not where a
    clip's trailing silence does); unchanged if nothing is voiced."""
    seg = audio[round(a_ms * audio.sr / 1000) : round(b_ms * audio.sr / 1000)]
    voiced = speech_segments(seg, rms_threshold=150.0, min_speech_ms=60, pad_ms=0)
    if not voiced:
        return a_ms, b_ms
    return a_ms + _ms(voiced[0][0]), min(b_ms, a_ms + _ms(voiced[-1][1]))


def _build(inp: Audio, spans: list[dict], exact: bool) -> tuple[list[dict], list[Background]]:
    """Turns cut from ``inp`` at ``spans`` (``start_ms``, ``end_ms``, ``text``, ``kind``, ``expects``) and the
    rest of ``inp`` as a background track, so turns + background == ``inp``."""
    sr, turns = inp.sr, []
    residual = list(inp.samples) if exact else None
    for s in sorted(spans, key=lambda x: x["start_ms"]):
        a, b = round(s["start_ms"] * sr / 1000), round(s["end_ms"] * sr / 1000)
        if b <= a:
            continue
        turn = {"t": s["start_ms"], "text": s.get("text", ""), "audio": inp[a:b]}
        for k in ("kind", "expects"):
            if s.get(k):
                turn[k] = s[k]
        turns.append(turn)
        if residual is not None:
            residual[a:b] = [0] * (b - a)
    bg = []
    if residual is not None and any(residual):
        from array import array

        bg = [Background(Audio(array("h", residual), sr))]
    return turns, bg


def _words(path: Path) -> list[tuple[str, float, float]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("chunks", [])
    if data and "item" in data[0]:  # {"speaker", "item": [...]} (candor turn taking example format)
        data = [w for d in data for w in d["item"]]
    return [(w["text"], w["timestamp"][0], w["timestamp"][1]) for w in data if w["timestamp"][0] is not None]


def _text(words, a: float, b: float) -> str:
    return " ".join(w for w, s, _ in words if a <= s < b).strip()


def _spans_pause(d: Path, dur_ms: int, inp: Audio) -> tuple[list[dict], dict]:
    pauses = sorted((p["timestamp"] for p in json.loads((d / "pause.json").read_text())), key=lambda x: x[0])
    words = _words(d / "transcription.json")
    spans, start = [], 0
    for i, (p0, p1) in enumerate(pauses + [[dur_ms / 1000, None]]):
        end = min(_ms(p0), dur_ms)
        if end > start:
            a, b = _trim(inp, start, end)
            spans.append({"start_ms": a, "end_ms": b, "text": _text(words, start / 1000, end / 1000),
                          "expects": "wait" if p1 is not None else None})
        if p1 is not None:
            start = max(start, _ms(p1))
    return spans, {"pause.json": pauses}


def _spans_turn_taking(d: Path, dur_ms: int, inp: Audio) -> tuple[list[dict], dict]:
    ann = json.loads((d / "turn_taking.json").read_text())
    end = _ms(ann[0]["timestamp"][0])  # the official input_end_time
    a, _ = _trim(inp, 0, end)  # the turn ends where the annotation says (the reference for latency)
    words = _words(d / "transcription.json")  # absent in the released data
    return [{"start_ms": a, "end_ms": end, "text": _text(words, 0, end / 1000)}], {"turn_taking.json": ann}


def _spans_icc(d: Path, dur_ms: int, inp: Audio, gap_s: float) -> tuple[list[dict], dict]:
    words = _words(d / "transcription.json")
    groups: list[list] = []
    for w in words:
        if groups and w[1] - groups[-1][-1][2] < gap_s:
            groups[-1].append(w)
        else:
            groups.append([w])
    spans = []
    for i, g in enumerate(groups):
        a, b = _ms(g[0][1]), min(_ms(g[-1][2]), dur_ms)
        if b > a:
            spans.append({"start_ms": a, "end_ms": b, "text": " ".join(w[0] for w in g),
                          "expects": "wait" if i < len(groups) - 1 else None})
    return spans, {"transcription_words": len(words)}


def _spans_interrupt_v10(d: Path, dur_ms: int, inp: Audio) -> tuple[list[dict], dict]:
    ann = json.loads((d / "interrupt.json").read_text())
    ctx = read_wav(d / "context.wav").dur_ms
    a0, b0 = _trim(inp, 0, ctx)
    i0, i1 = (_ms(x) for x in ann[0]["timestamp"])
    a1, b1 = _trim(inp, i0, min(i1, dur_ms))
    return [{"start_ms": a0, "end_ms": b0, "text": ann[0]["context"]},
            {"start_ms": a1, "end_ms": b1, "text": ann[0]["interrupt"], "expects": "yield"}], {"interrupt.json": ann}


def _spans_v15(d: Path, dur_ms: int, inp: Audio, task: str, clean: bool) -> tuple[list[dict], dict]:
    meta = json.loads((d / "metadata.json").read_text())
    ctx = read_wav(d / "context.wav").dur_ms
    a0, b0 = _trim(inp, 0, min(ctx, dur_ms))
    spans = [{"start_ms": a0, "end_ms": b0, "text": meta.get("context_text", "")}]
    if not clean:
        key, kind, expects = V15_OVERLAP[task]
        o0, o1 = (_ms(x) for x in meta["timestamps"])
        a1, b1 = _trim(inp, o0, min(o1, dur_ms))
        spans.append({"start_ms": a1, "end_ms": b1, "text": meta.get(key, ""), "kind": kind, "expects": expects})
    return spans, {"metadata.json": meta}


def load_sample(root: str | Path, subset: str, sample_id: str, *, sr: int = 16000, exact: bool = True,
                clean: bool = False, icc_split_gap_s: float = 1.0) -> Sample:
    """One sample of ``subset`` (a key of ``SUBSETS``) from the unpacked data at ``root``
    (``<root>/v1.0/<subset>/<id>/input.wav`` …). ``sr`` is the rate audio is resampled to (the agent's
    microphone rate). ``exact``: keep everything outside the turns as a background track, so the agent hears
    ``input.wav`` exactly. ``clean`` (v1.5 only): play ``clean_input.wav`` — the same sample without the overlap
    event, which v1.5's behaviour judge compares against."""
    version, task, lic = SUBSETS[subset]
    d = Path(root) / subset / sample_id
    wav = d / ("clean_input.wav" if clean else "input.wav")
    if clean and version != "1.5":
        raise ValueError("clean inputs exist only in v1.5")
    src = read_wav(wav)
    inp = resample(src, sr)
    dur = inp.dur_ms
    if task == "pause_handling":
        spans, ann = _spans_pause(d, dur, inp)
    elif task == "smooth_turn_taking":
        spans, ann = _spans_turn_taking(d, dur, inp)
    elif task == "backchannel":
        spans, ann = _spans_icc(d, dur, inp, icc_split_gap_s)
    elif version == "1.0":
        spans, ann = _spans_interrupt_v10(d, dur, inp)
    else:
        spans, ann = _spans_v15(d, dur, inp, task, clean)
    turns, bg = _build(inp, spans, exact)
    bench = {"name": NAME, "version": version, "subset": subset, "task": task, "sample_id": sample_id,
             "clean": clean, "source": str(wav), "source_sr": src.sr, "duration_ms": dur, "license": lic,
             "annotation": ann}
    scenario = {"persona": PERSONA, "benchmark": bench,
                "turns": [{k: v for k, v in t.items() if k != "audio"} | {"dur": t["audio"].dur_ms} for t in turns]}
    tid = f"fdb-{subset.replace('/', '-')}-{sample_id}" + ("-clean" if clean else "")
    return Sample(tid, subset, Task(id=tid, scenario=scenario, criteria={"benchmark": NAME, "task": task}), turns, dur, bg)


def load(root: str | Path, subset: str, *, limit: int | None = None, ids: list[str] | None = None,
         split: str | None = None, material_frac: float = 0.0, **kw) -> list[Sample]:
    """All samples of ``subset`` (or ``ids``, or the first ``limit``). ``split="eval"`` / ``"material"``
    keeps only that side of ``split_of(…, material_frac)``."""
    ids = ids if ids is not None else _ids(Path(root) / subset)
    if split is not None:
        ids = [i for i in ids if split_of(subset, i, material_frac) == split]
    return [load_sample(root, subset, i, **kw) for i in ids[:limit]]


def make_env(sample: Sample, spec: AgentSpec, tail_ms: int = 0) -> Env:
    """The episode as the benchmark runs it: exactly ``input.wav`` long (``tail_ms`` more to see what the
    agent does after the input ends — not part of the official protocol). It never ends early on silence."""
    horizon = sample.duration_ms + tail_ms
    return Env({"user": ReplayUser(sample.turns)}, spec, max_ms=horizon, end_idle_ms=horizon + 1, background=sample.background)


# ---------------------------------------------------------------- the agent's output, as the official scripts see it


def bench_meta(ep: dict) -> dict:
    return ep["meta"]["task"]["scenario"]["benchmark"]


def speech_spans(ep: dict, role: str = "agent", media=None, horizon_s: float | None = None) -> list[tuple[float, float]]:
    """Voiced spans (s) of ``role``'s turns: an energy VAD over each turn's audio when ``media`` (a
    ``MediaStore``) is given and the turn has audio, else the turn span. Clipped to ``horizon_s``."""
    out = []
    for t in ep["turns"]:
        if t["role"] != role or t["end_time"] <= t["start_time"]:
            continue
        t0 = t["start_time"] / 1000
        spans = None
        if media is not None and "media" in t:
            spans = [(t0 + a, t0 + b) for a, b in speech_segments(media.load(t["media"]))]
        if spans is None:
            spans = [(t0, t["end_time"] / 1000)]
        for a, b in spans:
            if horizon_s is not None:
                b = min(b, horizon_s)
            if b > a:
                out.append((round(a, 3), round(b, 3)))
    return sorted(out)


def agent_words(ep: dict, media=None, horizon_s: float | None = None) -> list[dict]:
    """The agent's words as ``{"text", "timestamp": [s, e]}`` (the official ``chunks`` format): each agent
    turn's text spread linearly over its voiced spans (whole span without audio). Stand-in for the official
    ASR alignment of ``output.wav``."""
    chunks = []
    for t in ep["turns"]:
        if t["role"] != "agent" or not t["text"].strip() or t["end_time"] <= t["start_time"]:
            continue
        t0 = t["start_time"] / 1000
        spans = None
        if media is not None and "media" in t:
            spans = [(t0 + a, t0 + b) for a, b in speech_segments(media.load(t["media"]))]
        if not spans:
            spans = [(t0, t["end_time"] / 1000)]
        words = t["text"].split()
        per = sum(b - a for a, b in spans) / len(words)

        def at(off, spans=spans):  # an offset into voiced time -> a time on the timeline
            for a, b in spans:
                if off <= b - a + 1e-9:
                    return a + off
                off -= b - a
            return spans[-1][1]

        for i, w in enumerate(words):
            s, e = at(i * per), at((i + 1) * per)
            if horizon_s is not None:
                if s >= horizon_s:
                    break
                e = min(e, horizon_s)
            chunks.append({"text": w, "timestamp": [round(s, 3), round(e, 3)]})
    return sorted(chunks, key=lambda c: c["timestamp"][0])


# ---------------------------------------------------------------- per-sample official metrics


def _pause_handling(ep, media):
    """eval_pause_handling.py: TOR over the whole output (lower is better)."""
    h = bench_meta(ep)["duration_ms"] / 1000
    return {"TOR": _v1.take_turn(agent_words(ep, media, h))}


def _smooth_turn_taking(ep, media):
    """eval_smooth_turn_taking.py: TOR (higher is better) and latency = first output word − end of the
    user's turn (negative clipped to 0), only when TOR = 1."""
    b = bench_meta(ep)
    h = b["duration_ms"] / 1000
    end = b["annotation"]["turn_taking.json"][0]["timestamp"][0]
    chunks = agent_words(ep, media, h)
    tor = _v1.take_turn(chunks)
    out = {"TOR": tor}
    if tor:
        out["latency"] = max(0.0, chunks[0]["timestamp"][0] - end)
    return out


def _user_interruption_v10(ep, media):
    """eval_user_interruption.py (after asr.py --task user_interruption crops the output at the end of the
    interruption): TOR and latency on the output after the interruption ends."""
    b = bench_meta(ep)
    h = b["duration_ms"] / 1000
    end = b["annotation"]["interrupt.json"][0]["timestamp"][1]
    chunks = [c for c in agent_words(ep, media, h) if c["timestamp"][0] >= end]
    tor = _v1.take_turn(chunks)
    out = {"TOR": tor, "response_text": " ".join(c["text"] for c in chunks)}
    if tor:
        out["latency"] = max(0.0, chunks[0]["timestamp"][0] - end)
    return out


def _backchannel(ep, media):
    """eval_backchannel.py, step for step (including its quirk that a later short segment can reset TOR to
    0 unless a > 3 s segment broke the loop). Returns TOR, frequency (backchannels / s of output) and the
    predicted backchannel intervals (for JSD against the human distribution, ``official_metrics``)."""
    h = bench_meta(ep)["duration_ms"] / 1000
    segments = speech_spans(ep, "agent", media, h)
    chunks = agent_words(ep, media, h)
    tor, pred = _v1.backchannel_tor(segments, chunks)
    return {"TOR": tor, "freq": len(pred) / h, "backchannels": pred, "duration_s": h}


def _v15_timing(ep, media):
    """get_timing.py: VAD speech of input (merged ≤ 0.6 s) and output (merged ≤ 0.5 s) → ``latency_stop_list``
    (user∩model overlaps) and ``latency_resp_list`` (user end → next model start). Paper definitions
    (v1.5 §3): stop latency = overlap onset → model stops; response latency = overlap end → model's next
    utterance. ``stop`` / ``resp`` pick the intervals belonging to the overlap event (the user speech span that
    contains the annotated overlap window)."""
    b = bench_meta(ep)
    h = b["duration_ms"] / 1000
    user = _v1.merge(speech_spans(ep, "user", media, h), _v1.USER_MERGE_GAP)
    model = _v1.merge(speech_spans(ep, "agent", media, h), _v1.MODEL_MERGE_GAP)
    stops, resps = _v1.overlaps(user, model), _v1.response_gaps(user, model)
    out = {"latency_stop_list": stops, "latency_resp_list": resps}
    ts = b["annotation"].get("metadata.json", {}).get("timestamps")
    if ts and not b.get("clean"):
        ev = next((u for u in user if u[0] < ts[1] and u[1] > ts[0]), None)
        if ev is not None:
            st = [iv for iv in stops if ev[0] <= iv[0] and iv[1] <= ev[1]]
            if st:
                out["stop"] = round(st[0][1] - st[0][0], 3)
            rp = [iv for iv in resps if abs(iv[0] - round(ev[1], 3)) < 1e-6]
            if rp:
                out["resp"] = round(rp[0][1] - rp[0][0], 3)
    return out


def sample_metrics(ep: dict, media=None) -> dict:
    """The benchmark's per-sample metrics for one episode (no LLM judge; see ``judge_*``)."""
    b = bench_meta(ep)
    task, version = b["task"], b["version"]
    if task == "pause_handling":
        return _pause_handling(ep, media)
    if task == "smooth_turn_taking":
        return _smooth_turn_taking(ep, media)
    if task == "backchannel":
        return _backchannel(ep, media)
    if task == "user_interruption" and version == "1.0":
        return _user_interruption_v10(ep, media)
    return _v15_timing(ep, media)


# ---------------------------------------------------------------- aggregation


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def _interp(xp: list[float], fp: list[float], x: list[float]) -> list[float]:
    """scipy interp1d(kind="linear", fill_value="extrapolate") on sorted ``xp``."""
    out = []
    for v in x:
        j = 0
        while j < len(xp) - 2 and v > xp[j + 1]:
            j += 1
        x0, x1 = xp[j], xp[j + 1]
        out.append(fp[j] + (fp[j + 1] - fp[j]) * (v - x0) / (x1 - x0))
    return out


def jensenshannon(p: list[float], q: list[float]) -> float:
    """scipy.spatial.distance.jensenshannon (natural log; both normalized; returns the distance = sqrt)."""
    sp, sq = sum(p), sum(q)
    p, q = [x / sp for x in p], [x / sq for x in q]
    m = [(a + b) / 2 for a, b in zip(p, q)]
    kl = lambda a, b: sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0)  # noqa: E731
    return math.sqrt(max(0.0, (kl(p, m) + kl(q, m)) / 2))


def backchannel_jsd(pred: list[list[float]], duration_s: float, gt_dist: list[float]) -> float:
    """eval_backchannel.py: histogram of predicted backchannels in 0.2 s windows vs the human distribution
    (``icc_gt_distribution.json`` in the official repo, keyed by sample id), resized by linear interpolation."""
    if not pred:
        return 1.0
    hist = _v1.backchannel_histogram(pred, duration_s)
    n_gt, n = len(gt_dist), len(hist)
    xg = [i / (n_gt - 1) for i in range(n_gt)] if n_gt > 1 else [0.0, 1.0]
    xp = [i / (n - 1) for i in range(n)] if n > 1 else [0.0]
    gt = _interp(xg, gt_dist if n_gt > 1 else gt_dist * 2, xp)
    return jensenshannon(hist, gt)


def official_metrics(episodes: list[dict], media=None, gt_distribution: dict | None = None,
                     ratings: dict[str, int] | None = None, behaviours: dict[str, str] | None = None) -> dict:
    """Aggregate official metrics of one subset's episodes, as the official scripts print them.

    - pause_handling: ``TOR`` (↓)
    - smooth_turn_taking: ``TOR`` (↑), ``latency`` (s, ↓; over TOR=1 samples)
    - backchannel: ``TOR`` (↓), ``freq`` (↑), ``JSD`` (↓; needs ``gt_distribution`` = icc_gt_distribution.json)
    - v1.0 user_interruption: ``TOR`` (↑), ``latency`` (↓), ``rating`` (0–5 judge, from ``ratings``: episode id → score)
    - v1.5 subsets: ``stop`` / ``resp`` latency means (s) over the overlap event, the same over every interval
      (``stop_all`` / ``resp_all``), and behaviour ratios ``C_RESPOND`` … from ``behaviours`` (episode id → tag).
    """
    per = {ep["meta"]["episode_id"]: sample_metrics(ep, media) for ep in episodes}
    b = bench_meta(episodes[0])
    task, version = b["task"], b["version"]
    out: dict = {"subset": b["subset"], "n": len(per)}
    vals = list(per.values())
    if task in ("pause_handling", "smooth_turn_taking") or (task == "user_interruption" and version == "1.0"):
        out["TOR"] = _mean([v["TOR"] for v in vals])
        if task != "pause_handling":
            out["latency"] = _mean([v.get("latency") for v in vals])
        if ratings and task == "user_interruption":
            out["rating"] = _mean(list(ratings.values()))
    elif task == "backchannel":
        out["TOR"] = _mean([v["TOR"] for v in vals])
        out["freq"] = _mean([v["freq"] for v in vals])
        if gt_distribution is not None:
            jsd = []
            for ep in episodes:
                v = per[ep["meta"]["episode_id"]]
                sid = bench_meta(ep)["sample_id"]
                if sid in gt_distribution:
                    jsd.append(backchannel_jsd(v["backchannels"], v["duration_s"], gt_distribution[sid]))
                    v["JSD"] = round(jsd[-1], 4)
            out["JSD"] = _mean(jsd)
    else:
        out["stop"] = _mean([v.get("stop") for v in vals])
        out["resp"] = _mean([v.get("resp") for v in vals])
        out["stop_all"] = _mean([e - s for v in vals for s, e in v["latency_stop_list"]])
        out["resp_all"] = _mean([e - s for v in vals for s, e in v["latency_resp_list"]])
        out["stop_n"] = sum("stop" in v for v in vals)
        out["resp_n"] = sum("resp" in v for v in vals)
        if behaviours:
            tags = list(behaviours.values())
            out["behaviour"] = {t: round(tags.count(t) / len(tags), 2) for t in sorted(set(tags))}
    out["per_sample"] = per
    return out


# ---------------------------------------------------------------- LLM judges (official prompts, from the fdbench component)

async def judge_interruption(ep: dict, llm, media=None) -> int | None:
    """eval_user_interruption.py: rate (0–5) the response after the interruption; only for TOR = 1 samples."""
    m = _user_interruption_v10(ep, media)
    if not m["TOR"]:
        return None
    ann = bench_meta(ep)["annotation"]["interrupt.json"][0]
    user = _v1.interruption_user_message(ann["context"], ann["interrupt"], m["response_text"])
    reply = await llm.chat([{"role": "system", "content": _v1.INTERRUPTION_JUDGE}, {"role": "user", "content": user}])
    return _v1.parse_interruption_rating(reply)


def behaviour_instruction(repo: str | Path) -> str:
    """v1.5's judge instruction (``v1_v1.5/evaluation/instruction/behavior.txt`` of a checkout of the official repo)."""
    return (Path(repo) / "v1_v1.5/evaluation/instruction/behavior.txt").read_text()


async def judge_behaviour(ep: dict, clean_ep: dict, llm, instruction: str, media=None) -> str | None:
    """eval_behavior.py: one C_* tag from the noisy and clean inputs and outputs (official judge: gpt-4o-2024-08-06).
    The user side uses the turns' transcripts with linear word timing (the official script uses ASR of the inputs)."""
    def user_chunks(e):
        out = []
        for t in e["turns"]:
            if t["role"] == "user" and t["text"]:
                ws = t["text"].split()
                d = (t["end_time"] - t["start_time"]) / 1000 / len(ws)
                out += [{"text": w, "timestamp": [round(t["start_time"] / 1000 + i * d, 3), round(t["start_time"] / 1000 + (i + 1) * d, 3)]}
                        for i, w in enumerate(ws)]
        return out
    h = bench_meta(ep)["duration_ms"] / 1000
    final_input = _v1.behaviour_input(user_chunks(clean_ep), user_chunks(ep),
                                  agent_words(clean_ep, media, bench_meta(clean_ep)["duration_ms"] / 1000), agent_words(ep, media, h))
    reply = await llm.chat([{"role": "system", "content": instruction}, {"role": "user", "content": final_input}])
    return _v1.parse_behaviour(reply)
