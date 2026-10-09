"""Label MUSAN noise clips with our event labels, model-assisted, for ``fetch_noise_banks.py --labels``.

MUSAN's ``noise/free-sound`` clips (845 of 930) carry no metadata at all (only ids); ``noise/sound-bible`` clips have
titles (``LICENSE``), which ``fetch_noise_banks.EVENT_RULES`` match. This script scores every clip with a zero-shot
audio-text model (CLAP, ``laion/clap-htsat-unfused``, Apache-2.0) against our labels' descriptions plus many
distractor classes (music, speech, rain, engines, birds ...), and keeps a label only when it is the clear top class
(probability >= ``--min-p`` and >= ``--ratio`` times the runner-up). The titled sound-bible clips serve as a check:
the script reports how often CLAP agrees with the title rules there.

    python scripts/label_musan_noise.py --musan DIR/downloads/musan/musan --model DIR/models/clap-htsat-unfused \
        --out scripts/noise_labels.json --scores DIR/noise_scores.json

Needs torch + transformers (not dependencies of the package). The output ({clip id: label}) is a proposal: it was
reviewed by hand before use — per-label contact sheets of the clips' spectrograms, keeping only clips whose pattern fits
the label (ring cadences, siren sweeps, horn harmonics, bark bursts, impacts, typing clicks) and dropping the rest as
``null`` — and the result is ``scripts/noise_labels.json`` (2026-10-07: 164 proposed, 79 kept, 74 dropped; cough,
throat clearing and turn signals had no convincing clip). The scores file keeps every clip's top classes.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_noise_banks import label_of, musan_titles  # noqa: E402

OURS = {
    "cough": ["a person coughing", "someone coughs"],
    "sneeze": ["a person sneezing"],
    "throat_clear": ["a person clearing their throat"],
    "door_slam": ["a door slamming shut", "a door being closed hard"],
    "dog_bark": ["a dog barking", "dogs barking"],
    "phone_ring": ["a telephone ringing", "a mobile phone ringtone"],
    "dishes": ["dishes and plates clattering", "cutlery and plates clinking in a kitchen"],
    "cup": ["a cup put down on a saucer", "a spoon stirring in a cup"],
    "keyboard": ["typing on a computer keyboard"],
    "horn": ["a car horn honking", "a vehicle horn beeping"],
    "siren": ["an emergency vehicle siren", "a police siren wailing"],
    "indicator": ["a car turn signal ticking"],
}
DISTRACTORS = ["music", "a person speaking", "a crowd of people talking", "applause", "rain", "wind", "thunder",
               "birds chirping", "insects buzzing", "a car engine idling", "traffic noise in a city", "a train passing",
               "an airplane flying", "a helicopter", "gunshots", "an explosion", "machine hum", "electronic beeps",
               "a dial tone", "static noise", "running water", "footsteps", "keys jingling", "hand clapping",
               "paper rustling", "a vacuum cleaner", "an alarm bell ringing", "a church bell", "a cat meowing",
               "glass breaking", "hammering", "a clock ticking", "silence", "white noise", "laughter",
               "a baby crying", "animal sounds", "a toilet flushing", "a motorcycle", "wood creaking"]


def load(p: Path, sr: int = 48000, win_s: float = 10.0, n_win: int = 3) -> list[np.ndarray]:
    with wave.open(str(p)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        src_sr, ch = w.getframerate(), w.getnchannels()
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    x = np.interp(np.arange(0, len(x), src_sr / sr), np.arange(len(x)), x).astype(np.float32)
    n = int(win_s * sr)
    if len(x) <= n:
        return [x]
    # the loudest windows (an event clip's event, not the silence around it)
    hop = n // 2
    starts = list(range(0, len(x) - n + 1, hop))
    energy = [float(np.mean(x[s:s + n] ** 2)) for s in starts]
    best = sorted(np.argsort(energy)[::-1][:n_win])
    return [x[starts[i]:starts[i] + n] for i in best]


def _emb(x):
    """The projected embedding (transformers 5 wraps it in an output object)."""
    return x if hasattr(x, "norm") else x.pooler_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--musan", required=True, help="the extracted musan/ directory")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--min-p", type=float, default=0.5)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    from transformers import ClapModel, ClapProcessor

    dev = a.device if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(a.model).to(dev).eval()
    proc = ClapProcessor.from_pretrained(a.model)
    classes = [(lab, t) for lab, ts in OURS.items() for t in ts] + [("other:" + d, d) for d in DISTRACTORS]
    with torch.no_grad():
        ti = proc(text=[f"the sound of {t}" for _, t in classes], return_tensors="pt", padding=True).to(dev)
        temb = _emb(model.get_text_features(**ti))
        temb = temb / temb.norm(dim=-1, keepdim=True)
    noise = Path(a.musan) / "noise"
    titles = musan_titles(noise)
    scores, labels = {}, {}
    clips = sorted(noise.glob("*/*.wav"))
    for k, p in enumerate(clips):
        wins = load(p)
        with torch.no_grad():
            ai = proc(audio=wins, sampling_rate=48000, return_tensors="pt").to(dev)
            aemb = _emb(model.get_audio_features(**ai))
            aemb = aemb / aemb.norm(dim=-1, keepdim=True)
            logits = (aemb @ temb.T) * model.logit_scale_a.exp()
            prob = logits.softmax(-1).mean(0).cpu().numpy()
        by = {}
        for (lab, _), pr in zip(classes, prob):
            by[lab] = by.get(lab, 0.0) + float(pr)  # a label's prompts pooled
        top = sorted(by.items(), key=lambda kv: -kv[1])[:4]
        rec = {"top": [(lab, round(v, 3)) for lab, v in top], "title": titles.get(p.stem, "")}
        (l1, p1), (_, p2) = top[0], top[1]
        if not l1.startswith("other:") and p1 >= a.min_p and p1 >= a.ratio * p2 and not rec["title"]:  # titled: by title
            labels[p.stem] = l1
            rec["label"] = l1
        scores[p.stem] = rec
        if k % 50 == 0:
            print(k, len(clips), p.stem, rec, flush=True)
    # the check: CLAP vs the title rules on the titled clips
    agree, n = [], 0
    for cid, rec in scores.items():
        rule = label_of(rec["title"])[0] if rec["title"] else None
        if rule is not None:
            n += 1
            agree.append((cid, rec["title"], rule, rec.get("label")))
    report = {"labelled": len(labels), "clips": len(clips),
              "per_label": {lab: sum(v == lab for v in labels.values()) for lab in OURS},
              "title_check": {"titled_with_rule_label": n, "clap_agrees": sum(r == c for _, _, r, c in agree),
                              "clap_other_label": sum(c is not None and r != c for _, _, r, c in agree),
                              "clap_none": sum(c is None for _, _, r, c in agree), "rows": agree}}
    Path(a.scores).write_text(json.dumps({"report": report, "scores": scores}, indent=1))
    Path(a.out).write_text(json.dumps(dict(sorted(labels.items())), indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "title_check"} | {"title_check": {k: v for k, v in report["title_check"].items() if k != "rows"}}, indent=1))


if __name__ == "__main__":
    main()
