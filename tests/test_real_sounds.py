"""Recorded sounds for online users: third-party speech events and voice packs (a recorded speaker)."""

import json
import math
from array import array

import pytest

from interaction_gym.audio import Audio
from interaction_gym.benchmarks.clip_bank import speech_events, voice_packs
from interaction_gym.clients import FakeSpeech
from interaction_gym.soundscape import Soundscape, event_clip
from interaction_gym.user import Behaviors
from tests.test_behaviors import run, user_turns


def tone(path, ms=600, f=220.0, sr=16000, amp=8000):
    n = sr * ms // 1000
    Audio(array("h", (int(amp * math.sin(2 * math.pi * f * i / sr)) for i in range(n))), sr).write_wav(path)
    return str(path)


def test_speech_events_need_recordings(tmp_path):
    import random

    with pytest.raises(ValueError):
        event_clip("speech", 16000, random.Random(0), bank=tmp_path)
    with pytest.raises(ValueError):
        run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, speech=FakeSpeech(),
            behaviors=Behaviors(third_party_per_min=20.0), soundscape=Soundscape(bank=tmp_path))


def test_third_party_speech_is_a_recorded_noise_to_ignore(tmp_path):
    (tmp_path / "events" / "speech").mkdir(parents=True)
    clip = tone(tmp_path / "events" / "speech" / "a.wav", ms=3000)
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, speech=FakeSpeech(),
                  behaviors=Behaviors(third_party_per_min=20.0), soundscape=Soundscape(bank=tmp_path))
    log = [e for e in ep["meta"]["user"]["noise_events"] if e["label"] == "speech"]
    assert log and all(e["clip"] == f"file:{clip}" for e in log)
    turns = [t for t in user_turns(ep, "noise") if t.get("label") == "speech"]
    assert turns and all(t["expects"] == "ignore" for t in turns)


def pack(tmp_path, n_bc=3):
    ref = tone(tmp_path / "ref.wav", ms=4000, f=150.0)
    bcs = [{"audio": tone(tmp_path / f"bc{i}.wav", ms=500, f=300.0 + 50 * i), "text": f"yeah {i}"} for i in range(n_bc)]
    return {"id": "p1", "lang": "en", "ref": {"audio": ref, "text": "I would like to talk about my trip please"},
            "backchannel": bcs, "aside": [{"audio": tone(tmp_path / "aside.wav", ms=1500), "text": "one sec, honey"}]}


def test_a_voice_pack_plays_the_speakers_own_backchannels_and_asides(tmp_path):
    p = pack(tmp_path)
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], speech=FakeSpeech(), scenario={"voice_pack": p})
    bcs = user_turns(ep, "backchannel")
    assert bcs and all(t["text"] in {c["text"] for c in p["backchannel"]} for t in bcs)
    assert all(a["text"] != b["text"] for a, b in zip(bcs, bcs[1:]))  # never the same recording twice in a row
    rep = ep["meta"]["user"]
    assert rep["voice_pack"]["id"] == "p1" and rep["random_events"]["pack_backchannels"] == len(bcs)
    env, ep = run(["Hi, I need to move a booking.", "Bye ###STOP###"], listener=None, interrupt=None, speech=FakeSpeech(),
                  scenario={"voice_pack": p}, behaviors=Behaviors(aside_per_min=20, away_p=0.0))
    asides = user_turns(ep, "aside")
    assert asides and all(t["text"] == "one sec, honey" for t in asides)


def test_voice_packs_from_a_clip_bank(tmp_path):
    recs = []

    def rec(i, kind, use, ms, text, sid, lang="en", split="material"):
        rel = f"clips/{kind or use}/{i}.wav"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        tone(tmp_path / rel, ms=ms)
        recs.append({"id": i, "kind": kind, "text": text, "dur_ms": ms, "sr": 16000, "audio": rel, "use": use, "split": split,
                     "lang": lang, "source": {"benchmark": "HumDial-FDBench", "sample_id": sid, "license": "Apache-2.0", "synthetic": False}})

    rec("o1", None, "opening", 6000, "what are some good places to visit in spring", "0001_0001")
    rec("o2", None, "opening", 1000, "hi", "0001_0002")
    rec("b1", "backchannel", "clip", 600, "of course", "0001_0003")
    rec("b2", "backchannel", "clip", 3000, "that is very interesting indeed", "0001_0004")
    rec("n1", "noise", "clip", 3000, "do you want lunch", "0001_0005")
    rec("e1", "backchannel", "clip", 600, "yeah", "0009_0001", split="eval")
    (tmp_path / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    packs = voice_packs(tmp_path / "manifest.jsonl")
    assert [p["id"] for p in packs] == ["HumDial-FDBench-en-0001-r0"]
    assert packs[0]["ref"]["text"].startswith("what are") and [c["text"] for c in packs[0]["backchannel"]] == ["of course"]
    assert speech_events(tmp_path / "manifest.jsonl", tmp_path / "bank") == 1
    assert (tmp_path / "bank" / "events" / "speech" / "n1.wav").exists()
