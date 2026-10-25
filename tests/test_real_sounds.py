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
            "backchannel": bcs, "aside": [{"audio": tone(tmp_path / "aside.wav", ms=1500), "text": "one sec, honey", "to": "other"}]}


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
    assert (tmp_path / "bank" / "events" / "speech" / "n1.wav").exists() and (tmp_path / "bank" / "events" / "speech_en" / "n1.wav").exists()


def test_third_party_speech_in_the_callers_language_first(tmp_path):
    import random

    for d in ("speech", "speech_zh"):
        (tmp_path / "events" / d).mkdir(parents=True)
    tone(tmp_path / "events" / "speech" / "en.wav")
    zh = tone(tmp_path / "events" / "speech_zh" / "zh.wav")
    assert all(event_clip("speech", 16000, random.Random(i), bank=tmp_path, variant="zh")[1] == f"file:{zh}" for i in range(5))
    assert event_clip("speech", 16000, random.Random(0), bank=tmp_path, variant="de")[1].endswith("en.wav")


class WordSpeech(FakeSpeech):
    """A tone per word with 80 ms of silence between words, so the quiet places are the word gaps."""

    async def synth(self, text, *a, **k):
        self.calls += 1
        sr, out = self.sr, array("h")
        for _ in text.split():
            out += array("h", (int(8000 * math.sin(2 * math.pi * 200 * i / sr)) for i in range(sr * 250 // 1000)))
            out += array("h", [0] * (sr * 80 // 1000))
        return Audio(out, sr)


def test_take_cuts_fall_in_the_word_gaps():
    from interaction_gym.user import take_cuts

    import asyncio
    audio = asyncio.run(WordSpeech().synth("one two three four five six"))
    parts = take_cuts(audio, [3, 3])
    assert sum(len(p) for p in parts) == len(audio)
    k = len(parts[0])
    assert all(v == 0 for v in audio.samples[k - 100:k + 100])  # in the silence between two words


def test_a_one_take_pause_is_one_synthesis_cut_in_pieces():
    turns = ["I need to move, (pause) let me see, my Saturday booking.", "Bye ###STOP###"]
    sp = WordSpeech()
    env, ep = run(turns, listener=None, interrupt=None, speech=sp, behaviors=Behaviors(pause_one_take=True, pause_ms=(900, 900)))
    first, rest = user_turns(ep)[:2]
    assert first["text"] == "I need to move..." and first["expects"] == "wait" and rest["start_time"] - first["end_time"] == 900
    whole = (first["end_time"] - first["start_time"]) + (rest["end_time"] - rest["start_time"])
    assert abs(whole - 10 * 330) <= 2  # one take of the ten words, split between the two pieces
    calls = sp.calls
    sp2 = WordSpeech()
    run(turns, listener=None, interrupt=None, speech=sp2, behaviors=Behaviors(pause_ms=(900, 900)))
    assert calls == sp2.calls - 1  # off: one take a piece


def test_pack_clips_keep_only_listening_tokens_and_asides_to_others(tmp_path):
    from interaction_gym.user import VoicePack, is_listening_token

    assert all(is_listening_token(x) for x in ("Mm-hmm.", "Yeah", "uh huh", "嗯嗯", "对对。", "I see."))
    assert not any(is_listening_token(x) for x in ("Well said.", "应该没错", "Of course.", "I feel the same way.", "懂啦"))
    p = pack(tmp_path)
    p["backchannel"].append({"audio": p["backchannel"][0]["audio"], "text": "应该没错"})
    p["aside"].append({"audio": p["aside"][0]["audio"], "text": "do you think a navy suit is fine for a wedding?"})
    vp = VoicePack.load(p, 16000)
    assert [t for _, t in vp.clips["backchannel"]] == ["yeah 0", "yeah 1", "yeah 2"]
    assert [t for _, t in vp.clips["aside"]] == ["one sec, honey"]


def test_sampled_nudges_vary_and_stay_in_range():
    from interaction_gym.user import TurnTaking

    gaps = []
    for seed in range(6):
        env, ep = run(["Hello, is anyone there?", "Hello?", "Are you there?", "Bye ###STOP###"], listener=None, interrupt=None,
                      seed=seed, agent_ms=0, timing=TurnTaking(nudge_after_ms=4000, nudge_sigma=0.5))
        T = user_turns(ep)[1:]  # after its reply to the agent: the user speaks again unanswered
        gaps += [b["start_time"] - a["end_time"] for a, b in zip(T, T[1:])]
    assert len(set(gaps)) > 3 and all(2000 <= g <= 10000 for g in gaps)


def test_a_random_pause_never_follows_a_full_stop():
    import random

    from interaction_gym.user import _split_mid_thought

    for seed in range(30):
        a, b = _split_mid_thought("Right, I see. That actually sounds doable for us.", random.Random(seed))
        assert not a.endswith("....") and not a[:-3].endswith(".")
