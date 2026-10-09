"""Voice: first-turn clone by default, leveling (trim + loudness), style instructions, seeds."""

import asyncio
import math
from array import array

import pytest

from interaction_gym import Task
from interaction_gym.audio import Audio, active_level, normalize_level, speech_span, trim_silence
from interaction_gym.clients import Cached, FakeSpeech, OpenAISpeech
from interaction_gym.user import NEUTRAL_STYLE, Leveling, UserTurn, Voice, clones


def tone(ms: int, amp: float, sr: int = 16000, lead_ms: int = 0, trail_ms: int = 0) -> Audio:
    n = round(ms * sr / 1000)
    body = array("h", (round(amp * 32767 * math.sin(2 * math.pi * 220 * i / sr)) for i in range(n)))
    return Audio.silence(lead_ms, sr) + Audio(body, sr) + Audio.silence(trail_ms, sr)


class Recorder:
    """A TTS that records its calls and returns a tone with 1 s of trailing silence."""

    def __init__(self, model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", amp=0.05):
        self.model, self.sr, self.amp, self.calls = model, 16000, amp, []

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None, seed=None):
        self.calls.append({"text": text, "voice": voice, "instructions": instructions, "ref": ref_audio is not None,
                           "ref_text": ref_text, "seed": seed})
        return tone(800, self.amp, lead_ms=300, trail_ms=1000)


TASK = Task(id="t", scenario={"profile": {"name": "Ann", "gender": "female", "speaking_style": "energetic, talks fast"}})


def render_all(voice, texts):
    async def go():
        ref, segs = None, []
        for i, text in enumerate(texts):
            seg = await voice.render(f"u{i}", 0, UserTurn(text), TASK, ref)
            ref = voice.reference(seg, ref)
            segs.append(seg)
        return segs
    return asyncio.run(go())


def test_a_tts_without_a_clone_tts_fails_loudly():
    with pytest.raises(ValueError, match="clone"):
        Voice(OpenAISpeech("http://x/v1", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    with pytest.raises(ValueError, match="clone"):
        Voice(Cached(Recorder()))
    assert Voice(Recorder(), clone=False).clone is None  # the explicit opt-out
    assert Voice().clone is None  # no TTS: text-only / recorded turns


def test_tts_that_clone_themselves():
    assert clones(FakeSpeech()) and clones(Cached(OpenAISpeech("http://x/v1", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")))
    assert not clones(OpenAISpeech("http://x/v1", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
    fake = FakeSpeech()
    assert Voice(fake).clone is fake


def test_later_turns_are_cloned_from_the_first():
    tts, clone = Recorder(), Recorder("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    v = Voice(tts, clone=clone)
    render_all(v, ["Hello there, is this the pharmacy?", "mm-hmm", "Are you open on Sunday?"])
    assert [c["text"] for c in tts.calls] == ["Hello there, is this the pharmacy?"]
    assert all(c["ref"] and c["ref_text"] == "Hello there, is this the pharmacy?" for c in clone.calls) and len(clone.calls) == 2
    assert v.profile(TASK)["clone"]["reference"] == "first turn"


def test_style_modes():
    for style, want in (("persona", "energetic, talks fast"), ("neutral", NEUTRAL_STYLE), ("Whisper.", "Whisper."), (None, None)):
        tts = Recorder()
        render_all(Voice(tts, clone=False, style=style), ["Hello there."])
        assert tts.calls[0]["instructions"] == want


def test_leveling_trims_and_normalises():
    a = tone(800, 0.3, lead_ms=500, trail_ms=1500)
    t = trim_silence(a, 120)
    assert abs(t.dur_ms - (800 + 2 * 120)) <= 40
    assert abs(active_level(normalize_level(t, -23.0)) + 23.0) < 0.2
    loud, quiet = Recorder(amp=0.5), Recorder(amp=0.01)
    lv = Leveling()
    a1 = render_all(Voice(loud, clone=False, leveling=lv), ["Hello there."])[0].data
    a2 = render_all(Voice(quiet, clone=False, leveling=lv), ["Hello there."])[0].data
    assert abs(active_level(a1) - active_level(a2)) < 0.2 and a1.dur_ms < 1300  # same level; 1 s trailing silence gone
    assert speech_span(a1)[0] / a1.sr < 0.15


def test_persona_level_offset_is_small_and_fixed():
    lv = Leveling()
    quiet = Task(id="q", scenario={"profile": {"name": "B", "speaking_style": "shy, quiet"}})
    loud = Task(id="l", scenario={"profile": {"name": "C", "speaking_style": "booming, confident"}})
    assert lv.offset(quiet, "x") == -2.0 and lv.offset(loud, "x") == 2.0
    other = Task(id="o", scenario={"profile": {"name": "D", "speaking_style": "relaxed"}})
    assert abs(lv.offset(other, "x")) <= 1.0 and lv.offset(other, "x") == lv.offset(other, "y")


def test_episode_seed():
    tts = Recorder()
    render_all(Voice(tts, clone=False, seed=True), ["One.", "Two."])
    assert tts.calls[0]["seed"] is not None and tts.calls[0]["seed"] == tts.calls[1]["seed"]
    tts = Recorder()
    render_all(Voice(tts, clone=False, seed=False), ["One."])
    assert tts.calls[0]["seed"] is None


class NoSeed:
    model, sr = "x", 16000

    async def synth(self, text, voice="default", instructions=None, ref_audio=None, ref_text=None, language=None):
        return tone(500, 0.1)


def test_defaults_and_the_old_behaviour():
    v = Voice(Recorder(), clone=Recorder("Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    assert v.style == "neutral" and v.instructions(TASK) == NEUTRAL_STYLE and v.leveling == Leveling() and v.seed
    old = Voice(Recorder(), clone=False, style="persona", leveling=None, seed=False)
    tts = old.speech
    a = render_all(old, ["Hello there."])[0].data
    assert tts.calls[0]["instructions"] == "energetic, talks fast" and tts.calls[0]["seed"] is None and a.dur_ms == 2100
    render_all(Voice(NoSeed(), clone=False), ["Hello there."])  # a TTS client that takes no seed gets none
