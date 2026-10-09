"""The default sound bank: DEMAND ambience in a user cache, fetched on first use, synthetic if that fails. No network."""

import math
import random
import warnings

import pytest

import interaction_gym.noisebank as NB
from interaction_gym.soundscape import Soundscape, background_spec, colored_noise, event_clip, render_background, _bank


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """The default bank at ``tmp_path/soundbank`` (auto-fetch allowed, but the network is never touched)."""
    root = tmp_path / "soundbank"
    monkeypatch.setenv("IG_SOUNDBANK", str(root))
    monkeypatch.delenv("IG_SOUNDBANK_FETCH", raising=False)
    NB.reset()
    calls = []

    def no_net(*a, **k):
        calls.append(a)
        raise OSError("offline (test)")

    monkeypatch.setattr(NB, "build_demand_bank", no_net)
    yield root, calls
    NB.reset()


def _ambience(root, kind="street"):
    (root / "ambience" / kind).mkdir(parents=True)
    colored_noise("white", 8000, -10.0, seconds=1).write_wav(root / "ambience" / kind / "x.wav")


def test_default_bank_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("IG_SOUNDBANK", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert NB.default_bank_dir() == tmp_path / "interaction_gym" / "soundbank"
    for off in ("synthetic", "none", "0"):
        monkeypatch.setenv("IG_SOUNDBANK", off)
        assert NB.default_bank_dir() is None


def test_default_uses_the_cache_bank_when_present(cache):
    root, calls = cache
    _ambience(root)
    assert Soundscape().bank == "default"
    spec = background_spec(None, "street", Soundscape().bank)
    assert spec["source"] == "bank"
    bg = render_background(spec, 8000, 7, Soundscape().bank)
    assert spec["source"] == f"file:{root / 'ambience' / 'street' / 'x.wav'}"
    rms = math.sqrt(sum(x * x for x in bg.audio.samples) / len(bg.audio.samples))
    assert abs(20 * math.log10(rms / 32768) - spec["level_dbfs"]) < 0.5
    # seeded as before: the same seed, the same offset
    assert render_background(background_spec(None, "street", "default"), 8000, 7, "default").offset_ms == bg.offset_ms
    # no events in an ambience-only bank: synthetic ones
    assert event_clip("cough", 8000, random.Random(0), "default")[1] == "synthetic"
    assert not calls


def test_events_from_a_full_bank_in_the_cache(cache):
    root, calls = cache
    _ambience(root)
    (root / "events" / "cough").mkdir(parents=True)
    colored_noise("white", 8000, -20.0, seconds=1).write_wav(root / "events" / "cough" / "c.wav")
    _, src = event_clip("cough", 8000, random.Random(0), "default")
    assert src.startswith("file:") and src.endswith("c.wav") and not calls


def test_events_alone_never_fetch(cache):
    root, calls = cache
    assert event_clip("cough", 8000, random.Random(0), "default")[1] == "synthetic" and not calls


def test_falls_back_to_synthetic_when_the_fetch_fails(cache):
    root, calls = cache
    with pytest.warns(RuntimeWarning, match="could not fetch the DEMAND ambience"):
        spec = background_spec(None, "cafe", "default")
    assert spec["source"] == "synthetic:pink" and len(calls) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # once per process: no retry, no second warning
        assert background_spec(None, "street", "default")["source"] == "synthetic:brown"
    assert len(calls) == 1
    assert render_background(spec, 8000, 3, "default") is not None


def test_fetches_once_when_missing(cache, monkeypatch, capsys):
    root, _ = cache
    built = []
    monkeypatch.setattr(NB, "build_demand_bank", lambda r, **k: (built.append(r), _ambience(r)))
    assert _bank("default").root == root and built == [root]
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "DEMAND" in err and "CC BY-SA 3.0" in err and str(root) in err
    assert _bank("default").root == root and built == [root]


def test_opt_out(cache, monkeypatch):
    root, calls = cache
    _ambience(root)
    assert background_spec(None, "street", Soundscape(bank=None).bank)["source"] == "synthetic:brown"
    assert background_spec(None, "street", Soundscape(bank="synthetic").bank)["source"] == "synthetic:brown"
    monkeypatch.setenv("IG_SOUNDBANK", "synthetic")
    NB.reset()
    assert background_spec(None, "street", Soundscape().bank)["source"] == "synthetic:brown"
    monkeypatch.setenv("IG_SOUNDBANK", str(root.parent / "empty"))
    monkeypatch.setenv("IG_SOUNDBANK_FETCH", "0")
    NB.reset()
    assert background_spec(None, "street", "default")["source"] == "synthetic:brown" and not calls
