"""IG_* environment variables, with the pre-rename DIG_* names as a deprecated fallback."""

import warnings

import pytest

import interaction_gym.envvars as E


def test_ig_first_then_dig_then_default(monkeypatch):
    monkeypatch.setattr(E, "_warned", False)
    monkeypatch.delenv("IG_X_URL", raising=False)
    monkeypatch.delenv("DIG_X_URL", raising=False)
    assert E.getenv("IG_X_URL", "d") == "d"
    monkeypatch.setenv("DIG_X_URL", "old")
    with pytest.warns(FutureWarning, match="DIG_X_URL is deprecated"):
        assert E.getenv("IG_X_URL", "d") == "old"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # warns once per process
        assert E.getenv("X_URL") == "old"
    monkeypatch.setenv("IG_X_URL", "new")
    assert E.getenv("IG_X_URL", "d") == "new"
