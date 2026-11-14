"""Tests for the Cactus runtime wrappers.

No Home Assistant and no engine needed: ``needle.load_cactus`` is monkeypatched
with a stand-in, which is exactly how the wrapper's failure reporting is
exercised.
"""

from __future__ import annotations

import pytest

from custom_components.cactus_home import needle


class FakeEngine:
    """Minimal stand-in for the ``cactus`` module."""

    def __init__(self, *, fail: str | None = None):
        self.calls = 0
        self._fail = fail

    def set_threads(self, threads):  # pragma: no cover - not exercised here
        pass

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        if self._fail is not None:
            raise RuntimeError(self._fail)
        return {"text": ""}

    def Cactus(self, **kwargs):  # pragma: no cover - not exercised here
        raise AssertionError("text agent should not be built in this test")


def patch_engine(monkeypatch, engine):
    monkeypatch.setattr(needle, "load_cactus", lambda telemetry=False: engine)


def test_warm_up_loads_the_model_and_is_idempotent(monkeypatch):
    engine = FakeEngine()
    patch_engine(monkeypatch, engine)

    transcriber = needle.CactusTranscriber()
    transcriber.warm_up()
    transcriber.warm_up()

    assert engine.calls == 1
    assert transcriber.loaded


def test_warm_up_names_the_cause_when_the_model_fails(monkeypatch):
    patch_engine(monkeypatch, FakeEngine(fail="weights missing"))

    transcriber = needle.CactusTranscriber()
    with pytest.raises(needle.CactusUnavailable, match="Whistle speech model"):
        transcriber.warm_up()

    # A failed warm-up must not mark the model ready.
    assert transcriber.loaded
    with pytest.raises(needle.CactusUnavailable):
        transcriber.warm_up()


def test_transcribe_reports_engine_failures_as_cactus_errors(monkeypatch):
    patch_engine(monkeypatch, FakeEngine(fail="engine exploded"))

    transcriber = needle.CactusTranscriber()
    import numpy as np

    with pytest.raises(needle.CactusRuntimeError, match="engine exploded"):
        transcriber.transcribe(np.zeros(160, dtype=np.float32), "en")


def test_transcribe_returns_text_and_empty_for_silence(monkeypatch):
    engine = FakeEngine()
    patch_engine(monkeypatch, engine)

    transcriber = needle.CactusTranscriber()
    import numpy as np

    assert transcriber.transcribe(np.zeros(160, dtype=np.float32), "en") == ""


def test_missing_cactus_package_is_reported_clearly(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cactus":
            raise ImportError("no module named cactus")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(needle.CactusUnavailable, match="cactus-compute"):
        needle.load_cactus()
