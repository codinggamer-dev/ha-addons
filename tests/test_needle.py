"""Tests for the Cactus runtime wrappers.

No Home Assistant and no engine needed: ``needle.load_cactus`` is monkeypatched
with a stand-in, which is how the wrapper's argument handling and its failure
reporting are exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.cactus_home import needle

SAMPLES = np.zeros(160, dtype=np.float32)


class FakeEngine:
    """Stand-in for the ``cactus`` module that accepts arbitrary keywords."""

    def __init__(self, *, fail: str | None = None, text: str = ""):
        self.calls = 0
        self.kwarg_sets: list[dict] = []
        self.audio_types: list[type] = []
        self._fail = fail
        self._text = text

    def set_threads(self, threads):  # pragma: no cover - not exercised here
        pass

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        self.kwarg_sets.append(dict(kwargs))
        self.audio_types.append(type(audio))
        if self._fail is not None:
            raise RuntimeError(self._fail)
        return {"text": self._text}

    def Cactus(self, **kwargs):  # pragma: no cover - not exercised here
        raise AssertionError("text agent should not be built in this test")


class StrictEngine:
    """An engine build whose ``transcribe`` accepts only ``language``.

    This is the shape that breaks a caller passing ``keywords`` or ``weights``:
    the unexpected keyword raises immediately, and Home Assistant reports only
    "speech-to-text failed".
    """

    def __init__(self):
        self.seen: list[dict] = []

    def set_threads(self, threads):  # pragma: no cover - not exercised here
        pass

    def transcribe(self, audio, language=None):
        self.seen.append({"language": language})
        return {"text": "strict ok"}


def patch_engine(monkeypatch, engine):
    monkeypatch.setattr(needle, "load_cactus", lambda telemetry=False: engine)


def test_configured_keywords_are_used_when_no_override_is_given(monkeypatch):
    engine = FakeEngine()
    patch_engine(monkeypatch, engine)

    needle.CactusTranscriber(keywords=["configured"]).transcribe(SAMPLES, "de")

    assert engine.kwarg_sets[0]["keywords"] == ["configured"]


def test_per_call_keywords_override_the_configured_bias(monkeypatch):
    engine = FakeEngine()
    patch_engine(monkeypatch, engine)

    transcriber = needle.CactusTranscriber(keywords=["configured"])
    transcriber.transcribe(SAMPLES, "de", ["Lichtstreifen", "Stehlampe"])

    assert engine.kwarg_sets[0]["keywords"] == ["Lichtstreifen", "Stehlampe"]


def test_per_call_keywords_are_dropped_on_engines_without_that_argument(monkeypatch):
    engine = StrictEngine()
    patch_engine(monkeypatch, engine)

    needle.CactusTranscriber().transcribe(SAMPLES, "de", ["Lichtstreifen"])

    assert engine.seen == [{"language": "de"}]


def test_transcribe_returns_text(monkeypatch):
    patch_engine(monkeypatch, FakeEngine(text="turn on the light"))

    transcriber = needle.CactusTranscriber()
    assert transcriber.transcribe(SAMPLES, "en") == "turn on the light"


def test_empty_text_is_preserved_for_silence(monkeypatch):
    patch_engine(monkeypatch, FakeEngine(text=""))

    transcriber = needle.CactusTranscriber()
    assert transcriber.transcribe(SAMPLES, "en") == ""


def test_var_keyword_engines_receive_every_argument(monkeypatch):
    engine = FakeEngine()
    patch_engine(monkeypatch, engine)

    transcriber = needle.CactusTranscriber(keywords=["Cactus"], weights="/tmp/x.cact")
    transcriber.transcribe(SAMPLES, "de")

    assert engine.kwarg_sets == [
        {"language": "de", "keywords": ["Cactus"], "weights": "/tmp/x.cact"}
    ]
    assert transcriber.last_kwargs == ["keywords", "language", "weights"]


def test_optional_kwargs_are_filtered_to_the_engine_signature(monkeypatch):
    engine = StrictEngine()
    patch_engine(monkeypatch, engine)

    # keywords and weights would raise TypeError on this build.
    transcriber = needle.CactusTranscriber(keywords=["Cactus"], weights="/tmp/x.cact")
    assert transcriber.transcribe(SAMPLES, "de") == "strict ok"

    assert engine.seen == [{"language": "de"}]
    assert transcriber.last_kwargs == ["language"]


def test_introspection_is_safe_on_a_fresh_instance(monkeypatch):
    patch_engine(monkeypatch, FakeEngine())

    # Must not depend on the caller having warmed the engine up first.
    assert "language" in needle.CactusTranscriber()._accepted_kwargs()


def test_the_audio_goes_to_the_engine_as_samples(monkeypatch):
    engine = FakeEngine()
    patch_engine(monkeypatch, engine)

    needle.CactusTranscriber().transcribe(SAMPLES, "en")

    assert engine.audio_types == [np.ndarray]


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


def test_transcribe_reports_engine_failures_with_context(monkeypatch):
    patch_engine(monkeypatch, FakeEngine(fail="engine exploded"))

    transcriber = needle.CactusTranscriber(keywords=["Cactus"])
    with pytest.raises(needle.CactusRuntimeError) as err:
        transcriber.transcribe(SAMPLES, "de")

    # The message must carry enough to diagnose without a debugger.
    assert "engine exploded" in str(err.value)
    assert "160 samples" in str(err.value)
    assert "'de'" in str(err.value)
    assert "language" in str(err.value)


def test_unexpected_response_shape_is_reported(monkeypatch):
    class WeirdEngine(FakeEngine):
        def transcribe(self, audio, **kwargs):
            return 42

    patch_engine(monkeypatch, WeirdEngine())

    with pytest.raises(needle.CactusRuntimeError, match="unexpected Whistle response"):
        needle.CactusTranscriber().transcribe(SAMPLES, "en")


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
