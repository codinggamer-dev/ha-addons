"""Tests for the Whistle STT entity's pure helpers and debug dump.

These need Home Assistant only because ``stt.py`` subclasses
``SpeechToTextEntity``; they skip cleanly when it is absent.
"""

from __future__ import annotations

import asyncio
import wave

import numpy as np
import pytest

pytest.importorskip("homeassistant")

try:
    from homeassistant.components import stt as stt_component
except Exception:  # noqa: BLE001 - optional HA extras may be missing
    pytest.skip("Home Assistant stt component is unavailable", allow_module_level=True)

from custom_components.cactus_home import CactusHomeData  # noqa: E402
from custom_components.cactus_home.stt import (  # noqa: E402
    CactusWhistleSTTEntity,
    _async_dump_debug_audio,
    _base_language,
    _parse_keywords,
)
from custom_components.cactus_home.needle import CactusTextAgent  # noqa: E402


class FakeHass:
    async def async_add_executor_job(self, fn, *args):
        return fn(*args)


class FakeEntry:
    entry_id = "entry-1"
    title = "Cactus Home"


class FakeTranscriber:
    """Records the samples the entity hands to the engine."""

    def __init__(self, text="understood"):
        self.text = text
        self.received = None
        self.languages: list[str | None] = []

    def transcribe(self, samples, language=None):
        self.received = samples
        self.languages.append(language)
        return self.text


def make_entity(options=None, transcriber=None):
    data = CactusHomeData(
        options=dict(options or {}),
        runtime=CactusTextAgent(),
        transcriber=transcriber,
    )
    entity = CactusWhistleSTTEntity(FakeEntry(), data)
    entity.hass = FakeHass()
    return entity


def metadata(language="en", bit_rate=16, sample_rate=16000, channel=1):
    return stt_component.SpeechMetadata(
        language=language,
        format=stt_component.AudioFormats.WAV,
        codec=stt_component.AudioCodecs.PCM,
        bit_rate=stt_component.AudioBitRates(bit_rate),
        sample_rate=stt_component.AudioSampleRates(sample_rate),
        channel=stt_component.AudioChannels(channel),
    )


def pcm16_bytes(*values):
    out = bytearray()
    for value in values:
        out += int(value).to_bytes(2, "little", signed=True)
    return bytes(out)


def test_entity_decodes_the_metadata_format_before_transcribing():
    """Regression: bit_rate is bits, not bytes.

    Passing ``bit_rate // 8`` to the decoder made every single command fail
    with a decode error and the bare "speech-to-text failed" reply, because the
    decoder compares against bit depths.
    """
    transcriber = FakeTranscriber()
    entity = make_entity(transcriber=transcriber)

    async def stream():
        yield pcm16_bytes(0, 16384, -16384, 32767)

    result = asyncio.run(entity.async_process_audio_stream(metadata(), stream()))

    assert result.result == stt_component.SpeechResultState.SUCCESS
    assert result.text == "understood"
    assert transcriber.received is not None
    assert transcriber.received.shape == (4,)
    assert transcriber.received[0] == pytest.approx(0.0)
    assert transcriber.received[1] == pytest.approx(0.5)
    assert transcriber.received[3] == pytest.approx(1.0, abs=1e-4)
    assert entity._last_error is None


def test_entity_records_a_decode_failure(monkeypatch):
    """A decode failure must be visible, not silent.

    The decode branch was once the only error path that left ``last_error``
    null, which is what made the original failure impossible to diagnose.
    """
    import custom_components.cactus_home.stt as stt_module
    from custom_components.cactus_home.audio import AudioFormatError

    def boom(*args, **kwargs):
        raise AudioFormatError("simulated decode failure")

    monkeypatch.setattr(stt_module, "to_whistle_samples", boom)

    entity = make_entity(transcriber=FakeTranscriber())

    async def stream():
        yield pcm16_bytes(0, 1)

    result = asyncio.run(entity.async_process_audio_stream(metadata(), stream()))

    assert result.result == stt_component.SpeechResultState.ERROR
    assert entity.extra_state_attributes["last_error"] == "simulated decode failure"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("en", "en"),
        ("en-US", "en"),
        ("de_DE", "de"),
        ("pt-BR", None),
        (None, None),
        ("", None),
    ],
)
def test_base_language_mapping(tag, expected):
    assert _base_language(tag) == expected


def test_parse_keywords_accepts_strings_and_lists():
    assert _parse_keywords("Siobhan, Krzysztof") == ["Siobhan", "Krzysztof"]
    assert _parse_keywords("one\ntwo , three") == ["one", "two", "three"]
    assert _parse_keywords(["a", " b "]) == ["a", "b"]
    assert _parse_keywords("") == []


def test_supported_metadata_is_pipeline_compatible():
    entity = make_entity()

    assert entity.supported_languages == ["en", "de", "fr", "es", "it", "nl", "pl"]
    assert stt_component.AudioFormats.WAV in entity.supported_formats
    assert stt_component.AudioCodecs.PCM in entity.supported_codecs
    assert stt_component.AudioBitRates.BITRATE_16 in entity.supported_bit_rates
    assert stt_component.AudioSampleRates.SAMPLERATE_16000 in entity.supported_sample_rates
    assert stt_component.AudioChannels.CHANNEL_MONO in entity.supported_channels


def test_language_resolution_prefers_the_configured_language():
    entity = make_entity({"language": "de"})
    assert entity._resolve_language("en-US", entity._data.options) == "de"

    auto = make_entity({"language": "auto"})
    assert auto._resolve_language("en-US", auto._data.options) == "en"
    assert auto._resolve_language("pt-BR", auto._data.options) is None


def test_debug_dump_writes_a_16k_mono_wav(tmp_path):
    samples = np.zeros(16000, dtype=np.float32)

    asyncio.run(_async_dump_debug_audio(FakeHass(), samples, str(tmp_path)))

    written = tmp_path / "cactus_last_utterance.wav"
    assert written.exists()
    with wave.open(str(written), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 16000


def test_debug_dump_is_a_noop_without_a_directory(tmp_path):
    asyncio.run(_async_dump_debug_audio(FakeHass(), np.zeros(10, dtype=np.float32), ""))

    assert list(tmp_path.iterdir()) == []


def test_debug_dump_swallows_io_errors(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")

    # makedirs under a regular file raises NotADirectoryError (an OSError),
    # which the dump must absorb rather than fail the voice command.
    asyncio.run(
        _async_dump_debug_audio(
            FakeHass(), np.zeros(10, dtype=np.float32), str(blocker / "nested")
        )
    )


def test_entity_surfaces_the_last_error():
    entity = make_entity()

    assert entity.extra_state_attributes["last_error"] is None

    entity._record_error("Whistle speech model could not be loaded")

    assert entity.extra_state_attributes["last_error"] == (
        "Whistle speech model could not be loaded"
    )

    entity._record_error(None)

    assert entity.extra_state_attributes["last_error"] is None
