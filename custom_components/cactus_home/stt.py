"""Whistle speech-to-text as a Home Assistant STT entity.

Home Assistant hands the STT provider a negotiated audio stream. The pipeline's
default is 16 kHz mono 16-bit PCM, which Whistle consumes directly; other
rates, bit depths and stereo input are converted in :mod:`.audio` before the
samples reach the engine.

The blocking ``cactus.transcribe`` call always runs in an executor.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import CactusHomeData
from .audio import AudioFormatError, to_whistle_samples, write_wav16
from .const import (
    AUTO_STT_KEYWORD_LIMIT,
    AUTO_STT_KEYWORD_MAX_WORDS,
    CONF_ASSIST_ONLY,
    CONF_AUTO_STT_KEYWORDS,
    CONF_DEBUG_AUDIO_DIR,
    CONF_ENABLED_DOMAINS,
    CONF_EXCLUDE_ENTITIES,
    CONF_LANGUAGE,
    CONF_MAX_TARGETS,
    CONF_STT_KEYWORDS,
    CONF_WHISTLE_WEIGHTS,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_TARGETS,
    DOMAIN,
    MANUFACTURER,
    MODEL_WHISTLE,
    NAME,
    STT_KEYWORD_CACHE_TTL,
    SUPPORTED_LANGUAGES,
)
from .entities import async_collect_targets
from .needle import CactusError
from .tools import device_keywords

_LOGGER = logging.getLogger(__name__)

# Whistle only speaks these; anything else is handled by vocabulary/keywords.
_STT_SAMPLE_RATES = [
    AudioSampleRates.SAMPLERATE_16000,
    AudioSampleRates.SAMPLERATE_8000,
    AudioSampleRates.SAMPLERATE_22000,
    AudioSampleRates.SAMPLERATE_32000,
    AudioSampleRates.SAMPLERATE_44100,
    AudioSampleRates.SAMPLERATE_48000,
]


def _base_language(tag: str | None) -> str | None:
    """Map a Home Assistant language tag ('en-US') onto a Whistle code."""
    if not tag:
        return None
    base = tag.replace("_", "-").split("-", 1)[0].lower()
    return base if base in SUPPORTED_LANGUAGES else None


def _parse_keywords(raw: Any) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [part.strip() for part in str(raw).replace("\n", ",").split(",") if part.strip()]


async def _async_dump_debug_audio(hass: HomeAssistant, samples, directory: Any) -> None:
    """Write the converted 16 kHz clip to disk when debugging is enabled.

    Failures are logged and swallowed: a debugging aid must never affect a
    voice command.
    """
    folder = str(directory or "").strip()
    if not folder:
        return

    target = os.path.join(os.path.expanduser(folder), "cactus_last_utterance.wav")

    def _write() -> None:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        write_wav16(target, samples)

    try:
        await hass.async_add_executor_job(_write)
    except OSError as err:
        _LOGGER.warning("Could not write debug audio to %s: %s", target, err)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Whistle STT entity for a config entry."""
    data: CactusHomeData = hass.data[DOMAIN][entry.entry_id]
    async_add_entities([CactusWhistleSTTEntity(entry, data)])


class CactusWhistleSTTEntity(SpeechToTextEntity):
    """Local speech-to-text backed by Cactus Whistle."""

    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_name = MODEL_WHISTLE

    def __init__(self, entry: ConfigEntry, data: CactusHomeData) -> None:
        self._entry = entry
        self._data = data
        self._last_error: str | None = None
        self._auto_keywords_cache: list[str] | None = None
        self._auto_keywords_at = 0.0
        self._attr_unique_id = f"{entry.entry_id}_{MODEL_WHISTLE.lower()}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title or NAME,
            manufacturer=MANUFACTURER,
            model=MODEL_WHISTLE,
            entry_type=None,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Return the languages Whistle can transcribe."""
        return list(SUPPORTED_LANGUAGES)

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        # 8/16/24/32 bit integer PCM are all decoded in audio.py.
        return [
            AudioBitRates.BITRATE_8,
            AudioBitRates.BITRATE_16,
            AudioBitRates.BITRATE_24,
            AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return list(_STT_SAMPLE_RATES)

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream
    ) -> SpeechResult:
        """Buffer the stream, convert it, and transcribe it with Whistle."""
        raw = bytearray()
        stream_error: str | None = None
        try:
            async for chunk in stream:
                raw.extend(chunk)
        except Exception as err:  # noqa: BLE001 - the stream is external input
            # A stream that ends early is worth transcribing anyway: whatever
            # arrived is usually enough for a short command.
            stream_error = f"{type(err).__name__}: {err}"
            _LOGGER.warning(
                "Whistle audio stream ended early after %d bytes (%s); "
                "transcribing what arrived",
                len(raw),
                stream_error,
            )

        if not raw:
            if stream_error is not None:
                self._record_error(stream_error)
                return SpeechResult(text=None, result=SpeechResultState.ERROR)
            return SpeechResult(text="", result=SpeechResultState.SUCCESS)

        options = self._data.options
        language = self._resolve_language(metadata.language, options)

        try:
            samples = to_whistle_samples(
                bytes(raw),
                # metadata.bit_rate is bits per sample, which is what the
                # decoder wants. Dividing by 8 here was the bug that made every
                # command fail with "speech-to-text failed".
                bit_depth=int(metadata.bit_rate),
                channels=int(metadata.channel),
                sample_rate=int(metadata.sample_rate),
            )
        except AudioFormatError as err:
            self._record_error(str(err))
            _LOGGER.error(
                "Could not decode the audio stream (%d bytes, %s): %s",
                len(raw),
                metadata,
                err,
                exc_info=True,
            )
            return SpeechResult(text=None, result=SpeechResultState.ERROR)

        if samples.size == 0:
            return SpeechResult(text="", result=SpeechResultState.SUCCESS)

        # Optional troubleshooting aid: keep the exact clip Whistle saw.
        await _async_dump_debug_audio(
            self.hass, samples, self._data.options.get(CONF_DEBUG_AUDIO_DIR)
        )

        transcriber = self._data.transcriber
        keywords = self._transcription_keywords()
        try:
            text = await self.hass.async_add_executor_job(
                transcriber.transcribe, samples, language, keywords
            )
        except CactusError as err:
            self._record_error(str(err))
            _LOGGER.error(
                "Whistle transcription failed for a %.1f second clip: %s",
                samples.size / 16000,
                err,
                exc_info=True,
            )
            return SpeechResult(text=None, result=SpeechResultState.ERROR)
        except Exception as err:  # noqa: BLE001 - never raise into the pipeline
            self._record_error(f"{type(err).__name__}: {err}")
            _LOGGER.exception("Unexpected Whistle failure")
            return SpeechResult(text=None, result=SpeechResultState.ERROR)

        self._record_error(None)

        if not text:
            _LOGGER.debug("Whistle returned no speech for %s", metadata.language)
            return SpeechResult(text="", result=SpeechResultState.SUCCESS)

        return SpeechResult(text=text, result=SpeechResultState.SUCCESS)

    def _resolve_language(self, metadata_language: str, options: dict[str, Any]) -> str | None:
        """Prefer the configured language, else honour the pipeline's tag."""
        configured = options.get(CONF_LANGUAGE, DEFAULT_LANGUAGE)
        if configured and configured != "auto":
            return str(configured)
        return _base_language(metadata_language)

    def _transcription_keywords(self) -> list[str] | None:
        """Manual keywords first, then device names when the option is on."""
        combined = list(_parse_keywords(self._data.options.get(CONF_STT_KEYWORDS)))
        if self._data.options.get(CONF_AUTO_STT_KEYWORDS):
            combined.extend(self._auto_keywords())

        seen: set[str] = set()
        unique: list[str] = []
        for name in combined:
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(name)
        return unique or None

    def _auto_keywords(self) -> list[str]:
        """Device names for this home, cached briefly.

        Experimental. Any failure here is swallowed: a bias list is a nice to
        have and must never cost the user a voice command.
        """
        if self._auto_keywords_cache is not None and (
            time.monotonic() - self._auto_keywords_at < STT_KEYWORD_CACHE_TTL
        ):
            return self._auto_keywords_cache

        options = self._data.options
        try:
            targets = async_collect_targets(
                self.hass,
                {
                    "enabled_domains": options.get(CONF_ENABLED_DOMAINS),
                    "exclude_entities": options.get(CONF_EXCLUDE_ENTITIES),
                    "assist_only": options.get(CONF_ASSIST_ONLY, True),
                    "max_targets": options.get(CONF_MAX_TARGETS, DEFAULT_MAX_TARGETS),
                },
            )
        except Exception:  # noqa: BLE001 - never fail a command over a bias list
            _LOGGER.exception("Could not collect device names for keyword biasing")
            return []

        names = device_keywords(
            targets,
            limit=AUTO_STT_KEYWORD_LIMIT,
            max_words=AUTO_STT_KEYWORD_MAX_WORDS,
        )
        self._auto_keywords_cache = names
        self._auto_keywords_at = time.monotonic()
        _LOGGER.debug("Biasing speech recognition toward %d device names", len(names))
        return names

    def _record_error(self, message: str | None) -> None:
        """Keep the last failure on the entity so the UI can show it.

        Home Assistant reduces any STT failure to the same "speech-to-text
        failed" reply, so the real reason has to be surfaced somewhere visible.
        """
        self._last_error = message
        try:
            self.async_write_ha_state()
        except Exception:  # noqa: BLE001 - the entity may not be added yet
            pass

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Expose the active configuration and the last failure."""
        return {
            "weights": self._data.options.get(CONF_WHISTLE_WEIGHTS) or "base whistle.cact",
            "keywords": _parse_keywords(self._data.options.get(CONF_STT_KEYWORDS)),
            "auto_keywords_enabled": bool(self._data.options.get(CONF_AUTO_STT_KEYWORDS)),
            "auto_keywords": self._auto_keywords_cache or [],
            "debug_audio_dir": self._data.options.get(CONF_DEBUG_AUDIO_DIR) or None,
            "engine_kwargs": list(getattr(self._data.transcriber, "last_kwargs", []) or []),
            "last_error": self._last_error,
        }
