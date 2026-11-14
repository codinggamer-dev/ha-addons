"""Audio helpers for feeding Home Assistant STT streams into Whistle.

Whistle wants 16 kHz mono ``float32`` samples in ``[-1, 1]``. Home Assistant
hands the STT provider raw PCM bytes in whatever format the pipeline
negotiated, so this module converts bytes to samples, folds channels down to
mono and resamples when the negotiated rate is not 16 kHz.

``numpy`` is a hard dependency of Home Assistant, so it is imported eagerly.
``soxr`` is optional: when ``cactus-compute[mic]`` is installed it gives a
higher-quality resampler, otherwise a vectorised linear interpolation is used.
"""

from __future__ import annotations

import numpy as np

_WHISTLE_SAMPLE_RATE = 16000

try:  # pragma: no cover - depends on the optional [mic] extra
    import soxr as _soxr
except ImportError:  # pragma: no cover - the common case
    _soxr = None


class AudioFormatError(ValueError):
    """Raised when the incoming audio cannot be interpreted."""


def pcm_to_float32(
    data: bytes,
    *,
    bit_depth: int,
    channels: int,
    sample_rate: int,
) -> np.ndarray:
    """Decode interleaved little-endian PCM bytes into ``float32`` in [-1, 1].

    ``bit_depth`` is bits per sample: 8, 16, 24 or 32. It is deliberately not
    named ``sample_width``, because that name conventionally means bytes (as in
    ``wave.getsampwidth()``) and mixing the two units silently rejects 16-bit
    audio. 32-bit input is treated as signed integer PCM, which is what Home
    Assistant emits; signed 24-bit is unpacked manually because numpy has no
    native three-byte integer type.
    """
    if not data:
        raise AudioFormatError("empty audio stream")
    if channels < 1:
        raise AudioFormatError(f"invalid channel count {channels}")
    if sample_rate < 1:
        raise AudioFormatError(f"invalid sample rate {sample_rate}")

    if bit_depth == 8:
        # 8-bit PCM in Home Assistant is unsigned.
        samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif bit_depth == 16:
        if len(data) % 2:
            data = data[:-1]
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif bit_depth == 24:
        raw = np.frombuffer(data, dtype=np.uint8)
        raw = raw[: len(raw) - (len(raw) % 3)]
        triples = raw.reshape(-1, 3).astype(np.int32)
        # Little-endian sign extension: shift the top byte's sign bit down.
        values = (triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)) << 8
        samples = values.astype(np.float32) / 2147483648.0
    elif bit_depth == 32:
        if len(data) % 4:
            data = data[: len(data) - (len(data) % 4)]
        samples = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioFormatError(f"unsupported sample bit depth {bit_depth}")

    if channels > 1:
        # Drop a trailing partial frame so reshape cannot fail on odd input.
        samples = samples[: samples.size - (samples.size % channels)]
        samples = samples.reshape(-1, channels).mean(axis=1)

    return np.ascontiguousarray(samples, dtype=np.float32)


def resample(samples: np.ndarray, source_rate: int, target_rate: int = _WHISTLE_SAMPLE_RATE) -> np.ndarray:
    """Resample mono ``float32`` samples to ``target_rate``."""
    if source_rate == target_rate or samples.size == 0:
        return samples
    if source_rate <= 0 or target_rate <= 0:
        raise AudioFormatError(f"invalid resample {source_rate} -> {target_rate}")

    if _soxr is not None:  # pragma: no cover - optional dependency
        return np.asarray(_soxr.resample(samples, source_rate, target_rate, quality="HQ"), dtype=np.float32)

    duration = samples.size / float(source_rate)
    target_count = max(1, int(round(duration * target_rate)))
    source_positions = np.linspace(0.0, duration, num=samples.size, endpoint=False, dtype=np.float64)
    target_positions = np.linspace(0.0, duration, num=target_count, endpoint=False, dtype=np.float64)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def to_whistle_samples(
    data: bytes,
    *,
    bit_depth: int,
    channels: int,
    sample_rate: int,
) -> np.ndarray:
    """Full decode pipeline used by the STT entity.

    ``bit_depth`` is bits per sample (8, 16, 24 or 32), not bytes.
    """
    samples = pcm_to_float32(
        data, bit_depth=bit_depth, channels=channels, sample_rate=sample_rate
    )
    return resample(samples, source_rate=sample_rate, target_rate=_WHISTLE_SAMPLE_RATE)


def write_wav16(path: str, samples: np.ndarray, sample_rate: int = _WHISTLE_SAMPLE_RATE) -> None:
    """Write mono float samples to a 16 kHz 16-bit WAV file.

    Used by the smoke script and handy when a caller prefers a path over raw
    samples.
    """
    import wave

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
