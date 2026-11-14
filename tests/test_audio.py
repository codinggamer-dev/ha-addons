"""Unit tests for the PCM decoding and resampling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.cactus_home.audio import (
    AudioFormatError,
    pcm_to_float32,
    resample,
    to_whistle_samples,
)


def pcm16(values):
    return np.asarray(values, dtype="<i2").tobytes()


def test_decodes_signed_16_bit_mono():
    samples = pcm_to_float32(pcm16([0, 16384, -16384, 32767]), sample_width=16, channels=1, sample_rate=16000)
    assert samples.dtype == np.float32
    assert samples[0] == pytest.approx(0.0)
    assert samples[1] == pytest.approx(0.5)
    assert samples[2] == pytest.approx(-0.5)
    assert samples[3] == pytest.approx(1.0, abs=1e-4)


def test_folds_stereo_to_mono():
    interleaved = pcm16([16384, -16384, 0, 0])
    samples = pcm_to_float32(interleaved, sample_width=16, channels=2, sample_rate=16000)
    assert samples.shape == (2,)
    assert samples[0] == pytest.approx(0.0)


def test_decodes_unsigned_8_bit():
    samples = pcm_to_float32(bytes([128, 255, 0]), sample_width=8, channels=1, sample_rate=16000)
    assert samples[0] == pytest.approx(0.0)
    assert samples[1] == pytest.approx(1.0, abs=1e-2)
    assert samples[2] == pytest.approx(-1.0, abs=1e-2)


def test_decodes_24_bit():
    raw = b"\x00\x00\x00" + b"\xff\xff\x7f" + b"\x00\x00\x80"
    samples = pcm_to_float32(raw, sample_width=24, channels=1, sample_rate=16000)
    assert samples[0] == pytest.approx(0.0)
    assert samples[1] == pytest.approx(1.0, abs=1e-4)
    assert samples[2] == pytest.approx(-1.0, abs=1e-4)


def test_empty_stream_raises():
    with pytest.raises(AudioFormatError):
        pcm_to_float32(b"", sample_width=16, channels=1, sample_rate=16000)


def test_unsupported_width_raises():
    with pytest.raises(AudioFormatError):
        pcm_to_float32(b"\x00\x00", sample_width=12, channels=1, sample_rate=16000)


def test_resample_changes_length():
    samples = np.zeros(44100, dtype=np.float32)
    resampled = resample(samples, source_rate=44100, target_rate=16000)
    assert resampled.shape[0] == pytest.approx(16000, abs=2)


def test_resample_is_a_noop_at_the_target_rate():
    samples = np.ones(160, dtype=np.float32)
    assert resample(samples, 16000, 16000) is samples


def test_full_pipeline_from_8k_stereo():
    frames = pcm16([1000, 2000] * 8000)
    samples = to_whistle_samples(frames, sample_width=16, channels=2, sample_rate=8000)
    assert samples.dtype == np.float32
    assert samples.shape[0] == pytest.approx(16000, abs=4)
