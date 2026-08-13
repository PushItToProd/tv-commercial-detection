"""Tests for silent-audio detection."""

import io
import math
import wave

import numpy as np
import pytest

from tv_commercial_detector import audio_health
from tv_commercial_detector.config import app_config


def make_wav(samples: np.ndarray, sample_rate: int = 44100, width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def silent_wav(seconds: float = 4.0) -> bytes:
    return make_wav(np.zeros(int(44100 * seconds), dtype=np.int16))


def tone_wav(seconds: float = 4.0, amplitude: int = 8000) -> bytes:
    t = np.arange(int(44100 * seconds)) / 44100
    return make_wav((amplitude * np.sin(2 * math.pi * 440 * t)).astype(np.int16))


def test_peak_amplitude_silence():
    assert audio_health.peak_amplitude(silent_wav(0.1)) == 0.0


def test_peak_amplitude_tone():
    peak = audio_health.peak_amplitude(tone_wav(0.1, amplitude=16384))
    assert peak == pytest.approx(0.5, abs=0.01)


def test_peak_amplitude_full_scale_is_capped():
    """int16's minimum is -32768, one past positive full scale."""
    samples = np.full(100, -32768, dtype=np.int16)
    assert audio_health.peak_amplitude(make_wav(samples)) == 1.0


def test_peak_amplitude_8_bit_is_offset_from_128():
    assert (
        audio_health.peak_amplitude(
            make_wav(np.full(100, 128, dtype=np.uint8), width=1)
        )
        == 0.0
    )
    assert audio_health.peak_amplitude(
        make_wav(np.full(100, 255, dtype=np.uint8), width=1)
    ) == pytest.approx(127 / 128)


def test_peak_amplitude_rejects_garbage():
    assert audio_health.peak_amplitude(b"not a wav file") is None


def test_peak_amplitude_rejects_empty_clip():
    assert audio_health.peak_amplitude(make_wav(np.zeros(0, dtype=np.int16))) is None


def test_no_warning_before_the_streak_is_reached():
    for _ in range(app_config.audio_silence_clips - 1):
        audio_health.record_clip(silent_wav(0.1))
    assert audio_health.warning() is None


def test_warning_after_consecutive_silent_clips():
    for _ in range(app_config.audio_silence_clips):
        audio_health.record_clip(silent_wav(0.1))
    warning = audio_health.warning()
    assert warning is not None
    assert warning["silent_clips"] == app_config.audio_silence_clips
    assert warning["last_peak"] == 0.0
    assert warning["silent_seconds"] >= 0


def test_signal_clears_the_warning():
    for _ in range(app_config.audio_silence_clips):
        audio_health.record_clip(silent_wav(0.1))
    audio_health.record_clip(tone_wav(0.1))
    assert audio_health.warning() is None
    assert audio_health.health.silent_since is None


def test_streak_requires_consecutive_silence():
    for _ in range(app_config.audio_silence_clips - 1):
        audio_health.record_clip(silent_wav(0.1))
    audio_health.record_clip(tone_wav(0.1))
    audio_health.record_clip(silent_wav(0.1))
    assert audio_health.warning() is None


def test_unparseable_clip_does_not_break_the_streak():
    """A corrupt clip says nothing about whether the source is live."""
    for _ in range(app_config.audio_silence_clips - 1):
        audio_health.record_clip(silent_wav(0.1))
    audio_health.record_clip(b"garbage")
    assert audio_health.warning() is None
    audio_health.record_clip(silent_wav(0.1))
    assert audio_health.warning() is not None


def test_missing_audio_is_not_silence():
    """Runs without the native host send no clips at all; that isn't a fault."""
    for _ in range(app_config.audio_silence_clips + 2):
        audio_health.record_clip(None)
        audio_health.record_clip(b"")
    assert audio_health.warning() is None
    assert audio_health.health.clips_seen == 0


def test_warning_logged_once_per_transition(caplog):
    with caplog.at_level("WARNING", logger="tv_commercial_detector.audio_health"):
        for _ in range(app_config.audio_silence_clips + 5):
            audio_health.record_clip(silent_wav(0.1))
    dead = [r for r in caplog.records if "Audio capture appears dead" in r.message]
    assert len(dead) == 1
