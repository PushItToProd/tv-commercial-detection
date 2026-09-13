"""Tests for audio_features — above all, that it matches what the models were fit on.

A shipped model's weights are only meaningful against the exact feature
arithmetic of `experiments/structure/extract_audio.py`, so the package version
is checked against that script directly rather than against stored numbers.
"""

import importlib.util
import io
import math
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from tv_commercial_detector.audio_features import (
    FEATURES,
    feature_vector,
    features,
    trailing_block,
)

EXPERIMENT_DIR = Path(__file__).parent.parent / "experiments" / "structure"


def make_wav(samples: np.ndarray, sample_rate: int = 44100, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.astype("<i2").tobytes())
    return buf.getvalue()


def broadcastish_clip(seconds: float = 4.0, sample_rate: int = 44100) -> np.ndarray:
    """Noise, a tone, and a near-silent gap — something for every feature to see."""
    rng = np.random.default_rng(0)
    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    x = 3000 * rng.standard_normal(n) + 6000 * np.sin(2 * math.pi * 220 * t)
    x[n // 3 : n // 3 + sample_rate // 2] *= 0.001
    return np.clip(x, -32768, 32767)


def load_experiment_module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_features_match_the_experiment_extractor(tmp_path):
    extract_audio = load_experiment_module("extract_audio")
    wav = make_wav(broadcastish_clip())
    path = tmp_path / "clip.wav"
    path.write_bytes(wav)

    expected = extract_audio.features(path)
    actual = features(wav)

    assert actual is not None
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1e-9, abs=1e-12), key


def test_trailing_block_matches_the_experiment_layout():
    sys.path.insert(0, str(EXPERIMENT_DIR))
    try:
        evidence = load_experiment_module("evidence")
    finally:
        sys.path.remove(str(EXPERIMENT_DIR))
    X = np.random.default_rng(1).standard_normal((15, len(FEATURES)))
    np.testing.assert_allclose(trailing_block(X), evidence._trailing(X, win=15)[-1])


def test_feature_vector_is_in_features_order():
    wav = make_wav(broadcastish_clip())
    feats = features(wav)
    vec = feature_vector(wav)
    assert feats is not None and vec is not None
    assert vec.tolist() == [feats[name] for name in FEATURES]


def test_stereo_is_downmixed_to_mono():
    mono = broadcastish_clip()
    stereo = np.column_stack([mono, mono]).ravel()
    np.testing.assert_allclose(
        feature_vector(make_wav(stereo, channels=2)), feature_vector(make_wav(mono))
    )


@pytest.mark.parametrize(
    "wav",
    [
        b"not a wav",
        make_wav(np.zeros(0)),
        make_wav(np.zeros(100)),  # shorter than one STFT frame
    ],
    ids=["garbage", "empty", "too-short"],
)
def test_unanalyzable_clips_return_none(wav):
    assert features(wav) is None
    assert feature_vector(wav) is None


def test_non_16_bit_clips_return_none():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)
        wf.setframerate(44100)
        wf.writeframes(bytes([128]) * 44100)
    assert features(buf.getvalue()) is None
