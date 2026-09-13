"""Tests for audio_sensor: the model artifact, the rolling window, and abstention."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tv_commercial_detector import audio_health, audio_sensor
from tv_commercial_detector.audio_features import FEATURES
from tv_commercial_detector.audio_sensor import AudioWindow, load_model

DIMS = 4 * len(FEATURES)
REAL_MODEL = (
    Path(audio_sensor.__file__).parent
    / "classifiers"
    / "audio_models"
    / "nascar_on_nbc.json"
)


def model_json(**overrides) -> dict:
    raw = {
        "version": 1,
        "features": FEATURES,
        "block": "mean,sd,current,delta",
        "window_seconds": 30.0,
        "min_clips": 12,
        "min_span_seconds": 24.0,
        "mu": [0.0] * DIMS,
        "sd": [1.0] * DIMS,
        # Only the current clip's first feature counts, so a test can set p(ad)
        # by choosing that one value.
        "w": [0.0] * (2 * len(FEATURES)) + [1.0] + [0.0] * (2 * len(FEATURES) - 1),
        "b": 0.0,
        "ad_threshold": 0.8,
        "content_threshold": 0.2,
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model_json()))
    loaded = load_model(path)
    assert loaded is not None
    return loaded


@pytest.fixture
def fake_features(mocker):
    """Make each clip's feature vector carry *value* in its first slot."""

    def vector(wav_bytes):
        if wav_bytes == b"bad":
            return None
        vec = np.zeros(len(FEATURES))
        vec[0] = float(wav_bytes.decode())
        return vec

    return mocker.patch.object(audio_sensor, "feature_vector", side_effect=vector)


def clip(value: float) -> bytes:
    # A fresh bytes object each call, so identity checks see distinct clips.
    return f"{value}".encode()


def feed(n: int, value: float = 0.0, start: float = 0.0, step: float = 2.0) -> bytes:
    """Observe *n* clips at a steady cadence; returns the last one."""
    last = b""
    for i in range(n):
        last = clip(value)
        audio_sensor.sensor.observe(last, start + i * step, is_seeking=False)
    return last


# --- the model artifact -----------------------------------------------------


def test_shipped_nbc_model_loads():
    model = load_model(REAL_MODEL)
    assert model is not None
    assert 0.0 < model.content_threshold < model.ad_threshold < 1.0


def test_p_ad_is_logistic_regression_on_standardized_block(tmp_path):
    rng = np.random.default_rng(0)
    mu, sd, w = rng.normal(size=DIMS), rng.uniform(0.5, 2, DIMS), rng.normal(size=DIMS)
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(model_json(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), b=0.3))
    )
    model = load_model(path)
    block = rng.normal(size=DIMS)
    expected = 1 / (1 + math.exp(-(((block - mu) / sd) @ w + 0.3)))
    assert model.p_ad(block) == pytest.approx(expected)


def test_p_ad_does_not_overflow(model):
    block = np.zeros(DIMS)
    block[2 * len(FEATURES)] = -1e6
    assert model.p_ad(block) == 0.0
    block[2 * len(FEATURES)] = 1e6
    assert model.p_ad(block) == 1.0


def test_missing_model_is_none(tmp_path):
    assert load_model(tmp_path / "nope.json") is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"version": 2},
        {"features": list(reversed(FEATURES))},
        {"block": "mean,sd"},
        {"w": [0.0] * 3},
        {"ad_threshold": 0.1, "content_threshold": 0.9},
    ],
    ids=["version", "features", "block", "dims", "band"],
)
def test_mismatched_model_is_none(tmp_path, overrides):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(model_json(**overrides)))
    assert load_model(path) is None


def test_unreadable_model_is_none(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json")
    assert load_model(path) is None


# --- the window -------------------------------------------------------------


def test_window_keeps_only_clips_inside_its_span():
    window = AudioWindow()
    for i in range(40):
        window.add(i * 2.0, np.full(len(FEATURES), float(i)))
    block, clips, span = window.block(30.0, 12, 24.0)
    # Offsets 50..78 lie within 30 s of 78; 48 doesn't.
    assert clips == 15
    assert span == pytest.approx(28.0)
    assert block[0] == pytest.approx(np.mean(range(25, 40)))


def test_window_is_cold_until_enough_clips_span_enough_time():
    window = AudioWindow()
    for i in range(12):
        window.add(i * 2.0, np.zeros(len(FEATURES)))
    assert window.block(30.0, 12, 24.0)[0] is None  # 12 clips over 22 s
    window.add(24.0, np.zeros(len(FEATURES)))
    assert window.block(30.0, 12, 24.0)[0] is not None


def test_window_resets_on_backward_step():
    window = AudioWindow()
    for i in range(20):
        window.add(i * 2.0, np.zeros(len(FEATURES)))
    assert window.add(10.0, np.zeros(len(FEATURES))) is True
    assert len(window) == 1


def test_forward_jump_ages_out_the_old_window():
    window = AudioWindow()
    for i in range(20):
        window.add(i * 2.0, np.zeros(len(FEATURES)))
    window.add(500.0, np.zeros(len(FEATURES)))
    assert window.block(30.0, 1, 0.0)[1] == 1


# --- the sensor ---------------------------------------------------------------


def test_warm_sensor_reports_p_ad(model, fake_features):
    last = feed(15, value=5.0)
    reading = audio_sensor.sensor.reading(model, last)
    assert reading.abstain is None
    assert reading.p_ad == pytest.approx(1 / (1 + math.exp(-5.0)))
    assert reading.clips == 15


def test_cold_sensor_abstains(model, fake_features):
    last = feed(5)
    reading = audio_sensor.sensor.reading(model, last)
    assert reading.p_ad is None
    assert reading.abstain == "cold"


def test_coarse_capture_interval_never_warms(model, fake_features):
    last = feed(30, step=10.0)
    assert audio_sensor.sensor.reading(model, last).abstain == "cold"


def test_no_model_abstains(fake_features):
    last = feed(15)
    assert audio_sensor.sensor.reading(None, last).abstain == "no_model"


def test_reading_for_a_different_clip_abstains(model, fake_features):
    feed(15)
    # Equal contents, but not the clip that just went into the window.
    assert audio_sensor.sensor.reading(model, clip(0.0)).abstain == "no_clip"
    assert audio_sensor.sensor.reading(model, None).abstain == "no_clip"


def test_frame_without_audio_abstains(model, fake_features):
    feed(15)
    audio_sensor.sensor.observe(None, 100.0, is_seeking=False)
    assert audio_sensor.sensor.reading(model, None).abstain == "no_clip"


def test_seek_resets_the_window(model, fake_features):
    feed(15)
    last = clip(0.0)
    audio_sensor.sensor.observe(last, 30.0, is_seeking=True)
    assert audio_sensor.sensor.reading(model, last).abstain == "cold"


def test_backward_offset_resets_the_window(model, fake_features):
    feed(15, start=100.0)
    last = clip(0.0)
    audio_sensor.sensor.observe(last, 50.0, is_seeking=False)
    reading = audio_sensor.sensor.reading(model, last)
    assert reading.abstain == "cold"
    assert reading.clips == 1


def test_silent_capture_abstains_and_resets(model, fake_features, mocker):
    feed(15)
    mocker.patch.object(
        type(audio_health.health),
        "is_silent",
        new_callable=mocker.PropertyMock,
        return_value=True,
    )
    last = clip(0.0)
    audio_sensor.sensor.observe(last, 30.0, is_seeking=False)
    assert audio_sensor.sensor.reading(model, last).abstain == "silent"
    assert len(audio_sensor.sensor.window) == 0


def test_unparseable_clip_abstains_but_keeps_the_window(model, fake_features):
    feed(15)
    bad = b"bad"
    audio_sensor.sensor.observe(bad, 30.0, is_seeking=False)
    assert audio_sensor.sensor.reading(model, bad).abstain == "unparseable"
    assert len(audio_sensor.sensor.window) == 15


def test_missing_offset_falls_back_to_wall_clock(model, fake_features, mocker):
    clock = iter(range(0, 1000, 2))
    mocker.patch.object(
        audio_sensor.time, "monotonic", side_effect=lambda: float(next(clock))
    )
    last = b""
    for _ in range(15):
        last = clip(1.0)
        audio_sensor.sensor.observe(last, None, is_seeking=False)
    assert audio_sensor.sensor.reading(model, last).p_ad is not None


def test_status_reflects_the_last_reading(model, fake_features):
    assert audio_sensor.sensor.status() is None
    last = feed(15, value=-5.0)
    audio_sensor.sensor.reading(model, last)
    status = audio_sensor.sensor.status()
    assert status["abstain"] is None
    assert status["p_ad"] == pytest.approx(1 / (1 + math.exp(5.0)))
    # A new clip clears it until a profile reads again.
    feed(1, start=40.0)
    assert audio_sensor.sensor.status() is None
