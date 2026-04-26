"""Tests for phash_override module and the /flag_frames route."""

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import tv_commercial_detector.phash_override as phash_override_module
from tv_commercial_detector.config import app_config
from tv_commercial_detector import phash_override
from tv_commercial_detector.state import FrameEntry, recent_frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(color: tuple[int, int, int] = (128, 64, 32), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _write_jpeg(path, color=(128, 64, 32), size=(64, 64)):
    img = Image.new("RGB", size, color)
    img.save(str(path), format="JPEG")
    return str(path)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_check_override_no_overrides_returns_none(tmp_path):
    app_config.save_dir = tmp_path
    phash_override_module.reset()

    result = phash_override.check_override(_write_jpeg(tmp_path / "frame.jpg"))
    assert result is None


def test_round_trip_add_then_check(tmp_path):
    app_config.save_dir = tmp_path
    phash_override_module.reset()

    img_bytes = _jpeg_bytes(color=(10, 20, 30))
    phash_override.add_override(img_bytes, "ad")

    # Write the same image to a temp file and check it
    img_path = tmp_path / "same.jpg"
    img_path.write_bytes(img_bytes)

    result = phash_override.check_override(str(img_path))
    assert result == "ad"


def test_hamming_distance_above_threshold_returns_none(tmp_path):
    app_config.save_dir = tmp_path
    app_config.phash_threshold = 0  # only exact matches
    phash_override_module.reset()

    # Store one image
    stored_bytes = _jpeg_bytes(color=(0, 0, 0))
    phash_override.add_override(stored_bytes, "ad")

    # Check a very different image — solid white vs solid black
    different_path = tmp_path / "different.jpg"
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    img.save(str(different_path), format="JPEG")

    result = phash_override.check_override(str(different_path))
    assert result is None


def test_add_override_writes_json(tmp_path):
    app_config.save_dir = tmp_path
    phash_override_module.reset()

    img_bytes = _jpeg_bytes(color=(50, 100, 150))
    returned_phash = phash_override.add_override(img_bytes, "content")

    overrides_path = tmp_path / "phash_overrides.json"
    assert overrides_path.exists()

    data = json.loads(overrides_path.read_text())
    assert len(data) == 1
    assert data[0]["label"] == "content"
    assert data[0]["phash"] == returned_phash


def test_cache_reload_after_reset(tmp_path):
    app_config.save_dir = tmp_path
    phash_override_module.reset()

    img_bytes = _jpeg_bytes(color=(77, 88, 99))
    phash_override.add_override(img_bytes, "ad")

    # Reset the cache — next call should re-read from disk
    phash_override_module.reset()

    img_path = tmp_path / "reload.jpg"
    img_path.write_bytes(img_bytes)

    result = phash_override.check_override(str(img_path))
    assert result == "ad"


# ---------------------------------------------------------------------------
# HTTP test — /flag_frames
# ---------------------------------------------------------------------------


def test_flag_frames_ignore_saves_nothing(client: TestClient, tmp_path):
    app_config.save_dir = tmp_path
    phash_override_module.reset()

    # Populate recent_frames with one entry via /receive would require more
    # setup; instead test the ignore shortcut directly by posting with no
    # matching timestamp (so saved=0) and an ignore label.
    payload = {
        "frames": [
            {"timestamp": "2099-01-01T00:00:00", "label": "ignore"},
        ]
    }
    resp = client.post("/flag_frames", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"saved": 0}

    overrides_path = tmp_path / "phash_overrides.json"
    assert not overrides_path.exists()


def test_flag_frames_skips_phash_when_unchecked(client: TestClient, tmp_path):
    app_config.save_dir = tmp_path
    phash_override_module.reset()

    recent_frames.append(
        FrameEntry(
            timestamp="2099-01-01T00:00:00",
            frame_bytes=_jpeg_bytes(color=(10, 200, 10)),
            ext=".jpg",
            result=None,
            page_title="?",
            video_title="",
            network_name="",
            video_offset=None,
            state_classification=None,
        )
    )

    payload = {
        "frames": [
            {
                "timestamp": "2099-01-01T00:00:00",
                "label": "ad",
                "phash": False,
            }
        ]
    }
    resp = client.post("/flag_frames", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"saved": 0}

    overrides_path = tmp_path / "phash_overrides.json"
    assert not overrides_path.exists()


def test_flag_frames_defaults_phash_to_true(client: TestClient, tmp_path):
    app_config.save_dir = tmp_path
    phash_override_module.reset()

    recent_frames.append(
        FrameEntry(
            timestamp="2099-01-01T00:00:01",
            frame_bytes=_jpeg_bytes(color=(200, 10, 10)),
            ext=".jpg",
            result=None,
            page_title="?",
            video_title="",
            network_name="",
            video_offset=None,
            state_classification=None,
        )
    )

    payload = {
        "frames": [
            {
                "timestamp": "2099-01-01T00:00:01",
                "label": "content",
            }
        ]
    }
    resp = client.post("/flag_frames", json=payload)
    assert resp.status_code == 200
    assert resp.json() == {"saved": 1}

    overrides_path = tmp_path / "phash_overrides.json"
    assert overrides_path.exists()
    data = json.loads(overrides_path.read_text())
    assert len(data) == 1
    assert data[0]["label"] == "content"
