"""Tests for /review, /frames, /classify, /save, and /features endpoints."""

import io
import json

import pytest
from PIL import Image

from tv_commercial_detector.config import app_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(color=(128, 128, 128), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _save_jpeg(filename: str) -> None:
    """Write a small test JPEG into app_config.save_dir."""
    path = app_config.save_dir / filename
    path.write_bytes(_jpeg_bytes())


# ---------------------------------------------------------------------------
# /review (GET)
# ---------------------------------------------------------------------------


def test_review_page_returns_html(client):
    resp = client.get("/review")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /frames/{filename} (thumbnail) and /frames/full/{filename}
# ---------------------------------------------------------------------------


def test_serve_frame_returns_image(client):
    _save_jpeg("test_frame.jpg")
    resp = client.get("/frames/test_frame.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")


def test_serve_frame_path_traversal_blocked(client):
    resp = client.get("/frames/../../etc/passwd")
    assert resp.status_code in (400, 404, 422)


def test_serve_frame_full_returns_image(client):
    _save_jpeg("test_full.jpg")
    resp = client.get("/frames/full/test_full.jpg")
    assert resp.status_code == 200


def test_serve_frame_full_missing(client):
    resp = client.get("/frames/full/does_not_exist.jpg")
    assert resp.status_code == 404


def test_serve_frame_rejects_non_image_extension(client):
    resp = client.get("/frames/malicious.txt")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /classify (POST)
# ---------------------------------------------------------------------------


def test_classify_labels_frame(client):
    _save_jpeg("label_me.jpg")
    resp = client.post("/classify", json={"filename": "label_me.jpg", "label": "ad"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "ad"
    # Verify labels.json was written
    labels_file = app_config.save_dir / "labels.json"
    assert labels_file.exists()
    data = json.loads(labels_file.read_text())
    assert data["label_me.jpg"] == "ad"


def test_classify_valid_labelsaccept_ignore(client):
    resp = client.post("/classify", json={"filename": "x.jpg", "label": "ignore"})
    assert resp.status_code == 200


def test_classify_rejects_invalid_label(client):
    resp = client.post("/classify", json={"filename": "x.jpg", "label": "maybe"})
    assert resp.status_code == 400


def test_classify_rejects_path_traversal(client):
    resp = client.post(
        "/classify", json={"filename": "../etc/passwd", "label": "ad"}
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /save (POST)
# ---------------------------------------------------------------------------


def test_save_stores_image(client):
    files = {"image": ("snap.jpg", _jpeg_bytes(), "image/jpeg")}
    data = {"page_title": "Test Page"}
    resp = client.post("/save", data=data, files=files)
    assert resp.status_code == 200
    saved = resp.json()["saved"]
    assert (app_config.save_dir / saved).exists()


# ---------------------------------------------------------------------------
# /features (POST)
# ---------------------------------------------------------------------------


def test_features_stores_valid_record(client):
    resp = client.post(
        "/features",
        json={
            "filename": "feat.jpg",
            "network_logo": "Fox",
            "logo_position": "upper_right",
            "scoreboard_position": "bottom",
        },
    )
    assert resp.status_code == 200


def test_features_rejects_invalid_logo(client):
    resp = client.post(
        "/features",
        json={"filename": "feat.jpg", "network_logo": "BadNetwork"},
    )
    assert resp.status_code == 400


def test_features_rejects_path_traversal(client):
    resp = client.post(
        "/features",
        json={"filename": "../bad.jpg", "network_logo": "Fox"},
    )
    assert resp.status_code == 400
