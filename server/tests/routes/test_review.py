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


@pytest.fixture
def frames_dir(tmp_path):
    """Give a test its own empty save dir.

    The `client` fixture is session-scoped, so its save dir accumulates frames
    across tests — pagination and filter assertions need a known-empty one.
    """
    original = app_config.save_dir
    app_config.save_dir = tmp_path
    yield tmp_path
    app_config.save_dir = original


def _seed_frames(count: int, prefix: str = "2026-01-01T00-00-") -> list[str]:
    """Write `count` frames with sortable timestamp-style names."""
    names = [f"{prefix}{i:02d}.jpg" for i in range(count)]
    for name in names:
        _save_jpeg(name)
    return names


def _rendered_filenames(resp) -> list[str]:
    """Pull the filenames out of the IMAGES blob embedded in the page."""
    body = resp.text
    blob = body.split("const IMAGES = ", 1)[1].split(";\n", 1)[0]
    return [item["filename"] for item in json.loads(blob)]


# ---------------------------------------------------------------------------
# /review (GET)
# ---------------------------------------------------------------------------


def test_review_page_returns_html(client):
    resp = client.get("/review")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_review_paginates(client, frames_dir):
    _seed_frames(25)
    resp = client.get("/review?per_page=10&page=2")
    assert resp.status_code == 200
    names = _rendered_filenames(resp)
    assert len(names) == 10
    assert names[0] == "2026-01-01T00-00-10.jpg"


def test_review_last_page_is_partial(client, frames_dir):
    _seed_frames(25)
    assert len(_rendered_filenames(client.get("/review?per_page=10&page=3"))) == 5


def test_review_clamps_page_beyond_last(client, frames_dir):
    _seed_frames(5)
    resp = client.get("/review?per_page=10&page=999")
    assert resp.status_code == 200
    assert len(_rendered_filenames(resp)) == 5


def test_review_sort_desc_reverses_order(client, frames_dir):
    _seed_frames(5)
    names = _rendered_filenames(client.get("/review?sort=desc"))
    assert names == sorted(names, reverse=True)


def test_review_rejects_invalid_page_and_per_page(client):
    assert client.get("/review?page=0").status_code == 422
    assert client.get("/review?per_page=0").status_code == 422
    assert client.get("/review?per_page=99999").status_code == 422
    assert client.get("/review?sort=sideways").status_code == 422


def test_review_filters_by_label(client, frames_dir):
    _seed_frames(3)
    client.post(
        "/classify", json={"filename": "2026-01-01T00-00-01.jpg", "label": "ad"}
    )
    assert _rendered_filenames(client.get("/review?label=ad")) == [
        "2026-01-01T00-00-01.jpg"
    ]
    unlabeled = _rendered_filenames(client.get("/review?label=__unset__"))
    assert "2026-01-01T00-00-01.jpg" not in unlabeled
    assert "2026-01-01T00-00-00.jpg" in unlabeled


def test_review_filters_by_feature(client, frames_dir):
    _seed_frames(3)
    client.post(
        "/features",
        json={"filename": "2026-01-01T00-00-02.jpg", "network_logo": "Fox"},
    )
    assert _rendered_filenames(client.get("/review?network_logo=Fox")) == [
        "2026-01-01T00-00-02.jpg"
    ]
    assert "2026-01-01T00-00-02.jpg" not in _rendered_filenames(
        client.get("/review?network_logo=__unset__")
    )


def test_review_rejects_invalid_filter_values(client):
    assert client.get("/review?label=maybe").status_code == 400
    assert client.get("/review?network_logo=BadNetwork").status_code == 400
    assert client.get("/review?logo_position=sideways").status_code == 400
    assert client.get("/review?scoreboard_position=diagonal").status_code == 400


def test_review_filters_by_filename_substring(client, frames_dir):
    _save_jpeg("2026-01-01T00-00-00.jpg")
    _save_jpeg("2026-02-02T00-00-00.jpg")
    assert _rendered_filenames(client.get("/review?q=2026-02")) == [
        "2026-02-02T00-00-00.jpg"
    ]


def test_review_filters_by_date_range(client, frames_dir):
    for day in ("01", "02", "03"):
        _save_jpeg(f"2026-01-{day}T00-00-00.jpg")
    names = _rendered_filenames(client.get("/review?start=2026-01-02&end=2026-01-02"))
    assert names == ["2026-01-02T00-00-00.jpg"]


def test_review_incomplete_filter_excludes_fully_labeled(client, frames_dir):
    _seed_frames(2)
    done = "2026-01-01T00-00-00.jpg"
    client.post("/classify", json={"filename": done, "label": "ad"})
    client.post(
        "/features",
        json={
            "filename": done,
            "network_logo": "Fox",
            "logo_position": "upper_right",
            "scoreboard_position": "bottom",
        },
    )
    names = _rendered_filenames(client.get("/review?incomplete=1"))
    assert done not in names
    assert "2026-01-01T00-00-01.jpg" in names


def test_review_excludes_compressed_thumbnails(client, frames_dir):
    _save_jpeg("2026-01-01T00-00-00.jpg")
    _save_jpeg("compressed_2026-01-01T00-00-00.jpg")
    assert _rendered_filenames(client.get("/review")) == ["2026-01-01T00-00-00.jpg"]


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
    resp = client.post("/classify", json={"filename": "../etc/passwd", "label": "ad"})
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
