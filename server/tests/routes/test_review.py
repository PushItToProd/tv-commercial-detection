"""Tests for /review, /frames, /classify, /save, and /features endpoints."""

import io
import json

import pytest
from PIL import Image

from tv_commercial_detector.config import app_config, images_dir, thumbnails_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(color=(128, 128, 128), size=(64, 64)) -> bytes:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _save_jpeg(filename: str) -> None:
    """Write a small test JPEG into the images dir."""
    path = images_dir() / filename
    path.parent.mkdir(parents=True, exist_ok=True)
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


def test_review_bare_date_bounds_cover_the_whole_day(client, frames_dir):
    """A date with no time keeps meaning 00:00:00–23:59:59, as it always did."""
    for hour in ("00", "12", "23"):
        _save_jpeg(f"2026-01-02T{hour}-30-00-000001_0.jpg")
    _save_jpeg("2026-01-03T00-00-00-000001_0.jpg")
    names = _rendered_filenames(client.get("/review?start=2026-01-02&end=2026-01-02"))
    assert names == [
        "2026-01-02T00-30-00-000001_0.jpg",
        "2026-01-02T12-30-00-000001_0.jpg",
        "2026-01-02T23-30-00-000001_0.jpg",
    ]


def test_review_filters_by_time_of_day(client, frames_dir):
    for hour in ("08", "09", "10", "11"):
        _save_jpeg(f"2026-01-02T{hour}-00-00-000001_0.jpg")
    names = _rendered_filenames(
        client.get("/review?start=2026-01-02T09:00&end=2026-01-02T10:00")
    )
    assert names == [
        "2026-01-02T09-00-00-000001_0.jpg",
        "2026-01-02T10-00-00-000001_0.jpg",
    ]


def test_review_time_bounds_span_midnight(client, frames_dir):
    for stamp in (
        "2026-01-02T23-00-00",
        "2026-01-02T23-59-59",
        "2026-01-03T00-00-01",
        "2026-01-03T02-00-00",
    ):
        _save_jpeg(f"{stamp}-000001_0.jpg")
    names = _rendered_filenames(
        client.get("/review?start=2026-01-02T23:30&end=2026-01-03T01:00")
    )
    assert names == [
        "2026-01-02T23-59-59-000001_0.jpg",
        "2026-01-03T00-00-01-000001_0.jpg",
    ]


def test_review_end_bound_without_seconds_covers_the_whole_minute(client, frames_dir):
    for second in ("00", "30", "59"):
        _save_jpeg(f"2026-01-02T09-15-{second}-000001_0.jpg")
    _save_jpeg("2026-01-02T09-16-00-000001_0.jpg")
    names = _rendered_filenames(
        client.get("/review?start=2026-01-02T09:15&end=2026-01-02T09:15")
    )
    assert names == [
        "2026-01-02T09-15-00-000001_0.jpg",
        "2026-01-02T09-15-30-000001_0.jpg",
        "2026-01-02T09-15-59-000001_0.jpg",
    ]


def test_review_accepts_seconds_in_time_bounds(client, frames_dir):
    for second in ("00", "30", "59"):
        _save_jpeg(f"2026-01-02T09-15-{second}-000001_0.jpg")
    names = _rendered_filenames(
        client.get("/review?start=2026-01-02T09:15:30&end=2026-01-02T09:15:30")
    )
    assert names == ["2026-01-02T09-15-30-000001_0.jpg"]


def test_review_time_bounds_apply_across_filename_formats(client, frames_dir):
    """Both naming conventions are placed in time by the same parser."""
    _save_jpeg("2026-01-02_09-30-00.jpg")
    _save_jpeg("2026-01-02T09-45-00-000001_0.jpg")
    _save_jpeg("2026-01-02_11-00-00.jpg")
    names = _rendered_filenames(
        client.get("/review?start=2026-01-02T09:00&end=2026-01-02T10:00")
    )
    assert names == [
        "2026-01-02_09-30-00.jpg",
        "2026-01-02T09-45-00-000001_0.jpg",
    ]


def test_review_time_bounds_exclude_untimestamped_names(client, frames_dir):
    _save_jpeg("2026-01-02T09-30-00-000001_0.jpg")
    _save_jpeg("no_timestamp_here.jpg")
    names = _rendered_filenames(client.get("/review?start=2026-01-01"))
    assert names == ["2026-01-02T09-30-00-000001_0.jpg"]


def test_review_rejects_malformed_time_bounds(client):
    for bad in ("nonsense", "2026-13", "01-02-2026", "2026-01-02T09", "2026-01-02 9:5"):
        resp = client.get("/review", params={"start": bad})
        assert resp.status_code == 400, f"{bad!r} should be rejected"
        resp = client.get("/review", params={"end": bad})
        assert resp.status_code == 400, f"{bad!r} should be rejected"


def test_review_accepts_space_separated_time_bound(client, frames_dir):
    """A space instead of "T" saves percent-encoding in a hand-written URL."""
    _save_jpeg("2026-01-02T09-30-00-000001_0.jpg")
    _save_jpeg("2026-01-02T11-30-00-000001_0.jpg")
    names = _rendered_filenames(
        client.get("/review", params={"start": "2026-01-02 10:00"})
    )
    assert names == ["2026-01-02T11-30-00-000001_0.jpg"]


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


def test_review_excludes_thumbnails(client, frames_dir):
    """Thumbnails live in their own dir, so listing frames never sees them."""
    _save_jpeg("2026-01-01T00-00-00.jpg")
    client.get("/frames/2026-01-01T00-00-00.jpg")  # generates the thumbnail
    assert (thumbnails_dir() / "2026-01-01T00-00-00.jpg").exists()
    assert _rendered_filenames(client.get("/review")) == ["2026-01-01T00-00-00.jpg"]


def test_review_sorts_chronologically_across_filename_formats(client, frames_dir):
    """`_`-style names must interleave with `T`-style ones by timestamp.

    Sorting on the raw name would put every `_` name after every `T` name from
    the same date, since "_" > "T" in ASCII.
    """
    for name in (
        "2026-01-01T10-00-00-000001_0.jpg",
        "2026-01-01_09-30-00.jpg",
        "2026-01-01T08-00-00-000001_0.jpg",
        "2026-01-01_11-15.jpg",  # older format: no seconds
    ):
        _save_jpeg(name)
    assert _rendered_filenames(client.get("/review")) == [
        "2026-01-01T08-00-00-000001_0.jpg",
        "2026-01-01_09-30-00.jpg",
        "2026-01-01T10-00-00-000001_0.jpg",
        "2026-01-01_11-15.jpg",
    ]


def test_review_sorts_batch_index_numerically(client, frames_dir):
    """Batch suffixes are numbers, so _10 must follow _9, not precede it."""
    for i in (0, 2, 9, 10, 11):
        _save_jpeg(f"2026-01-01T08-00-00-000001_{i}.jpg")
    assert _rendered_filenames(client.get("/review")) == [
        f"2026-01-01T08-00-00-000001_{i}.jpg" for i in (0, 2, 9, 10, 11)
    ]


def test_review_unparseable_names_sort_last(client, frames_dir):
    _save_jpeg("aaa_not_a_timestamp.jpg")
    _save_jpeg("2026-01-01T08-00-00-000001_0.jpg")
    assert _rendered_filenames(client.get("/review")) == [
        "2026-01-01T08-00-00-000001_0.jpg",
        "aaa_not_a_timestamp.jpg",
    ]


def test_review_ignores_files_outside_images_dir(client, frames_dir):
    """Metadata and audio at the save_dir root must not appear as frames."""
    _save_jpeg("2026-01-01T08-00-00-000001_0.jpg")
    (frames_dir / "stray.jpg").write_bytes(_jpeg_bytes())
    assert _rendered_filenames(client.get("/review")) == [
        "2026-01-01T08-00-00-000001_0.jpg"
    ]


def test_serve_frame_writes_thumbnail_to_thumbnails_dir(client, frames_dir):
    _save_jpeg("thumb_me.jpg")
    resp = client.get("/frames/thumb_me.jpg")
    assert resp.status_code == 200
    assert (thumbnails_dir() / "thumb_me.jpg").exists()
    assert not (images_dir() / "compressed_thumb_me.jpg").exists()


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
    assert (images_dir() / saved).exists()


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
