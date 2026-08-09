"""Tests for frame_saver.save_frames_batch."""

import json

import pytest

from tv_commercial_detector.config import app_config, audio_dir, images_dir
from tv_commercial_detector.frame_saver import CLASSIFICATIONS_FILE, save_frames_batch
from tv_commercial_detector.state import FrameEntry


@pytest.fixture
def save_dir(tmp_path):
    original = app_config.save_dir
    app_config.save_dir = tmp_path
    yield tmp_path
    app_config.save_dir = original


def _entry(timestamp="2026-01-01T08:00:00.000001", audio=None) -> FrameEntry:
    return FrameEntry(
        timestamp=timestamp,
        frame_bytes=b"fake-image-bytes",
        ext=".jpg",
        result=None,
        page_title="Test",
        video_title="",
        network_name="",
        video_offset=None,
        state_classification=None,
        audio_bytes=audio,
    )


def test_saves_frame_into_images_dir(save_dir):
    saved = save_frames_batch([_entry()], "test")
    assert saved == ["2026-01-01T08-00-00-000001_0.jpg"]
    assert (images_dir() / saved[0]).read_bytes() == b"fake-image-bytes"
    # Not at the save_dir root, where it would inflate every frame listing.
    assert not (save_dir / saved[0]).exists()


def test_saves_audio_into_audio_dir_with_matching_stem(save_dir):
    saved = save_frames_batch([_entry(audio=b"fake-wav")], "test")
    stem = saved[0].removesuffix(".jpg")
    assert (audio_dir() / f"{stem}.wav").read_bytes() == b"fake-wav"
    assert not (images_dir() / f"{stem}.wav").exists()


def test_omits_audio_file_when_no_audio_captured(save_dir):
    save_frames_batch([_entry()], "test")
    assert not audio_dir().exists() or list(audio_dir().iterdir()) == []


def test_writes_classifications_metadata_at_save_dir_root(save_dir):
    saved = save_frames_batch([_entry()], "periodic", extra={"note": "hi"})
    records = [
        json.loads(line)
        for line in (save_dir / CLASSIFICATIONS_FILE).read_text().splitlines()
    ]
    assert len(records) == 1
    # Records key on the bare filename, not a path, so the layout change is
    # invisible to labels.json / features.jsonl lookups.
    assert records[0]["filename"] == saved[0]
    assert records[0]["save_reason"] == "periodic"
    assert records[0]["note"] == "hi"


def test_batch_index_distinguishes_frames_sharing_a_timestamp(save_dir):
    saved = save_frames_batch([_entry(), _entry()], "test")
    assert saved == [
        "2026-01-01T08-00-00-000001_0.jpg",
        "2026-01-01T08-00-00-000001_1.jpg",
    ]
