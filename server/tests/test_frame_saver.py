"""Tests for frame_saver.save_frames_batch."""

import json

import pytest

from tv_commercial_detector.config import app_config, audio_dir, images_dir
from tv_commercial_detector.frame_saver import CLASSIFICATIONS_FILE, save_frames_batch
from tv_commercial_detector.state import FrameEntry
from tv_commercial_detector.video_timebase import VideoTimebase

TIMEBASE_KEYS = (
    "video_id",
    "video_duration",
    "is_live",
    "seekable_start",
    "seekable_end",
)


@pytest.fixture
def save_dir(tmp_path):
    original = app_config.save_dir
    app_config.save_dir = tmp_path
    yield tmp_path
    app_config.save_dir = original


def _entry(
    timestamp="2026-01-01T08:00:00.000001", audio=None, timebase=None
) -> FrameEntry:
    return FrameEntry(
        timestamp=timestamp,
        frame_bytes=b"fake-image-bytes",
        ext=".jpg",
        result=None,
        page_title="Test",
        page_url="https://tv.youtube.com/watch/abc",
        video_title="",
        network_name="",
        video_offset=None,
        state_classification=None,
        audio_bytes=audio,
        **({"timebase": timebase} if timebase is not None else {}),
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
    assert records[0]["page_url"] == "https://tv.youtube.com/watch/abc"
    assert records[0]["note"] == "hi"


def test_writes_timebase_fields(save_dir):
    save_frames_batch(
        [
            _entry(
                timebase=VideoTimebase(
                    video_id="1LaATJR0CeM",
                    duration=11982.4,
                    is_live=False,
                    seekable_start=0.0,
                    seekable_end=11982.4,
                )
            )
        ],
        "test",
    )
    record = json.loads((save_dir / CLASSIFICATIONS_FILE).read_text())
    assert record["video_id"] == "1LaATJR0CeM"
    assert record["video_duration"] == pytest.approx(11982.4)
    assert record["is_live"] is False
    assert record["seekable_start"] == pytest.approx(0.0)
    assert record["seekable_end"] == pytest.approx(11982.4)


def test_timebase_fields_present_but_null_when_unreported(save_dir):
    """A frame from an extension predating these fields still saves, and the
    keys are there so a reader never has to distinguish absent from unknown."""
    save_frames_batch([_entry()], "test")
    record = json.loads((save_dir / CLASSIFICATIONS_FILE).read_text())
    for key in TIMEBASE_KEYS:
        assert key in record and record[key] is None


def test_batch_index_distinguishes_frames_sharing_a_timestamp(save_dir):
    saved = save_frames_batch([_entry(), _entry()], "test")
    assert saved == [
        "2026-01-01T08-00-00-000001_0.jpg",
        "2026-01-01T08-00-00-000001_1.jpg",
    ]
