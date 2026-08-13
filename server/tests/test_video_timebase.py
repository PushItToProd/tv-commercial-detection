"""Tests for parsing the timebase fields the extension reports.

The interesting cases are all about `duration`, which arrives as a string
because Infinity and NaN are exactly the readings that matter and neither
survives JSON.
"""

import json

import pytest

from tv_commercial_detector.video_timebase import VideoTimebase, parse_timebase


def test_finite_duration_is_a_recording():
    tb = parse_timebase(video_duration="11982.4")
    assert tb.duration == pytest.approx(11982.4)
    assert tb.is_live is False


def test_infinite_duration_is_live():
    """A live stream reports Infinity, which has no JSON form — hence is_live."""
    tb = parse_timebase(video_duration="Infinity")
    assert tb.duration is None
    assert tb.is_live is True


def test_nan_duration_is_unknown_not_vod():
    """NaN means metadata hasn't loaded, which says nothing about live-ness."""
    tb = parse_timebase(video_duration="NaN")
    assert tb.duration is None
    assert tb.is_live is None


@pytest.mark.parametrize("value", ["", "null", "undefined", "abc"])
def test_unusable_duration_leaves_both_unknown(value):
    tb = parse_timebase(video_duration=value)
    assert tb.duration is None
    assert tb.is_live is None


def test_seekable_range():
    tb = parse_timebase(seekable_start="0", seekable_end="16263.19")
    assert tb.seekable_start == pytest.approx(0.0)
    assert tb.seekable_end == pytest.approx(16263.19)


def test_non_finite_seekable_bounds_are_dropped():
    """A bound of Infinity is not a position, so it is no more use than none."""
    tb = parse_timebase(seekable_start="NaN", seekable_end="Infinity")
    assert tb.seekable_start is None
    assert tb.seekable_end is None


def test_video_id_passthrough_and_blank():
    assert parse_timebase(video_id="1LaATJR0CeM").video_id == "1LaATJR0CeM"
    assert parse_timebase(video_id="").video_id is None


def test_no_fields_at_all():
    """An extension predating these fields yields an all-unknown timebase."""
    assert parse_timebase() == VideoTimebase()


def test_record_is_json_serializable_for_every_case():
    """The whole point of splitting duration: `classifications.jsonl` must stay
    parseable by things stricter than Python's json module."""
    for duration in ("11982.4", "Infinity", "NaN", ""):
        record = parse_timebase(video_duration=duration).as_record()
        text = json.dumps(record)
        assert "Infinity" not in text and "NaN" not in text
        assert json.loads(text) == record
