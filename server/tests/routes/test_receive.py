"""Tests for POST /receive.

classify_image is patched so no real LLM or CV calls are made.
matrix.apply_matrix_settings is also patched so no HTTP calls go out.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from tv_commercial_detector import state as state_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(color: tuple[int, int, int] = (0, 0, 0), size=(64, 64)) -> bytes:
    """Return a tiny JPEG image as bytes."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _post_frame(
    client: TestClient, result: str, mocker, audio: bytes | None = None, **form_fields
):
    """POST a synthetic JPEG to /receive with classify_image mocked to *result*."""
    from tv_commercial_detector.classify import ClassificationResult

    mocker.patch(
        "tv_commercial_detector.routes.receive.classify_image",
        return_value=ClassificationResult(
            source="opencv", type=result, reason="test", reply=None
        ),
    )
    mocker.patch("tv_commercial_detector.routes.receive.apply_matrix_settings")

    data = {
        "is_paused": "false",
        "is_seeking": "false",
        **form_fields,
    }
    files = {"image": ("frame.jpg", _jpeg_bytes(), "image/jpeg")}
    if audio is not None:
        files["audio"] = ("audio.wav", audio, "audio/wav")
    return client.post("/receive", data=data, files=files)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_receive_classifies_as_ad(client, mocker):
    resp = _post_frame(client, "ad", mocker)
    assert resp.status_code == 200
    # With source="opencv", debounce is bypassed → classification should flip.
    assert state_module.state.classification == "ad"


def test_receive_classifies_as_content(client, mocker):
    resp = _post_frame(client, "content", mocker)
    assert resp.status_code == 200
    assert state_module.state.classification == "content"


def test_receive_records_page_metadata_on_the_frame(client, mocker):
    # page_url is what identifies the source site once a frame is on disk;
    # nothing else in the record says which service the broadcast came from.
    _post_frame(
        client,
        "content",
        mocker,
        page_title="Home - YouTube TV",
        page_url="https://tv.youtube.com/watch/abc",
        video_title="Autotrader 400",
        network_name="Oregon's FOX",
    )
    entry = state_module.recent_frames[-1]
    assert entry.page_url == "https://tv.youtube.com/watch/abc"
    assert entry.page_title == "Home - YouTube TV"


def test_receive_records_timebase_on_the_frame(client, mocker):
    # These are what say whether video_offset is a position in the program, so
    # they have to survive the trip from the form onto the frame entry.
    _post_frame(
        client,
        "content",
        mocker,
        video_offset="160.99",
        video_id="1LaATJR0CeM",
        video_duration="11982.4",
        seekable_start="0",
        seekable_end="11982.4",
    )
    entry = state_module.recent_frames[-1]
    assert entry.video_offset == pytest.approx(160.99)
    assert entry.timebase.video_id == "1LaATJR0CeM"
    assert entry.timebase.duration == pytest.approx(11982.4)
    assert entry.timebase.is_live is False
    assert entry.timebase.seekable_end == pytest.approx(11982.4)


def test_receive_records_live_stream_timebase(client, mocker):
    """A live stream's infinite duration arrives as a string and becomes a flag."""
    _post_frame(client, "content", mocker, video_duration="Infinity")
    entry = state_module.recent_frames[-1]
    assert entry.timebase.is_live is True
    assert entry.timebase.duration is None


def test_receive_without_timebase_fields_still_works(client, mocker):
    """An extension predating these fields must keep posting successfully."""
    resp = _post_frame(client, "content", mocker)
    assert resp.status_code == 200
    assert state_module.recent_frames[-1].timebase.video_id is None


def test_receive_no_image_paused(client):
    resp = client.post("/receive", data={"is_paused": "true", "is_seeking": "false"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["paused"] is True


def test_receive_no_image_seeking(client):
    resp = client.post("/receive", data={"is_paused": "false", "is_seeking": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["seeking"] is True


def test_receive_no_image_no_paused_no_seeking(client):
    resp = client.post("/receive", data={"is_paused": "false", "is_seeking": "false"})
    assert resp.status_code == 400


def test_receive_triggers_matrix_on_state_change(client, mocker):
    state_module.state.classification = None
    state_module.state.auto_switch = True
    state_module.state.auto_switch_paused_until = None

    from tv_commercial_detector.classify import ClassificationResult

    mocker.patch(
        "tv_commercial_detector.routes.receive.classify_image",
        return_value=ClassificationResult(
            source="opencv", type="ad", reason="test", reply=None
        ),
    )
    mock_matrix = mocker.patch(
        "tv_commercial_detector.routes.receive.apply_matrix_settings"
    )

    files = {"image": ("frame.jpg", _jpeg_bytes(), "image/jpeg")}
    client.post("/receive", data={"is_paused": "false", "is_seeking": "false"}, files=files)

    mock_matrix.assert_called_once_with("ad")


def test_receive_debounce_enabled_no_immediate_switch(client, mocker):
    """With debounce on and LLM source, the first frame should not commit."""
    from tv_commercial_detector.classify import ClassificationResult

    state_module.state.classification = "content"
    state_module.state.last_result = "content"
    state_module.state.enable_debounce = True

    mocker.patch(
        "tv_commercial_detector.routes.receive.classify_image",
        return_value=ClassificationResult(
            source="llm", type="ad", reason="test", reply=None
        ),
    )
    mock_matrix = mocker.patch(
        "tv_commercial_detector.routes.receive.apply_matrix_settings"
    )

    files = {"image": ("frame.jpg", _jpeg_bytes(), "image/jpeg")}
    client.post("/receive", data={"is_paused": "false", "is_seeking": "false"}, files=files)

    # First "ad" from LLM should not commit because prev was "content"
    assert state_module.state.classification == "content"
    mock_matrix.assert_not_called()


# ---------------------------------------------------------------------------
# GET /recent_frames
# ---------------------------------------------------------------------------


def test_get_recent_frames_empty(client):
    resp = client.get("/recent_frames")
    assert resp.status_code == 200
    assert resp.json() == {"frames": []}


def test_get_recent_frames_populated(client):
    from datetime import datetime

    from tv_commercial_detector.classify import ClassificationResult
    from tv_commercial_detector.state import FrameEntry, recent_frames

    ts = datetime.now().isoformat()
    recent_frames.append(
        FrameEntry(
            timestamp=ts,
            frame_bytes=b"fake",
            ext=".jpg",
            result=ClassificationResult(source="opencv", type="ad", reason="x", reply=None),
            page_title="",
            page_url="",
            video_title="",
            network_name="",
            video_offset=None,
            state_classification="content",
        )
    )
    resp = client.get("/recent_frames")
    assert resp.status_code == 200
    frames = resp.json()["frames"]
    assert len(frames) == 1
    assert frames[0]["timestamp"] == ts
    assert frames[0]["classification"] == "ad"
    assert frames[0]["state_classification"] == "content"


def test_get_recent_frames_no_result_is_null(client):
    from datetime import datetime

    from tv_commercial_detector.state import FrameEntry, recent_frames

    ts = datetime.now().isoformat()
    recent_frames.append(
        FrameEntry(
            timestamp=ts,
            frame_bytes=b"fake",
            ext=".jpg",
            result=None,
            page_title="",
            page_url="",
            video_title="",
            network_name="",
            video_offset=None,
            state_classification=None,
        )
    )
    resp = client.get("/recent_frames")
    assert resp.status_code == 200
    frames = resp.json()["frames"]
    assert frames[0]["classification"] is None


# ---------------------------------------------------------------------------
# GET /recent_frames/{timestamp}/image
# ---------------------------------------------------------------------------


def test_get_recent_frame_image_found(client):
    from tv_commercial_detector.state import FrameEntry, recent_frames

    ts = "2024-01-01T00:00:00.000000"
    recent_frames.append(
        FrameEntry(
            timestamp=ts,
            frame_bytes=b"\xff\xd8\xff",
            ext=".jpg",
            result=None,
            page_title="",
            page_url="",
            video_title="",
            network_name="",
            video_offset=None,
            state_classification=None,
        )
    )
    resp = client.get(f"/recent_frames/{ts}/image")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/jpeg")
    assert resp.content == b"\xff\xd8\xff"


def test_get_recent_frame_image_not_found(client):
    resp = client.get("/recent_frames/no-such-timestamp/image")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /flag_frames
# ---------------------------------------------------------------------------


def test_flag_frames_valid(client):
    resp = client.post(
        "/flag_frames",
        json={"frames": [{"timestamp": "ts1", "label": "ad"}, {"timestamp": "ts2", "label": "content"}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"saved": 0}


def test_flag_frames_invalid_label(client):
    resp = client.post(
        "/flag_frames",
        json={"frames": [{"timestamp": "ts1", "label": "unknown"}]},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Audio health
# ---------------------------------------------------------------------------


def test_silent_audio_surfaces_a_warning_on_status(client, mocker):
    from tests.test_audio_health import silent_wav, tone_wav
    from tv_commercial_detector.config import app_config

    for _ in range(app_config.audio_silence_clips):
        assert _post_frame(client, "content", mocker, audio=silent_wav(0.1)).status_code == 200
    assert client.get("/is_ad/status").json()["audio_warning"] is not None

    _post_frame(client, "content", mocker, audio=tone_wav(0.1))
    assert client.get("/is_ad/status").json()["audio_warning"] is None


def test_frames_without_audio_leave_status_clean(client, mocker):
    from tv_commercial_detector.config import app_config

    for _ in range(app_config.audio_silence_clips + 1):
        _post_frame(client, "content", mocker)
    assert client.get("/is_ad/status").json()["audio_warning"] is None
