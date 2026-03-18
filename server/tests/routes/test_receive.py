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


def _post_frame(client: TestClient, result: str, mocker, **form_fields):
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
