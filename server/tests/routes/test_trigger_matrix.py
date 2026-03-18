"""Tests for POST /trigger_matrix and POST /settings/resume_auto_switch."""

import pytest

from tv_commercial_detector import state as state_module


def test_trigger_matrix_ad(client, mocker):
    mock_matrix = mocker.patch("tv_commercial_detector.routes.trigger_matrix.matrix")
    mock_matrix.apply_matrix_settings = mocker.AsyncMock()

    resp = client.post("/trigger_matrix", json={"classification": "ad"})
    assert resp.status_code == 200
    assert resp.json()["triggered"] == "ad"
    assert state_module.state.auto_switch is False


def test_trigger_matrix_content(client, mocker):
    mock_matrix = mocker.patch("tv_commercial_detector.routes.trigger_matrix.matrix")
    mock_matrix.apply_matrix_settings = mocker.AsyncMock()

    resp = client.post("/trigger_matrix", json={"classification": "content"})
    assert resp.status_code == 200
    assert resp.json()["triggered"] == "content"


def test_trigger_matrix_invalid_classification(client):
    resp = client.post("/trigger_matrix", json={"classification": "unknown"})
    assert resp.status_code == 400


def test_resume_auto_switch(client, mocker):
    import time

    state_module.state.auto_switch = False
    state_module.state.auto_switch_paused_until = time.time() + 100
    state_module.state.classification = "ad"

    mock_matrix = mocker.patch("tv_commercial_detector.routes.trigger_matrix.matrix")
    mock_matrix.apply_matrix_settings = mocker.AsyncMock()

    resp = client.post("/settings/resume_auto_switch")
    assert resp.status_code == 200
    assert state_module.state.auto_switch is True
    assert state_module.state.auto_switch_paused_until is None
