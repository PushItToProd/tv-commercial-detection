"""Tests for SSE and JSON status endpoints."""

import json

import pytest

from tv_commercial_detector import state as state_module


def test_is_ad_status_default(client):
    resp = client.get("/is_ad/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["classification"] is None
    assert body["paused"] is True
    assert body["auto_switch"] is True


def test_is_ad_status_reflects_state(client):
    state_module.state.classification = "ad"
    state_module.state.paused = False
    resp = client.get("/is_ad/status")
    body = resp.json()
    assert body["classification"] == "ad"
    assert body["paused"] is False


def test_is_ad_html(client):
    resp = client.get("/is_ad")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_is_ad_stream_endpoint_exists(client):
    """Smoke-test: confirm the /is_ad/stream route is registered.

    Reading the full streaming body is not feasible in unit tests because the
    generator has an infinite keep-alive loop. We verify the endpoint is reachable
    by inspecting the route list instead.
    """
    from tv_commercial_detector.routes.status import router

    routes_paths = [r.path for r in router.routes]  # type: ignore[attr-defined]
    assert "/is_ad/stream" in routes_paths


def test_settings_auto_switch(client):
    resp = client.post("/settings/auto_switch", json={"enabled": False})
    assert resp.status_code == 200
    assert state_module.state.auto_switch is False


def test_settings_enable_debounce(client):
    resp = client.post("/settings/enable_debounce", json={"enabled": True})
    assert resp.status_code == 200
    assert state_module.state.enable_debounce is True

