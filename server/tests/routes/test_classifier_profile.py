"""Tests for classifier profile settings endpoints and list_profiles()."""

from tv_commercial_detector.classify import list_profiles
from tv_commercial_detector.config import app_config


# ---------------------------------------------------------------------------
# list_profiles()
# ---------------------------------------------------------------------------


def test_list_profiles_contains_nascar_on_fox():
    profiles = list_profiles()
    assert "nascar_on_fox" in profiles


def test_list_profiles_sorted():
    profiles = list_profiles()
    assert profiles == sorted(profiles)


def test_list_profiles_no_dunder():
    profiles = list_profiles()
    assert "__init__" not in profiles


# ---------------------------------------------------------------------------
# GET /settings/classifier_profile
# ---------------------------------------------------------------------------


def test_get_classifier_profile_default(client):
    resp = client.get("/settings/classifier_profile")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current"] == "nascar_on_fox"
    assert "nascar_on_fox" in body["available"]


def test_get_classifier_profile_available_is_list(client):
    resp = client.get("/settings/classifier_profile")
    body = resp.json()
    assert isinstance(body["available"], list)
    assert len(body["available"]) >= 1


# ---------------------------------------------------------------------------
# POST /settings/classifier_profile
# ---------------------------------------------------------------------------


def test_post_classifier_profile_valid(client):
    resp = client.post(
        "/settings/classifier_profile", json={"profile": "nascar_on_fox"}
    )
    assert resp.status_code == 200
    assert app_config.classifier_profile == "nascar_on_fox"


def test_post_classifier_profile_invalid_name_rejected(client):
    # Names with uppercase, path-separator chars, or starting with a digit
    for bad in ["NASCAR_on_Fox", "../evil", "1bad", "bad name", ""]:
        resp = client.post(
            "/settings/classifier_profile", json={"profile": bad}
        )
        assert resp.status_code in (400, 422), f"expected error for {bad!r}"


def test_post_classifier_profile_nonexistent_profile_rejected(client):
    resp = client.post(
        "/settings/classifier_profile", json={"profile": "does_not_exist"}
    )
    assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# DETECTOR_CLASSIFIER_PROFILE env-var wiring (startup)
# ---------------------------------------------------------------------------


def test_env_var_sets_classifier_profile(tmp_path, monkeypatch):
    """Verify that DETECTOR_CLASSIFIER_PROFILE is applied during lifespan startup."""
    import os

    from fastapi.testclient import TestClient

    monkeypatch.setenv("DETECTOR_CLASSIFIER_PROFILE", "nascar_on_fox")
    monkeypatch.setenv("DETECTOR_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "no_config.json"))
    monkeypatch.delenv("DETECTOR_ENABLE_DEBOUNCE", raising=False)
    monkeypatch.delenv("DETECTOR_MATRIX_URL", raising=False)

    # Import here to avoid polluting the session-scoped app fixture
    from tv_commercial_detector.main import create_app

    fresh_app = create_app()
    with TestClient(fresh_app) as c:
        resp = c.get("/settings/classifier_profile")
        assert resp.status_code == 200
        assert resp.json()["current"] == "nascar_on_fox"
