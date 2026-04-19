"""Shared fixtures for the tv_commercial_detector test suite."""

import os

import pytest
from fastapi.testclient import TestClient

import tv_commercial_detector.phash_override as phash_override_module
from tv_commercial_detector import state as state_module
from tv_commercial_detector.config import app_config
from tv_commercial_detector.main import create_app


@pytest.fixture(scope="session")
def app(tmp_path_factory):
    """Create the FastAPI app once per test session.

    Using session scope avoids re-registering Prometheus metrics on each test.
    """
    frames = tmp_path_factory.mktemp("frames")
    os.environ["DETECTOR_SAVE_DIR"] = str(frames)
    # Point to a non-existent file so the lifespan skips JSON config loading.
    os.environ["CONFIG_FILE"] = str(frames / "no_config.json")
    os.environ.pop("DETECTOR_ENABLE_DEBOUNCE", None)
    os.environ.pop("DETECTOR_MATRIX_URL", None)
    return create_app()


@pytest.fixture(scope="session")
def client(app):
    """Session-scoped HTTP test client (lifespan runs once)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_state():
    """Reset mutable module-level singletons before every test function."""
    s = state_module.state
    s.classification = None
    s.paused = True
    s.seeking = False
    s.auto_switch = True
    s.enable_debounce = False
    s.last_result = None
    s.matrix_switching = False
    s.last_periodic_save = None
    s.auto_switch_paused_until = None
    state_module.sse_clients.clear()
    state_module.recent_frames.clear()
    phash_override_module.reset()

    app_config.matrix_url = "http://localhost:5000"
    app_config.enable_debounce = False
    # Empty output_settings means matrix.apply_matrix_settings is a no-op.
    app_config.output_settings = {"ad": {}, "content": {}}
    yield
