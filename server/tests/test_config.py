"""Tests for AppConfig — defaults, JSON loading, and env-var overrides."""

import json
import os

from tv_commercial_detector.config import AppConfig


def test_defaults():
    config = AppConfig()
    assert config.matrix_url == "http://localhost:5000"
    assert config.save_dir.name == "frames"
    assert config.enable_debounce is False
    assert config.output_settings == {"ad": {}, "content": {}}


def test_custom_values():
    config = AppConfig(
        matrix_url="http://my-matrix:9000",
        enable_debounce=True,
    )
    assert config.matrix_url == "http://my-matrix:9000"
    assert config.enable_debounce is True


def test_config_json_loading(tmp_path):
    """Simulate the JSON-loading loop from lifespan."""
    config_data = {
        "matrix_url": "http://json-matrix:9999",
        "enable_debounce": True,
        "output_settings": {"ad": {"1": 2}, "content": {"1": 1}},
    }
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = AppConfig()
    with config_file.open() as f:
        for k, v in json.load(f).items():
            if hasattr(cfg, k.lower()):
                setattr(cfg, k.lower(), v)

    assert cfg.matrix_url == "http://json-matrix:9999"
    assert cfg.enable_debounce is True
    assert cfg.output_settings == {"ad": {"1": 2}, "content": {"1": 1}}


def test_config_json_ignores_unknown_keys(tmp_path):
    """Unknown keys in config.json are silently skipped."""
    config_data = {"matrix_url": "http://ok:1", "nonexistent_key": "boom"}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_data))

    cfg = AppConfig()
    with config_file.open() as f:
        for k, v in json.load(f).items():
            if hasattr(cfg, k.lower()):
                setattr(cfg, k.lower(), v)

    assert cfg.matrix_url == "http://ok:1"


def test_env_var_matrix_url(monkeypatch):
    """DETECTOR_MATRIX_URL is applied by the lifespan env-var loop."""
    monkeypatch.setenv("DETECTOR_MATRIX_URL", "http://env-matrix:8080")

    env_map = {
        "DETECTOR_MATRIX_URL": "matrix_url",
        "DETECTOR_SAVE_DIR": "save_dir",
        "DETECTOR_ENABLE_DEBOUNCE": "enable_debounce",
    }
    cfg = AppConfig()
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            setattr(cfg, attr, val)

    assert cfg.matrix_url == "http://env-matrix:8080"


def test_env_var_save_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DETECTOR_SAVE_DIR", str(tmp_path / "custom_frames"))

    env_map = {
        "DETECTOR_MATRIX_URL": "matrix_url",
        "DETECTOR_SAVE_DIR": "save_dir",
        "DETECTOR_ENABLE_DEBOUNCE": "enable_debounce",
    }
    cfg = AppConfig()
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            setattr(cfg, attr, val)

    assert str(cfg.save_dir) == str(tmp_path / "custom_frames")
