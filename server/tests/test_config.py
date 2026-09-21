"""Tests for AppConfig and load_config — defaults, JSON loading, and env-var overrides."""

import json
from pathlib import Path

import pytest

from tv_commercial_detector.config import AppConfig, load_config


def test_defaults():
    config = AppConfig()
    assert config.matrix_url == "http://localhost:5000"
    assert config.save_dir.name == "frames"
    assert config.enable_debounce is False
    assert config.auto_switch is True
    assert config.output_settings == {"ad": {}, "content": {}}


def test_custom_values():
    config = AppConfig(
        matrix_url="http://my-matrix:9000",
        enable_debounce=True,
    )
    assert config.matrix_url == "http://my-matrix:9000"
    assert config.enable_debounce is True


def _write_config(tmp_path, data) -> Path:
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(data))
    return config_file


def test_config_json_loading(tmp_path):
    config_file = _write_config(
        tmp_path,
        {
            "matrix_url": "http://json-matrix:9999",
            "enable_debounce": True,
            "auto_switch": False,
            "output_settings": {"ad": {"1": 2}, "content": {"1": 1}},
        },
    )
    cfg = AppConfig()
    load_config(cfg, config_file, environ={})

    assert cfg.matrix_url == "http://json-matrix:9999"
    assert cfg.enable_debounce is True
    assert cfg.auto_switch is False
    assert cfg.output_settings == {"ad": {"1": 2}, "content": {"1": 1}}


def test_config_json_ignores_unknown_keys(tmp_path):
    """Unknown keys in config.json are silently skipped."""
    config_file = _write_config(
        tmp_path, {"matrix_url": "http://ok:1", "nonexistent_key": "boom"}
    )
    cfg = AppConfig()
    load_config(cfg, config_file, environ={})

    assert cfg.matrix_url == "http://ok:1"
    assert not hasattr(cfg, "nonexistent_key")


def test_missing_config_file_keeps_defaults(tmp_path):
    cfg = AppConfig()
    load_config(cfg, tmp_path / "absent.json", environ={})
    assert cfg == AppConfig()


def test_env_var_matrix_url(tmp_path):
    cfg = AppConfig()
    load_config(
        cfg,
        tmp_path / "absent.json",
        environ={"DETECTOR_MATRIX_URL": "http://env-matrix:8080"},
    )
    assert cfg.matrix_url == "http://env-matrix:8080"


def test_env_var_save_dir_becomes_path(tmp_path):
    cfg = AppConfig()
    load_config(
        cfg,
        tmp_path / "absent.json",
        environ={"DETECTOR_SAVE_DIR": str(tmp_path / "custom_frames")},
    )
    assert cfg.save_dir == tmp_path / "custom_frames"


def test_env_var_overrides_config_json(tmp_path):
    config_file = _write_config(tmp_path, {"enable_debounce": True})
    cfg = AppConfig()
    load_config(cfg, config_file, environ={"DETECTOR_ENABLE_DEBOUNCE": "0"})
    assert cfg.enable_debounce is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("True", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("off", False),
    ],
)
def test_env_var_enable_debounce_is_parsed(tmp_path, value, expected):
    """Every env value is a string, and bool("0") is True."""
    cfg = AppConfig(enable_debounce=not expected)
    load_config(
        cfg, tmp_path / "absent.json", environ={"DETECTOR_ENABLE_DEBOUNCE": value}
    )
    assert cfg.enable_debounce is expected


def test_empty_env_var_counts_as_unset(tmp_path):
    """docker-compose.yml passes an unset RECEIVER_ENABLE_DEBOUNCE through as ""."""
    config_file = _write_config(tmp_path, {"enable_debounce": True})
    cfg = AppConfig()
    load_config(cfg, config_file, environ={"DETECTOR_ENABLE_DEBOUNCE": ""})
    assert cfg.enable_debounce is True


def test_unparseable_bool_env_var_raises(tmp_path):
    with pytest.raises(ValueError, match="enable_debounce"):
        load_config(
            AppConfig(),
            tmp_path / "absent.json",
            environ={"DETECTOR_ENABLE_DEBOUNCE": "maybe"},
        )


def test_string_bool_in_config_json_is_parsed(tmp_path):
    config_file = _write_config(tmp_path, {"auto_switch": "false"})
    cfg = AppConfig()
    load_config(cfg, config_file, environ={})
    assert cfg.auto_switch is False
