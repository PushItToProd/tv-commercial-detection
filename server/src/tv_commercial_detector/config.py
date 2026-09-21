import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    matrix_url: str = "http://localhost:5000"
    llm_url: str = "http://localhost:3002"
    save_dir: Path = field(default_factory=lambda: Path("frames"))
    enable_debounce: bool = False
    auto_switch: bool = True
    enable_llm_audio: bool = False
    llm_model_name: str = field(default_factory=lambda: os.environ.get("LLAMA_MODEL_NAME", "local"))
    output_settings: dict = field(default_factory=lambda: {"ad": {}, "content": {}})
    classifier_profile: str = "nascar_on_fox"
    phash_threshold: int = 10
    # Peak amplitude (fraction of full scale) below which a clip counts as
    # silent. Dead capture reads exactly 0; the margin covers dither on a live
    # but very quiet source.
    audio_silence_threshold: float = 0.001
    # Consecutive silent clips before the server calls audio capture dead. A
    # real broadcast can be quiet for one clip; three in a row can't be.
    audio_silence_clips: int = 3
    # How long without any report from the extension before its last reading is
    # called stale rather than current. The capture interval is a few seconds,
    # so this is several missed ticks. 0 disables the check.
    video_report_stale_seconds: float = 30.0


app_config = AppConfig()


ENV_OVERRIDES = {
    "DETECTOR_MATRIX_URL": "matrix_url",
    "LLAMA_SERVER_URL": "llm_url",
    "DETECTOR_SAVE_DIR": "save_dir",
    "DETECTOR_ENABLE_DEBOUNCE": "enable_debounce",
    "DETECTOR_CLASSIFIER_PROFILE": "classifier_profile",
}

_FIELD_TYPES = {f.name: f.type for f in fields(AppConfig)}
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off"}


def _parse_bool(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_STRINGS:
        return True
    if lowered in _FALSE_STRINGS:
        return False
    raise ValueError(f"{name}: expected a boolean, got {value!r}")


def _coerce(name: str, value: Any) -> Any:
    """Convert a value from config.json or the environment to the field's type.

    Strings are parsed explicitly because every environment value is one, and
    bool("0") is True.
    """
    field_type = _FIELD_TYPES[name]
    if field_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return _parse_bool(name, value)
        raise ValueError(f"{name}: expected a boolean, got {value!r}")
    if field_type in (int, float) and isinstance(value, str):
        return field_type(value)
    if field_type is Path:
        return Path(value)
    return value


def load_config(
    config: AppConfig,
    config_path: Path,
    environ: Mapping[str, str] = os.environ,
) -> None:
    """Apply config.json, then environment overrides, to config in place.

    config is mutated rather than replaced because other modules hold a
    reference to app_config from import time. An empty environment variable
    counts as unset, since docker-compose.yml passes optional settings through
    as `${VAR:-}`.
    """
    if config_path.exists():
        with config_path.open() as f:
            for k, v in json.load(f).items():
                name = k.lower()
                if hasattr(config, name):
                    setattr(config, name, _coerce(name, v))

    for env_key, name in ENV_OVERRIDES.items():
        val = environ.get(env_key)
        if val:
            setattr(config, name, _coerce(name, val))


# Media lives in per-type subdirectories of save_dir so that listing frames
# doesn't have to walk tens of thousands of thumbnails and audio clips.
# Metadata (labels.json, features.jsonl, classifications.jsonl) stays at the
# save_dir root.
IMAGES_SUBDIR = "images"
THUMBNAILS_SUBDIR = "thumbnails"
AUDIO_SUBDIR = "audio"


# Resolved on each call rather than cached, since save_dir is set during app
# startup and reassigned by tests.
def images_dir() -> Path:
    """Full-size frames."""
    return app_config.save_dir / IMAGES_SUBDIR


def thumbnails_dir() -> Path:
    """Thumbnails generated on demand by the review UI."""
    return app_config.save_dir / THUMBNAILS_SUBDIR


def audio_dir() -> Path:
    """Audio clips captured alongside a frame; same stem as the image."""
    return app_config.save_dir / AUDIO_SUBDIR
