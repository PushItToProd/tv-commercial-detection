import io
import json

import imagehash
from PIL import Image

from .config import app_config

_overrides: list[dict] | None = None


def _get_overrides_path():
    return app_config.save_dir / "phash_overrides.json"


def get_overrides() -> list[dict]:
    global _overrides
    if _overrides is None:
        path = _get_overrides_path()
        if path.exists():
            with open(path) as f:
                _overrides = json.load(f)
        else:
            _overrides = []
    return _overrides


def add_override(image_bytes: bytes, label: str) -> str:
    h = imagehash.phash(Image.open(io.BytesIO(image_bytes)))
    overrides = get_overrides()
    overrides.append({"phash": str(h), "label": label})
    path = _get_overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(overrides, f)
    return str(h)


def check_override(image_path: str) -> str | None:
    overrides = get_overrides()
    if not overrides:
        return None
    h = imagehash.phash(Image.open(image_path))
    for entry in overrides:
        stored = imagehash.hex_to_hash(entry["phash"])
        if h - stored <= app_config.phash_threshold:
            return entry["label"]
    return None


def reset() -> None:
    global _overrides
    _overrides = None
