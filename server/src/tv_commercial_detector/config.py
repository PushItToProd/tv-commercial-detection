import os
from dataclasses import dataclass, field
from pathlib import Path


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


app_config = AppConfig()


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
