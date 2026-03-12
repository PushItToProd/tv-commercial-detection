from collections import deque
from dataclasses import dataclass
from pathlib import Path
import tempfile


@dataclass
class AppState:
    classification: str | None = None  # None | "ad" | "content" | "unknown"
    paused: bool = True
    auto_switch: bool = True
    last_result: str | None = None  # Immediately previous result, used for debounce


state = AppState()

# Rolling buffer of recent frames: each entry is (iso_timestamp: str, png_bytes: bytes)
recent_frames: deque = deque(maxlen=5)

last_image_path = Path(tempfile.gettempdir()) / "tv_detector_last_frame.png"
