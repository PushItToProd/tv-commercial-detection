import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import tempfile
import time


@dataclass
class FrameEntry:
    timestamp: str              # ISO 8601, from datetime.now().isoformat()
    frame_bytes: bytes
    ext: str                    # ".jpg" or ".png"
    result: Any                 # ClassificationResult | None
    page_title: str
    video_title: str
    network_name: str
    video_offset: float | None
    state_classification: str | None  # state.classification at time of receipt


@dataclass
class AppState:
    classification: str | None = None  # None | "ad" | "content" | "unknown"
    paused: bool = True
    seeking: bool = False
    auto_switch: bool = True
    enable_debounce: bool = True
    last_result: str | None = None  # Immediately previous result, used for debounce
    matrix_switching: bool = False
    last_periodic_save: datetime | None = None
    auto_switch_paused_until: float | None = None  # Unix timestamp; auto-switch temporarily suppressed until this time

    def is_pending_change(self) -> bool:
        return self.last_result is not None and self.last_result != self.classification

    def is_auto_switch_paused(self) -> bool:
        return self.auto_switch_paused_until is not None and self.auto_switch_paused_until > time.time()


state = AppState()

sse_clients: set[asyncio.Queue] = set()

# Rolling buffer of recent frames
recent_frames: deque[FrameEntry] = deque(maxlen=5)

last_image_path = Path(tempfile.gettempdir()) / "tv_detector_last_frame.png"
