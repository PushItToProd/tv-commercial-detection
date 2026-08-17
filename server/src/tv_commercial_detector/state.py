import asyncio
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .video_timebase import VideoTimebase


@dataclass
class FrameEntry:
    timestamp: str  # ISO 8601, from datetime.now().isoformat()
    frame_bytes: bytes
    ext: str  # ".jpg" or ".png"
    result: Any  # ClassificationResult | None
    page_title: str
    page_url: str
    video_title: str
    network_name: str
    video_offset: float | None
    state_classification: str | None  # state.classification at time of receipt
    audio_bytes: bytes | None = None  # WAV audio captured alongside this frame
    # What `video_offset` is measured against; see video_timebase.py. Defaults
    # to an all-unknown timebase, so a frame from an extension predating these
    # fields still saves.
    timebase: VideoTimebase = field(default_factory=VideoTimebase)


# What the player is doing, as far as the server can tell. The extension
# reports the last three plus `no_video`; the first two are inferred from how
# long it has been since it reported anything at all. `video_status()` collapses
# them into one value, most severe first, because a reading nobody has confirmed
# recently says nothing about the player regardless of what it holds.
VIDEO_WAITING = "waiting"  # nothing has ever been reported
VIDEO_STALE = "stale"  # reported once, but not recently
VIDEO_NO_VIDEO = "no_video"  # extension is reporting; it can't find a player
VIDEO_PAUSED = "paused"
VIDEO_SEEKING = "seeking"
VIDEO_PLAYING = "playing"


@dataclass
class AppState:
    classification: str | None = None  # None | "ad" | "content" | "unknown"
    classification_reason: str | None = None  # Reason for the current classification
    # The last reading the extension sent. Prefer `video_status()` over reading
    # these directly: on their own they can't distinguish "the player is paused"
    # from "nobody has told us anything", which is why `paused` starts True.
    paused: bool = True
    seeking: bool = False
    no_video: bool = False
    # time.monotonic() of the last report from the extension, or None if it has
    # never reported. Monotonic because only the elapsed time matters, and it's
    # served to clients as an age rather than an absolute time.
    last_report_at: float | None = None
    auto_switch: bool = True
    enable_debounce: bool = True
    last_result: str | None = None  # Immediately previous result, used for debounce
    matrix_switching: bool = False
    last_periodic_save: datetime | None = None
    auto_switch_paused_until: float | None = (
        None  # Unix timestamp; auto-switch temporarily suppressed until this time
    )

    def is_pending_change(self) -> bool:
        return self.last_result is not None and self.last_result != self.classification

    def mark_report(self) -> None:
        """Record that the extension just reported in."""
        self.last_report_at = time.monotonic()

    def report_age(self) -> float | None:
        """Seconds since the extension last reported, or None if it never has."""
        if self.last_report_at is None:
            return None
        return time.monotonic() - self.last_report_at

    def video_status(self, stale_after: float) -> str:
        """Collapse the reported flags and report age into one status.

        `stale_after` is a number of seconds; 0 or less disables the staleness
        check, leaving the last reading in place however old it is.
        """
        age = self.report_age()
        if age is None:
            return VIDEO_WAITING
        if stale_after > 0 and age > stale_after:
            return VIDEO_STALE
        if self.no_video:
            return VIDEO_NO_VIDEO
        # Paused outranks seeking to match what the status page has always
        # shown: a scrub on a paused video reads as paused.
        if self.paused:
            return VIDEO_PAUSED
        if self.seeking:
            return VIDEO_SEEKING
        return VIDEO_PLAYING

    def is_auto_switch_paused(self) -> bool:
        return (
            self.auto_switch_paused_until is not None
            and self.auto_switch_paused_until > time.time()
        )


state = AppState()

sse_clients: set[asyncio.Queue] = set()

# Rolling buffer of recent frames
recent_frames: deque[FrameEntry] = deque(maxlen=5)

last_image_path = Path(tempfile.gettempdir()) / "tv_detector_last_frame.png"
