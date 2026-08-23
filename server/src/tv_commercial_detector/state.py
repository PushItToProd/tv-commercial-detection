import asyncio
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
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


class VideoStatus(StrEnum):
    """What the player is doing, as far as the server can tell.

    Declared most severe first, which is the order `video_status()` resolves
    them in: a reading nobody has confirmed recently says nothing about the
    player regardless of what it holds, and an explanation the extension gave
    us beats one the server had to infer from silence.

    A StrEnum so the members serialize as their own values over JSON and
    compare equal to the strings clients already send and receive.
    """

    WAITING = "waiting"  # nothing has ever been reported
    STOPPED = "stopped"  # the extension said it stopped capturing
    STALE = "stale"  # reported once, but not recently
    NO_VIDEO = "no_video"  # extension is reporting; it can't find a player
    PAUSED = "paused"
    SEEKING = "seeking"
    PLAYING = "playing"


class StopReason(StrEnum):
    """Why capture stopped, as reported by the extension.

    Validated against this set rather than passed through as free text: the
    value reaches the status page, and a fixed vocabulary is the difference
    between a label and whatever a client felt like sending.
    """

    USER = "user"  # Stop clicked in the popup
    TAB_CLOSED = "tab_closed"  # monitored tab went away
    NO_ENDPOINTS = "no_endpoints"  # nothing configured to post to


def parse_stop_reason(value: str) -> StopReason | None:
    """Coerce a reported stop reason, or None if it isn't one we know."""
    try:
        return StopReason(value)
    except ValueError:
        return None


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
    # Set when the extension says it stopped capturing, cleared by any report
    # that isn't a stop. Knowing capture ended on purpose beats inferring
    # something went wrong from the silence that follows.
    capture_stopped: bool = False
    capture_stop_reason: StopReason | None = None
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

    def video_status(self, stale_after: float) -> VideoStatus:
        """Collapse the reported flags and report age into one status.

        `stale_after` is a number of seconds; 0 or less disables the staleness
        check, leaving the last reading in place however old it is.
        """
        age = self.report_age()
        if age is None:
            return VideoStatus.WAITING
        # An explicit stop outranks staleness: the silence that follows it is
        # expected, and already accounted for.
        if self.capture_stopped:
            return VideoStatus.STOPPED
        if stale_after > 0 and age > stale_after:
            return VideoStatus.STALE
        if self.no_video:
            return VideoStatus.NO_VIDEO
        # Paused outranks seeking to match what the status page has always
        # shown: a scrub on a paused video reads as paused.
        if self.paused:
            return VideoStatus.PAUSED
        if self.seeking:
            return VideoStatus.SEEKING
        return VideoStatus.PLAYING

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
