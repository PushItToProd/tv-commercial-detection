"""The player's timebase, as reported by the extension alongside each frame.

`video_offset` (the player's `currentTime`) is only comparable across separate
capture passes if it measures a position in the *program* rather than time since
the player loaded. These fields settle which one it is: a live stream reports an
infinite `duration` and a DVR window whose `seekable` start creeps forward,
while a recording reports a finite duration and a range starting at 0. Recording
them is what makes it possible to treat two discontinuous passes over the same
program — capture, reboot, capture again the next day — as one timeline, and to
line up passes taken at different capture intervals.

`video_id` names the program itself. It is stable where the title is not: the
title is briefly empty while the player navigates, which is enough to send a
frame to a directory of its own.

Infinity and NaN have no JSON representation, so `duration` is split on the way
in — writing a bare `Infinity` into `classifications.jsonl` produces a file that
Python will read back but `jq` and `JSON.parse` will not:

    duration    is_live   what the player reported
    <float>     False     a recording, of this length
    None        True      a live stream (duration was infinite)
    None        None      nothing usable yet (NaN, absent or unparseable)
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoTimebase:
    video_id: str | None = None
    duration: float | None = None
    is_live: bool | None = None
    seekable_start: float | None = None
    seekable_end: float | None = None

    def as_record(self) -> dict:
        """The fields under the names they carry in `classifications.jsonl`."""
        return {
            "video_id": self.video_id,
            "video_duration": self.duration,
            "is_live": self.is_live,
            "seekable_start": self.seekable_start,
            "seekable_end": self.seekable_end,
        }


def _finite(value: str) -> float | None:
    """A form field as a finite float, or None if it is empty or unusable."""
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_timebase(
    video_id: str = "",
    video_duration: str = "",
    seekable_start: str = "",
    seekable_end: str = "",
) -> VideoTimebase:
    """Build a `VideoTimebase` from the raw `/receive` form fields.

    Every field is optional and independently recoverable: an extension that
    predates them, or a player that has not loaded metadata yet, yields a
    timebase of all-None rather than an error.
    """
    duration: float | None = None
    is_live: bool | None = None
    try:
        raw = float(video_duration) if video_duration else None
    except ValueError:
        raw = None
    if raw is not None:
        if math.isinf(raw):
            is_live = True
        elif math.isfinite(raw):
            duration, is_live = raw, False
        # NaN means metadata hasn't loaded; leave both unknown.

    return VideoTimebase(
        video_id=video_id or None,
        duration=duration,
        is_live=is_live,
        seekable_start=_finite(seekable_start),
        seekable_end=_finite(seekable_end),
    )
