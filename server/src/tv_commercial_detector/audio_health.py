"""Track whether the audio arriving with each frame actually carries signal.

When the native host ends up monitoring a sink the browser isn't playing to, the
capture source is perfectly healthy and completely silent: clips keep arriving,
on schedule and the right length, holding nothing but zeros. Nothing downstream
notices — the LLM just gets a silent clip and the save dir fills with useless
WAVs — so the server checks the clips it receives and surfaces a warning.
"""

import io
import logging
import time
import wave
from dataclasses import dataclass, field

import numpy as np

from .config import app_config

logger = logging.getLogger(__name__)

# How long to stay quiet between repeat warnings once silence is established.
_REWARN_INTERVAL_SECS = 300.0

# WAV sample widths we can read, mapped to their numpy dtype and full-scale
# amplitude. 8-bit WAV is unsigned with a midpoint of 128.
_SAMPLE_DTYPES: dict[int, tuple[str, float]] = {
    1: ("uint8", 128.0),
    2: ("int16", 32768.0),
    4: ("int32", 2147483648.0),
}


def peak_amplitude(wav_bytes: bytes) -> float | None:
    """Peak sample amplitude of a WAV clip as a fraction of full scale.

    Returns None if the clip can't be parsed or has an unsupported sample width,
    which is reported as "unknown" rather than treated as silence.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            width = wf.getsampwidth()
            spec = _SAMPLE_DTYPES.get(width)
            if spec is None:
                return None
            frames = wf.readframes(wf.getnframes())
    except wave.Error, EOFError, ValueError:
        return None

    dtype, full_scale = spec
    samples = np.frombuffer(frames, dtype=dtype)
    if samples.size == 0:
        return None
    if dtype == "uint8":
        deviation = np.abs(samples.astype(np.int32) - 128).max()
    else:
        # int16/int32 minimums are one below -full_scale; clamp so the result
        # can't exceed 1.0.
        deviation = min(np.abs(samples.astype(np.int64)).max(), full_scale)
    return float(deviation) / full_scale


@dataclass
class AudioHealth:
    """Rolling view of recent clips. Reset between test functions."""

    clips_seen: int = 0
    silent_streak: int = 0
    last_peak: float | None = None
    last_clip_at: float | None = None
    silent_since: float | None = None
    _last_warned_at: float | None = field(default=None, repr=False)

    @property
    def is_silent(self) -> bool:
        return self.silent_streak >= app_config.audio_silence_clips


health = AudioHealth()


def reset() -> None:
    global health
    health = AudioHealth()


def record_clip(wav_bytes: bytes | None) -> None:
    """Fold one received clip into the rolling health view.

    A clip that can't be parsed leaves the streak alone: it says nothing about
    whether the capture source is live.
    """
    if not wav_bytes:
        return

    peak = peak_amplitude(wav_bytes)
    now = time.time()
    health.clips_seen += 1
    health.last_clip_at = now
    health.last_peak = peak

    if peak is None:
        logger.warning("Received an audio clip that couldn't be parsed as WAV")
        return

    was_silent = health.is_silent
    if peak > app_config.audio_silence_threshold:
        if was_silent:
            logger.info(
                "Audio capture recovered after %d silent clip(s) (peak %.4f)",
                health.silent_streak,
                peak,
            )
        health.silent_streak = 0
        health.silent_since = None
        health._last_warned_at = None
        return

    health.silent_streak += 1
    if health.silent_since is None:
        health.silent_since = now
    if not health.is_silent:
        return

    # Warn on the transition into silence, then only occasionally, so a whole
    # race's worth of dead clips doesn't bury the classification log.
    if (
        health._last_warned_at is None
        or now - health._last_warned_at >= _REWARN_INTERVAL_SECS
    ):
        health._last_warned_at = now
        logger.warning(
            "Audio capture appears dead: last %d clip(s) are silent"
            " (peak %.4f, silent for %.0fs). The native host is probably"
            " monitoring a sink the browser isn't playing to — check"
            " `pactl list sink-inputs` and native_host/audio_capture.log.",
            health.silent_streak,
            peak,
            now - (health.silent_since or now),
        )


def warning() -> dict | None:
    """Payload for the status endpoint, or None while audio looks fine.

    Absent audio isn't reported: the extension only sends clips when the native
    host is connected, and a run without it isn't a fault.
    """
    if not health.is_silent:
        return None
    return {
        "silent_clips": health.silent_streak,
        "silent_since": health.silent_since,
        "silent_seconds": time.time() - health.silent_since
        if health.silent_since
        else None,
        "last_peak": health.last_peak,
    }
