"""p(ad) from the audio of the last ~30 s, as a sensor classifier profiles can consult.

A single 4 s clip is the wrong unit: a break lasts minutes, so the useful
question is whether the recent past has sounded like commercials. So `/receive`
feeds every clip to a rolling window here via `observe()`, and a profile asks
for a `reading()` against its own fitted `AudioModel`.

The window is kept in broadcast time (`video_offset`), not frame count, because
the capture interval is configurable and a model's trailing statistics were
learned at one particular cadence. It follows from that the sensor has to be
honest about when it has no opinion, and it abstains rather than guessing:

- `no_model`     the profile ships no usable model
- `no_clip`      the frame being classified isn't the clip last observed — a
                 re-classify of a saved frame must not borrow the live window
- `unparseable`  the clip couldn't be analyzed
- `silent`       `audio_health` says capture is dead; zeros carry no evidence
- `cold`         too few clips in the window to match what the model was fit
                 on — after a restart or a seek, or permanently at a capture
                 interval much coarser than the training cadence

Models are fit by `scripts/fit_audio_model.py`.
"""

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio_health
from .audio_features import FEATURES, feature_vector, trailing_block

logger = logging.getLogger(__name__)

MODEL_VERSION = 1
BLOCK_LAYOUT = "mean,sd,current,delta"

# History kept regardless of model. Longer than any window a model uses, so the
# window itself is the model's choice.
HISTORY_SECONDS = 120.0

# Feature extraction runs on the request path every capture tick; it measures
# a few ms, so anything near this is worth hearing about.
SLOW_FEATURES_SECONDS = 0.1


@dataclass(frozen=True)
class AudioModel:
    """L2 logistic regression over one trailing block of audio features."""

    window_seconds: float
    min_clips: int
    min_span_seconds: float
    mu: np.ndarray
    sd: np.ndarray
    w: np.ndarray
    b: float
    # p(ad) at or above `ad_threshold` is a confident ad, at or below
    # `content_threshold` a confident content; between them the sensor has a
    # score but not a verdict. Fit per model: the ranking transfers between
    # broadcasts better than any fixed threshold does.
    ad_threshold: float
    content_threshold: float
    meta: dict = field(default_factory=dict, compare=False)

    def p_ad(self, block: np.ndarray) -> float:
        z = float(((block - self.mu) / self.sd) @ self.w + self.b)
        # Numerically stable sigmoid either side of zero.
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)


def load_model(path: Path) -> AudioModel | None:
    """Load a fitted model, or None — logged — if it's missing or doesn't fit.

    A bad artifact degrades the profile to its next check instead of breaking
    the import of the whole profile.
    """
    try:
        raw = json.loads(Path(path).read_text())
    except FileNotFoundError:
        logger.warning("No audio model at %s; audio sensor disabled", path)
        return None
    except (OSError, ValueError) as exc:
        logger.warning("Can't read audio model %s: %s", path, exc)
        return None

    problem = _validate(raw)
    if problem:
        logger.warning("Ignoring audio model %s: %s", path, problem)
        return None
    return AudioModel(
        window_seconds=float(raw["window_seconds"]),
        min_clips=int(raw["min_clips"]),
        min_span_seconds=float(raw["min_span_seconds"]),
        mu=np.array(raw["mu"], dtype=np.float64),
        sd=np.array(raw["sd"], dtype=np.float64),
        w=np.array(raw["w"], dtype=np.float64),
        b=float(raw["b"]),
        ad_threshold=float(raw["ad_threshold"]),
        content_threshold=float(raw["content_threshold"]),
        meta={k: raw[k] for k in ("trained_on", "metrics") if k in raw},
    )


def _validate(raw: dict) -> str | None:
    if raw.get("version") != MODEL_VERSION:
        return f"unsupported version {raw.get('version')!r}"
    if raw.get("features") != FEATURES:
        return "feature list doesn't match audio_features.FEATURES"
    if raw.get("block") != BLOCK_LAYOUT:
        return f"unsupported block layout {raw.get('block')!r}"
    dims = 4 * len(FEATURES)
    for key in ("mu", "sd", "w"):
        if len(raw.get(key) or []) != dims:
            return f"{key} must have {dims} entries"
    if not raw["content_threshold"] < raw["ad_threshold"]:
        return "content_threshold must be below ad_threshold"
    return None


class AudioWindow:
    """Feature vectors for recent clips, keyed by broadcast time."""

    def __init__(self, history_seconds: float = HISTORY_SECONDS) -> None:
        self.history_seconds = history_seconds
        self._entries: deque[tuple[float, np.ndarray]] = deque()

    def __len__(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        self._entries.clear()

    def add(self, t: float, vec: np.ndarray) -> bool:
        """Append a clip at time *t*. Returns True if that reset the window.

        A step backward in time is a seek the extension didn't flag; carrying
        the window across it would mix two unrelated stretches of audio. A
        forward jump needs no rule of its own, because eviction by age drops
        whatever it left behind.
        """
        reset = bool(self._entries) and t < self._entries[-1][0]
        if reset:
            self._entries.clear()
        self._entries.append((t, vec))
        while self._entries and self._entries[0][0] <= t - self.history_seconds:
            self._entries.popleft()
        return reset

    def block(
        self, window_seconds: float, min_clips: int, min_span_seconds: float
    ) -> tuple[np.ndarray | None, int, float]:
        """(model input, clips in window, span in seconds) for the newest clip.

        The input is None while the window is too thin to match what a model
        with these parameters was fit on.
        """
        if not self._entries:
            return None, 0, 0.0
        t_now = self._entries[-1][0]
        recent = [(t, v) for t, v in self._entries if t > t_now - window_seconds]
        span = t_now - recent[0][0]
        if len(recent) < min_clips or span < min_span_seconds:
            return None, len(recent), span
        return trailing_block(np.vstack([v for _, v in recent])), len(recent), span


@dataclass(frozen=True)
class AudioReading:
    p_ad: float | None  # None whenever `abstain` is set
    clips: int = 0
    span_seconds: float = 0.0
    abstain: str | None = None

    def signals(self) -> dict:
        """The reading as it's recorded alongside a classification."""
        return {"p_audio": self.p_ad, "audio_abstain": self.abstain}


class AudioSensor:
    def __init__(self) -> None:
        self.window = AudioWindow()
        self._last_clip: bytes | None = None
        self._last_problem: str | None = None
        self._warm = False
        self._cadence_warned = False
        self.last_reading: AudioReading | None = None

    def observe(
        self, wav_bytes: bytes | None, video_offset: float | None, is_seeking: bool
    ) -> None:
        """Fold the clip that arrived with a frame into the window.

        Call after `audio_health.record_clip`, so a clip that completes a
        silent streak is judged against it.
        """
        self.last_reading = None
        self._last_clip = wav_bytes
        self._last_problem = None
        if not wav_bytes:
            return
        if is_seeking:
            self._reset("seek")
        if audio_health.health.is_silent:
            self._reset("audio capture is silent")
            self._last_problem = "silent"
            return

        started = time.perf_counter()
        vec = feature_vector(wav_bytes)
        elapsed = time.perf_counter() - started
        if elapsed > SLOW_FEATURES_SECONDS:
            logger.warning("Audio feature extraction took %.0f ms", elapsed * 1000)
        if vec is None:
            # Says nothing about the audio around it, so the window stands.
            self._last_problem = "unparseable"
            return

        t = video_offset if video_offset is not None else time.monotonic()
        if self.window.add(t, vec):
            self._reset("video offset stepped backward", clear=False)

    def reading(
        self, model: AudioModel | None, wav_bytes: bytes | None
    ) -> AudioReading:
        """The sensor's opinion of the clip last passed to `observe()`."""
        reading = self._reading(model, wav_bytes)
        self.last_reading = reading
        return reading

    def _reading(
        self, model: AudioModel | None, wav_bytes: bytes | None
    ) -> AudioReading:
        if model is None:
            return AudioReading(p_ad=None, abstain="no_model")
        # Identity, not equality: the question is whether this is the clip that
        # just went into the window, not whether two clips happen to match.
        if wav_bytes is None or wav_bytes is not self._last_clip:
            return AudioReading(p_ad=None, abstain="no_clip")
        if self._last_problem:
            return AudioReading(p_ad=None, abstain=self._last_problem)

        block, clips, span = self.window.block(
            model.window_seconds, model.min_clips, model.min_span_seconds
        )
        if block is None:
            # A window spanning its full length with half the clips it needs
            # will never fill: the capture interval is too coarse for the model.
            coarse = span >= model.min_span_seconds and clips * 2 <= model.min_clips
            if coarse and not self._cadence_warned:
                self._cadence_warned = True
                logger.warning(
                    "Audio sensor has only %d clips over %.0f s and needs %d;"
                    " the capture interval is too long for it to give an opinion",
                    clips,
                    span,
                    model.min_clips,
                )
            return AudioReading(
                p_ad=None, clips=clips, span_seconds=span, abstain="cold"
            )
        if not self._warm:
            self._warm = True
            logger.info("Audio sensor warm: %d clips over %.0f s", clips, span)
        return AudioReading(p_ad=model.p_ad(block), clips=clips, span_seconds=span)

    def reset_window(self, why: str) -> None:
        """Discard the window, so the sensor abstains until it refills."""
        self._reset(why)

    def _reset(self, why: str, clear: bool = True) -> None:
        if clear:
            self.window.reset()
        if self._warm:
            logger.info(
                "Audio sensor window reset (%s); abstaining until it refills", why
            )
        self._warm = False

    def status(self) -> dict | None:
        """Payload for the status endpoint; None until a profile has asked."""
        if self.last_reading is None:
            return None
        r = self.last_reading
        return {
            "p_ad": r.p_ad,
            "abstain": r.abstain,
            "clips": r.clips,
            "span_seconds": r.span_seconds,
        }


sensor = AudioSensor()


def reset() -> None:
    global sensor
    sensor = AudioSensor()
