"""Tests for classifiers.nascar_on_fox — using synthetic images, no real broadcast frames.

Strategy:
- Negative test: a blank (all-black) image should never match any logo.
- Positive test: paste the actual masked logo template at the exact
  coordinates that the detector expects; the match score should exceed the
  detection threshold.
"""

import numpy as np
import pytest

from tv_commercial_detector.classification.logo_match import (
    LOGOS_DIR,
    load_masked,
)
from tv_commercial_detector.classifiers.nascar_on_fox import (
    has_network_logo,
    has_side_by_side_logo,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def blank_bgr(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Return a black BGR image at the given size."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def frame_with_logo_at(
    logo: np.ndarray,
    abs_x: int,
    abs_y: int,
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    """Place *logo* at (abs_x, abs_y) in an otherwise black 1920×1080 frame."""
    frame = blank_bgr(width, height)
    lh, lw = logo.shape[:2]
    # Clamp to frame bounds
    x2 = min(abs_x + lw, width)
    y2 = min(abs_y + lh, height)
    frame[abs_y:y2, abs_x:x2] = logo[: y2 - abs_y, : x2 - abs_x]
    return frame


# ---------------------------------------------------------------------------
# Negative tests — blank image must not trigger any logo detection
# ---------------------------------------------------------------------------


def test_no_network_logo_in_blank_frame():
    # Use only logos that fit inside the crop region (135h × 320w).
    small_logos = {
        "fox": load_masked(str(LOGOS_DIR / "fox_logo_crop.png")),
        "fs1": load_masked(str(LOGOS_DIR / "fs1_logo_crop.png")),
    }
    assert has_network_logo(blank_bgr(), small_logos) is False


def test_no_side_by_side_logo_in_blank_frame():
    assert has_side_by_side_logo(blank_bgr()) is False


# ---------------------------------------------------------------------------
# Positive tests — logo placed at the expected position must be detected.
#
# Network logo crop region (after resize to 1920×1080):
#   img[0 : 1080//8, 1920*5//6 : 1920]  →  img[0:135, 1600:1920]
# Detection expects: 110 ≤ tl_x ≤ 140, 15 ≤ tl_y ≤ 35  (in crop coords)
# Absolute position in full frame: (1600 + tl_x, tl_y)  →  ~(1720, 20)
#
# Side-by-side logo crop region:
#   img[0 : 1080//5, 0 : 1920//5]  →  img[0:216, 0:384]
# Detection expects: max_val >= 0.8
# We place the logo at (0, 0) in the crop (= (0, 0) in absolute coords).
# ---------------------------------------------------------------------------

NETWORK_LOGO_ABS_X = 1720  # 1600 + 120
NETWORK_LOGO_ABS_Y = 20
SIDE_BY_SIDE_LOGO_ABS_X = 0
SIDE_BY_SIDE_LOGO_ABS_Y = 0


@pytest.mark.parametrize(
    "logo_path",
    [
        LOGOS_DIR / "fox_logo_crop.png",
        LOGOS_DIR / "fs1_logo_crop.png",
        # cw_sports_logo.png is 154px tall, larger than the 135px crop region;
        # matchTemplate would raise an error, so it belongs in a broader
        # integration test once the detector crops are made configurable.
    ],
    ids=["fox", "fs1"],
)
def test_network_logo_detected(logo_path):
    masked = load_masked(str(logo_path))
    frame = frame_with_logo_at(masked, NETWORK_LOGO_ABS_X, NETWORK_LOGO_ABS_Y)
    assert has_network_logo(frame, {"logo": masked}) is True


@pytest.mark.parametrize(
    "logo_path",
    [
        LOGOS_DIR / "fox_side_by_side_logo_crop.png",
        LOGOS_DIR / "fs1_side_by_side_logo_crop.png",
        LOGOS_DIR / "nbc_nascar_non_stop_side_by_side_logo.png",
    ],
    ids=["fox_sbs", "fs1_sbs", "nbc_sbs"],
)
def test_side_by_side_logo_detected(logo_path):
    masked = load_masked(str(logo_path))
    frame = frame_with_logo_at(masked, SIDE_BY_SIDE_LOGO_ABS_X, SIDE_BY_SIDE_LOGO_ABS_Y)
    assert has_side_by_side_logo(frame, {"logo": masked}) is True
