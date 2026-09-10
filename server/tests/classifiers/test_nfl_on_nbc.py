"""Tests for classifiers.nfl_on_nbc — synthetic images, no real broadcast frames.

Same strategy as test_nascar_on_nbc: paste the real peacock template into an
otherwise blank frame at the coordinates this graphics package uses, and check
that a blank frame stays quiet. See that module's docstring for why grey
(rather than black) is the honest stand-in for a blank frame here.
"""

import cv2
import numpy as np
import pytest

from tv_commercial_detector.classification.logo_match import LOGOS_DIR, mask_non_white
from tv_commercial_detector.classifiers.nfl_on_nbc import (
    PEACOCK_REGION,
    PEACOCK_TEMPLATE,
    has_peacock_logo,
    peacock_score,
)

# Where the peacock actually sits in a 1920x1080 frame on this graphics
# package — noticeably higher and further right than on the NASCAR feed.
PEACOCK_ABS_X = 1814
PEACOCK_ABS_Y = 27


def blank_bgr(width: int = 1920, height: int = 1080, fill: int = 128) -> np.ndarray:
    return np.full((height, width, 3), fill, dtype=np.uint8)


def frame_with_logo_at(logo: np.ndarray, abs_x: int, abs_y: int, fill: int = 128) -> np.ndarray:
    frame = blank_bgr(fill=fill)
    lh, lw = logo.shape[:2]
    frame[abs_y : abs_y + lh, abs_x : abs_x + lw] = logo
    return frame


# --- negative ---------------------------------------------------------------


@pytest.mark.parametrize("fill", [0, 128, 255], ids=["black", "grey", "white"])
def test_no_peacock_in_blank_frame(fill):
    assert has_peacock_logo(blank_bgr(fill=fill)) is False


# --- positive ---------------------------------------------------------------


def test_peacock_detected_at_expected_position():
    frame = frame_with_logo_at(PEACOCK_TEMPLATE, PEACOCK_ABS_X, PEACOCK_ABS_Y)
    assert has_peacock_logo(frame) is True


def test_peacock_scores_near_perfect_on_exact_paste():
    frame = frame_with_logo_at(PEACOCK_TEMPLATE, PEACOCK_ABS_X, PEACOCK_ABS_Y)
    assert peacock_score(frame) > 0.95


# --- regressions on the things that make this profile different from NASCAR -


def test_peacock_template_survives_colour():
    """The peacock must not be white-masked; masking would zero it out."""
    assert PEACOCK_TEMPLATE.any()
    assert not mask_non_white(PEACOCK_TEMPLATE.copy()).any()


def test_peacock_outside_search_window_is_ignored():
    """A peacock elsewhere on screen must not count as the network bug.

    Ads and promos can show the logo anywhere; only the corner bug means the
    broadcast is live.
    """
    frame = frame_with_logo_at(PEACOCK_TEMPLATE, 400, 600)
    assert has_peacock_logo(frame) is False


def test_search_window_contains_the_template_bbox():
    x0, x1, y0, y1 = PEACOCK_REGION
    th, tw = PEACOCK_TEMPLATE.shape[:2]
    assert x0 <= PEACOCK_ABS_X and PEACOCK_ABS_X + tw <= x1
    assert y0 <= PEACOCK_ABS_Y and PEACOCK_ABS_Y + th <= y1


def test_search_window_differs_from_nascar_on_nbc():
    """Guards the whole point of this profile existing as a separate module.

    If this window ever collapses back to nascar_on_nbc's, reusing that
    profile directly would have worked and this one wouldn't be needed — the
    NFL bug sits high and right enough that NASCAR's window clips it out.
    """
    from tv_commercial_detector.classifiers.nascar_on_nbc import (
        PEACOCK_REGION as NASCAR_PEACOCK_REGION,
    )

    assert PEACOCK_REGION != NASCAR_PEACOCK_REGION


def test_peacock_template_matches_its_own_file():
    on_disk = cv2.imread(str(LOGOS_DIR / "nbc_peacock_logo.png"))
    assert np.array_equal(PEACOCK_TEMPLATE, on_disk)


def test_peacock_check_does_not_mutate_caller_frame():
    frame = frame_with_logo_at(PEACOCK_TEMPLATE, PEACOCK_ABS_X, PEACOCK_ABS_Y)
    before = frame.copy()
    has_peacock_logo(frame)
    assert np.array_equal(frame, before)
