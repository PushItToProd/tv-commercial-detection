"""Tests for classifiers.nascar_on_nbc — synthetic images, no real broadcast frames.

Same strategy as test_nascar_on_fox: paste the real template into an otherwise
blank frame at the coordinates the detector expects, and check that a blank
frame stays quiet.

The peacock differs from the Fox-style logos in that it is matched in colour,
so these tests load it with a plain imread rather than load_masked. A blank
frame here is mid-grey rather than black: black would make the "does masking
leak" question untestable, and grey is a more honest stand-in for the sky and
asphalt the bug actually sits over.
"""

import cv2
import numpy as np
import pytest

from tv_commercial_detector.classification.logo_match import LOGOS_DIR, load_masked
from tv_commercial_detector.classifiers.nascar_on_nbc import (
    PEACOCK_REGION,
    PEACOCK_TEMPLATE,
    USA_TEMPLATE,
    has_network_logo,
    has_peacock_logo,
    has_side_by_side_logo,
    has_usa_logo,
    peacock_score,
    usa_score,
)

# Where the peacock actually sits in a 1920x1080 frame.
PEACOCK_ABS_X = 1771
PEACOCK_ABS_Y = 66

SIDE_BY_SIDE_ABS_X = 0
SIDE_BY_SIDE_ABS_Y = 0


def blank_bgr(width: int = 1920, height: int = 1080, fill: int = 128) -> np.ndarray:
    return np.full((height, width, 3), fill, dtype=np.uint8)


def frame_with_logo_at(logo: np.ndarray, abs_x: int, abs_y: int, fill: int = 128) -> np.ndarray:
    frame = blank_bgr(fill=fill)
    lh, lw = logo.shape[:2]
    frame[abs_y : abs_y + lh, abs_x : abs_x + lw] = logo
    return frame


# --- negative -------------------------------------------------------------


@pytest.mark.parametrize("fill", [0, 128, 255], ids=["black", "grey", "white"])
def test_no_peacock_in_blank_frame(fill):
    assert has_peacock_logo(blank_bgr(fill=fill)) is False


def test_no_side_by_side_in_blank_frame():
    assert has_side_by_side_logo(blank_bgr()) is False


# --- positive -------------------------------------------------------------


def test_peacock_detected_at_expected_position():
    frame = frame_with_logo_at(PEACOCK_TEMPLATE, PEACOCK_ABS_X, PEACOCK_ABS_Y)
    assert has_peacock_logo(frame) is True


def test_peacock_scores_near_perfect_on_exact_paste():
    frame = frame_with_logo_at(PEACOCK_TEMPLATE, PEACOCK_ABS_X, PEACOCK_ABS_Y)
    assert peacock_score(frame) > 0.95


@pytest.mark.parametrize(
    "logo_path",
    [
        LOGOS_DIR / "nbc_nascar_non_stop_side_by_side_logo.png",
        LOGOS_DIR / "nbc_nascar_non_stop_full_logo.png",
    ],
    ids=["non_stop", "non_stop_full"],
)
def test_side_by_side_logo_detected(logo_path):
    masked = load_masked(str(logo_path))
    frame = frame_with_logo_at(masked, SIDE_BY_SIDE_ABS_X, SIDE_BY_SIDE_ABS_Y, fill=0)
    assert has_side_by_side_logo(frame, {"logo": masked}) is True


# --- regressions on the two things that make this profile different --------


def test_peacock_template_survives_colour():
    """The peacock must not be white-masked; masking would zero it out.

    Guards against someone "consistently" switching this template over to
    load_masked, which silently reduces it to an all-black patch.
    """
    from tv_commercial_detector.classification.logo_match import mask_non_white

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


def test_side_by_side_check_does_not_mutate_caller_frame():
    """The peacock check runs after this one and needs the untouched frame."""
    frame = frame_with_logo_at(PEACOCK_TEMPLATE, PEACOCK_ABS_X, PEACOCK_ABS_Y)
    before = frame.copy()
    has_side_by_side_logo(frame)
    assert np.array_equal(frame, before)


def test_peacock_template_matches_its_own_file():
    on_disk = cv2.imread(str(LOGOS_DIR / "nbc_peacock_logo.png"))
    assert np.array_equal(PEACOCK_TEMPLATE, on_disk)


# --- USA Network bug -------------------------------------------------------

USA_ABS_X = 1784
USA_ABS_Y = 76


def test_usa_detected_at_expected_position():
    frame = frame_with_logo_at(USA_TEMPLATE, USA_ABS_X, USA_ABS_Y, fill=0)
    assert has_usa_logo(frame) is True


@pytest.mark.parametrize("fill", [0, 128], ids=["black", "grey"])
def test_no_usa_in_blank_frame(fill):
    assert has_usa_logo(blank_bgr(fill=fill)) is False


def test_usa_score_is_zero_on_blown_out_frame():
    """A saturated upper right must score 0, not a degenerate 1.0.

    White-masking an all-white region keeps every pixel, leaving a uniform
    patch; TM_CCOEFF_NORMED divides by zero there and can report a perfect
    match. That would turn every blown-out sky into a false `content`.
    """
    assert usa_score(blank_bgr(fill=255)) == 0.0
    assert has_usa_logo(blank_bgr(fill=255)) is False


def test_usa_template_is_white_masked():
    """Unlike the peacock, the USA bug is white and does survive masking."""
    assert USA_TEMPLATE.any()
    assert np.array_equal(USA_TEMPLATE, load_masked(LOGOS_DIR / "usa_network_logo.png"))


def test_either_bug_satisfies_has_network_logo():
    peacock = frame_with_logo_at(PEACOCK_TEMPLATE, PEACOCK_ABS_X, PEACOCK_ABS_Y)
    usa = frame_with_logo_at(USA_TEMPLATE, USA_ABS_X, USA_ABS_Y, fill=0)
    assert has_network_logo(peacock) is True
    assert has_network_logo(usa) is True
    assert has_network_logo(blank_bgr(fill=60)) is False


def test_usa_outside_search_window_is_ignored():
    frame = frame_with_logo_at(USA_TEMPLATE, 400, 600, fill=0)
    assert has_usa_logo(frame) is False
