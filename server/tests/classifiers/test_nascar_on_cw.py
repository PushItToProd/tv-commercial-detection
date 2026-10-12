"""Tests for classifiers.nascar_on_cw — synthetic images, no real broadcast frames.

The bug is drawn from the profile's own silhouette template, tinted and laid
over a flat background; the side-by-side break is a bright rectangle at the
race panel's position on a dark frame.
"""

import cv2
import numpy as np
import pytest

from tv_commercial_detector.classification.result import ClassificationResult
from tv_commercial_detector.classifiers import nascar_on_cw
from tv_commercial_detector.classifiers.nascar_on_cw import (
    BUG_REGION,
    BUG_TEMPLATE,
    SIDE_BY_SIDE_RACE_PANEL,
    bug_score,
    has_bug,
    is_side_by_side,
    side_by_side_score,
)

# Where the template's top-left corner sits in a 1920x1080 frame.
BUG_ABS_X = 1711
BUG_ABS_Y = 43

GREY_BUG = (190, 190, 190)
# BGR. Luminance matches LIGHT_ASPHALT, which is what defeats a grayscale match.
YELLOW_BUG = (0, 215, 255)
LIGHT_ASPHALT = 196


def blank_bgr(fill: int = 128) -> np.ndarray:
    return np.full((1080, 1920, 3), fill, dtype=np.uint8)


def frame_with_bug(
    color: tuple[int, int, int], fill: int = 128, abs_x: int = BUG_ABS_X, abs_y: int = BUG_ABS_Y
) -> np.ndarray:
    frame = blank_bgr(fill)
    h, w = BUG_TEMPLATE.shape
    frame[abs_y : abs_y + h, abs_x : abs_x + w][BUG_TEMPLATE > 0] = color
    return frame


def side_by_side_frame(shift: int = 0, background: int = 28) -> np.ndarray:
    frame = blank_bgr(background)
    x0, x1, y0, y1 = SIDE_BY_SIDE_RACE_PANEL
    frame[y0:y1, x0 + shift : x1 + shift] = 150
    return frame


# --- bug ---------------------------------------------------------------------


@pytest.mark.parametrize("fill", [0, 128, 255], ids=["black", "grey", "white"])
def test_no_bug_in_blank_frame(fill):
    """A flat corner has no variance to correlate against and must score 0."""
    assert bug_score(blank_bgr(fill)) == 0.0


def test_grey_bug_detected():
    assert bug_score(frame_with_bug(GREY_BUG)) > 0.95


def test_grey_bug_detected_over_dark_background():
    assert has_bug(frame_with_bug(GREY_BUG, fill=28)) is True


def test_yellow_bug_detected_at_equal_luminance():
    """The caution-period bug over sunlit asphalt: invisible in grayscale."""
    frame = frame_with_bug(YELLOW_BUG, fill=LIGHT_ASPHALT)
    x0, x1, y0, y1 = BUG_REGION
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    assert nascar_on_cw._shape_score(gray, BUG_TEMPLATE) < nascar_on_cw.BUG_THRESHOLD
    assert has_bug(frame) is True


def test_bug_outside_search_window_is_ignored():
    """Promos show the CW logo elsewhere; only the corner bug counts."""
    assert has_bug(frame_with_bug(GREY_BUG, abs_x=400, abs_y=600)) is False


def test_search_window_contains_the_template():
    x0, x1, y0, y1 = BUG_REGION
    h, w = BUG_TEMPLATE.shape
    assert x0 <= BUG_ABS_X and BUG_ABS_X + w <= x1
    assert y0 <= BUG_ABS_Y and BUG_ABS_Y + h <= y1


def test_bug_check_does_not_mutate_caller_frame():
    frame = frame_with_bug(GREY_BUG)
    before = frame.copy()
    has_bug(frame)
    assert np.array_equal(frame, before)


# --- side-by-side ------------------------------------------------------------


@pytest.mark.parametrize("fill", [0, 128, 255], ids=["black", "grey", "white"])
def test_no_side_by_side_in_blank_frame(fill):
    assert side_by_side_score(blank_bgr(fill)) == 0.0


def test_side_by_side_panel_detected():
    assert is_side_by_side(side_by_side_frame()) is True


def test_side_by_side_tolerates_the_plate_variant_offset():
    """The panel sits 3 px further left under the "CW NASCAR" plate."""
    assert is_side_by_side(side_by_side_frame(shift=-3)) is True


def test_full_width_bands_are_not_side_by_side():
    """Edges along the panel's top and bottom only, as a wall or ticker gives."""
    frame = blank_bgr(28)
    _, _, y0, y1 = SIDE_BY_SIDE_RACE_PANEL
    frame[y0:y1, :] = 150
    assert is_side_by_side(frame) is False


# --- classify_image: ordering ------------------------------------------------


@pytest.fixture
def classify(tmp_path, mocker):
    """Classify a frame through the profile with the LLM stubbed out."""
    llm = mocker.patch.object(
        nascar_on_cw.llm_match,
        "classify_by_prompt",
        return_value=ClassificationResult(source="llm", type="ad", reason="llm", reply="ad"),
    )
    mocker.patch.object(nascar_on_cw.llm_match, "load_image_b64", return_value="")

    def run(frame: np.ndarray) -> ClassificationResult:
        path = tmp_path / "frame.png"
        cv2.imwrite(str(path), frame)
        return nascar_on_cw.classify_image(str(path), None)

    return run, llm


def test_bug_alone_is_content(classify):
    run, llm = classify
    result = run(frame_with_bug(GREY_BUG))
    assert (result.type, result.reason) == ("content", "network_logo")
    llm.assert_not_called()


def test_side_by_side_outranks_the_bug(classify):
    """The bug stays up through the break, so the panel has to win."""
    run, llm = classify
    frame = side_by_side_frame()
    h, w = BUG_TEMPLATE.shape
    frame[BUG_ABS_Y : BUG_ABS_Y + h, BUG_ABS_X : BUG_ABS_X + w][BUG_TEMPLATE > 0] = GREY_BUG
    assert has_bug(frame) is True
    result = run(frame)
    assert (result.type, result.reason) == ("ad", "side_by_side")
    llm.assert_not_called()


def test_undecided_frame_goes_to_the_llm_with_the_cw_prompt(classify):
    run, llm = classify
    result = run(blank_bgr())
    assert result.source == "llm"
    assert llm.call_args.kwargs["prompt"] == nascar_on_cw.PROMPT


def test_disabled_checks_fall_through(classify, mocker):
    run, llm = classify
    mocker.patch.object(nascar_on_cw, "ENABLE_BUG_CHECK", False)
    assert run(frame_with_bug(GREY_BUG)).source == "llm"
