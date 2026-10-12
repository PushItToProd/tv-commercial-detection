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
    ANNIVERSARY_BUG_TEMPLATE,
    BUG_REGION,
    BUG_TEMPLATE,
    SIDE_BY_SIDE_RACE_PANEL,
    TICKER_COLOR,
    TICKER_X,
    bug_score,
    has_bug,
    has_ticker,
    is_side_by_side,
    side_by_side_score,
    ticker_score,
)

# Where the template's top-left corner sits in a 1920x1080 frame.
BUG_ABS_X = 1711
BUG_ABS_Y = 43

GREY_BUG = (190, 190, 190)
# BGR. Luminance matches LIGHT_ASPHALT, which is what defeats a grayscale match.
YELLOW_BUG = (0, 215, 255)
LIGHT_ASPHALT = 196
RED_BUG = (0, 0, 255)


def blank_bgr(fill: int = 128) -> np.ndarray:
    return np.full((1080, 1920, 3), fill, dtype=np.uint8)


def frame_with_bug(
    color: tuple[int, int, int],
    fill: int = 128,
    abs_x: int = BUG_ABS_X,
    abs_y: int = BUG_ABS_Y,
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


def anniversary_frame(color: tuple[int, int, int], fill: int = 128) -> np.ndarray:
    frame = blank_bgr(fill)
    h, w = ANNIVERSARY_BUG_TEMPLATE.shape
    frame[43 : 43 + h, 1633 : 1633 + w][ANNIVERSARY_BUG_TEMPLATE > 0] = color
    return frame


@pytest.mark.parametrize("color", [GREY_BUG, YELLOW_BUG, RED_BUG])
def test_anniversary_bug_detected(color):
    assert bug_score(anniversary_frame(color, fill=40)) > 0.95


def test_red_bug_detected_at_equal_luminance():
    # BGR (0, 0, 255) has luminance 76; a grey background at 76 hides it from
    # the grayscale and yellowness passes.
    frame = frame_with_bug(RED_BUG, fill=76)
    x0, x1, y0, y1 = BUG_REGION
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    assert nascar_on_cw._shape_score(gray, BUG_TEMPLATE) < nascar_on_cw.BUG_THRESHOLD
    assert has_bug(frame) is True


def test_bugs_do_not_match_each_other():
    sports = frame_with_bug(GREY_BUG, fill=40)
    anniversary = anniversary_frame(GREY_BUG, fill=40)
    assert (
        nascar_on_cw._bug_score(sports, *nascar_on_cw.BUGS[1])
        < nascar_on_cw.BUG_THRESHOLD
    )
    assert (
        nascar_on_cw._bug_score(anniversary, *nascar_on_cw.BUGS[0])
        < nascar_on_cw.BUG_THRESHOLD
    )


def ticker_frame(covered_from_x: int | None = None) -> np.ndarray:
    """Dark frame with the bottom two ticker rows: slate cells, black separators."""
    frame = blank_bgr(20)
    x0, x1 = TICKER_X
    frame[992:1024, x0:x1] = TICKER_COLOR
    frame[1027:1057, x0:x1] = TICKER_COLOR
    # Green name cells take about a third of each row.
    frame[992:1057, x0 : x0 + 500] = (40, 160, 40)
    frame[1024:1027] = 20
    if covered_from_x is not None:
        frame[900:1080, covered_from_x:] = 230
    return frame


def side_by_side_ticker_frame() -> np.ndarray:
    """Dark frame with the two-row ticker of a side-by-side break and no panel border.

    The rows sit 2 px lower than the live ticker's bottom two rows.
    """
    frame = blank_bgr(20)
    x0, x1 = TICKER_X
    frame[994:1026, x0:x1] = TICKER_COLOR
    frame[1029:1058, x0:x1] = TICKER_COLOR
    frame[994:1058, x0 : x0 + 500] = (40, 160, 40)
    frame[1026:1029] = 20
    return frame


def test_side_by_side_ticker_detected_without_a_panel_border():
    frame = side_by_side_ticker_frame()
    assert side_by_side_score(frame) < nascar_on_cw.SIDE_BY_SIDE_THRESHOLD
    assert nascar_on_cw.side_by_side_ticker_score(frame) > 0.6
    assert is_side_by_side(frame) is True


def test_the_two_tickers_are_not_mistaken_for_each_other():
    assert nascar_on_cw.side_by_side_ticker_score(ticker_frame()) <= 0.0
    assert ticker_score(side_by_side_ticker_frame()) <= 0.0


def test_no_side_by_side_ticker_in_blank_frame():
    assert nascar_on_cw.side_by_side_ticker_score(blank_bgr(20)) == 0.0


def test_no_ticker_in_blank_frame():
    assert ticker_score(blank_bgr(20)) == 0.0


def test_ticker_detected():
    assert ticker_score(ticker_frame()) > 0.6
    assert has_ticker(ticker_frame()) is True


def test_slate_colored_area_without_rows_is_not_a_ticker():
    frame = blank_bgr(20)
    frame[900:1080] = TICKER_COLOR
    assert ticker_score(frame) == 0.0
    assert has_ticker(frame) is False


def test_mostly_covered_ticker_is_not_claimed():
    assert has_ticker(ticker_frame(covered_from_x=600)) is False


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
        return_value=ClassificationResult(
            source="llm", type="ad", reason="llm", reply="ad"
        ),
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


def test_side_by_side_ticker_outranks_the_bug(classify):
    # A night race on the dark side-by-side background shows no panel border,
    # and the bug is on screen.
    run, _ = classify
    frame = side_by_side_ticker_frame()
    h, w = BUG_TEMPLATE.shape
    frame[BUG_ABS_Y : BUG_ABS_Y + h, BUG_ABS_X : BUG_ABS_X + w][BUG_TEMPLATE > 0] = (
        GREY_BUG
    )
    assert has_bug(frame) is True
    result = run(frame)
    assert result.type == "ad"
    assert result.reason == "side_by_side"


def test_ticker_alone_is_content(classify):
    run, llm = classify
    result = run(ticker_frame())
    assert result.type == "content"
    assert result.reason == "scoring_ticker"
    llm.assert_not_called()


def test_disabled_ticker_check_falls_through(classify, mocker):
    run, _ = classify
    mocker.patch.object(nascar_on_cw, "ENABLE_TICKER_CHECK", False)
    assert run(ticker_frame()).source == "llm"


def test_side_by_side_outranks_the_bug(classify):
    """The bug stays up through the break, so the panel has to win."""
    run, llm = classify
    frame = side_by_side_frame()
    h, w = BUG_TEMPLATE.shape
    frame[BUG_ABS_Y : BUG_ABS_Y + h, BUG_ABS_X : BUG_ABS_X + w][BUG_TEMPLATE > 0] = (
        GREY_BUG
    )
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
