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
    SIDE_BY_SIDE_REGION,
    SIDE_BY_SIDE_TEMPLATE,
    USA_SPORTS_REGION,
    USA_SPORTS_TEMPLATE,
    USA_TEMPLATE,
    has_network_logo,
    has_peacock_logo,
    has_side_by_side_logo,
    has_usa_logo,
    has_usa_sports_logo,
    peacock_score,
    side_by_side_score,
    usa_score,
    usa_sports_score,
)

# Where the peacock actually sits in a 1920x1080 frame.
PEACOCK_ABS_X = 1771
PEACOCK_ABS_Y = 66

# Where the NASCAR NON STOP banner actually sits in a 1920x1080 frame.
SIDE_BY_SIDE_ABS_X = 55
SIDE_BY_SIDE_ABS_Y = 33


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


def test_side_by_side_banner_detected_at_expected_position():
    frame = frame_with_logo_at(
        cv2.cvtColor(SIDE_BY_SIDE_TEMPLATE, cv2.COLOR_GRAY2BGR),
        SIDE_BY_SIDE_ABS_X,
        SIDE_BY_SIDE_ABS_Y,
        fill=0,
    )
    assert has_side_by_side_logo(frame) is True
    assert side_by_side_score(frame) > 0.95


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


def test_side_by_side_banner_must_not_be_white_masked():
    """The banner is matched unmasked, and white-masking would gut it.

    Guards against someone "consistently" switching this template over to
    load_masked alongside the USA bug. The mask keeps ~1% of the banner, and
    nothing at all once the glyph edges are soft — which is exactly what the
    Prime Video feed's anti-aliased rendering looks like, and why the masked
    version scored 0.16-0.22 there against a 0.80 threshold.
    """
    from tv_commercial_detector.classification.logo_match import mask_non_white

    colour = cv2.cvtColor(SIDE_BY_SIDE_TEMPLATE, cv2.COLOR_GRAY2BGR)
    assert colour.any(axis=2).mean() > 0.9
    assert mask_non_white(colour.copy()).any(axis=2).mean() < 0.05

    softened = cv2.GaussianBlur(colour, (5, 5), 0)
    assert not mask_non_white(softened).any()


def test_side_by_side_banner_outside_search_window_is_ignored():
    """The banner only means an ad break in the upper left.

    Commercials shown during a break can carry NASCAR branding anywhere on
    screen; only the banner in its own corner marks the side-by-side layout.
    """
    frame = frame_with_logo_at(
        cv2.cvtColor(SIDE_BY_SIDE_TEMPLATE, cv2.COLOR_GRAY2BGR), 900, 600, fill=0
    )
    assert has_side_by_side_logo(frame) is False


def test_side_by_side_search_window_contains_the_template_bbox():
    x0, x1, y0, y1 = SIDE_BY_SIDE_REGION
    th, tw = SIDE_BY_SIDE_TEMPLATE.shape[:2]
    assert x0 <= SIDE_BY_SIDE_ABS_X and SIDE_BY_SIDE_ABS_X + tw <= x1
    assert y0 <= SIDE_BY_SIDE_ABS_Y and SIDE_BY_SIDE_ABS_Y + th <= y1


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


# --- "usa SPORTS" lockup ---------------------------------------------------

# Where the lockup sits on the September feed. It sat 9 px left and 5 px up in
# August, which is why the window carries margin.
USA_SPORTS_ABS_X = 1789
USA_SPORTS_ABS_Y = 57
USA_SPORTS_AUGUST_ABS_X = 1780
USA_SPORTS_AUGUST_ABS_Y = 52


def test_usa_sports_detected_at_expected_position():
    frame = frame_with_logo_at(USA_SPORTS_TEMPLATE, USA_SPORTS_ABS_X, USA_SPORTS_ABS_Y, fill=0)
    assert has_usa_sports_logo(frame) is True
    assert usa_sports_score(frame) > 0.95


@pytest.mark.parametrize("fill", [0, 128], ids=["black", "grey"])
def test_no_usa_sports_in_blank_frame(fill):
    assert has_usa_sports_logo(blank_bgr(fill=fill)) is False


def test_usa_sports_score_is_zero_on_blown_out_frame():
    """Same degenerate-match trap as the plain wordmark; see that test."""
    assert usa_sports_score(blank_bgr(fill=255)) == 0.0
    assert has_usa_sports_logo(blank_bgr(fill=255)) is False


def test_usa_sports_template_is_white_masked():
    """The lockup is translucent white, so it is masked like the wordmark.

    Guards against someone matching it in color alongside the peacock: that
    does detect it, but pins the match to whatever backdrop the template was
    cropped over, and measured 513 detections against the mask's 570.
    """
    assert USA_SPORTS_TEMPLATE.any()
    assert np.array_equal(USA_SPORTS_TEMPLATE, load_masked(LOGOS_DIR / "usa_sports_logo.png"))


def test_usa_sports_outside_search_window_is_ignored():
    frame = frame_with_logo_at(USA_SPORTS_TEMPLATE, 400, 600, fill=0)
    assert has_usa_sports_logo(frame) is False


@pytest.mark.parametrize(
    ("abs_x", "abs_y"),
    [
        (USA_SPORTS_ABS_X, USA_SPORTS_ABS_Y),
        (USA_SPORTS_AUGUST_ABS_X, USA_SPORTS_AUGUST_ABS_Y),
    ],
    ids=["september", "august"],
)
def test_usa_sports_search_window_contains_both_observed_positions(abs_x, abs_y):
    x0, x1, y0, y1 = USA_SPORTS_REGION
    th, tw = USA_SPORTS_TEMPLATE.shape[:2]
    assert x0 <= abs_x and abs_x + tw <= x1
    assert y0 <= abs_y and abs_y + th <= y1
    frame = frame_with_logo_at(USA_SPORTS_TEMPLATE, abs_x, abs_y, fill=0)
    assert has_usa_sports_logo(frame) is True


def test_usa_sports_bug_satisfies_has_network_logo():
    frame = frame_with_logo_at(USA_SPORTS_TEMPLATE, USA_SPORTS_ABS_X, USA_SPORTS_ABS_Y, fill=0)
    assert has_network_logo(frame) is True


def test_usa_sports_check_does_not_mutate_caller_frame():
    """The side-by-side and peacock checks share the frame with this one."""
    frame = frame_with_logo_at(USA_SPORTS_TEMPLATE, USA_SPORTS_ABS_X, USA_SPORTS_ABS_Y, fill=0)
    before = frame.copy()
    has_usa_sports_logo(frame)
    assert np.array_equal(frame, before)


# --- classify_image: where the audio sensor sits in the pipeline --------------
#
# The OpenCV checks and the LLM are patched out: these tests are about ordering
# and the confidence band, not about any one detector.


@pytest.fixture
def pipeline(tmp_path, mocker):
    """A blank frame on disk, with every detector patched to a controllable stub."""
    from tv_commercial_detector import audio_sensor
    from tv_commercial_detector.classification.result import ClassificationResult
    from tv_commercial_detector.classifiers import nascar_on_nbc

    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), blank_bgr())

    stubs = {
        "banner": mocker.patch.object(nascar_on_nbc, "has_side_by_side_logo", return_value=False),
        "bug": mocker.patch.object(nascar_on_nbc, "has_network_logo", return_value=False),
        "quick": mocker.patch.object(
            nascar_on_nbc.llm_match, "_report_racing_related", return_value=True
        ),
        "prompt": mocker.patch.object(
            nascar_on_nbc.llm_match,
            "classify_by_prompt",
            return_value=ClassificationResult(
                source="llm", type="content", reason="llm", reply="racing"
            ),
        ),
        "reading": mocker.patch.object(audio_sensor.sensor, "reading"),
    }
    mocker.patch.object(nascar_on_nbc.llm_match, "load_image_b64", return_value="")
    assert nascar_on_nbc.AUDIO_MODEL is not None

    def set_audio(p_ad, abstain=None):
        stubs["reading"].return_value = audio_sensor.AudioReading(
            p_ad=p_ad, clips=15, span_seconds=28.0, abstain=abstain
        )

    set_audio(None, "cold")

    def classify():
        return nascar_on_nbc.classify_image(str(path), b"wav")

    return nascar_on_nbc, stubs, set_audio, classify


def test_confident_audio_ad_skips_the_llm(pipeline):
    profile, stubs, set_audio, classify = pipeline
    set_audio(profile.AUDIO_MODEL.ad_threshold)
    result = classify()
    assert (result.source, result.type) == ("audio", "ad")
    assert result.signals == {"p_audio": profile.AUDIO_MODEL.ad_threshold, "audio_abstain": None}
    stubs["quick"].assert_not_called()
    stubs["prompt"].assert_not_called()


def test_confident_audio_content_skips_the_llm(pipeline):
    profile, stubs, set_audio, classify = pipeline
    set_audio(profile.AUDIO_MODEL.content_threshold)
    result = classify()
    assert (result.source, result.type) == ("audio", "content")
    stubs["quick"].assert_not_called()


def test_audio_between_the_thresholds_goes_to_the_llm(pipeline):
    profile, stubs, set_audio, classify = pipeline
    model = profile.AUDIO_MODEL
    p = (model.ad_threshold + model.content_threshold) / 2
    set_audio(p)
    result = classify()
    assert result.source == "llm"
    # The score still rides along with the LLM's verdict.
    assert result.signals == {"p_audio": p, "audio_abstain": None}
    stubs["prompt"].assert_called_once()


def test_abstaining_audio_goes_to_the_llm(pipeline):
    _, stubs, _, classify = pipeline
    result = classify()
    assert result.source == "llm"
    assert result.signals == {"p_audio": None, "audio_abstain": "cold"}


def test_banner_outranks_confident_audio_content(pipeline):
    profile, stubs, set_audio, classify = pipeline
    stubs["banner"].return_value = True
    set_audio(0.0)
    result = classify()
    assert (result.source, result.type, result.reason) == ("opencv", "ad", "side_by_side")
    assert result.signals["p_audio"] == 0.0


def test_bug_outranks_confident_audio_ad(pipeline):
    # Sponsor reads over live pictures at a rejoin: vision is the authority.
    profile, stubs, set_audio, classify = pipeline
    stubs["bug"].return_value = True
    set_audio(1.0)
    result = classify()
    assert (result.source, result.type) == ("opencv", "content")


def test_audio_check_can_be_disabled(pipeline, mocker):
    profile, stubs, set_audio, classify = pipeline
    mocker.patch.object(profile, "ENABLE_AUDIO_CHECK", False)
    set_audio(1.0)
    result = classify()
    assert result.source == "llm"
    assert stubs["reading"].call_args.args[0] is None
