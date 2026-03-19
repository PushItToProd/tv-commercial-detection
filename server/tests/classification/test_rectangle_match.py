"""Tests for rectangle_match — pure logic and synthetic-image detection."""

import cv2
import numpy as np

from tv_commercial_detector.classification.rectangle_match import (
    EPS,
    KNOWN_RECTANGLES,
    detect_rectangles,
    find_matching_rectangle,
    find_matching_rectangles,
    image_has_known_ad_rectangle,
)

# ---------------------------------------------------------------------------
# find_matching_rectangle — pure maths, no images
# ---------------------------------------------------------------------------


def test_exact_match():
    """A rectangle that exactly matches a known normalised box is returned."""
    box = (0.0, 0.239, 0.387, 0.381)  # fox-side-by-side-left
    width, height = 1920, 1080
    rx, ry, rw, rh = box
    rect = (int(rx * width), int(ry * height), int(rw * width), int(rh * height))
    matches = find_matching_rectangle(box, [rect], width, height)
    assert len(matches) == 1
    assert matches[0][1] == rect


def test_no_match_far_rectangle():
    """A rectangle far from the target returns no matches."""
    box = (0.0, 0.239, 0.387, 0.381)
    # Place the test rect in the opposite corner
    rect = (1800, 900, 100, 100)
    matches = find_matching_rectangle(box, [rect], 1920, 1080)
    assert matches == []


def test_match_within_eps():
    """A rectangle slightly off but within EPS is accepted."""
    box = (0.1, 0.1, 0.4, 0.4)
    width, height = 1000, 1000
    # Offset each coordinate by (EPS / 4) in normalised space → dist < EPS
    offset = EPS / 4
    rect = (
        int((0.1 + offset) * width),
        int((0.1 + offset) * height),
        int((0.4 - offset) * width),
        int((0.4 - offset) * height),
    )
    matches = find_matching_rectangle(box, [rect], width, height)
    assert len(matches) == 1


def test_no_match_just_outside_eps():
    """A rectangle just beyond EPS in L2 distance is rejected."""
    box = (0.1, 0.1, 0.4, 0.4)
    width, height = 1000, 1000
    # Offset each coordinate by (EPS / 2) → L2 distance = sqrt(4*(eps/2)^2) = eps > EPS
    offset = EPS / 2 + 0.01
    rect = (
        int((0.1 + offset) * width),
        int((0.1 + offset) * height),
        int((0.4 + offset) * width),
        int((0.4 + offset) * height),
    )
    matches = find_matching_rectangle(box, [rect], width, height)
    assert matches == []


def test_multiple_rects_only_close_one_matches():
    box = (0.0, 0.239, 0.387, 0.381)
    width, height = 1920, 1080
    rx, ry, rw, rh = box
    close_rect = (
        int(rx * width),
        int(ry * height),
        int(rw * width),
        int(rh * height),
    )
    far_rect = (1800, 900, 50, 50)
    matches = find_matching_rectangle(box, [close_rect, far_rect], width, height)
    assert len(matches) == 1
    assert matches[0][1] == close_rect


# ---------------------------------------------------------------------------
# find_matching_rectangles — dict variant
# ---------------------------------------------------------------------------


def test_find_matching_rectangles_returns_name_keys():
    width, height = 1920, 1080
    box = KNOWN_RECTANGLES["fox-side-by-side-left"]
    rx, ry, rw, rh = box
    rect = (int(rx * width), int(ry * height), int(rw * width), int(rh * height))

    result = find_matching_rectangles(KNOWN_RECTANGLES, [rect], width, height)
    assert "fox-side-by-side-left" in result
    # Other known rectangles should NOT match this single rect
    for name in ("fox-side-by-side-right", "fox-side-by-side-scoreboard"):
        assert name not in result


def test_find_matching_rectangles_empty_list():
    result = find_matching_rectangles(KNOWN_RECTANGLES, [], 1920, 1080)
    assert result == {}


# ---------------------------------------------------------------------------
# detect_rectangles — synthetic image with clear rectangle
# ---------------------------------------------------------------------------


def _make_rect_image(
    width: int = 1920,
    height: int = 1080,
    rect_x: int = 0,
    rect_y: int = 258,
    rect_w: int = 742,
    rect_h: int = 412,
) -> np.ndarray:
    """Create a black image with a white-bordered (hollow) rectangle.

    A hollow border is used rather than a filled rect because detect_rectangles
    uses edge detection + RETR_EXTERNAL contours; a filled region on a plain
    background produces lines at the border — but the contourArea of those
    open edge contours is 0, which fails the > 5000 filter.  A thick hollow
    border gives closed inner/outer contours whose area is well above 5000.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(
        img,
        (rect_x, rect_y),
        (rect_x + rect_w, rect_y + rect_h),
        (255, 255, 255),
        thickness=20,
    )
    return img


def test_detect_rectangles_finds_bordered_rect():
    """detect_rectangles should find at least the large white-bordered rectangle."""
    img = _make_rect_image()
    rects = detect_rectangles(img)
    assert len(rects) >= 1


def test_detect_rectangles_blank_image():
    """A featureless grey image has no detectable rectangles."""
    img = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    rects = detect_rectangles(img)
    assert rects == []


# ---------------------------------------------------------------------------
# image_has_known_ad_rectangle — end-to-end with a synthetic frame
# ---------------------------------------------------------------------------


def _make_fox_sbs_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    """Draw rects approximating the fox-side-by-side-left box on a dark frame."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    box = KNOWN_RECTANGLES["fox-side-by-side-left"]
    rx, ry, rw, rh = box
    x = int(rx * width)
    y = int(ry * height)
    w = int(rw * width)
    h = int(rh * height)
    # Draw a thick white border so Canny + contour detection picks it up
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=10)
    return img


def test_image_has_known_ad_rectangle_detects_fox_sbs():
    img = _make_fox_sbs_frame()
    match = image_has_known_ad_rectangle(img)
    assert match == "fox-side-by-side-left"


def test_image_has_known_ad_rectangle_blank():
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert image_has_known_ad_rectangle(img) is None
