"""Tests for logo_match — using synthetic images, no real broadcast frames."""

import numpy as np

from tv_commercial_detector.classification.logo_match import (
    mask_non_white,
)

# ---------------------------------------------------------------------------
# mask_non_white unit test
# ---------------------------------------------------------------------------


def test_mask_non_white_keeps_white():
    img = np.full((10, 10, 3), 255, dtype=np.uint8)
    result = mask_non_white(img)
    assert np.all(result == 255)


def test_mask_non_white_zeroes_grey():
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    result = mask_non_white(img)
    assert np.all(result == 0)


def test_mask_non_white_mixed():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = [255, 255, 255]  # white
    img[0, 1] = [200, 200, 200]  # borderline white (threshold is 200, exclusive)
    img[1, 0] = [199, 255, 255]  # one channel below threshold
    result = mask_non_white(img, min_thresh=200)
    assert np.all(result[0, 0] == 255)  # kept
    assert np.all(result[0, 1] == 200)  # kept (all channels >= 200)
    assert np.all(result[1, 0] == 0)    # zeroed (first channel = 199 < 200)


def test_mask_non_white_leaves_argument_unchanged():
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    original = img.copy()
    mask_non_white(img)
    assert np.array_equal(img, original)


def test_mask_non_white_does_not_reach_through_a_view():
    """A crop is a numpy view; masking it must not touch the parent frame."""
    img = np.full((10, 10, 3), 100, dtype=np.uint8)
    original = img.copy()
    masked = mask_non_white(img[0:5, 0:5])
    assert np.all(masked == 0)
    assert np.array_equal(img, original)
