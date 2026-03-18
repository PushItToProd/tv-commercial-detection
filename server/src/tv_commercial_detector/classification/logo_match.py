import time
from pathlib import Path
from typing import NamedTuple

import cv2

MATCH_METHOD = cv2.TM_CCOEFF_NORMED

LOGOS_DIR = Path(__file__).parent.parent / "prompt" / "logos"

# FIXME: load the logo(s) from config; also support loading just the logos for
# the current network

# We search for these in the upper right. Positioning is hardcoded for Fox.
NETWORK_LOGO_PATHS = [
    LOGOS_DIR / "fox_logo_crop.png",
    LOGOS_DIR / "fs1_logo_crop.png",
]

# We search for these in the upper left.
SIDE_BY_SIDE_LOGO_PATHS = [
    LOGOS_DIR / "fox_side_by_side_logo_crop.png",
    LOGOS_DIR / "fs1_side_by_side_logo_crop.png",
    LOGOS_DIR / "nbc_nascar_non_stop_side_by_side_logo.png",
]


class MatchResult(NamedTuple):
    res: cv2.typing.MatLike
    top_left: cv2.typing.Point
    bottom_right: cv2.typing.Point
    max_val: float
    min_val: float
    elapsed_time: float

    def __str__(self):
        max_val_str = f"{self.max_val * 100:.2f}%"
        min_val_str = f"{self.min_val * 100:.2f}%"
        return (
            f"top_left={self.top_left}, bottom_right={self.bottom_right}, "
            f"max_val={max_val_str}, min_val={min_val_str}, "
            f"elapsed_time={self.elapsed_time:.4f}s"
        )


def match_template(img, template, method) -> MatchResult:
    h, w, *_ = template.shape

    # Apply template Matching
    start_time = time.perf_counter()
    res = cv2.matchTemplate(img, template, method)
    elapsed_time = time.perf_counter() - start_time

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return MatchResult(res, top_left, bottom_right, max_val, min_val, elapsed_time)


def mask_non_white(img, min_thresh=200):
    non_white_mask = (
        (img[:, :, 0] < min_thresh)
        | (img[:, :, 1] < min_thresh)
        | (img[:, :, 2] < min_thresh)
    )
    img[non_white_mask] = [0, 0, 0]
    return img


def load_masked(img_path):
    return mask_non_white(cv2.imread(img_path))


MASKED_NETWORK_LOGOS = [*map(load_masked, NETWORK_LOGO_PATHS)]
MASKED_SIDE_BY_SIDE_LOGOS = [*map(load_masked, SIDE_BY_SIDE_LOGO_PATHS)]


def _has_network_logo(img, masked_logo):
    # scale to fixed size to ensure coordinates for logo match are consistent
    img = cv2.resize(img, (1920, 1080))

    masked_img = mask_non_white(img)

    h, w = masked_img.shape[:2]
    masked_img_crop = masked_img[0 : h // 8, w * 5 // 6 : w]

    result = match_template(masked_img_crop, masked_logo, method=MATCH_METHOD)

    tl_x, tl_y = result.top_left
    br_x, br_y = result.bottom_right

    # FIXME: make this configurable and more robust
    return (
        110 <= tl_x <= 140
        and 15 <= tl_y <= 35
        and 245 <= br_x <= 290
        and 50 <= br_y <= 75
        and result.max_val >= 0.39
    )


def has_network_logo(img, masked_logos=MASKED_NETWORK_LOGOS):
    return any(_has_network_logo(img, masked_logo) for masked_logo in masked_logos)


def _has_side_by_side_logo(img, masked_logo):
    # scale to fixed size to ensure coordinates for logo match are consistent
    img = cv2.resize(img, (1920, 1080))

    masked_img = mask_non_white(img)

    h, w = masked_img.shape[:2]
    masked_img_crop = masked_img[0 : h // 5, 0 : w // 5]

    result = match_template(masked_img_crop, masked_logo, method=MATCH_METHOD)

    tl_x, tl_y = result.top_left
    br_x, br_y = result.bottom_right

    # FIXME: make this configurable and more robust
    return (
        # 110 <= tl_x <= 140 and
        # 15 <= tl_y <= 35 and
        # 245 <= br_x <= 290 and
        # 50 <= br_y <= 75 and
        result.max_val >= 0.8
    )


def has_side_by_side_logo(img, masked_logos=MASKED_SIDE_BY_SIDE_LOGOS):
    return any(_has_side_by_side_logo(img, masked_logo) for masked_logo in masked_logos)
