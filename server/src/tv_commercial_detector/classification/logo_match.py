import time
from pathlib import Path
from typing import NamedTuple

import cv2
from cv2.typing import MatLike, Point

TEMPLATE_MATCH_METHOD = cv2.TM_CCOEFF_NORMED

LOGOS_DIR = Path(__file__).parent.parent / "prompt" / "logos"

# TODO: move network-specific logo details into per-network packages.

# We search for these in the upper right. Positioning is hardcoded for Fox.
NETWORK_LOGO_PATHS = [
    LOGOS_DIR / "fox_logo_crop.png",
    LOGOS_DIR / "fs1_logo_crop.png",
    LOGOS_DIR / "cw_sports_logo_crop.png",
]

# We search for these in the upper left.
SIDE_BY_SIDE_LOGO_PATHS = [
    LOGOS_DIR / "fox_side_by_side_logo_crop.png",
    LOGOS_DIR / "fs1_side_by_side_logo_crop.png",
    LOGOS_DIR / "nbc_nascar_non_stop_side_by_side_logo.png",
]


class MatchResult(NamedTuple):
    res: MatLike
    top_left: Point
    bottom_right: Point
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


def _match_template(img: MatLike, template: MatLike, method=TEMPLATE_MATCH_METHOD) -> MatchResult:
    """
    Perform template matching on the given image using the given method.
    """
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


def mask_non_white(img: MatLike, min_thresh=200):
    non_white_mask = (
        (img[:, :, 0] < min_thresh)
        | (img[:, :, 1] < min_thresh)
        | (img[:, :, 2] < min_thresh)
    )
    img[non_white_mask] = [0, 0, 0]
    return img


class ImageLoadError(Exception):
    """
    Indicates a failure to load the image using cv2.imread.
    """
    def __init__(self, img_path):
        super().__init__(f"Failed to load image at path: {img_path}")
        self.img_path = img_path


def _load_masked(img_path: Path | str) -> MatLike:
    img = cv2.imread(img_path)
    if img is None:
        raise ImageLoadError(img_path)
    return mask_non_white(img)


MASKED_NETWORK_LOGOS = [*map(_load_masked, NETWORK_LOGO_PATHS)]
MASKED_SIDE_BY_SIDE_LOGOS = [*map(_load_masked, SIDE_BY_SIDE_LOGO_PATHS)]


def _has_network_logo(img: MatLike, masked_logo: MatLike) -> bool:
    # scale to fixed size to ensure coordinates for logo match are consistent
    img = cv2.resize(img, (1920, 1080))

    masked_img = mask_non_white(img)

    h, w = masked_img.shape[:2]
    masked_img_crop = masked_img[0 : h // 8, w * 5 // 6 : w]

    result = _match_template(masked_img_crop, masked_logo, method=TEMPLATE_MATCH_METHOD)

    tl_x, tl_y = result.top_left
    br_x, br_y = result.bottom_right

    # TODO: these values are hardcoded for Fox -- move them into nascar_on_fox
    return (
        110 <= tl_x <= 140
        and 15 <= tl_y <= 35
        and 245 <= br_x <= 290
        and 50 <= br_y <= 75
        and result.max_val >= 0.39
    )


def has_network_logo(img, masked_logos=MASKED_NETWORK_LOGOS):
    return any(_has_network_logo(img, masked_logo) for masked_logo in masked_logos)


def _has_side_by_side_logo(img: MatLike, masked_logo: MatLike) -> bool:
    # scale to fixed size to ensure coordinates for logo match are consistent
    img = cv2.resize(img, (1920, 1080))

    masked_img = mask_non_white(img)

    h, w = masked_img.shape[:2]
    masked_img_crop = masked_img[0 : h // 5, 0 : w // 5]

    result = _match_template(masked_img_crop, masked_logo, method=TEMPLATE_MATCH_METHOD)

    return result.max_val >= 0.8


def has_side_by_side_logo(img: MatLike, masked_logos=MASKED_SIDE_BY_SIDE_LOGOS) -> bool:
    return any(_has_side_by_side_logo(img, masked_logo) for masked_logo in masked_logos)
