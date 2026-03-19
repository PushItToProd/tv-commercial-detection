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


# MatchResult is a NamedTuple rather than a dataclass because it is intended to
# be immutable and lightweight, and it benefits from the built-in tuple behavior
# (e.g., unpacking, indexing).
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


def match_template(img: MatLike, template: MatLike, method=TEMPLATE_MATCH_METHOD) -> MatchResult:
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


def load_masked(img_path: Path | str) -> MatLike:
    img = cv2.imread(img_path)
    if img is None:
        raise ImageLoadError(img_path)
    return mask_non_white(img)


MASKED_NETWORK_LOGOS = [*map(load_masked, NETWORK_LOGO_PATHS)]
MASKED_SIDE_BY_SIDE_LOGOS = [*map(load_masked, SIDE_BY_SIDE_LOGO_PATHS)]
