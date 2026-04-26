import cv2

from ..classification import llm_match, logo_match, rectangle_match
from ..classification.result import ClassificationResult

# XXX: I've stored these logo locations in dicts so that in the future we could
# theoretically configure this classifier to only look for the logo of the the
# network we're currently watching.


# We search for these in the upper right.
NETWORK_LOGO_PATHS = {
    "fox": logo_match.LOGOS_DIR / "fox_logo_crop.png",
    "fs1": logo_match.LOGOS_DIR / "fs1_logo_crop.png",
}

# We search for these in the upper left.
SIDE_BY_SIDE_LOGO_PATHS = {
    "fox": logo_match.LOGOS_DIR / "fox_logo_crop.png",
    "fox": logo_match.LOGOS_DIR / "fox_side_by_side_logo_crop.png",
    "fs1": logo_match.LOGOS_DIR / "fs1_side_by_side_logo_crop.png",
    "trucks": logo_match.LOGOS_DIR / "truck_series_logo.png",
}

MASKED_NETWORK_LOGOS = {
    name: logo_match.load_masked(path)
    for name, path in NETWORK_LOGO_PATHS.items()
}
MASKED_SIDE_BY_SIDE_LOGOS = {
    name: logo_match.load_masked(path)
    for name, path in SIDE_BY_SIDE_LOGO_PATHS.items()
}


def _extract_fox_logo_region(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Crop the region of the image where the Fox network logo would appear if
    present.
    """
    h, w = img.shape[:2]
    return img[0 : h // 8, w * 5 // 6 : w]


def _has_network_logo(img: cv2.typing.MatLike, masked_logo: cv2.typing.MatLike) -> bool:
    # scale to fixed size to ensure coordinates for logo match are consistent.
    # XXX: also, template matching -- all cropped logos are from 1920x1080
    # images.
    # img = cv2.resize(img, (1920, 1080))

    img_crop = _extract_fox_logo_region(img)
    masked_img_crop = logo_match.mask_non_white(img_crop.copy())

    result = logo_match.match_template(masked_img_crop, masked_logo)

    tl_x, tl_y = result.top_left
    br_x, br_y = result.bottom_right

    return (
        # TODO: add an option to override or temporarily disable checking the pixels
        True
        and result.max_val >= 0.39
        # and 110 <= tl_x <= 140
        # and 15 <= tl_y <= 35
        # and 245 <= br_x <= 290
        # and 50 <= br_y <= 75
    )


def has_network_logo(img, masked_logos=MASKED_NETWORK_LOGOS):
    return any(_has_network_logo(img, masked_logo) for masked_logo in masked_logos.values())


def _has_side_by_side_logo(img: cv2.typing.MatLike, masked_logo: cv2.typing.MatLike) -> bool:
    # scale to fixed size to ensure coordinates for logo match are consistent
    # img = cv2.resize(img, (1920, 1080))

    masked_img = logo_match.mask_non_white(img)

    h, w = masked_img.shape[:2]
    masked_img_crop = masked_img[0 : h // 5, 0 : w // 5]

    result = logo_match.match_template(masked_img_crop, masked_logo)

    return result.max_val >= 0.8


def has_side_by_side_logo(img: cv2.typing.MatLike, masked_logos=MASKED_SIDE_BY_SIDE_LOGOS) -> bool:
    return any(_has_side_by_side_logo(img, masked_logo) for masked_logo in masked_logos.values())



def classify_image(image_path: str) -> ClassificationResult:
    """Three-pass classification: logo detection, scoreboard detection,
    then prompt-based fallback."""
    # FIXME: don't pass the image as a path to every function here. Load it once
    # and pass the image object or bytes to each function.

    cv_img = cv2.imread(image_path)
    cv_img_1080p = cv2.resize(cv_img, (1920, 1080))

    # Network logo in the upper right -> not an ad break
    if has_network_logo(cv_img_1080p):
        return ClassificationResult(
            source="opencv", type="content", reason="network_logo", reply="(opencv)"
        )

    # Side-by-side logo in the upper left -> very likely an ad break
    if has_side_by_side_logo(cv_img_1080p):
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="side_by_side",
            reply="side-by-side logo match (opencv)",
        )

    # check for bounding boxes used during side-by-side ad breaks
    matched_rect = rectangle_match.image_has_known_ad_rectangle(cv_img_1080p)
    if matched_rect is not None:
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="matched_rectangle",
            reply=f"{matched_rect} (opencv)",
        )

    image_data = llm_match.load_image_b64(image_path)

    racing_related = llm_match._report_racing_related(image_data)
    if not racing_related:
        return ClassificationResult(
            source="llm",
            type="ad",
            reason="model_quick_reject",
            reply="No NASCAR-related content detected",
        )

    ## XXX: uncomment to test without using the slower model prompt.
    ## (It would be better if classification was more configurable so I could
    ## just toggle off the final lengthy step.)
    # return ClassificationResult(type="content", reason="assume_content", reply="")

    return llm_match.classify_by_prompt(image_data)
