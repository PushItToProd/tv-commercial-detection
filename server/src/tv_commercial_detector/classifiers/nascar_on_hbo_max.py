import base64

import cv2

from ..classification import llm_match, logo_match, rectangle_match
from ..classification.result import ClassificationResult


# full screen ad breaks
WELL_BE_BACK_LOGO = logo_match.LOGOS_DIR / "tnt-sports-full-screen-brb.png"
WELL_BE_BACK_LOGO_MASKED = logo_match.load_masked(WELL_BE_BACK_LOGO)

# side-by-side (full screen racing on HBO)
COMMERCIAL_BRB_LOGO = logo_match.LOGOS_DIR / "tnt-sports-sbs-commercial-break-in-progress.png"
COMMERCIAL_BRB_LOGO_MASKED = logo_match.load_masked(COMMERCIAL_BRB_LOGO)



def has_well_be_back(img: cv2.typing.MatLike, masked_logo: cv2.typing.MatLike = WELL_BE_BACK_LOGO_MASKED):
    result = logo_match.match_template(img, masked_logo)
    return result.max_val >= 0.85


def has_commercial_break_logo(img: cv2.typing.MatLike, masked_logo: cv2.typing.MatLike = COMMERCIAL_BRB_LOGO_MASKED):
    result = logo_match.match_template(img, masked_logo)
    return result.max_val >= 0.85


def classify_image(image_path: str, audio_bytes: bytes | None = None) -> ClassificationResult:
    cv_img = cv2.imread(image_path)
    cv_img_1080p = cv2.resize(cv_img, (1920, 1080))

    if has_commercial_break_logo(cv_img_1080p):
        return ClassificationResult(
            source="opencv", type="content", reason="commercial_break_racing", reply="(opencv)"
        )

    if has_well_be_back(cv_img_1080p):
        return ClassificationResult(
            source="opencv", type="ad", reason="well_be_back", reply="(opencv)"
        )

    return ClassificationResult(
        source="default",
        type="content",
        reason="default_fallback",
        reply="No ad logo signifiers detected"
    )
