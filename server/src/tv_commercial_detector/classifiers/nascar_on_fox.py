import cv2

from ..classification import llm_match, logo_match, rectangle_match
from ..classification.result import ClassificationResult


def classify_image(image_path: str) -> ClassificationResult:
    """Three-pass classification: logo detection, scoreboard detection,
    then prompt-based fallback."""
    # FIXME: don't pass the image as a path to every function here. Load it once
    # and pass the image object or bytes to each function.

    cv_img = cv2.imread(image_path)

    # Network logo in the upper right -> not an ad break
    if logo_match.has_network_logo(cv_img):
        return ClassificationResult(
            source="opencv", type="content", reason="network_logo", reply="(opencv)"
        )

    # Side-by-side logo in the upper left -> very likely an ad break
    if logo_match.has_side_by_side_logo(cv_img):
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="side_by_side",
            reply="side-by-side logo match (opencv)",
        )

    # check for bounding boxes used during side-by-side ad breaks
    matched_rect = rectangle_match.image_has_known_ad_rectangle(cv_img)
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
