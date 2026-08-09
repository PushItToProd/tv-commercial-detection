"""
NASCAR Cup Series on NBC / NBC Sports.

NBC keeps the peacock bug in the upper right during race content and drops it
during full-screen ads, so the peacock is the primary `content` signal — the
same role the Fox logo plays in `nascar_on_fox`.

Two things differ from the Fox profile and matter if you edit this:

1. The peacock is matched IN COLOR. `logo_match.load_masked` (and
   `mask_non_white`) zero out everything that isn't near-white, which erases a
   six-colour logo completely. Load this template with a plain `cv2.imread`.

2. The search window is deliberately tight. Matching the peacock over the same
   wide region the Fox profile uses (the upper-right sixth of the frame) does
   not separate: the weakest true positive scores below the strongest random
   negative. Constraining the window to where the bug actually sits is what
   makes the match reliable.

Threshold rationale: measured over the 21 labelled NBC frames in `frames/`
against 3000 random non-NBC frames. Weakest true positive scores 0.573, the
strongest false positive 0.539, so 0.55 separates them completely on that set.
Leave-one-out (template rebuilt without the frame under test) is the honest
estimate for unseen frames and puts two positives at 0.52-0.54, i.e. about 10%
of frames would fall through instead of short-circuiting.

That asymmetry is deliberate. A false positive claims content during an ad and
costs a missed break; a miss only falls through to the LLM, which will most
likely reach the same verdict for a few hundred ms. Prefer missing.
"""
import base64

import cv2

from ..classification import llm_match, logo_match
from ..classification.result import ClassificationResult

# --- Peacock (network bug, upper right) -> content --------------------------

PEACOCK_LOGO = logo_match.LOGOS_DIR / "nbc_peacock_logo.png"
# NOT load_masked: see module docstring.
PEACOCK_TEMPLATE = cv2.imread(str(PEACOCK_LOGO))

# Search window in 1920x1080 coordinates. The bug occupies roughly
# x 1774-1842, y 69-112; this leaves margin for broadcast-to-broadcast drift
# without reopening the false-positive rate.
PEACOCK_REGION = (1740, 1880, 40, 140)  # x0, x1, y0, y1
PEACOCK_THRESHOLD = 0.55

# --- "NASCAR NON STOP" (side-by-side ad break, upper left) -> ad ------------
#
# UNVALIDATED. Every labelled NBC frame available is `content`, so there is no
# side-by-side frame to test against. These templates are last season's assets
# and are checked only against content frames, where they correctly stay quiet
# (max score 0.35). Whether they fire on a real break is unknown — harvest a
# side-by-side frame and re-crop before trusting this.
SIDE_BY_SIDE_LOGO_PATHS = {
    "non_stop": logo_match.LOGOS_DIR / "nbc_nascar_non_stop_side_by_side_logo.png",
    "non_stop_full": logo_match.LOGOS_DIR / "nbc_nascar_non_stop_full_logo.png",
}
MASKED_SIDE_BY_SIDE_LOGOS = {
    name: logo_match.load_masked(path) for name, path in SIDE_BY_SIDE_LOGO_PATHS.items()
}
SIDE_BY_SIDE_THRESHOLD = 0.8

# Toggles so the unvalidated half can be turned off mid-broadcast without a
# code change, via `/settings/classifier_profile` swapping to a known-good
# profile or by editing these at the console.
ENABLE_PEACOCK_CHECK = True
ENABLE_SIDE_BY_SIDE_CHECK = True

PROMPT = llm_match.load_prompt("prompt_nbc.txt")


def peacock_score(img: cv2.typing.MatLike, template: cv2.typing.MatLike = PEACOCK_TEMPLATE) -> float:
    """Best match score for the peacock bug within its search window.

    *img* must already be resized to 1920x1080.
    """
    x0, x1, y0, y1 = PEACOCK_REGION
    result = logo_match.match_template(img[y0:y1, x0:x1], template)
    return result.max_val


def has_peacock_logo(
    img: cv2.typing.MatLike,
    template: cv2.typing.MatLike = PEACOCK_TEMPLATE,
    threshold: float = PEACOCK_THRESHOLD,
) -> bool:
    return peacock_score(img, template) >= threshold


def _has_side_by_side_logo(img: cv2.typing.MatLike, masked_logo: cv2.typing.MatLike) -> bool:
    masked_img = logo_match.mask_non_white(img.copy())
    h, w = masked_img.shape[:2]
    result = logo_match.match_template(masked_img[0 : h // 5, 0 : w // 5], masked_logo)
    return result.max_val >= SIDE_BY_SIDE_THRESHOLD


def has_side_by_side_logo(
    img: cv2.typing.MatLike, masked_logos=MASKED_SIDE_BY_SIDE_LOGOS
) -> bool:
    return any(_has_side_by_side_logo(img, logo) for logo in masked_logos.values())


def classify_image(image_path: str, audio_bytes: bytes | None = None) -> ClassificationResult:
    cv_img = cv2.imread(image_path)
    cv_img_1080p = cv2.resize(cv_img, (1920, 1080))

    # Side-by-side runs first: during a break the race stays on screen, so the
    # peacock can still be visible and would otherwise win.
    if ENABLE_SIDE_BY_SIDE_CHECK and has_side_by_side_logo(cv_img_1080p):
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="side_by_side",
            reply="NASCAR NON STOP side-by-side logo match (opencv)",
        )

    if ENABLE_PEACOCK_CHECK and has_peacock_logo(cv_img_1080p):
        return ClassificationResult(
            source="opencv", type="content", reason="network_logo", reply="(opencv)"
        )

    # No OpenCV verdict. Fall through to the LLM rather than assuming content:
    # the NBC graphics package hasn't been verified against this season, so a
    # silent content default would hide every case where the templates are stale.
    image_data = llm_match.load_image_b64(image_path)
    audio_data = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes is not None else None

    if not llm_match._report_racing_related(image_data, audio_data):
        return ClassificationResult(
            source="llm",
            type="ad",
            reason="model_quick_reject",
            reply="No NASCAR-related content detected",
        )

    return llm_match.classify_by_prompt(image_data, audio_data, prompt=PROMPT)
