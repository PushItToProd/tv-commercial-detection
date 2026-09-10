"""NFL on NBC / Sunday Night Football.

NBC's NFL broadcasts carry the same peacock corner bug as `nascar_on_nbc`, but
NOT at the same place: on this graphics package the bug sits noticeably higher
and further right, at roughly x 1814-1888, y 27-76 in a 1920x1080 frame,
against NASCAR's x 1771-1845, y 66-115. Reusing `nascar_on_nbc.PEACOCK_REGION`
against this broadcast clips the bug out of the search window entirely — its
top edge sits above the NASCAR region's `y0` and its right edge sits past the
NASCAR region's `x1` less the template width — so `matchTemplate` never even
sees the whole logo and scores land at 0.3-0.5 regardless of threshold. This
is a positioning problem, not a colour or saturation one: the same template,
searched at the right position, matches at 0.8-0.86 here, right in line with
NASCAR's own true positives.

There is no NASCAR NON STOP equivalent on this feed — NBC's NFL coverage cuts
away to a normal full-screen commercial rather than shrinking the game into a
side-by-side panel — so unlike `nascar_on_nbc` there is nothing to detect that
means `ad`. The peacock is the only OpenCV signal; its absence falls through
to the LLM rather than defaulting to `ad`, since a stale template should be
noticed rather than silently start rejecting every frame as a commercial.

Threshold rationale: measured over ~600 frames of an in-progress broadcast
(Patriots @ Seahawks, 2026-09-09). The bug's backing varies with what's
playing behind it — a neutral grey backing plate close to the template's own
background scores 0.8-0.86, but the same peacock over a bright graphic, a
player close-up, or a crowd shot loses correlation against that background and
can drop to 0.5-0.7 despite being clearly visible by eye. The highest score
found with no peacock actually present in the window was 0.52-0.59 (a couple
of dark transition frames scored a coincidental match against the window
edge), which sits inside that same weak-true-positive band -- there is no
clean gap to split on. 0.7 clears every confirmed false match with margin and
keeps ~62% of frames on the fast path; the weaker true positives it misses
fall through to the LLM instead of being misread, which is the safer failure
mode (see `nascar_on_nbc` docstring for the same asymmetry argument).
"""
import base64

import cv2

from ..classification import llm_match, logo_match
from ..classification.result import ClassificationResult

PEACOCK_LOGO = logo_match.LOGOS_DIR / "nbc_peacock_logo.png"
# NOT load_masked: the peacock is opaque and coloured; white-masking would
# zero it out. See nascar_on_nbc's module docstring for the same point.
PEACOCK_TEMPLATE = cv2.imread(str(PEACOCK_LOGO))

# Search window in 1920x1080 coordinates. The bug occupies roughly
# x 1814-1888, y 27-76 on this graphics package; this leaves margin for
# broadcast-to-broadcast drift without reopening the false-positive rate.
PEACOCK_REGION = (1795, 1905, 10, 95)  # x0, x1, y0, y1
PEACOCK_THRESHOLD = 0.7

PROMPT = llm_match.load_prompt("prompt_nfl_nbc.txt")


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


def classify_image(image_path: str, audio_bytes: bytes | None = None) -> ClassificationResult:
    cv_img = cv2.imread(image_path)
    cv_img_1080p = cv2.resize(cv_img, (1920, 1080))

    if has_peacock_logo(cv_img_1080p):
        return ClassificationResult(
            source="opencv", type="content", reason="network_logo", reply="(opencv)"
        )

    # No OpenCV verdict. Fall through to the LLM rather than assuming ad: the
    # peacock miss could just as easily be a weak match (see module docstring)
    # as a genuine ad, and a silent ad default would hide every case where the
    # template has gone stale against a new graphics package.
    image_data = llm_match.load_image_b64(image_path)
    audio_data = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes is not None else None

    if not llm_match._report_racing_related(image_data, audio_data, subject="NFL football"):
        return ClassificationResult(
            source="llm",
            type="ad",
            reason="model_quick_reject",
            reply="No NFL-related content detected",
        )

    return llm_match.classify_by_prompt(image_data, audio_data, prompt=PROMPT)
