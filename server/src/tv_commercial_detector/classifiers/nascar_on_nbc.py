"""
NASCAR Cup Series on NBC Sports — both the NBC and USA Network feeds.

NBC Sports carries Cup races on NBC proper and on USA. It's the same production
with the same "NASCAR NON STOP" side-by-side break, but the corner bug differs,
so this profile checks for either one. Whichever bug is present means the
broadcast is live, the same role the Fox logo plays in `nascar_on_fox`.

The two bugs need genuinely different matching and are NOT interchangeable:

- The NBC peacock is opaque and coloured -> matched in colour.
- The USA wordmark is translucent white -> matched on a white mask.

See each section below.

Three things differ from the Fox profile and matter if you edit this:

1. The peacock is matched IN COLOR. `logo_match.load_masked` (and
   `mask_non_white`) zero out everything that isn't near-white, which erases a
   six-colour logo completely. Load this template with a plain `cv2.imread`.

2. The search window is deliberately tight. Matching the peacock over the same
   wide region the Fox profile uses (the upper-right sixth of the frame) does
   not separate: the weakest true positive scores below the strongest random
   negative. Constraining the window to where the bug actually sits is what
   makes the match reliable.

3. The side-by-side banner is matched on UNMASKED GRAYSCALE, not a white mask.
   See that section.

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
import math

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

# --- USA Network wordmark (network bug, upper right) -> content -------------
#
# The USA bug is white, so unlike the peacock it takes the same white-masked
# match the Fox profile uses. What it does not tolerate is an unguarded match:
# the bug is translucent, so over a blown-out sky it becomes a near-invisible
# ghost and the masked region saturates to a uniform patch. TM_CCOEFF_NORMED
# divides by zero on a uniform patch and can report a perfect 1.0, which would
# turn every bright sky into a false `content`. The mask-fraction guard below
# is what makes this check safe, and it matters more than the mask threshold —
# dropping the threshold to 160 to chase the faint frames measured slightly
# WORSE than the stock 200 (79.0% vs 79.8% recall), so this uses the default.
#
# The ghost frames are not worth chasing at all: they carry almost no signal,
# and reaching them costs far more precision than the recall is worth. They
# fall through to the LLM instead.
#
# Measured over 188 frames from the USA broadcast against 3000 archive frames:
# ~80% recall with 0/3000 false positives, strongest false positive 0.419.
USA_LOGO = logo_match.LOGOS_DIR / "usa_network_logo.png"
USA_TEMPLATE = logo_match.load_masked(USA_LOGO)
USA_REGION = (1750, 1890, 50, 140)  # x0, x1, y0, y1
USA_THRESHOLD = 0.65

# A white mask only carries information when it isolates something. An empty
# mask has nothing to match; a saturated one has no structure to separate the
# glyph from its background. Treat either as "no detection".
USA_MIN_MASK_FRACTION = 0.01
USA_MAX_MASK_FRACTION = 0.90

# --- "NASCAR NON STOP" (side-by-side ad break, upper left) -> ad ------------
#
# Matched UNMASKED, in grayscale. This is the one place the white mask actively
# hurts. The banner is white-on-black, so masking it looks like the obvious
# move, but `mask_non_white` (min_thresh=200) only survives a crisp rendering of
# the glyphs. It holds on the NBC/USA feed and collapses on Prime Video, where
# the same graphic is softer and anti-aliased: masked scores drop to 0.16-0.22
# there, far below any usable threshold, while genuine NBC/USA breaks sit at
# only 0.89. Dropping the mask removes that dependence on how one feed happens
# to render its edges.
#
# Measured with the template below (cropped from the Prime feed at
# 2026-06-21T21-04-26-980026_0, x 55-345 / y 33-75 in 1920x1080 space):
#
#   Prime banner frames (crop source excluded, n=52): median 0.998, 90% >= 0.80
#   NBC/USA banner frames (n=94):                     min 0.892, median 0.955
#   Labelled `content` frames (n=239):                max 0.656, none >= 0.80
#   Aug 9 non-banner frames (n=2226):                 max 0.71
#
# One template, ~0.18 of margin on both feeds. The ~10% of Prime positives below
# threshold are the banner animating in and out, which is a frame or two either
# side of a break that the LLM handles anyway.
#
# The banner is small relative to the search region, so restrict the search to
# where it actually sits rather than the upper-left fifth: the region below is
# the band the banner occupies on both feeds, with margin.
SIDE_BY_SIDE_LOGO = logo_match.LOGOS_DIR / "nbc_nascar_non_stop_banner.png"
SIDE_BY_SIDE_TEMPLATE = cv2.cvtColor(cv2.imread(str(SIDE_BY_SIDE_LOGO)), cv2.COLOR_BGR2GRAY)
SIDE_BY_SIDE_REGION = (0, 500, 0, 160)  # x0, x1, y0, y1
SIDE_BY_SIDE_THRESHOLD = 0.8

# Toggles so any single check can be turned off mid-broadcast without a code
# change, via `/settings/classifier_profile` swapping to a known-good profile or
# by editing these at the console.
ENABLE_PEACOCK_CHECK = True
ENABLE_USA_CHECK = True
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


def usa_score(img: cv2.typing.MatLike, template: cv2.typing.MatLike = USA_TEMPLATE) -> float:
    """Best match score for the USA wordmark within its search window.

    Returns 0.0 when the white mask is empty or saturated; see
    USA_MIN_MASK_FRACTION. *img* must already be resized to 1920x1080.
    """
    x0, x1, y0, y1 = USA_REGION
    masked = logo_match.mask_non_white(img[y0:y1, x0:x1].copy())

    mask_fraction = masked.any(axis=2).mean()
    if not USA_MIN_MASK_FRACTION <= mask_fraction <= USA_MAX_MASK_FRACTION:
        return 0.0

    score = logo_match.match_template(masked, template).max_val
    # Guard anyway: a uniform patch that slips past the fraction check still
    # produces a divide-by-zero inside matchTemplate.
    return float(score) if math.isfinite(score) else 0.0


def has_usa_logo(
    img: cv2.typing.MatLike,
    template: cv2.typing.MatLike = USA_TEMPLATE,
    threshold: float = USA_THRESHOLD,
) -> bool:
    return usa_score(img, template) >= threshold


def has_network_logo(img: cv2.typing.MatLike) -> bool:
    """True if either NBC Sports corner bug is present."""
    if ENABLE_PEACOCK_CHECK and has_peacock_logo(img):
        return True
    return ENABLE_USA_CHECK and has_usa_logo(img)


def side_by_side_score(
    img: cv2.typing.MatLike, template: cv2.typing.MatLike = SIDE_BY_SIDE_TEMPLATE
) -> float:
    """Best match score for the NASCAR NON STOP banner within its search window.

    *img* must already be resized to 1920x1080.
    """
    x0, x1, y0, y1 = SIDE_BY_SIDE_REGION
    gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return logo_match.match_template(gray, template).max_val


def has_side_by_side_logo(
    img: cv2.typing.MatLike,
    template: cv2.typing.MatLike = SIDE_BY_SIDE_TEMPLATE,
    threshold: float = SIDE_BY_SIDE_THRESHOLD,
) -> bool:
    return side_by_side_score(img, template) >= threshold


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

    if has_network_logo(cv_img_1080p):
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
