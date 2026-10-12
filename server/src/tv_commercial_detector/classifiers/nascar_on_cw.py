"""
NASCAR O'Reilly Auto Parts Series on The CW.

PROVISIONAL: every threshold here was measured on a single broadcast, the
Las Vegas race, from frames saved while the Fox profile was running. Treat the
numbers as a starting point until a full `record_broadcast.py` recording has
been ruled and the checks re-measured against it.

Two things set this graphics package apart from Fox and NBC:

1. The "CW Sports" corner bug STAYS ON SCREEN during side-by-side breaks, so
   the bug can only mean `content` once side-by-side has been ruled out. The
   order of the checks below is what encodes that.

2. Side-by-side breaks carry no dependable banner. Most show a "CW NASCAR"
   plate in the upper left, but a sponsor takeover replaces it with the
   sponsor's logo and a plain variant shows nothing. What all of them share is
   the geometry: the race shrunk into the same panel on the left. So the break
   is detected from that panel's border, not from a logo.

A frame neither check settles goes to the LLM. There is no audio model for this
feed; fitting one needs a continuous recording.
"""
import math

import cv2
import numpy as np

from ..classification import llm_match, logo_match, rectangle_match
from ..classification.result import ClassificationResult

# --- Side-by-side race panel (left) -> ad -----------------------------------
#
# During a break the race plays in a fixed panel on the left. Its four sides
# are scored separately and the WEAKEST one is the score, because live racing
# routinely puts a strong edge along one or two of these lines (a wall, the
# top of the scoring ticker) but not along all four at once.
#
# The panel sits 3 px further left under the "CW NASCAR" plate than in the
# plain variant; `border_edge_fractions` allows for that drift.
#
# Measured over the 609 saved frames of the Las Vegas race, 509 frames of a CW
# pre-race show, and 3000 archive frames from other networks:
#
#   Las Vegas, side-by-side (all three variants, n=92):  min 0.93
#   Las Vegas, one break animating in (n=1):             0.72
#   Las Vegas, everything else (n=516):                  max 0.66
#   CW pre-race show (no side-by-side breaks, n=509):    max 0.67
#   Other networks (n=3000):                             max 0.85
#
# The 0.85 is an FS1 crowd shot, which is why the threshold is not lower.
SIDE_BY_SIDE_RACE_PANEL = (60, 790, 313, 725)  # x0, x1, y0, y1 at 1920x1080
SIDE_BY_SIDE_THRESHOLD = 0.9

# --- "CW Sports" bug (upper right) -> content -------------------------------
#
# The bug is translucent grey, well short of white, and turns yellow under
# caution. Neither a white mask nor a color template covers both, so it is
# matched as a SHAPE: the template is a binary silhouette of the glyphs, and
# TM_CCOEFF_NORMED ignores how bright the glyphs are as long as they stand out
# from what is behind them.
#
# Grayscale alone loses the yellow bug over sunlit asphalt, where yellow and
# grey have nearly the same luminance. A second pass matches the same
# silhouette against a "yellowness" map (min(R, G) - B), where the yellow bug
# is the only bright thing in the corner whatever is behind it.
#
# Measured over the same frames, bug presence judged by eye on contact sheets
# sorted by score, not ruled frame by frame:
#
#   Other networks, no CW bug (n=3000):       max 0.47 (the TNT Sports bug)
#   CW frames with no bug, highest seen:      0.45
#   CW frames with the bug, lowest seen:      0.31
#   CW frames scoring >= 0.55:                429 of 609 race, 346 of 509 pre-race
#
# Bug frames below the threshold are the grey bug ghosted over a bright sky or
# sunlit track, and some yellow ones over a yellow wall. They fall through to
# the LLM. A false match claims content during an ad and leaves a commercial on
# screen, so the threshold sits above every negative seen.
BUG_SHAPE = logo_match.LOGOS_DIR / "cw_sports_bug_shape.png"
BUG_TEMPLATE = cv2.imread(str(BUG_SHAPE), cv2.IMREAD_GRAYSCALE)
BUG_REGION = (1680, 1900, 20, 160)  # x0, x1, y0, y1
BUG_THRESHOLD = 0.55

# A patch with almost no variation has nothing to correlate against, and
# TM_CCOEFF_NORMED divides by its variance: a flat corner can report anything
# up to a perfect 1.0. Below this standard deviation the patch scores 0.
MIN_PATCH_STD = 4.0

# Toggles so a single check can be turned off mid-broadcast at the console.
ENABLE_SIDE_BY_SIDE_CHECK = True
ENABLE_BUG_CHECK = True

PROMPT = llm_match.load_prompt("prompt_cw.txt")


def side_by_side_score(img: cv2.typing.MatLike) -> float:
    """Weakest of the four sides of the side-by-side race panel, 0 to 1.

    *img* must already be resized to 1920x1080.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return min(rectangle_match.border_edge_fractions(gray, SIDE_BY_SIDE_RACE_PANEL))


def is_side_by_side(img: cv2.typing.MatLike, threshold: float = SIDE_BY_SIDE_THRESHOLD) -> bool:
    return side_by_side_score(img) >= threshold


def _shape_score(patch: cv2.typing.MatLike, template: cv2.typing.MatLike) -> float:
    """Best match for *template* in a single-channel *patch*; 0.0 if it is flat."""
    if patch.std() < MIN_PATCH_STD:
        return 0.0
    score = logo_match.match_template(patch, template).max_val
    return float(score) if math.isfinite(score) else 0.0


def bug_score(img: cv2.typing.MatLike, template: cv2.typing.MatLike = BUG_TEMPLATE) -> float:
    """Best match score for the CW Sports bug, grey or yellow.

    *img* must already be resized to 1920x1080.
    """
    x0, x1, y0, y1 = BUG_REGION
    corner = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    blue, green, red = (corner[:, :, c].astype(np.int16) for c in range(3))
    yellowness = np.clip(np.minimum(red, green) - blue, 0, 255).astype(np.uint8)
    return max(_shape_score(gray, template), _shape_score(yellowness, template))


def has_bug(
    img: cv2.typing.MatLike,
    template: cv2.typing.MatLike = BUG_TEMPLATE,
    threshold: float = BUG_THRESHOLD,
) -> bool:
    return bug_score(img, template) >= threshold


def classify_image(image_path: str, audio_bytes: bytes | None = None) -> ClassificationResult:
    cv_img = cv2.imread(image_path)
    cv_img_1080p = cv2.resize(cv_img, (1920, 1080))

    # Side-by-side runs first: the bug stays up through the break and would
    # otherwise win.
    if ENABLE_SIDE_BY_SIDE_CHECK and is_side_by_side(cv_img_1080p):
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="side_by_side",
            reply="CW side-by-side race panel (opencv)",
        )

    if ENABLE_BUG_CHECK and has_bug(cv_img_1080p):
        return ClassificationResult(
            source="opencv", type="content", reason="network_logo", reply="(opencv)"
        )

    # No OpenCV verdict. A missing bug usually means a full-screen commercial,
    # but the bug also ghosts out over bright backgrounds, so this goes to the
    # LLM instead of defaulting to ad.
    image_data = llm_match.load_image_b64(image_path)
    audio_data = llm_match.audio_b64(audio_bytes)
    return llm_match.classify_by_prompt(image_data, audio_data, prompt=PROMPT)
