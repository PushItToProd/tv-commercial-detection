"""
NASCAR O'Reilly Auto Parts Series on The CW.

PROVISIONAL: the thresholds were measured on the saved frames of one race (Las
Vegas) and on an unruled recording of another (Charlotte), with presence judged
by eye on contact sheets. Treat the numbers as a starting point until a full
`record_broadcast.py` recording has been ruled and the checks re-measured
against it.

Two things set this graphics package apart from Fox and NBC:

1. The "CW Sports" corner bug STAYS ON SCREEN during side-by-side breaks, so
   the bug can only mean `content` once side-by-side has been ruled out. The
   order of the checks below is what encodes that.

2. Side-by-side breaks carry no dependable banner. Most show a "CW NASCAR"
   plate in the upper left, but a sponsor takeover replaces it with the
   sponsor's logo and a plain variant shows nothing. What all of them share is
   the geometry: the race shrunk into the same panel on the left. So the break
   is detected from that panel's border, not from a logo.

3. The package changes between races. Las Vegas carried a "cw SPORTS" bug;
   Charlotte carried a "cw 20 YEARS" bug in live coverage, kept "cw SPORTS"
   inside side-by-side breaks, and had a scoring ticker Las Vegas did not.
   Both bugs are checked, and the ticker check only knows the Charlotte one.

A frame no OpenCV check settles goes to the audio sensor, then the LLM. The
audio model is borrowed from `nascar_on_nbc`; see the audio section below.
"""

import dataclasses
import math
from pathlib import Path

import cv2
import numpy as np

from .. import audio_sensor
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

# --- CW corner bugs (upper right) -> content --------------------------------
#
# Two bugs are in use: "cw SPORTS" and the wider "cw 20 YEARS" anniversary
# lockup. Each has its own silhouette and its own search window, because the
# anniversary lockup starts 80 px further left and a window wide enough for it
# raises the strongest false match of the "cw SPORTS" silhouette to 0.51.
#
# The bug is translucent grey, well short of white, and turns yellow under
# caution and red under a red flag. Neither a white mask nor a color template covers both, so it is
# matched as a SHAPE: the template is a binary silhouette of the glyphs, and
# TM_CCOEFF_NORMED ignores how bright the glyphs are as long as they stand out
# from what is behind them.
#
# Grayscale alone loses the yellow bug over sunlit asphalt, where yellow and
# grey have nearly the same luminance. A second pass matches the same
# silhouette against a "yellowness" map (min(R, G) - B), where the yellow bug
# is the only bright thing in the corner whatever is behind it. A third pass
# does the same for a red bug with a "redness" map (R - max(G, B)). No red-flag
# frame has been captured, so that pass is checked only against negatives.
#
# Measured over the same frames, bug presence judged by eye on contact sheets
# sorted by score, not ruled frame by frame:
#
#   Other networks, no CW bug (n=3000):       max 0.47 (the TNT Sports bug)
#   A second sample of other networks (n=2929): max 0.50, on the yellowness map
#   CW frames with no bug, highest seen:      0.45
#   CW frames with the bug, lowest seen:      0.31
#   CW frames scoring >= 0.55:                429 of 609 race, 346 of 509 pre-race
#
# The anniversary lockup, over 1889 Charlotte frames and 2929 from other
# networks, all three maps:
#
#   Other networks:                           max 0.43
#   Las Vegas ("cw SPORTS" bug, n=681):       max 0.38
#   Charlotte frames scoring >= 0.55:         1145 of 1889
#
# Bug frames below the threshold are the grey bug ghosted over a bright sky or
# sunlit track, and some yellow ones over a yellow wall. They fall through to
# the LLM. A false match claims content during an ad and leaves a commercial on
# screen, so the threshold sits above every negative seen.
BUG_SHAPE = logo_match.LOGOS_DIR / "cw_sports_bug_shape.png"
BUG_TEMPLATE = cv2.imread(str(BUG_SHAPE), cv2.IMREAD_GRAYSCALE)
BUG_REGION = (1680, 1900, 20, 160)  # x0, x1, y0, y1
ANNIVERSARY_BUG_SHAPE = logo_match.LOGOS_DIR / "cw_20_years_bug_shape.png"
ANNIVERSARY_BUG_TEMPLATE = cv2.imread(str(ANNIVERSARY_BUG_SHAPE), cv2.IMREAD_GRAYSCALE)
ANNIVERSARY_BUG_REGION = (1610, 1900, 20, 160)
BUGS = (
    (BUG_TEMPLATE, BUG_REGION),
    (ANNIVERSARY_BUG_TEMPLATE, ANNIVERSARY_BUG_REGION),
)
BUG_THRESHOLD = 0.55

# A patch with almost no variation has nothing to correlate against, and
# TM_CCOEFF_NORMED divides by its variance: a flat corner can report anything
# up to a perfect 1.0. Below this standard deviation the patch scores 0.
MIN_PATCH_STD = 4.0

# --- Scoring ticker (bottom) -> content -------------------------------------
#
# The Charlotte ticker is four rows of opaque slate cells, so its color does
# not vary with the picture behind it. Color alone is not enough: counting
# slate pixels in the ticker's area scores real ticker frames 0.21-0.49 and
# dark commercials up to 0.26, and other networks' tickers up to 0.75.
#
# What nothing else has is the row structure. Just inside the top and bottom
# of each cell there is no text, so about 63% of the line is slate (the rest
# is the green name cells); the 3 px separator between rows has none. The
# score is the weakest cell line minus the strongest separator line, across
# the bottom two rows, which are the ones the two-row pre-race ticker also
# has. A slate-colored region with no rows scores 0.
#
#   Other networks (n=2929):                  max 0.02
#   Las Vegas (a different ticker, n=681):    max 0.00
#   Charlotte side-by-side breaks:            max 0.00
#   Charlotte, everything else:               233 of 1811 score >= 0.2
#
# Every Charlotte frame sampled between 0.05 and 0.45 was live coverage with
# the ticker partly covered by a lower third or an inset. Of the frames at or
# above the threshold, 41 had no bug match.
TICKER_COLOR = (59, 54, 45)  # BGR
TICKER_COLOR_TOLERANCE = 8
TICKER_X = (370, 1870)
TICKER_CELL_ROWS = (995, 996, 1020, 1021, 1029, 1030, 1054)
TICKER_SEPARATOR_ROWS = (990, 991, 1024, 1025)
TICKER_THRESHOLD = 0.2

# A side-by-side break has its own two-row ticker in the same slate, and it is
# a second `ad` signal beside the race panel's border. The border check needs
# a brightness step on all four sides of the panel, which a night race on the
# break's dark background does not give: there the panel check scores 0.0-0.8
# and the "cw SPORTS" bug, shown over that background, would win as content.
#
# The break's ticker sits 2 px lower than the live ticker's bottom two rows.
# That is enough to tell them apart, because the graphics are opaque and land
# on the same pixels every time: rows 1023-1024 are cell here and separator in
# the live ticker, and row 1028 is separator here and cell there. Each ticker
# scores below 0 on the other's rows.
#
#   Charlotte side-by-side frames (n=262):    min 0.29
#   Every other Charlotte frame (n=2961):     max 0.02
#   Other networks and Las Vegas (n=3610):    max 0.00
#
# All 148 Charlotte frames the panel check finds are among the 262. Of the 114
# it misses, 82 had a bug match.
SIDE_BY_SIDE_TICKER_CELL_ROWS = (997, 1023, 1024, 1031)
SIDE_BY_SIDE_TICKER_SEPARATOR_ROWS = (992, 1026, 1028)

# --- Audio sensor -> ad or content ------------------------------------------
#
# This is `nascar_on_nbc`'s model, loaded from that profile's file: no CW
# recording has been ruled, so there is nothing to fit one on. It is borrowed
# on the observation that CW's mix sounds like NBC's, engine roar under live
# racing against the flat mastering of national spots. See `nascar_on_nbc.py`
# for the model's known weakness, live content without pack roar.
#
# Checked against the Charlotte recording with the OpenCV verdicts above as
# proxy labels; the numbers are in AGENTS.md. Set ENABLE_AUDIO_CHECK to False
# if it misbehaves, and the profile runs as it would with no audio at all.
AUDIO_MODEL = audio_sensor.load_model(
    Path(__file__).parent / "audio_models" / "nascar_on_nbc.json"
)

# Toggles so a single check can be turned off mid-broadcast at the console.
ENABLE_SIDE_BY_SIDE_CHECK = True
ENABLE_BUG_CHECK = True
ENABLE_TICKER_CHECK = True
ENABLE_AUDIO_CHECK = True

PROMPT = llm_match.load_prompt("prompt_cw.txt")


def side_by_side_score(img: cv2.typing.MatLike) -> float:
    """Weakest of the four sides of the side-by-side race panel, 0 to 1.

    *img* must already be resized to 1920x1080.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return min(rectangle_match.border_edge_fractions(gray, SIDE_BY_SIDE_RACE_PANEL))


def is_side_by_side(
    img: cv2.typing.MatLike, threshold: float = SIDE_BY_SIDE_THRESHOLD
) -> bool:
    """True if the race panel's border or the break's own ticker is on screen."""
    return (
        side_by_side_score(img) >= threshold
        or side_by_side_ticker_score(img) >= TICKER_THRESHOLD
    )


def _shape_score(patch: cv2.typing.MatLike, template: cv2.typing.MatLike) -> float:
    """Best match for *template* in a single-channel *patch*; 0.0 if it is flat."""
    if patch.std() < MIN_PATCH_STD:
        return 0.0
    score = logo_match.match_template(patch, template).max_val
    return float(score) if math.isfinite(score) else 0.0


def _bug_score(
    img: cv2.typing.MatLike,
    template: cv2.typing.MatLike,
    region: tuple[int, int, int, int],
) -> float:
    x0, x1, y0, y1 = region
    corner = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    blue, green, red = (corner[:, :, c].astype(np.int16) for c in range(3))
    yellowness = np.clip(np.minimum(red, green) - blue, 0, 255).astype(np.uint8)
    redness = np.clip(red - np.maximum(green, blue), 0, 255).astype(np.uint8)
    return max(
        _shape_score(channel, template) for channel in (gray, yellowness, redness)
    )


def bug_score(img: cv2.typing.MatLike) -> float:
    """Best match score for either CW bug, in grey, yellow or red.

    *img* must already be resized to 1920x1080.
    """
    return max(_bug_score(img, template, region) for template, region in BUGS)


def has_bug(img: cv2.typing.MatLike, threshold: float = BUG_THRESHOLD) -> bool:
    return bug_score(img) >= threshold


def _ticker_rows_score(
    img: cv2.typing.MatLike, cell_rows: tuple[int, ...], separator_rows: tuple[int, ...]
) -> float:
    """Weakest slate fraction over *cell_rows* minus the strongest over *separator_rows*."""
    x0, x1 = TICKER_X

    def slate_fraction(row: int) -> float:
        line = img[row, x0:x1].astype(np.int16)
        off_color = np.abs(line - np.array(TICKER_COLOR, dtype=np.int16)).max(axis=1)
        return float((off_color <= TICKER_COLOR_TOLERANCE).mean())

    cells = min(slate_fraction(row) for row in cell_rows)
    separators = max(slate_fraction(row) for row in separator_rows)
    return cells - separators


def ticker_score(img: cv2.typing.MatLike) -> float:
    """How strongly the bottom of the frame shows the live scoring ticker's rows.

    *img* must already be resized to 1920x1080.
    """
    return _ticker_rows_score(img, TICKER_CELL_ROWS, TICKER_SEPARATOR_ROWS)


def side_by_side_ticker_score(img: cv2.typing.MatLike) -> float:
    """How strongly the bottom of the frame shows a side-by-side break's ticker.

    *img* must already be resized to 1920x1080.
    """
    return _ticker_rows_score(
        img, SIDE_BY_SIDE_TICKER_CELL_ROWS, SIDE_BY_SIDE_TICKER_SEPARATOR_ROWS
    )


def has_ticker(img: cv2.typing.MatLike, threshold: float = TICKER_THRESHOLD) -> bool:
    return ticker_score(img) >= threshold


def has_side_by_side_logo(img: cv2.typing.MatLike) -> bool:
    """The profile's OpenCV `ad` verdict, under the name `fit_audio_model.py` calls."""
    return is_side_by_side(img)


def has_network_logo(img: cv2.typing.MatLike) -> bool:
    """The profile's OpenCV `content` verdict, under the name `fit_audio_model.py` calls."""
    return has_bug(img) or has_ticker(img)


def classify_image(
    image_path: str, audio_bytes: bytes | None = None
) -> ClassificationResult:
    cv_img = cv2.imread(image_path)
    cv_img_1080p = cv2.resize(cv_img, (1920, 1080))

    # Read for every frame, whatever ends up deciding it, so the score is on
    # record alongside OpenCV and LLM verdicts too.
    reading = audio_sensor.sensor.reading(
        AUDIO_MODEL if ENABLE_AUDIO_CHECK else None, audio_bytes
    )
    signals = reading.signals()

    # Side-by-side runs first: the bug stays up through the break and would
    # otherwise win.
    if ENABLE_SIDE_BY_SIDE_CHECK and is_side_by_side(cv_img_1080p):
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="side_by_side",
            reply="CW side-by-side break (opencv)",
            signals=signals,
        )

    if ENABLE_BUG_CHECK and has_bug(cv_img_1080p):
        return ClassificationResult(
            source="opencv",
            type="content",
            reason="network_logo",
            reply="(opencv)",
            signals=signals,
        )

    if ENABLE_TICKER_CHECK and has_ticker(cv_img_1080p):
        return ClassificationResult(
            source="opencv",
            type="content",
            reason="scoring_ticker",
            reply="(opencv)",
            signals=signals,
        )

    # Audio runs after every OpenCV check: vision is the better authority at a
    # rejoin, where a sponsor read over live pictures still sounds like an ad.
    if ENABLE_AUDIO_CHECK and AUDIO_MODEL is not None and reading.p_ad is not None:
        if reading.p_ad >= AUDIO_MODEL.ad_threshold:
            return ClassificationResult(
                source="audio",
                type="ad",
                reason="audio_sensor",
                reply=f"audio p(ad)={reading.p_ad:.3f}",
                signals=signals,
            )
        if reading.p_ad <= AUDIO_MODEL.content_threshold:
            return ClassificationResult(
                source="audio",
                type="content",
                reason="audio_sensor",
                reply=f"audio p(ad)={reading.p_ad:.3f}",
                signals=signals,
            )

    # No verdict. A missing bug and ticker usually mean a full-screen
    # commercial, but the bug also ghosts out over bright backgrounds, so this
    # goes to the LLM instead of defaulting to ad.
    image_data = llm_match.load_image_b64(image_path)
    audio_data = llm_match.audio_b64(audio_bytes)
    result = llm_match.classify_by_prompt(image_data, audio_data, prompt=PROMPT)
    return dataclasses.replace(result, signals=signals)
