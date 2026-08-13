"""Burst-level ground truth for the 2026-08-09 Iowa Corn 350 race window.

Assigned by reviewing contact sheets of one representative frame per burst
(`contact_sheet.py --mode burst`). Bursts are ~5 frames over ~8 s, so a single
label per burst holds except where a break boundary falls inside the window;
those are re-checked frame by frame in `refine_labels.py`.

Labelling rule, chosen to match what the operator actually wants the matrix to
do (see notes/misclassification-analysis-2026-08.md):

  ad      commercial-break material - third-party spots, network promos and
          bumpers aired inside a break, and NASCAR NON STOP side-by-side
          breaks where the race survives only as a small inset.
  content the broadcast's own programming - racing, studio, pit reporters,
          driver intros, victory lane, and sponsored squeezebacks that keep a
          large live race window.

UNCERTAIN holds the bursts where that rule genuinely does not decide: network
bumper cards and sponsor billboards that sit on the boundary between
programming and break. They are excluded from scoring rather than guessed at,
because a coin-flip label there would be indistinguishable from classifier
error.
"""

# Bursts showing commercial-break material.
AD_BURSTS = {
    0, 5, 6, 18, 19,
    34, 35, 36, 43, 44,
    51, 52, 56, 57, 58, 68, 69, 70,
    76, 77, 78, 88, 89, 90,
    99, 100, 107, 118, 119,
    134, 135, 138,
    150, 151, 159, 160, 161,
    185, 186,
    204,
}

# Genuinely undecidable under the rule above; excluded from scoring.
UNCERTAIN_BURSTS = {
    1,    # 19:31 Northern Iowa football graphic - broadcast bio package or promo
    50,   # 20:18 Ryan Blaney USA card - bumper, may sit either side of the break
    71,   # 20:37 Cheddar's billboard over a live crowd shot
    101,  # 21:06 aerial + IOWA CORN 350 bug, returning from break
    108,  # 21:13 Iowa Lottery "Chase to $1 million" promo over live midway b-roll
    133,  # 21:38 Christopher Bell USA card - bumper into the break
    137,  # 21:42 NASCAR CUP SERIES / USA SPORTS title card
    139,  # 21:44 graded crowd shot with USA bug
    205,  # 22:47 aerial of the speedway, returning from the F1 promo
}

# Position of the USA logo is the tell on network-promo material: the live bug
# sits in the UPPER right, so a USA logo in the LOWER right means a promo. That
# is what burst 0 is - an ad for USA's wrestling programming, not the driver
# intros it resembles at thumbnail size.
#
# Bursts the live classifier got wrong, kept here only as a review note:
#   114 21:19 Credit One squeezeback, large live race inset -> content, live said ad
#         (this is the burst the analysis flagged as the systematic error)
#   1595-1603 the Cheddar's side-by-side, same shape as the Credit One case


def label_for(burst_index: int) -> str:
    if burst_index in UNCERTAIN_BURSTS:
        return "uncertain"
    return "ad" if burst_index in AD_BURSTS else "content"
