"""The segment-level ground truth for the Iowa Corn 350 capture.

`COARSE` holds the final, verified boundaries. Reaching them took three passes,
because the first two were each wrong in a way the other caught:

1. Contact sheets over the whole broadcast at one frame in twelve
   (`sheet.py --step 12`), read by eye. This finds every break, but places its
   edges only to about +/-30 frames, and snapping them to the nearest picture cut
   made it worse - inside a commercial break the biggest cut is usually the join
   between two spots, not the edge of the break.
2. Boundaries proposed independently from the furniture signal
   (`refit_truth.py`). Precise at edges, but it splits a break in two whenever a
   spot holds a static title card long enough to look pinned.
3. Every frame where those two disagreed - 386 frames in 31 runs - reviewed
   individually (`sheet_disputes.py`), then a final single-frame pass over the
   eight boundaries still ambiguous after that.

The two methods fail in opposite directions, which is what makes their
disagreement set worth reviewing and what makes this ground truth rather than a
heuristic. Frames in the interior of a segment are not individually reviewed:
the coarse sheets cover one in twelve and the OpenCV anchors cover 70% of the
rest.

`SNAP_WINDOW` is 0 because the boundaries below are already exact. The snapping
machinery is kept so the earlier pass can be reproduced.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from timeline import fmt, load  # noqa: E402

HERE = Path(__file__).parent

# (start_frame, label, kind); runs until the next entry. `nonstop` marks a
# NASCAR NON STOP side-by-side break - still an `ad` for switching purposes
# (that is what the production profile does), but a structurally different thing
# from a full break, and tracked separately.
#
# Every boundary below was checked by eye at single-frame resolution. Two
# conventions had to be fixed to make the edges well defined:
#
# - The full-screen NASCAR wipe that opens and closes a NON STOP break counts as
#   part of the break. That makes all six of them 60 frames, i.e. 120 s.
# - Live track or crowd carrying only a sponsor card ("IOWA CORN" over an aerial,
#   the Progressive fan-cam) counts as *content*: the broadcast has come back and
#   is showing the race. A produced sponsor spot with actors and no live footage
#   counts as `ad` even when the sponsor is the race's own title sponsor - that
#   is what the Iowa Corn spot at 1662 is.
COARSE = [
    (0, "content", ""),
    (35, "ad", "full"),
    (143, "content", ""),  # show open / Iowa Speedway package
    (213, "ad", "full"),  # 208-212 is the T-38 flyover, still content
    (302, "content", ""),
    (629, "ad", "full"),
    (688, "content", ""),
    (1088, "ad", "nonstop"),
    (1148, "content", ""),
    (1341, "ad", "nonstop"),
    (1401, "content", ""),
    (1532, "ad", "full"),
    (1615, "content", ""),
    (1662, "ad", "full"),
    (1754, "content", ""),
    (2028, "ad", "full"),
    (2108, "content", ""),
    (2238, "ad", "full"),
    (2316, "content", ""),
    (2593, "ad", "nonstop"),
    (2654, "content", ""),
    (2897, "ad", "full"),
    (2966, "content", ""),
    (3109, "ad", "full"),
    (3174, "content", ""),  # bug-less pylon mode from here - see notes
    (3467, "ad", "nonstop"),
    (3527, "content", ""),
    (3903, "ad", "full"),
    (3993, "content", ""),  # Progressive fan-cam, then racing
    (4045, "ad", "full"),
    (4125, "content", ""),
    (4399, "ad", "nonstop"),
    (4459, "content", ""),
    (4691, "ad", "nonstop"),
    (4751, "content", ""),
]

# Boundaries are already exact; snapping them again would only move them off.
SNAP_WINDOW = 0


def cut_score(rows, i):
    """How strong a picture cut sits between frame i-1 and frame i."""
    r = rows[i]
    s = 0.0
    if r.get("phash_dist_prev") is not None:
        s += r["phash_dist_prev"] / 32.0
    if r.get("hist_corr_prev") is not None:
        s += 1.0 - max(0.0, r["hist_corr_prev"])
    if r.get("mad_prev") is not None:
        s += min(1.0, r["mad_prev"] / 60.0)
    # A near-black frame either side of the join is the strongest cue there is.
    if r["black_frac"] > 0.9 or rows[i - 1]["black_frac"] > 0.9:
        s += 1.5
    return s


def snap(rows, i0):
    lo, hi = max(1, i0 - SNAP_WINDOW), min(len(rows) - 1, i0 + SNAP_WINDOW)
    best, best_s = i0, -1.0
    for i in range(lo, hi + 1):
        s = cut_score(rows, i)
        # Prefer the cut nearest the eyeballed position when scores are close.
        s -= 0.02 * abs(i - i0)
        if s > best_s:
            best, best_s = i, s
    return best, best_s


def build(rows):
    segs = []
    for k, (i0, label, kind) in enumerate(COARSE):
        start = 0 if k == 0 else snap(rows, i0)[0]
        segs.append({"start": start, "label": label, "kind": kind})
    for k, s in enumerate(segs):
        s["end"] = (segs[k + 1]["start"] - 1) if k + 1 < len(segs) else len(rows) - 1
        s["t0"] = rows[s["start"]]["t"]
        s["t1"] = rows[s["end"]]["t"]
        s["dur"] = s["t1"] - s["t0"] + 2.0
        s["n"] = s["end"] - s["start"] + 1
    return segs


def per_frame(rows, segs):
    y = [None] * len(rows)
    for s in segs:
        for i in range(s["start"], s["end"] + 1):
            y[i] = s["label"]
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "truth.json"))
    args = ap.parse_args()

    rows = load()
    segs = build(rows)
    print(f"{len(segs)} segments over {fmt(rows[-1]['t'])}\n")
    print(f"{'#':>3} {'start':>9} {'end':>9} {'dur':>8}  {'label':8} {'kind':8} frames")
    for k, s in enumerate(segs):
        print(
            f"{k:3d} {fmt(s['t0']):>9} {fmt(s['t1']):>9} {s['dur']:7.0f}s  "
            f"{s['label']:8} {s['kind']:8} {s['start']}-{s['end']}"
        )

    y = per_frame(rows, segs)
    n_ad = sum(1 for v in y if v == "ad")
    print(
        f"\nframes: {len(y)}  ad {n_ad} ({n_ad / len(y):.1%})  content {len(y) - n_ad}"
    )

    with open(args.out, "w") as f:
        json.dump(
            {
                "segments": segs,
                "labels": {rows[i]["filename"]: y[i] for i in range(len(rows))},
            },
            f,
        )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
