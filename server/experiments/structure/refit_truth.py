"""Propose segment boundaries from broadcast furniture, for eyeball verification.

The hand-typed `COARSE` list in ground_truth.py came off contact sheets sampled
one frame in twelve, so its boundaries are only good to about +/-30 frames, and
snapping them to the nearest picture cut sometimes locks onto a cut *inside* a
commercial instead of the one at its edge.

This proposes them from the signal instead. A national spot is the one thing in
the broadcast with no pinned graphics at all, so:

    break = a maximal run of frames with no furniture and no OpenCV anchor,
            at least MIN_BREAK frames long, with runs separated by less than
            MERGE_GAP frames of furniture joined together

The gap merge matters because a spot can hold a static title card long enough to
look pinned for a few frames.

NASCAR NON STOP breaks are handled separately and need none of this: they are
bounded by a full-screen NASCAR wipe and the banner check finds them exactly.

This is a *proposal*. Every boundary it emits is then checked by eye
(`verify_boundaries.py`) before it becomes ground truth - the feature that
proposes a boundary must not also be the only thing certifying it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from timeline import PEACOCK_TH, SBS_TH, USA_TH, fmt, load  # noqa: E402

HERE = Path(__file__).parent

FURNITURE_TH = 0.004  # fs_all; content p25 = 0.006, ad-full p75 = 0.002
MIN_BREAK = 12  # 24 s - shorter than any real break here
MERGE_GAP = 6  # 12 s of apparent furniture inside a break is a title card


def load_furniture():
    return {
        r["filename"]: r
        for r in (json.loads(line) for line in open(HERE / "furniture.jsonl"))
    }


def propose(rows, fur):
    bare = []
    for i, r in enumerate(rows):
        f = fur.get(r["filename"], {})
        fs = f.get("fs_all")
        anchored = (
            r["peacock"] >= PEACOCK_TH or r["usa"] >= USA_TH or r["sbs"] >= SBS_TH
        )
        bare.append(not anchored and (fs is None or fs < FURNITURE_TH))

    runs, i = [], 0
    while i < len(bare):
        if not bare[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(bare) and bare[j + 1]:
            j += 1
        runs.append([i, j])
        i = j + 1

    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] - 1 <= MERGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged if r[1] - r[0] + 1 >= MIN_BREAK]


def nonstop_runs(rows):
    runs, i = [], 0
    while i < len(rows):
        if rows[i]["sbs"] < SBS_TH:
            i += 1
            continue
        j = i
        # Allow a couple of frames of banner dropout inside a break.
        while j + 1 < len(rows) and any(
            rows[k]["sbs"] >= SBS_TH for k in range(j + 1, min(j + 4, len(rows)))
        ):
            j += 1
        runs.append((i, j))
        i = j + 1
    return [r for r in runs if r[1] - r[0] + 1 >= 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "proposed.json"))
    args = ap.parse_args()

    rows = load()
    fur = load_furniture()
    full = propose(rows, fur)
    ns = nonstop_runs(rows)

    breaks = [{"start": a, "end": b, "kind": "full"} for a, b in full]
    breaks += [{"start": a, "end": b, "kind": "nonstop"} for a, b in ns]
    breaks.sort(key=lambda s: s["start"])
    # A full-break run that overlaps a NON STOP run is the same event seen twice.
    out = []
    for b in breaks:
        if out and b["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], b["end"])
            continue
        out.append(b)

    print(f"{len(out)} proposed breaks")
    print(f"{'#':>3} {'start':>9} {'end':>9} {'dur':>7}  kind      frames")
    for k, b in enumerate(out):
        d = rows[b["end"]]["t"] - rows[b["start"]]["t"] + 2
        print(
            f"{k:3d} {fmt(rows[b['start']]['t']):>9} {fmt(rows[b['end']]['t']):>9} "
            f"{d:6.0f}s  {b['kind']:9} {b['start']}-{b['end']}"
        )
        b["dur"] = d
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
