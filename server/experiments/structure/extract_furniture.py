"""Template-free detection of persistent broadcast furniture.

The premise: live coverage is *pinned* under graphics - a running-order pylon, a
corner bug, a lower third - that hold still while the picture behind them moves.
A national commercial has none of that. So instead of matching a known logo,
measure how much of the frame is edge that has not moved for several seconds.

That is the part of this work that should transfer. A logo template only knows
the broadcast it was cut from; "something is pinned to the screen" is true of
essentially every live sports production, whatever network it is on.

For each frame the edge map of the last `W` frames is intersected: a pixel
counts as furniture if it is an edge in *every* frame of the window. Two window
lengths are kept because they fail differently - a short window is fooled by a
held title card inside a spot, a long window smears across the boundary.

Region breakdown (fractions of the 1920x1080 frame):

    left    x 0.00-0.22            the running-order pylon
    topl    x 0.00-0.30, y 0-0.14  the "NASCAR CUP SERIES" header bar
    topr    x 0.85-1.00, y 0-0.16  where the network bug sits
    lower   y 0.78-1.00            lower thirds and the ticker
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

W_SHORT, W_LONG = 4, 12
SIZE = (480, 270)

REGIONS = {
    "left": (0.00, 0.22, 0.10, 0.92),
    "topl": (0.00, 0.30, 0.00, 0.14),
    "topr": (0.85, 1.00, 0.00, 0.16),
    "lower": (0.00, 1.00, 0.78, 1.00),
    "all": (0.00, 1.00, 0.00, 1.00),
}


def region_slices(shape):
    h, w = shape
    out = {}
    for name, (x0, x1, y0, y1) in REGIONS.items():
        out[name] = (slice(int(y0 * h), int(y1 * h)), slice(int(x0 * w), int(x1 * w)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.dir)
    recs = [
        json.loads(line)
        for line in open(root / "classifications.jsonl")
        if line.strip()
    ]
    recs.sort(key=lambda r: r["timestamp"])
    images = root / "images"

    hist: deque = deque(maxlen=W_LONG)
    sl = region_slices((SIZE[1], SIZE[0]))
    out = []
    for k, r in enumerate(recs):
        path = images / r["filename"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(cv2.resize(img, SIZE), cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200) > 0
        hist.append(edges)

        row = {"filename": r["filename"]}
        for tag, wlen in (("s", W_SHORT), ("l", W_LONG)):
            win = list(hist)[-wlen:]
            if len(win) < wlen:
                pers = None
            else:
                pers = np.logical_and.reduce(win)
            for name, (ys, xs) in sl.items():
                row[f"f{tag}_{name}"] = (
                    None if pers is None else float(pers[ys, xs].mean())
                )
        row["edge_all"] = float(edges.mean())
        out.append(row)
        if k % 500 == 0:
            print(f"  {k}/{len(recs)}", file=sys.stderr, flush=True)

    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
