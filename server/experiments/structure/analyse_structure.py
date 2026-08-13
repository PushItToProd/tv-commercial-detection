"""What shape is the broadcast, and which of those regularities are usable?

Everything here is measured against the verified segment list in
ground_truth.py. The point is to find constraints strong enough to drive a state
machine - a duration a break essentially always has, a gap it essentially never
violates - rather than to describe the broadcast for its own sake.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import build  # noqa: E402
from timeline import fmt, load  # noqa: E402

HERE = Path(__file__).parent


def section(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


def main():
    rows = load()
    segs = build(rows)
    aud = {
        r["filename"]: r
        for r in (json.loads(line) for line in open(HERE / "audio.jsonl"))
    }

    ads = [s for s in segs if s["label"] == "ad"]
    con = [s for s in segs if s["label"] == "content"]
    ns = [s for s in ads if s["kind"] == "nonstop"]
    full = [s for s in ads if s["kind"] == "full"]

    section("SEGMENT DURATIONS")
    for name, group in (("NON STOP break", ns), ("full break", full), ("content", con)):
        d = np.array([s["dur"] for s in group])
        print(
            f"{name:16s} n={len(d):2d}  min={d.min():5.0f}  med={np.median(d):5.0f}  "
            f"max={d.max():5.0f}  mean={d.mean():5.0f}  sd={d.std():5.1f}"
        )
        print(f"{'':16s} all: {sorted(int(x) for x in d)}")

    section("AD LOAD AND CADENCE")
    total = rows[-1]["t"]
    ad_t = sum(s["dur"] for s in ads)
    print(
        f"broadcast {fmt(total)}   ad {ad_t / 60:.1f} min ({ad_t / total:.1%})   "
        f"{len(ads)} breaks, one every {total / len(ads) / 60:.1f} min"
    )
    starts = [s["t0"] for s in ads]
    gaps = np.diff(starts)
    print(
        f"break-to-break start gap: min={gaps.min():.0f}s med={np.median(gaps):.0f}s "
        f"max={gaps.max():.0f}s"
    )
    print(
        "kind sequence:", " ".join("N" if s["kind"] == "nonstop" else "F" for s in ads)
    )

    section("THE CONSTRAINT A STATE MACHINE COULD USE")
    dmin_ad = min(s["dur"] for s in ads)
    dmin_con = min(s["dur"] for s in con[1:-1])  # first/last are truncated
    print(f"shortest break of any kind      {dmin_ad:5.0f}s")
    print(f"shortest interior content run   {dmin_con:5.0f}s")
    print(
        f"NON STOP breaks are {min(s['dur'] for s in ns):.0f}-{max(s['dur'] for s in ns):.0f}s "
        f"(n={len(ns)}) - effectively a fixed 120 s"
    )
    print("\nso, once a state is entered, a change of mind inside these windows is")
    print("noise by construction, not evidence.")

    section("SPOT GRID INSIDE FULL BREAKS")
    # A national break is a train of 15/30/60 s spots joined by a black frame
    # and a moment of near-silence. Both are visible here.
    for s in full:
        blacks, quiets = [], []
        for i in range(s["start"], s["end"] + 1):
            if rows[i]["black_frac"] > 0.90:
                blacks.append(round(rows[i]["t"] - s["t0"]))
            a = aud.get(rows[i]["filename"])
            if a and "error" not in a and a["min_rms_db"] < -55:
                quiets.append(round(rows[i]["t"] - s["t0"]))
        print(
            f"  break @{fmt(s['t0'])} {s['dur']:4.0f}s  black at {blacks}  "
            f"quiet at {quiets}"
        )

    section("BLACK FRAMES: WHERE DO THEY FALL AT ALL?")
    y = {}
    for s in segs:
        for i in range(s["start"], s["end"] + 1):
            y[i] = s["label"]
    bl = [i for i, r in enumerate(rows) if r["black_frac"] > 0.90]
    print(f"{len(bl)} near-black frames  ->  {Counter(y[i] for i in bl)}")
    near = sum(
        1
        for i in bl
        if any(abs(i - s["start"]) <= 3 or abs(i - s["end"]) <= 3 for s in segs)
    )
    print(f"{near} of them are within 3 frames of a segment edge")


if __name__ == "__main__":
    main()
