"""Ground truth and dataset for the continuous 51-minute recording.

Labelling is anchored on the two OpenCV checks whose precision was verified by
eye on this very recording (48/48 network-bug frames are live coverage, 16/16
side-by-side frames are breaks), and every frame those two leave undecided was
reviewed on contact sheets. That is 378 frames of hand labelling instead of
1572, without leaning on the classifier's own opinion anywhere.

Anchoring on the same checks the classifier uses does make those frames
trivially easy for every policy, so accuracy is also reported over the
fall-through subset alone, which is where the decisions are actually made.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent

# Index ranges for the frames neither OpenCV check settles. Inclusive.
FALL_RANGES: list[tuple[int, int, str]] = [
    (35, 110, "ad"),        # Sonic, Applebee's, a Richmond promo, USA/E! promos
    (111, 184, "content"),  # pre-race open: the Kurt Warner feature into the grid
    (213, 274, "ad"),       # DraftKings, Applebee's, Straight Talk, Liberty, Sonic
    (275, 582, "content"),  # race-open tease into the green flag and live racing
    (629, 687, "ad"),       # Mint Mobile, Ashley, AWS, United, Wendy's, Safelite
    (785, 1087, "content"),
    (1088, 1088, "ad"),     # NASCAR wordmark card going into the break
    (1147, 1282, "content"),
    (1341, 1341, "ad"),
    (1400, 1400, "ad"),
    (1401, 1405, "content"),
    (1532, 1532, "content"),
    (1536, 1571, "ad"),     # Iowa Corn, Arby's, Liberty Mutual, Skyrizi
]


def label(row: dict) -> str | None:
    if row["sbs"] >= 0.8:
        return "ad"
    if row["usa"] >= 0.65 or row["peacock"] >= 0.55:
        return "content"
    for lo, hi, lab in FALL_RANGES:
        if lo <= row["i"] <= hi:
            return lab
    return None


def main() -> None:
    rows = [json.loads(line) for line in open(HERE / "cont.jsonl")]
    out, missing = [], []
    for r in rows:
        gt = label(r)
        if gt is None:
            missing.append(r["i"])
            continue
        r["gt"] = gt
        r["episode"] = 0          # one unbroken episode
        out.append(r)

    if missing:
        print(f"WARNING: {len(missing)} unlabelled fall-through frames: "
              f"{missing[:20]}{'...' if len(missing) > 20 else ''}", file=sys.stderr)

    with open(HERE / "cont_dataset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print(f"{len(out)} frames  {dict(Counter(r['gt'] for r in out))}")
    fall = [r for r in out if r["sbs"] < 0.8 and r["usa"] < 0.65 and r["peacock"] < 0.55]
    print(f"fall-through (reaches the LLM): {len(fall)} "
          f"{dict(Counter(r['gt'] for r in fall))}")
    changes = sum(1 for a, b in zip(out, out[1:]) if a["gt"] != b["gt"])
    print(f"ground-truth changes: {changes}")


if __name__ == "__main__":
    main()
