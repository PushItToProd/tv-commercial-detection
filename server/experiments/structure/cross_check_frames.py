"""Does the template-free furniture signal transfer to a different broadcast?

`server/frames/` is the burst archive, and almost all of its 462 manual labels
are from March 2026 - that is *NASCAR on Fox*, a different network with a
different graphics package from the USA/NBC capture everything else here is
measured on. Nothing about the furniture detector is fitted to either: it has no
templates and no thresholds learned from Fox.

So this is the honest generalisation question. If "how much of the frame has not
moved for ten seconds" separates ad from content on Fox as well, the signal is
about live sports production in general rather than about one graphics package,
and it is worth carrying into profiles that do not yet have good OpenCV checks.

The archive is bursts, not a timeline: frames come in short runs about 4 s apart
with an incrementing suffix. A burst is long enough for the 4-frame window, and
that is all this needs - the 12-frame window is not computable here, and neither
is the audio model (the March frames predate audio capture entirely).
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audio_probe import auc, best_acc, blocked_cv  # noqa: E402
from extract_furniture import REGIONS, SIZE, region_slices  # noqa: E402

FRAMES = Path(__file__).resolve().parents[2] / "frames"
W = 4
MAX_SPAN = 20.0  # seconds; a wider gap means these frames are not one burst

NAME_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})-(\d+)_(\d+)\.(png|jpg)$"
)


def parse_time(name):
    m = NAME_RE.match(name)
    if not m:
        return None
    d, hh, mm, ss, us, _, _ = m.groups()
    return datetime.fromisoformat(f"{d}T{hh}:{mm}:{ss}.{us[:6].ljust(6, '0')}")


def main():
    labels = {
        k: v
        for k, v in json.load(open(FRAMES / "labels.json")).items()
        if v in ("ad", "content")
    }
    images = FRAMES / "images"
    names = os.listdir(images)
    timed = sorted(
        ((t, n) for t, n in ((parse_time(n), n) for n in names) if t),
        key=lambda t: t[0],
    )
    order = {n: i for i, (_, n) in enumerate(timed)}

    sl = region_slices((SIZE[1], SIZE[0]))
    feats, y, groups, used = [], [], [], []
    for name, lab in labels.items():
        i = order.get(name)
        if i is None or i < W - 1:
            continue
        win = timed[i - W + 1 : i + 1]
        if (win[-1][0] - win[0][0]).total_seconds() > MAX_SPAN:
            continue
        edges = []
        ok = True
        for _, n in win:
            img = cv2.imread(str(images / n))
            if img is None:
                ok = False
                break
            g = cv2.cvtColor(cv2.resize(img, SIZE), cv2.COLOR_BGR2GRAY)
            edges.append(cv2.Canny(g, 80, 200) > 0)
        if not ok:
            continue
        pers = np.logical_and.reduce(edges)
        row = [pers[ys, xs].mean() for ys, xs in (sl[k] for k in REGIONS)] + [
            edges[-1].mean()
        ]
        feats.append(row)
        y.append(1 if lab == "ad" else 0)
        groups.append(name[:10])  # block by capture date
        used.append(name)

    X = np.array(feats)
    y = np.array(y)
    g = np.array(groups)
    print(
        f"{len(y)} labelled frames usable  ({y.sum()} ad / {len(y) - y.sum()} content)"
    )
    print(f"dates: {sorted(set(g))}\n")

    cols = list(REGIONS) + ["edge_all"]
    print("single features:")
    for j, c in enumerate(cols):
        print(
            f"  {c:9s} AUC {auc(y, X[:, j]):.3f}   "
            f"ad median {np.median(X[y == 1, j]):.4f}  "
            f"content median {np.median(X[y == 0, j]):.4f}"
        )

    s, _ = blocked_cv(X, y, g, folds=min(4, len(set(g))))
    print(
        f"\nfused, blocked by capture date: AUC {auc(y, s):.3f}  acc {best_acc(y, s):.3f}"
    )

    # What the shipped Fox classifier recorded for these same frames, where known.
    recs = {}
    for line in open(FRAMES / "classifications.jsonl"):
        if line.strip():
            r = json.loads(line)
            recs[r["filename"]] = r
    both = [(n, labels[n], recs[n]["classification"]) for n in used if n in recs]
    if both:
        agree = sum(1 for _, a, b in both if a == b)
        print(
            f"\npipeline agreement on the {len(both)} of these with a stored verdict: "
            f"{agree}/{len(both)}"
        )


if __name__ == "__main__":
    main()
