"""Is a USA logo in the LOWER right a usable positive signal for an ad?

The operator's observation: the live network bug sits in the UPPER right, so a
USA logo in the LOWER right belongs to promo furniture, not to coverage. That
would make it a rare thing in this pipeline - a cheap OpenCV check that votes
`ad`, where today every OpenCV check votes `content` except the side-by-side
banner.

Promos render the logo at a different size from the on-air bug, so this sweeps
scales rather than matching once.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
SERVER = Path("/home/joe/Code/projects/tv-commercial-detector/server")
sys.path.insert(0, str(SERVER / "src"))

from tv_commercial_detector.classification import logo_match  # noqa: E402

IMAGES = SERVER / "frames" / "images"
LOWER_RIGHT = (1600, 1920, 900, 1080)   # x0, x1, y0, y1 in 1920x1080
SCALES = (0.8, 1.0, 1.3, 1.6, 2.0, 2.5)


def best_score(img1080, template) -> float:
    x0, x1, y0, y1 = LOWER_RIGHT
    region = logo_match.mask_non_white(img1080[y0:y1, x0:x1].copy())
    frac = region.any(axis=2).mean()
    if not 0.01 <= frac <= 0.90:
        return 0.0
    best = 0.0
    for s in SCALES:
        h, w = template.shape[:2]
        t = cv2.resize(template, (int(w * s), int(h * s)))
        if t.shape[0] >= region.shape[0] or t.shape[1] >= region.shape[1]:
            continue
        v = logo_match.match_template(region, t).max_val
        if np.isfinite(v):
            best = max(best, float(v))
    return best


def main() -> None:
    rows = [json.loads(line) for line in open(HERE / "dataset.jsonl")]
    template = logo_match.load_masked(logo_match.LOGOS_DIR / "usa_network_logo.png")

    scores = {"ad": [], "content": []}
    for n, r in enumerate(rows):
        if r["gt"] == "uncertain":
            continue
        img = cv2.imread(str(IMAGES / r["filename"]))
        if img is None:
            continue
        s = best_score(cv2.resize(img, (1920, 1080)), template)
        scores[r["gt"]].append((s, r["filename"]))
        if n % 300 == 0:
            print(f"  {n}/{len(rows)}", file=sys.stderr, flush=True)

    for lab in ("ad", "content"):
        v = sorted(x[0] for x in scores[lab])
        print(f"{lab:8s} n={len(v):4d}  median {v[len(v)//2]:.3f} "
              f"p90 {v[int(.9*len(v))]:.3f} p99 {v[int(.99*len(v))]:.3f} max {v[-1]:.3f}")

    print("\nthreshold sweep (fires => predict ad):")
    print(f"{'thr':>5s} {'ad recall%':>11s} {'content FP%':>12s}")
    for thr in (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        tp = sum(1 for s, _ in scores["ad"] if s >= thr)
        fp = sum(1 for s, _ in scores["content"] if s >= thr)
        print(f"{thr:5.2f} {100*tp/len(scores['ad']):11.1f} "
              f"{100*fp/len(scores['content']):12.2f}")

    json.dump({k: v for k, v in scores.items()}, open(HERE / "usa_corner.json", "w"))


if __name__ == "__main__":
    main()
