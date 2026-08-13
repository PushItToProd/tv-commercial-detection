"""Per-frame visual signals for a continuous `record_broadcast.py` capture.

Superset of `hysteresis/extract_cont.py`. The extra signals are the ones that
describe *broadcast structure* rather than the content of a single frame:

- `letterbox_top/bottom` - height of the black bars. Produced montages and many
  national spots are letterboxed; live coverage never is.
- `black_frac`, `is_black` - a nearly-black frame is the join between two
  commercial spots, which is what makes the 15/30/60 s spot grid visible.
- `ticker_edges` - edge density in the bottom strip where the leaderboard sits.
  The leaderboard is on for essentially all live coverage and never during a
  national spot.
- `edge_density`, `sat_mean` - coarse scene descriptors; ads are more saturated
  and more graphic than a track shot.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

SERVER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SERVER / "src"))

from tv_commercial_detector.classifiers import nascar_on_nbc as nbc  # noqa: E402

SMALL = (64, 36)
BLACK_LEVEL = 24  # 0-255; below this a pixel counts as black


def letterbox_bars(gray1080: np.ndarray) -> tuple[int, int]:
    """Height in pixels of the black bars at top and bottom of a 1080p frame."""
    rowmax = gray1080.max(axis=1)
    dark = rowmax < 40
    top = 0
    while top < 540 and dark[top]:
        top += 1
    bottom = 0
    while bottom < 540 and dark[1079 - bottom]:
        bottom += 1
    return top, bottom


def main() -> None:
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

    prev_hash = prev_small = prev_hist = None
    out = []
    for i, r in enumerate(recs):
        path = images / r["filename"]
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        img1080 = cv2.resize(img, (1920, 1080))
        gray1080 = cv2.cvtColor(img1080, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, SMALL).astype(np.int16)
        ph = imagehash.phash(Image.fromarray(gray))
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        top, bot = letterbox_bars(gray1080)
        # Interior = the frame minus any letterbox, so brightness stats describe
        # the picture rather than the bars.
        interior = gray1080[top : 1080 - bot] if top + bot < 1000 else gray1080
        black_frac = float((gray1080 < BLACK_LEVEL).mean())

        # Leaderboard strip: full width, bottom ~12% of the active picture.
        y1 = 1080 - bot
        y0 = max(0, y1 - 130)
        ticker = gray1080[y0:y1]
        ticker_edges = float((cv2.Canny(ticker, 80, 200) > 0).mean())

        hsv = cv2.cvtColor(cv2.resize(img1080, (480, 270)), cv2.COLOR_BGR2HSV)

        out.append(
            {
                "i": i,
                "filename": r["filename"],
                "timestamp": r["timestamp"],
                "video_offset": r.get("video_offset"),
                "video_title": r.get("video_title"),
                "peacock": float(nbc.peacock_score(img1080)),
                "usa": float(nbc.usa_score(img1080)),
                "sbs": float(nbc.side_by_side_score(img1080)),
                "phash": str(ph),
                "mad_prev": None
                if prev_small is None
                else float(np.abs(small - prev_small).mean()),
                "hist_corr_prev": None
                if prev_hist is None
                else float(cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)),
                "phash_dist_prev": None if prev_hash is None else int(ph - prev_hash),
                "letterbox_top": top,
                "letterbox_bottom": bot,
                "black_frac": black_frac,
                "is_black": bool(black_frac > 0.98),
                "mean_lum": float(interior.mean()),
                "ticker_edges": ticker_edges,
                "edge_density": float((cv2.Canny(gray1080, 80, 200) > 0).mean()),
                "sat_mean": float(hsv[:, :, 1].mean()),
            }
        )
        prev_hash, prev_small, prev_hist = ph, small, hist
        if i % 500 == 0:
            print(f"  {i}/{len(recs)}", file=sys.stderr, flush=True)

    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
