"""Per-frame signals for a continuous `record_broadcast.py` capture.

Same signals as extract_features.py, but for a recording that has no
classification metadata of its own and no gaps - so the whole thing is one
episode and long-horizon temporal policies can finally be exercised.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

SERVER = Path("/home/joe/Code/projects/tv-commercial-detector/server")
sys.path.insert(0, str(SERVER / "src"))

from tv_commercial_detector.classifiers import nascar_on_nbc as nbc  # noqa: E402

SMALL = (64, 36)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.dir)
    recs = [json.loads(line) for line in open(root / "classifications.jsonl") if line.strip()]
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, SMALL).astype(np.int16)
        ph = imagehash.phash(Image.fromarray(gray))
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(hist, hist)

        out.append({
            "i": i,
            "filename": r["filename"],
            "timestamp": r["timestamp"],
            "video_offset": r.get("video_offset"),
            "video_title": r.get("video_title"),
            "peacock": float(nbc.peacock_score(img1080)),
            "usa": float(nbc.usa_score(img1080)),
            "sbs": float(nbc.side_by_side_score(img1080)),
            "phash": str(ph),
            "mad_prev": None if prev_small is None else float(np.abs(small - prev_small).mean()),
            "hist_corr_prev": None if prev_hist is None else float(
                cv2.compareHist(hist, prev_hist, cv2.HISTCMP_CORREL)),
            "phash_dist_prev": None if prev_hash is None else int(ph - prev_hash),
        })
        prev_hash, prev_small, prev_hist = ph, small, hist
        if i % 250 == 0:
            print(f"  {i}/{len(recs)}", file=sys.stderr, flush=True)

    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
