"""Extract per-frame signals for the temporal-classifier experiment.

Emits one JSON record per frame of a chosen capture day: the recorded live
verdict, every OpenCV score the nascar_on_nbc profile computes, a perceptual
hash, and cheap frame-to-frame difference metrics. Everything downstream
(hysteresis simulation, scene segmentation, near-duplicate caching) reads this
table instead of touching the JPEGs again.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

SERVER = Path("/home/joe/Code/projects/tv-commercial-detector/server")
sys.path.insert(0, str(SERVER / "src"))

from tv_commercial_detector.classifiers import nascar_on_nbc as nbc  # noqa: E402

FRAMES = SERVER / "frames"
IMAGES = FRAMES / "images"

# Downsampled grayscale size used for the cheap inter-frame metrics. Small
# enough that the diff costs microseconds, large enough to survive JPEG noise.
SMALL = (64, 36)


def load_records(day: str) -> list[dict]:
    recs = {}
    with (FRAMES / "classifications.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["timestamp"].startswith(day):
                # Later records win: a frame re-saved under manual_report
                # carries the operator's correction.
                prev = recs.get(r["filename"])
                if prev is not None:
                    prev.update({k: v for k, v in r.items() if v is not None})
                else:
                    recs[r["filename"]] = r
    return sorted(recs.values(), key=lambda r: (r["timestamp"], r["filename"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-08-09")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = load_records(args.day)
    print(f"{len(recs)} records for {args.day}", file=sys.stderr)

    prev_small = None
    prev_hash = None
    out = []
    t_start = time.perf_counter()

    for i, r in enumerate(recs):
        path = IMAGES / r["filename"]
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue

        t0 = time.perf_counter()
        img1080 = cv2.resize(img, (1920, 1080))
        t_resize = time.perf_counter() - t0

        t0 = time.perf_counter()
        peacock = float(nbc.peacock_score(img1080))
        t_peacock = time.perf_counter() - t0

        t0 = time.perf_counter()
        usa = float(nbc.usa_score(img1080))
        t_usa = time.perf_counter() - t0

        t0 = time.perf_counter()
        sbs = float(nbc.side_by_side_score(img1080))
        t_sbs = time.perf_counter() - t0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, SMALL).astype(np.int16)

        t0 = time.perf_counter()
        ph = imagehash.phash(Image.fromarray(gray))
        t_phash = time.perf_counter() - t0

        if prev_small is None:
            mad = None
            hist_corr = None
            hamming = None
        else:
            mad = float(np.abs(small - prev_small).mean())
            h1 = cv2.calcHist([gray], [0], None, [64], [0, 256])
            h2 = prev_hist
            cv2.normalize(h1, h1)
            hist_corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
            hamming = int(ph - prev_hash)

        h = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(h, h)
        prev_hist = h
        prev_small = small
        prev_hash = ph

        out.append({
            "i": i,
            "filename": r["filename"],
            "timestamp": r["timestamp"],
            "video_offset": r.get("video_offset"),
            "video_title": r.get("video_title"),
            "save_reason": r.get("save_reason"),
            "live_class": r.get("classification"),
            "live_reason": r.get("classification_reason"),
            "state_class": r.get("state_classification"),
            "correct_label": r.get("correct_label"),
            "model_reply": (r.get("model_reply") or "")[:400],
            "peacock": peacock,
            "usa": usa,
            "sbs": sbs,
            "phash": str(ph),
            "mad_prev": mad,
            "hist_corr_prev": hist_corr,
            "phash_dist_prev": hamming,
            "t_resize": t_resize,
            "t_peacock": t_peacock,
            "t_usa": t_usa,
            "t_sbs": t_sbs,
            "t_phash": t_phash,
        })

        if i % 200 == 0:
            print(f"  {i}/{len(recs)}  {time.perf_counter() - t_start:.0f}s", file=sys.stderr)

    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
