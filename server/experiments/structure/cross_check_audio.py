"""Does the audio model transfer to a different network?

The audio result is measured entirely on one USA/NBC broadcast, so the obvious
objection is that it has learned that production's music beds rather than
anything about commercials. This is the closest available test of that.

Train on the 2026-08-13 USA capture, with its hand-verified labels. Test on a
*Fox/FS1* broadcast from the burst archive, using the shipped OpenCV checks as
proxy labels - `network_logo` means content, `side_by_side` means ad. Those are
the two checks the Fox profile is most confident in, and they are independent of
audio, which is what makes them usable here.

Three things this cannot be:

- It is not ground truth. The Fox anchors have their own error rate, and nobody
  has reviewed these frames by eye.
- The anchored frames are a biased, *easy* subset by construction: they are
  exactly the frames where a logo was clearly visible. Frames the Fox profile
  finds hard are absent, which is the population that matters most.
- Only the instantaneous 4 s model can be tested. The archive is short bursts,
  not a timeline, so the 30 s trailing window - worth about a point of AUC on the
  continuous capture - is not computable.

So a good number here is encouraging and not conclusive; a bad number would be
decisive. Read it in that direction.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audio_probe import FEATURES, auc, best_acc, fit_logreg, predict  # noqa: E402
from extract_audio import features as audio_features  # noqa: E402
from ground_truth import build as build_segs  # noqa: E402
from timeline import load  # noqa: E402

HERE = Path(__file__).parent
FRAMES = Path(__file__).resolve().parents[2] / "frames"
TEST_DATE = "2026-05-17"
ANCHOR = {"network_logo": "content", "side_by_side": "ad"}


def train_matrix():
    rows = load()
    segs = build_segs(rows)
    aud = {
        r["filename"]: r
        for r in (json.loads(line) for line in open(HERE / "audio.jsonl"))
    }
    lab = {}
    for s in segs:
        for i in range(s["start"], s["end"] + 1):
            lab[i] = s["label"]
    X, y = [], []
    for i, r in enumerate(rows):
        a = aud.get(r["filename"])
        if not a or "error" in a:
            continue
        X.append([a[f] for f in FEATURES])
        y.append(1 if lab[i] == "ad" else 0)
    return np.array(X), np.array(y)


def test_matrix():
    recs = [
        json.loads(line)
        for line in open(FRAMES / "classifications.jsonl")
        if line.strip()
    ]
    have = set(os.listdir(FRAMES / "audio"))
    X, y, skipped = [], [], Counter()
    for r in recs:
        if not r["timestamp"].startswith(TEST_DATE):
            continue
        want = ANCHOR.get(r.get("classification_reason") or "")
        if want is None or r.get("classification") != want:
            continue
        wav = Path(r["filename"]).stem + ".wav"
        if wav not in have:
            skipped["no audio"] += 1
            continue
        f = audio_features(FRAMES / "audio" / wav)
        if "error" in f:
            skipped[f["error"]] += 1
            continue
        X.append([f[k] for k in FEATURES])
        y.append(1 if want == "ad" else 0)
    return np.array(X), np.array(y), skipped


def main():
    Xtr, ytr = train_matrix()
    print(f"train: {len(ytr)} USA/NBC clips ({ytr.sum()} ad)")
    Xte, yte, skipped = test_matrix()
    print(f"test:  {len(yte)} FS1 anchored clips ({yte.sum()} ad) on {TEST_DATE}")
    if skipped:
        print(f"       skipped {dict(skipped)}")
    if len(yte) < 50 or yte.sum() == 0 or yte.sum() == len(yte):
        print("not enough of both classes to score")
        return

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    w = fit_logreg((Xtr - mu) / sd, ytr)
    s = predict((Xte - mu) / sd, w)
    print(
        f"\ntransferred, instantaneous 4 s model:  AUC {auc(yte, s):.3f}  "
        f"acc {best_acc(yte, s):.3f}"
    )
    print(
        f"  accuracy at the untuned 0.5 threshold: "
        f"{(((s >= 0.5).astype(int)) == yte).mean():.3f}"
    )
    print(f"  majority-class baseline: {max(yte.mean(), 1 - yte.mean()):.3f}")

    # Same model architecture trained and tested within FS1, as a ceiling.
    n = len(yte)
    half = n // 2
    mu2, sd2 = Xte[:half].mean(0), Xte[:half].std(0) + 1e-9
    w2 = fit_logreg((Xte[:half] - mu2) / sd2, yte[:half])
    s2 = predict((Xte[half:] - mu2) / sd2, w2)
    if 0 < yte[half:].sum() < len(yte[half:]):
        print(
            f"\nfor reference, trained within FS1 (first half -> second): "
            f"AUC {auc(yte[half:], s2):.3f}"
        )
    print("\nSee the module docstring before quoting either number.")


if __name__ == "__main__":
    main()
