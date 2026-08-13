"""Would the furniture signal have caught the operator's 349 reported errors?

`frames/incorrect_labels.json` is the record of every time the operator hit
"report wrong" - the classifier's own failure set, in its own words. That makes
it the most adversarial evaluation available here: these are precisely the
frames the shipped pipeline got wrong, on a network (Fox, March 2026) that
nothing in this experiment was fitted to.

Entries where `classified_as` equals `correct_label` are skipped; those record a
disagreement with the *state* rather than with the frame, and are not per-frame
errors.

No audio: the March captures predate audio collection entirely, so this tests
the furniture signal alone - which is also the only one of the two new signals
that could work here at all, since a 4-frame burst is enough for a persistence
window and nothing else.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audio_probe import auc, best_acc, blocked_cv  # noqa: E402
from cross_check_frames import FRAMES, MAX_SPAN, W, parse_time  # noqa: E402
from extract_furniture import REGIONS, SIZE, region_slices  # noqa: E402


def features_for(names, timed, order, images, sl):
    out = {}
    for name in names:
        i = order.get(name)
        if i is None or i < W - 1:
            continue
        win = timed[i - W + 1 : i + 1]
        if (win[-1][0] - win[0][0]).total_seconds() > MAX_SPAN:
            continue
        edges, ok = [], True
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
        out[name] = [pers[ys, xs].mean() for ys, xs in (sl[k] for k in REGIONS)] + [
            edges[-1].mean()
        ]
    return out


def main():
    inc = json.load(open(FRAMES / "incorrect_labels.json"))
    real = {k: v for k, v in inc.items() if v["classified_as"] != v["correct_label"]}
    print(f"{len(inc)} reports, {len(real)} of them per-frame errors")
    print(Counter((v["classified_as"], v["correct_label"]) for v in real.values()))

    images = FRAMES / "images"
    names_all = os.listdir(images)
    timed = sorted(
        ((t, n) for t, n in ((parse_time(n), n) for n in names_all) if t),
        key=lambda t: t[0],
    )
    order = {n: i for i, (_, n) in enumerate(timed)}
    sl = region_slices((SIZE[1], SIZE[0]))

    feats = features_for(list(real), timed, order, images, sl)
    print(f"{len(feats)} of them have a usable burst window\n")
    if not feats:
        return

    names = sorted(feats)
    X = np.array([feats[n] for n in names])
    y = np.array([1 if real[n]["correct_label"] == "ad" else 0 for n in names])
    g = np.array([n[:10] for n in names])
    print(f"  {y.sum()} truly ad / {len(y) - y.sum()} truly content")

    cols = list(REGIONS) + ["edge_all"]
    print("\nsingle features on the error set:")
    for j, c in enumerate(cols):
        print(f"  {c:9s} AUC {auc(y, X[:, j]):.3f}")

    s, _ = blocked_cv(X, y, g, folds=min(4, len(set(g))))
    a, acc = auc(y, s), best_acc(y, s)
    print(f"\nfused, blocked by capture date: AUC {a:.3f}  acc {acc:.3f}")
    base = max(y.mean(), 1 - y.mean())
    print(f"majority-class baseline on this set: {base:.3f}")
    print(
        f"\nThe shipped pipeline scores 0.000 on these by construction - every one\n"
        f"is a frame it got wrong. Anything above {base:.3f} is signal it did not have."
    )


if __name__ == "__main__":
    main()
