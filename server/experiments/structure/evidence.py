"""Per-frame evidence, as out-of-fold probabilities.

Keeps two kinds of evidence apart, because they behave differently:

*Anchors* are the existing rule-based OpenCV checks. They are deterministic, they
fire on a minority of frames, and when they fire they are almost always right.

*Sensors* are the continuous features - furniture persistence and audio. These
are fused by an L2 logistic regression whose predictions are produced by
segment-blocked cross-validation, so no frame is ever scored by a model that saw
its own segment. Using in-sample scores here would flatter every policy built on
top of them by several points.

`build_evidence` returns one row per frame: the anchor verdict (or None) and
p(ad) from each sensor set.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from audio_probe import FEATURES as AUDIO_FEATURES  # noqa: E402
from audio_probe import blocked_cv  # noqa: E402
from ground_truth import build as build_segs  # noqa: E402
from timeline import cv_verdict, load  # noqa: E402

HERE = Path(__file__).parent

FURN_FEATURES = [
    "fs_all",
    "fs_left",
    "fs_topl",
    "fs_topr",
    "fs_lower",
    "fl_all",
    "fl_left",
    "fl_topl",
    "edge_all",
]
VIS_FEATURES = [
    "black_frac",
    "mean_lum",
    "ticker_edges",
    "edge_density",
    "sat_mean",
    "letterbox_top",
    "letterbox_bottom",
]
WIN = 15


def _trailing(X, win=WIN):
    """Causal feature block: where we are, and how that differs from recently.

    The trailing mean and sd say what the last ~30 s sounded and looked like,
    which is what a break *is*. But a block of only trailing statistics is slow
    off the mark at a break's first frame, because 29 of its 30 seconds are
    still content - and that lag is paid in commercials left on screen.

    So the raw current frame goes in as well, along with its difference from the
    trailing mean. The difference is the change detector: a commercial starting
    is a step in exactly these features, and it is visible on the first frame
    even though the window average has barely moved.
    """
    out = np.zeros((len(X), X.shape[1] * 4))
    d = X.shape[1]
    for k in range(len(X)):
        blk = X[max(0, k - win + 1) : k + 1]
        m, s = blk.mean(0), blk.std(0)
        out[k, :d] = m
        out[k, d : 2 * d] = s
        out[k, 2 * d : 3 * d] = X[k]
        out[k, 3 * d :] = X[k] - m
    return out


def build_evidence():
    rows = load()
    segs = build_segs(rows)
    aud = {
        r["filename"]: r
        for r in (json.loads(line) for line in open(HERE / "audio.jsonl"))
    }
    fur = {
        r["filename"]: r
        for r in (json.loads(line) for line in open(HERE / "furniture.jsonl"))
    }

    seg_of, lab = {}, {}
    for k, s in enumerate(segs):
        for i in range(s["start"], s["end"] + 1):
            seg_of[i], lab[i] = k, s["label"]

    def val(d, keys):
        return [0.0 if d.get(k) is None else float(d[k]) for k in keys]

    A, F, V, y, g = [], [], [], [], []
    for i, r in enumerate(rows):
        a = aud.get(r["filename"], {})
        a = {} if "error" in a else a
        A.append(val(a, AUDIO_FEATURES))
        F.append(val(fur.get(r["filename"], {}), FURN_FEATURES))
        V.append(val(r, VIS_FEATURES))
        y.append(1 if lab[i] == "ad" else 0)
        g.append(seg_of[i])
    A, F, V = np.array(A), np.array(F), np.array(V)
    y, g = np.array(y), np.array(g)

    sets = {
        "audio": _trailing(A),
        "furniture": _trailing(np.hstack([F, V])),
        "audio+furniture": _trailing(np.hstack([A, F, V])),
    }
    p = {}
    for name, X in sets.items():
        s, _ = blocked_cv(X, y, g)
        p[name] = s

    ev = []
    for i, r in enumerate(rows):
        ev.append(
            {
                "i": i,
                "t": r["t"],
                "y": lab[i],
                "anchor": cv_verdict(r),
                "banner": bool(r["banner"]),
                "bug": bool(r["bug"]),
                "black": bool(r["black_frac"] > 0.90),
                **{f"p_{k}": float(v[i]) for k, v in p.items()},
            }
        )
    return rows, segs, ev


if __name__ == "__main__":
    rows, segs, ev = build_evidence()
    with open(HERE / "evidence.jsonl", "w") as f:
        for e in ev:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {len(ev)} rows")
