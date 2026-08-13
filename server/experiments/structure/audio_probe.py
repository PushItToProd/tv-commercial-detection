"""How much of the ad/content decision is carried by the audio alone?

Two things make a naive answer here badly wrong, and both are handled below.

**Overlap.** Clips are 4 s long and arrive every 2 s, so consecutive rows share
half their samples. A random train/test split therefore tests on audio it has
already trained on. Every split here is *by segment*: whole ad breaks and whole
content runs go to one fold, so nothing in a test fold overlaps anything in
training. On this data random splitting inflates accuracy by ~8 points, which is
the entire effect being measured.

**Scale.** A single 4 s clip is a poor unit. A break lasts at least 118 s, so the
question that actually matters is not "does this clip sound like an ad" but
"has the last half-minute sounded like ads". Both are reported; the trailing
window is worth about 10 points and is strictly causal, so it is usable live.

The model is plain L2 logistic regression on standardised features - the point
is to measure how much signal is present, not to squeeze out the last point,
and coefficients that can be read off matter more here than accuracy.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import build  # noqa: E402
from timeline import cv_verdict, load  # noqa: E402

HERE = Path(__file__).parent

FEATURES = [
    "rms_db",
    "crest",
    "dyn_range",
    "env_p50_db",
    "env_p10_db",
    "silence_frac",
    "min_rms_db",
    "centroid",
    "rolloff",
    "flatness",
    "stationarity",
    "flux_mean",
    "flux_max",
    "zcr",
    "b0_60",
    "b60_150",
    "b150_400",
    "b400_800",
    "b800_2000",
    "b2000_5000",
    "b5000_11025",
]
WIN = 15  # trailing frames, ~30 s


def fit_logreg(X, y, l2=1.0, iters=400, lr=0.5):
    n, d = X.shape
    w = np.zeros(d + 1)
    Xb = np.hstack([X, np.ones((n, 1))])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / n
        g[:d] += l2 * w[:d] / n
        w -= lr * g
    return w


def predict(X, w):
    return 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def blocked_cv(X, y, groups, folds=5):
    uniq = np.unique(groups)
    assign = {g: i % folds for i, g in enumerate(uniq)}
    fold = np.array([assign[g] for g in groups])
    scores = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = fit_logreg((X[tr] - mu) / sd, y[tr])
        scores[te] = predict((X[te] - mu) / sd, w)
    return scores, fold


def random_cv(X, y, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(y))
    scores = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = fit_logreg((X[tr] - mu) / sd, y[tr])
        scores[te] = predict((X[te] - mu) / sd, w)
    return scores


def best_acc(y, s):
    ths = np.unique(np.round(s, 3))
    return max(((s >= t).astype(int) == y).mean() for t in ths)


def main():
    rows = load()
    segs = build(rows)
    aud = {
        r["filename"]: r
        for r in (json.loads(line) for line in open(HERE / "audio.jsonl"))
    }

    seg_of, lab = {}, {}
    for k, s in enumerate(segs):
        for i in range(s["start"], s["end"] + 1):
            seg_of[i], lab[i] = k, s["label"]

    idx, X, y, g = [], [], [], []
    for i, r in enumerate(rows):
        a = aud.get(r["filename"])
        if not a or "error" in a:
            continue
        idx.append(i)
        X.append([a[f] for f in FEATURES])
        y.append(1 if lab[i] == "ad" else 0)
        g.append(seg_of[i])
    idx = np.array(idx)
    X = np.array(X)
    y = np.array(y)
    g = np.array(g)
    print(
        f"{len(y)} clips, {y.sum()} ad / {len(y) - y.sum()} content, "
        f"{len(np.unique(g))} segments"
    )

    # Trailing-window version: causal mean and sd of each feature over WIN frames.
    Xw = np.zeros((len(X), X.shape[1] * 2))
    for k in range(len(X)):
        lo = max(0, k - WIN + 1)
        blk = X[lo : k + 1]
        Xw[k] = np.concatenate([blk.mean(0), blk.std(0)])

    print("\n--- single features, AUC (blocked CV not needed for a rank stat) ---")
    aucs = sorted(
        ((auc(y, X[:, j]), FEATURES[j]) for j in range(X.shape[1])),
        key=lambda t: -abs(t[0] - 0.5),
    )
    for a, n in aucs[:10]:
        print(f"  {n:14s} AUC {a:.3f}")

    print("\n--- model comparison ---")
    for name, M in (("instant (4 s clip)", X), (f"trailing {WIN * 2} s window", Xw)):
        s_b, _ = blocked_cv(M, y, g)
        s_r = random_cv(M, y)
        print(
            f"  {name:24s} blocked AUC {auc(y, s_b):.3f} acc {best_acc(y, s_b):.3f}"
            f"   |  random-split (leaky) AUC {auc(y, s_r):.3f} "
            f"acc {best_acc(y, s_r):.3f}"
        )

    print("\n--- where it matters: frames OpenCV cannot settle ---")
    und = np.array([cv_verdict(rows[i]) is None for i in idx])
    s_b, _ = blocked_cv(Xw, y, g)
    print(
        f"  {und.sum()} undecided clips ({y[und].sum()} ad / {(1 - y[und]).sum()} content)"
    )
    print(
        f"  trailing-window AUC on those {auc(y[und], s_b[und]):.3f}  "
        f"acc {best_acc(y[und], s_b[und]):.3f}"
    )

    print("\n--- and on the two hardest stretches ---")
    for name, lo, hi in (
        ("bug-less pylon racing", 3174, 3466),
        ("full break @2:10", 3903, 3992),
    ):
        m = (idx >= lo) & (idx <= hi)
        pred = (s_b[m] >= 0.5).astype(int)
        print(
            f"  {name:24s} n={m.sum():4d} truth={'ad' if y[m][0] else 'content'}  "
            f"audio says ad on {pred.mean():.0%} of frames"
        )

    np.save(HERE / "audio_scores.npy", np.vstack([idx, s_b]))


if __name__ == "__main__":
    main()
