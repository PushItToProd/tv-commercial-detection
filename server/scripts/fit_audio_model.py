"""Fit, and sanity-check, the audio model a classifier profile consults.

The model is L2 logistic regression over one trailing block of the DSP features
in `audio_features.py` — see `audio_sensor.py` for how it's used live and
`notes/broadcast-structure-2026-08.md` for why audio is worth asking at all.

Everything here replays a continuous `record_broadcast.py` recording through the
same `AudioWindow` the live sensor uses, keyed on `video_offset`, so a model is
fit and checked on exactly the input it will see in production — including which
frames it abstains on while the window is still filling.

fit
    Trains on a recording plus the operator's rulings from the ground-truth
    review app. Frames ruled `other` stay in the timeline — they are still audio
    their neighbors' windows hear — but are not training rows.

    The confidence band is chosen from *segment-blocked out-of-fold* scores.
    Clips are 4 s long and arrive every 2 s, so neighbors share half their
    samples; a random split, or in-sample scores, would tune the band on audio
    the model has already heard and set it far too loose. The content side takes
    a stricter precision target than the ad side because a false `content`
    leaves a commercial on screen, which costs more than returning to the race a
    little late.

    The report breaks errors out on the frames the profile's OpenCV checks
    leave undecided, since those are the only ones the audio verdict can reach,
    and separately in the first 20 s after each break onset and each rejoin:
    network bumpers sound like the show going into a break, and sponsor reads
    sound like ads coming out of one.

check
    Scores a model against a recording it wasn't fit on, using the profile's
    OpenCV verdicts as proxy labels. Those are the easy frames by construction —
    a logo was plainly visible — so a good number here is encouraging rather
    than conclusive, and a bad one is decisive.

Features and OpenCV verdicts are cached per recording, so re-runs take seconds.

Usage:
    B=/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com
    uv run python scripts/fit_audio_model.py fit --dir $B/USA_4K_Iowa_Corn_350 \\
        --verdicts experiments/review_verdicts.json \\
        --out src/tv_commercial_detector/classifiers/audio_models/nascar_on_nbc.json
    uv run python scripts/fit_audio_model.py check --dir $B/USA_4K_Cook_Out_400 \\
        --model src/tv_commercial_detector/classifiers/audio_models/nascar_on_nbc.json
"""

import argparse
import importlib
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tv_commercial_detector.audio_features import (  # noqa: E402
    FEATURES,
    feature_vector,
)
from tv_commercial_detector.audio_sensor import (  # noqa: E402
    BLOCK_LAYOUT,
    MODEL_VERSION,
    AudioWindow,
    load_model,
)

# Matches the ~15 clips at a 2 s cadence the notes measured, with the sensor
# abstaining until at least 12 clips span 24 s.
WINDOW_SECONDS = 30.0
MIN_CLIPS = 12
MIN_SPAN_SECONDS = 24.0

EDGE_SECONDS = 20.0


# --- loading ----------------------------------------------------------------


def load_rows(broadcast: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in open(broadcast / "classifications.jsonl")
        if line.strip()
    ]
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def _cached(cache: Path, rows: list[dict], compute, workers: int) -> dict:
    """{filename: value}, computing only what *cache* doesn't already hold."""
    have = {}
    if cache.exists():
        for line in open(cache):
            rec = json.loads(line)
            have[rec["filename"]] = rec["value"]
    todo = [r["filename"] for r in rows if r["filename"] not in have]
    if todo:
        print(f"  computing {len(todo)} of {len(rows)} -> {cache}", file=sys.stderr)
        # Threads, not processes: the numpy/scipy/OpenCV work releases the GIL,
        # and a process pool needs a socket the sandbox won't grant.
        with ThreadPoolExecutor(max_workers=workers) as ex, open(cache, "a") as f:
            for k, (name, value) in enumerate(zip(todo, ex.map(compute, todo))):
                have[name] = value
                f.write(json.dumps({"filename": name, "value": value}) + "\n")
                if k % 1000 == 0:
                    print(f"    {k}/{len(todo)}", file=sys.stderr, flush=True)
    return have


def load_features(broadcast: Path, rows, cache_dir: Path, workers: int) -> dict:
    def compute(filename: str):
        wav = broadcast / "audio" / (Path(filename).stem + ".wav")
        if not wav.exists():
            return None
        vec = feature_vector(wav.read_bytes())
        return None if vec is None else vec.tolist()

    cache = cache_dir / f"audio_features_{broadcast.name}.jsonl"
    return _cached(cache, rows, compute, workers)


def load_anchors(broadcast: Path, rows, cache_dir: Path, workers: int, profile: str):
    """The profile's OpenCV verdict per frame: `ad`, `content`, or None."""
    mod = importlib.import_module(f"tv_commercial_detector.classifiers.{profile}")

    def compute(filename: str):
        img = cv2.imread(str(broadcast / "images" / filename))
        if img is None:
            return None
        img = cv2.resize(img, (1920, 1080))
        if mod.has_side_by_side_logo(img):
            return "ad"
        if mod.has_network_logo(img):
            return "content"
        return None

    cache = cache_dir / f"opencv_anchors_{profile}_{broadcast.name}.jsonl"
    return _cached(cache, rows, compute, workers)


def replay(rows, feats, window_seconds, min_clips, min_span_seconds):
    """Model input per frame as the live sensor would build it; None = abstain."""
    window = AudioWindow()
    blocks: list[np.ndarray | None] = []
    for r in rows:
        vec = feats.get(r["filename"])
        if r.get("is_seeking"):
            window.reset()
        if vec is None:
            blocks.append(None)
            continue
        window.add(float(r["video_offset"]), np.array(vec))
        block, _, _ = window.block(window_seconds, min_clips, min_span_seconds)
        blocks.append(block)
    return blocks


# --- model ------------------------------------------------------------------


def fit_logreg(X, y, l2=1.0, iters=400, lr=0.5):
    """The same gradient descent `experiments/structure/audio_probe.py` measured."""
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


def standardizer(X):
    return X.mean(0), X.std(0) + 1e-9


def auc(y, s):
    pos, neg = y.sum(), (1 - y).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    return (r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)


def blocked_cv(X, y, groups, folds=5):
    uniq = np.unique(groups)
    assign = {g: i % folds for i, g in enumerate(uniq)}
    fold = np.array([assign[g] for g in groups])
    scores = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = standardizer(X[tr])
        w = fit_logreg((X[tr] - mu) / sd, y[tr])
        scores[te] = predict((X[te] - mu) / sd, w)
    return scores


def pick_threshold(y, s, target, side, min_support):
    """The widest threshold whose selection holds *target* precision throughout.

    For `ad`, the lowest t such that every t' >= t (with enough frames selected
    to mean anything) keeps precision >= target; mirrored for `content`.
    Requiring it of every stricter threshold, not just t itself, stops the band
    from opening onto a lucky dip in a noisy precision curve.
    """
    if side == "content":
        # p(content) = 1 - p(ad); pick on that scale and map back.
        t = pick_threshold(1 - y, 1 - s, target, "ad", min_support)
        return None if t is None else 1 - t
    order = np.argsort(s)
    ss, yy = s[order], y[order]
    n = len(ss)
    support = n - np.arange(n)
    precision = np.cumsum(yy[::-1])[::-1] / support
    precision = np.where(support >= min_support, precision, np.inf)
    suffix_min = np.minimum.accumulate(precision[::-1])[::-1]
    ok = np.nonzero(suffix_min >= target)[0]
    return float(ss[ok[0]]) if len(ok) else None


def band_report(label, y, s, ad_t, content_t):
    n = len(y)
    is_ad, is_content = s >= ad_t, s <= content_t
    parts = [f"  {label:30s} n={n:5d}"]
    if n == 0:
        print(parts[0])
        return
    parts.append(f"decided {(is_ad | is_content).mean():6.1%}")
    if is_ad.any():
        parts.append(f"ad {is_ad.sum():5d} @ {y[is_ad].mean():6.1%} right")
    if is_content.any():
        parts.append(
            f"content {is_content.sum():5d} @ {(1 - y[is_content]).mean():6.1%} right"
        )
    print("  ".join(parts))


# --- commands ---------------------------------------------------------------


def manual_labels(rows, verdicts_path: Path):
    """Per-frame (label or None, segment id) from the operator's rulings.

    `other` carries the previous ruling forward for segmentation, so a
    sponsor read inside a content run doesn't split it into three segments and
    leak the run across folds — the same rule `reeval_manual.py` scores by.
    Frames past the last ruling get no segment.
    """
    store = json.load(open(verdicts_path))["structure"]
    labels, segs = [], []
    carried, seg = "content", -1
    for r in rows:
        v = (store.get(r["filename"]) or {}).get("verdict")
        if v is None:
            labels.append(None)
            segs.append(None)
            continue
        if v in ("ad", "content"):
            if v != carried or seg < 0:
                seg += 1
            carried = v
        elif seg < 0:
            seg = 0
        labels.append(v if v in ("ad", "content") else None)
        segs.append(seg)
    return labels, segs


def cmd_fit(args):
    broadcast = Path(args.dir)
    rows = load_rows(broadcast)
    labels, segs = manual_labels(rows, Path(args.verdicts))
    ruled = [i for i, s in enumerate(segs) if s is not None]
    rows = rows[: ruled[-1] + 1]
    labels, segs = labels[: len(rows)], segs[: len(rows)]
    print(f"{len(rows)} frames ruled in {broadcast.name}")

    feats = load_features(broadcast, rows, args.cache, args.workers)
    anchors = load_anchors(broadcast, rows, args.cache, args.workers, args.profile)
    blocks = replay(rows, feats, WINDOW_SECONDS, MIN_CLIPS, MIN_SPAN_SECONDS)

    idx = [
        i for i in range(len(rows)) if blocks[i] is not None and labels[i] is not None
    ]
    X = np.vstack([blocks[i] for i in idx])
    y = np.array([labels[i] == "ad" for i in idx], dtype=float)
    g = np.array([segs[i] for i in idx])
    unanchored = np.array([anchors.get(rows[i]["filename"]) is None for i in idx])
    print(
        f"{len(idx)} training rows ({int(y.sum())} ad), {len(np.unique(g))} segments; "
        f"{sum(b is None for b in blocks)} frames abstained (cold or no clip), "
        f"{sum(label is None for label in labels)} ruled `other`"
    )

    oof = blocked_cv(X, y, g)
    print(f"\nsegment-blocked out-of-fold AUC {auc(y, oof):.3f}")
    print(
        f"  on the {unanchored.sum()} frames OpenCV leaves undecided: "
        f"{auc(y[unanchored], oof[unanchored]):.3f}"
    )

    # Chosen on the undecided frames alone: they're the only ones an audio
    # verdict ever reaches, and they're harder than the anchored ones — a band
    # tuned over every frame holds its target overall and misses it here.
    yb, sb = y[unanchored], oof[unanchored]
    ad_t = pick_threshold(yb, sb, args.ad_precision, "ad", args.min_support)
    content_t = pick_threshold(
        yb, sb, args.content_precision, "content", args.min_support
    )
    if ad_t is None or content_t is None or not content_t < ad_t:
        sys.exit(f"no usable band: ad_threshold={ad_t} content_threshold={content_t}")
    print(
        f"\nband: ad >= {ad_t:.4f} (target {args.ad_precision:.1%}), "
        f"content <= {content_t:.4f} (target {args.content_precision:.1%}), "
        "out-of-fold on OpenCV-undecided frames"
    )
    band_report("all frames", y, oof, ad_t, content_t)
    band_report("OpenCV undecided", y[unanchored], oof[unanchored], ad_t, content_t)

    # Confident-wrong calls the audio verdict could actually make, near edges.
    pos = {i: k for k, i in enumerate(idx)}
    t = [float(r["video_offset"]) for r in rows]
    onset_wrong = rejoin_wrong = onset_n = rejoin_n = 0
    prev = None
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        if prev is not None and lab != prev:
            j = i
            while j < len(rows) and t[j] - t[i] < EDGE_SECONDS:
                k = pos.get(j)
                if k is not None and unanchored[k] and labels[j] == lab:
                    if lab == "ad":
                        onset_n += 1
                        onset_wrong += oof[k] <= content_t
                    else:
                        rejoin_n += 1
                        rejoin_wrong += oof[k] >= ad_t
                j += 1
        prev = lab
    print(
        f"\nfirst {EDGE_SECONDS:.0f} s after an edge, OpenCV undecided:"
        f"\n  break onsets: {onset_wrong} of {onset_n} frames confidently called content"
        f"\n  rejoins:      {rejoin_wrong} of {rejoin_n} frames confidently called ad"
    )

    mu, sd = standardizer(X)
    w = fit_logreg((X - mu) / sd, y)
    decided = (oof >= ad_t) | (oof <= content_t)
    model = {
        "version": MODEL_VERSION,
        "features": FEATURES,
        "block": BLOCK_LAYOUT,
        "window_seconds": WINDOW_SECONDS,
        "min_clips": MIN_CLIPS,
        "min_span_seconds": MIN_SPAN_SECONDS,
        "mu": mu.tolist(),
        "sd": sd.tolist(),
        "w": w[:-1].tolist(),
        "b": float(w[-1]),
        "ad_threshold": ad_t,
        "content_threshold": content_t,
        "trained_on": {
            "broadcast": broadcast.name,
            "video_title": rows[0].get("video_title"),
            "network": rows[0].get("network_name"),
            "frames": len(idx),
            "labels": Path(args.verdicts).name,
        },
        "metrics": {
            "blocked_cv_auc": round(float(auc(y, oof)), 4),
            "blocked_cv_auc_opencv_undecided": round(
                float(auc(y[unanchored], oof[unanchored])), 4
            ),
            "band_coverage": round(float(decided.mean()), 4),
            "band_coverage_opencv_undecided": round(
                float(decided[unanchored].mean()), 4
            ),
            "ad_precision_target": args.ad_precision,
            "content_precision_target": args.content_precision,
            "note": "out-of-fold within one broadcast; not a cross-broadcast estimate",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, indent=1) + "\n")
    print(f"\nwrote {out}")


def cmd_check(args):
    broadcast = Path(args.dir)
    model = load_model(Path(args.model))
    if model is None:
        sys.exit(f"can't load {args.model}")
    rows = load_rows(broadcast)
    print(f"{len(rows)} frames in {broadcast.name}")
    feats = load_features(broadcast, rows, args.cache, args.workers)
    anchors = load_anchors(broadcast, rows, args.cache, args.workers, args.profile)
    blocks = replay(
        rows, feats, model.window_seconds, model.min_clips, model.min_span_seconds
    )

    scores = [None if b is None else model.p_ad(b) for b in blocks]
    ad_t, content_t = model.ad_threshold, model.content_threshold
    anchored = [
        (anchors[r["filename"]] == "ad", s)
        for r, s in zip(rows, scores)
        if s is not None and anchors.get(r["filename"]) is not None
    ]
    y = np.array([a for a, _ in anchored], dtype=float)
    s = np.array([s for _, s in anchored])
    print(
        f"\nproxy labels from {args.profile}'s OpenCV checks: "
        f"{len(y)} scored frames ({int(y.sum())} banner=ad, {int(len(y) - y.sum())} bug=content)"
    )
    print(f"  AUC {auc(y, s):.3f}")
    band_report("anchored frames", y, s, ad_t, content_t)

    free = [s for r, s in zip(rows, scores) if anchors.get(r["filename"]) is None]
    n_cold = sum(s is None for s in free)
    warm = np.array([s for s in free if s is not None])
    print(
        f"\n{len(free)} frames OpenCV leaves undecided: {n_cold} abstained, "
        f"{(warm >= ad_t).sum()} audio ad, {(warm <= content_t).sum()} audio content, "
        f"{((warm > content_t) & (warm < ad_t)).sum()} to the LLM"
    )

    if args.dump:
        with open(args.dump, "w") as f:
            for r, sc in zip(rows, scores):
                f.write(
                    json.dumps(
                        {
                            "filename": r["filename"],
                            "video_offset": r["video_offset"],
                            "anchor": anchors.get(r["filename"]),
                            "p_audio": sc,
                        }
                    )
                    + "\n"
                )
        print(f"wrote per-frame scores to {args.dump}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--profile", default="nascar_on_nbc", help="whose OpenCV checks to use"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit")
    fit.add_argument("--dir", required=True, help="record_broadcast.py recording")
    fit.add_argument("--verdicts", required=True, help="review_verdicts.json")
    fit.add_argument("--out", required=True)
    fit.add_argument("--ad-precision", type=float, default=0.98)
    fit.add_argument("--content-precision", type=float, default=0.99)
    fit.add_argument("--min-support", type=int, default=50)
    fit.set_defaults(func=cmd_fit)

    check = sub.add_parser("check")
    check.add_argument("--dir", required=True, help="record_broadcast.py recording")
    check.add_argument("--model", required=True)
    check.add_argument("--dump", help="write per-frame scores as jsonl")
    check.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
