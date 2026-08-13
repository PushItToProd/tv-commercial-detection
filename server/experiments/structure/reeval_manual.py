#!/usr/bin/env python3
"""Re-score every policy against the operator's own frame-by-frame rulings.

The labels in `ground_truth.py` were assigned by an agent reading contact
sheets. `review_verdicts.json` holds a full manual pass over the same 4775
frames, so every number in `notes/broadcast-structure-2026-08.md` can be
re-derived against labels a human actually stands behind.

Two questions, and the second is the reason this exists:

1. Do the conclusions survive the label change at all?
2. Are they *load-bearing on the choices still open* - specifically how `other`
   is scored, and how `ad_shown` is weighted against `race_missed`?

`other` is the operator's "the binary does not apply here" ruling, and it covers
three unrelated things (post-ad-break ad reads, pre-race hype, transitions) that
pull in opposite directions. There is no single correct treatment, so all three
are run: excluded from scoring, folded to `ad`, folded to `content`.

`--weights` scores the same runs under an explicit cost asymmetry. The published
table reports `ad_shown` and `race_missed` side by side and declines to pick a
winner, which silently weights them equally; the operator's stated preference is
that leaving a commercial up is much worse than returning to the race late.

Nothing here writes to the committed experiment outputs.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import evidence as ev_mod  # noqa: E402
import ground_truth  # noqa: E402
from evaluate import DT, score, transitions  # noqa: E402
from policies import POLICIES  # noqa: E402
from timeline import load  # noqa: E402

VERDICTS = HERE.parent / "review_verdicts.json"


def manual_labels(rows: list[dict], other: str) -> tuple[list[str], list[bool]]:
    """Per-frame labels from the operator's rulings, plus a scoring mask.

    `other` frames still need *a* label, because the segment list they feed
    defines where the true transitions are, and a third value there would invent
    transitions at both ends of every ad read. Under `exclude` they carry the
    previous ruling forward - the timeline stays binary - and the mask drops them
    from accuracy and from the ad_shown / race_missed accounting.
    """
    store = json.load(open(VERDICTS))["structure"]
    y: list[str] = []
    mask: list[bool] = []
    last = "content"
    for r in rows:
        v = (store.get(r["filename"]) or {}).get("verdict")
        if v in ("ad", "content"):
            y.append(v)
            mask.append(True)
            last = v
        elif v == "other":
            if other == "exclude":
                y.append(last)
                mask.append(False)
            else:
                y.append(other)
                mask.append(True)
                last = other
        else:  # never ruled; fall back to the agent's label
            y.append(last)
            mask.append(False)
    return y, mask


def segments_from(y: list[str], rows: list[dict]) -> list[dict]:
    """Collapse a per-frame label run into the segment shape `build` returns."""
    segs = []
    i = 0
    for label, grp in itertools.groupby(y):
        n = len(list(grp))
        s = {"start": i, "end": i + n - 1, "label": label, "kind": ""}
        s["t0"], s["t1"] = rows[s["start"]]["t"], rows[s["end"]]["t"]
        s["dur"] = s["t1"] - s["t0"] + DT
        s["n"] = n
        segs.append(s)
        i += n
    return segs


def masked_score(y, pred, mask, w_ad: float, w_race: float) -> dict:
    """`score` restricted to the frames being counted, plus a weighted cost.

    Switches and flaps are deliberately measured on the *full* prediction: the
    relay moves whether or not the frame beneath it is being scored.
    """
    r = score(y, pred)
    ya, pa, m = np.array(y), np.array(pred), np.array(mask)
    if m.sum():
        r["acc"] = float((ya[m] == pa[m]).mean())
        r["ad_shown"] = float(((ya == "ad") & (pa == "content") & m).sum()) * DT
        r["race_missed"] = float(((ya == "content") & (pa == "ad") & m).sum()) * DT
    r["cost"] = w_ad * r["ad_shown"] + w_race * r["race_missed"]
    r["scored"] = int(m.sum())
    return r


def run(other: str, w_ad: float, w_race: float, quiet: bool = False) -> dict:
    rows = load()
    y, mask = manual_labels(rows, other)
    segs = segments_from(y, rows)

    # Both evidence and evaluation take their segments from here, so overriding
    # the one function re-runs the whole chain against the manual labels -
    # including the out-of-fold models, which are blocked on these segments.
    ground_truth.build = lambda _rows: segs
    ev_mod.build_segs = lambda _rows: segs
    _, _, ev = ev_mod.build_evidence()

    y, mask = y[: len(ev)], mask[: len(ev)]
    n_tr = len(transitions(y))
    if not quiet:
        n_ad = sum(1 for v in y if v == "ad")
        print(f"  {len(y)} frames · {n_tr} true transitions · "
              f"{n_ad * DT / 60:.1f} min ad ({n_ad / len(y):.1%}) · "
              f"{sum(mask)} frames scored ({sum(mask) / len(mask):.1%})")

    out = {}
    for sname, key in (("opencv only", None), ("opencv+furniture", "p_furniture"),
                       ("opencv+audio", "p_audio"),
                       ("opencv+audio+furniture", "p_audio+furniture")):
        for pname, fn in POLICIES.items():
            out[f"{sname}|{pname}"] = masked_score(y, fn(ev, key), mask, w_ad, w_race)
    return out


def table(res: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    hdr = (f"{'evidence':24s} {'policy':10s} {'acc%':>6} {'sw':>4} {'flap':>5} "
           f"{'ad_shown':>9} {'race_miss':>10} {'cost':>8}")
    print(hdr)
    print("-" * len(hdr))
    for k, r in res.items():
        s, p = k.split("|")
        print(f"{s:24s} {p:10s} {r['acc'] * 100:6.2f} {r['switches']:4d} "
              f"{r['flaps']:5d} {r['ad_shown']:8.0f}s {r['race_missed']:9.0f}s "
              f"{r['cost']:8.0f}")


def ranking(res: dict, by: str) -> list[str]:
    rev = by == "acc"
    return sorted(res, key=lambda k: -res[k][by] if rev else res[k][by])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-ad", type=float, default=3.0,
                    help="cost per second of commercial left on screen")
    ap.add_argument("--w-race", type=float, default=1.0,
                    help="cost per second of racing switched away from")
    ap.add_argument("--out", default=str(HERE / "results_manual.json"))
    args = ap.parse_args()

    runs = {}
    for other in ("exclude", "ad", "content"):
        print(f"\n### other -> {other}")
        runs[other] = run(other, args.w_ad, args.w_race)
        table(runs[other], f"manual labels, other={other}")

    print("\n\n################ is the choice load-bearing? ################")
    print(f"\ncost = {args.w_ad:g}*ad_shown + {args.w_race:g}*race_missed\n")
    for by in ("acc", "cost"):
        print(f"top 5 by {by}:")
        for other in runs:
            top = ranking(runs[other], by)[:5]
            print(f"  other={other:8s} " + " > ".join(t.split("|")[1] for t in top))
        same = len({tuple(ranking(r, by)[:5]) for r in runs.values()}) == 1
        print(f"  -> top-5 identical across all three treatments: {same}\n")

    print("accuracy vs cost disagree on the winner:")
    for other, r in runs.items():
        a, c = ranking(r, "acc")[0], ranking(r, "cost")[0]
        print(f"  other={other:8s} best by acc: {a:38s} best by cost: {c}")

    with open(args.out, "w") as f:
        json.dump({"weights": {"ad": args.w_ad, "race": args.w_race}, "runs": runs}, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
