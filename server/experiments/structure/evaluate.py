"""Score every (evidence set x policy) pair against the verified ground truth.

Frame accuracy is the least interesting number here and is reported mainly for
continuity with the earlier hysteresis work. What the operator actually
experiences is two things:

    ad_shown      seconds of commercial left on screen, because the policy was
                  still saying `content`
    race_missed   seconds of racing switched away from, because the policy was
                  still (or wrongly) saying `ad`

Those are not symmetric in cost and should not be summed. `switches` and `flaps`
matter too - a policy can score well on frames while thrashing the matrix relay.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import build as build_segs  # noqa: E402
from policies import POLICIES  # noqa: E402
from timeline import load  # noqa: E402

HERE = Path(__file__).parent
DT = 2.0
FLAP_FRAMES = 4  # a switch reversed inside 8 s
EDGE = 5  # frames either side of a true boundary, excluded from "steady"


def transitions(y):
    return [i for i in range(1, len(y)) if y[i] != y[i - 1]]


def score(y, pred):
    y = np.array(y)
    p = np.array(pred)
    acc = (y == p).mean()

    tr = transitions(list(y))
    steady = np.ones(len(y), bool)
    for t in tr:
        steady[max(0, t - EDGE) : t + EDGE + 1] = False
    steady_acc = (y[steady] == p[steady]).mean()

    sw = transitions(list(p))
    flaps = sum(1 for a, b in zip(sw, sw[1:]) if b - a <= FLAP_FRAMES)

    ad_shown = int(((y == "ad") & (p == "content")).sum()) * DT
    race_missed = int(((y == "content") & (p == "ad")).sum()) * DT

    lat = []
    for t in tr:
        hit = next((k for k in range(t, min(len(y), t + 60)) if p[k] == y[t]), None)
        lat.append((hit - t) if hit is not None else 60)
    return {
        "acc": acc,
        "steady": steady_acc,
        "switches": len(sw),
        "flaps": flaps,
        "ad_shown": ad_shown,
        "race_missed": race_missed,
        "lat_med": float(np.median(lat)),
        "lat_max": int(max(lat)),
        "missed_tr": int(sum(1 for x in lat if x >= 60)),
    }


def main():
    rows = load()
    segs = build_segs(rows)
    ev = [json.loads(line) for line in open(HERE / "evidence.jsonl")]
    y = []
    for s in segs:
        y.extend([s["label"]] * (s["end"] - s["start"] + 1))
    y = y[: len(ev)]
    n_tr = len(transitions(y))
    print(
        f"{len(y)} frames, {n_tr} true transitions, "
        f"{sum(1 for v in y if v == 'ad') * DT / 60:.1f} min of ad\n"
    )

    sets = [
        ("opencv only", None),
        ("opencv+furniture", "p_furniture"),
        ("opencv+audio", "p_audio"),
        ("opencv+audio+furniture", "p_audio+furniture"),
    ]

    hdr = (
        f"{'evidence':24s} {'policy':10s} {'acc%':>6} {'steady%':>8} {'sw':>4} "
        f"{'flap':>5} {'ad_shown':>9} {'race_miss':>10} {'lat_med':>8} {'miss':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for sname, key in sets:
        for pname, fn in POLICIES.items():
            pred = fn(ev, key)
            r = score(y, pred)
            results[f"{sname}|{pname}"] = r
            print(
                f"{sname:24s} {pname:10s} {r['acc'] * 100:6.2f} "
                f"{r['steady'] * 100:8.2f} {r['switches']:4d} {r['flaps']:5d} "
                f"{r['ad_shown']:8.0f}s {r['race_missed']:9.0f}s "
                f"{r['lat_med'] * DT:7.0f}s {r['missed_tr']:5d}"
            )
        print()
    print(f"(true switch count is {n_tr}; 'miss' counts transitions never followed)")

    with open(HERE / "results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
